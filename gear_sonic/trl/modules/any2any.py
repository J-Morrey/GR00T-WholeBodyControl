"""Any2Any cross-embodiment transfer: kinematic alignment + LoRA dynamics adaptation.

Implements the two stages of "Any2Any: Efficient Cross-Embodiment Transfer for
Humanoid Whole-Body Tracking" so a WBT policy pretrained on one humanoid can be
reused on another:

1. **Kinematic alignment** -- a fixed, non-learned map that rearranges the target
   robot's joint-space observations into the source robot's joint layout, and
   maps the source policy's action back to the target's actuated joints. This is
   the paper's ``Phi = J * D^-1 * S``.
2. **Dynamics adaptation** -- LoRA on only the dynamics-sensitive modules
   (the action decoder and the critic), everything else frozen.

For the Unitree G1 (29 DOF, source) -> Unitree R1 (26 DOF, target) pair that this
module targets, ``Phi`` collapses to the sparse scattering matrix ``S`` alone:

* Both robots have orthogonal hip axes ((0,1,0), (1,0,0), (0,0,1)), so the paper's
  inclined-hip decoupling ``D`` is the identity.
* R1's MJCF/URDF is a serialized model (its parallel ankle linkage is resolved
  below the policy), so the closed-chain Jacobian correction ``J`` is the identity.

24 of R1's 26 joints have a G1 counterpart. G1's ``waist_pitch`` and its four
wrist pitch/yaw joints have no R1 counterpart (zero-padded rows); R1's two head
joints have no G1 counterpart (dropped columns -- the head holds its default pose).

The alignment is what makes the pretrained checkpoint load with *zero* shape
mismatch: it turns R1's 840/1495-dim observations into G1's 930/1645, and the
policy's 29-dim action back into 26.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# =============================================================================
# Joint-name ground truth
# =============================================================================
# The integer maps below are always derived from joint *names*; nothing here
# hardcodes an index. G1 in particular has an irregularity that makes a naive
# "<link> -> <joint>" string rule wrong: its ``torso_link`` is driven by
# ``waist_pitch_joint``. So the link -> joint relation is read from the MJCF.

_ROBOTS_DIR = Path(__file__).resolve().parents[2] / "envs" / "manager_env" / "robots"
_MJCF_DIR = Path(__file__).resolve().parents[2] / "data" / "assets" / "robot_description" / "mjcf"

# The pair is read from the environment so a *control* transfer can be run
# without touching the code: setting both to the same robot makes every map
# below the identity, which isolates the Stage-2 LoRA/freeze machinery from the
# Stage-1 alignment. Defaults reproduce the G1 -> R1 configuration.
SOURCE_ROBOT = os.environ.get("SONIC_ANY2ANY_SOURCE", "g1")
TARGET_ROBOT = os.environ.get("SONIC_ANY2ANY_TARGET", "r1")

_SPEC = {
    "g1": ("g1.py", "G1_ISAACLAB_JOINTS", "g1_29dof_rev_1_0.xml"),
    "r1": ("r1.py", "R1_ISAACLAB_JOINTS", "r1.xml"),
}


def _read_isaaclab_bodies(robot: str) -> list[str]:
    """Read a robot's IsaacLab body-order list from its config module.

    Imports the module when possible; otherwise falls back to parsing the
    literal out of the source. The fallback exists so this module is testable
    outside a running Isaac Sim app, where ``import isaaclab`` fails on ``pxr``.
    """
    filename, var, _ = _SPEC[robot]
    try:
        import importlib

        mod = importlib.import_module(f"gear_sonic.envs.manager_env.robots.{robot}")
        return list(getattr(mod, var))
    except Exception:  # noqa: BLE001 - isaaclab unavailable outside the sim app
        tree = ast.parse((_ROBOTS_DIR / filename).read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == var:
                return list(ast.literal_eval(node.value))
        raise RuntimeError(f"{var} not found in {filename}") from None


def _body_to_joint(mjcf_name: str) -> dict[str, str]:
    """Map each body to the single joint that drives it, read from the MJCF.

    Parses the XML directly rather than going through ``mujoco``: importing that
    package inside a running Isaac Sim process pulls in ``glfw`` -> ``cffi``,
    which collides with the ``cffi`` Isaac bundles and raises a version mismatch.
    """
    from lxml import etree

    parser = etree.XMLParser(remove_comments=True, recover=True)
    worldbody = etree.parse(str(_MJCF_DIR / mjcf_name), parser).getroot().find("worldbody")
    out = {}
    for body in worldbody.iter("body"):
        joints = [j for j in body.findall("joint") if j.get("type") != "free"]
        if len(joints) == 1 and joints[0].get("name"):
            out[body.get("name")] = joints[0].get("name")
    return out


def isaaclab_dof_names(robot: str) -> list[str]:
    """Return a robot's actuated joint names in IsaacLab DOF order."""
    bodies = _read_isaaclab_bodies(robot)
    b2j = _body_to_joint(_SPEC[robot][2])
    return [b2j[b] for b in bodies[1:]]


def _canonical(joint_name: str) -> str:
    """Strip the ``_joint`` suffix so the two robots' names can be compared."""
    return joint_name.removesuffix("_joint")


def build_joint_maps() -> tuple[list[str], list[str], list[int], list[int]]:
    """Build the source/target DOF name lists and the two index maps.

    Returns:
        (source_names, target_names, src_from_tgt, tgt_from_src) where
        ``src_from_tgt[i]`` is the target DOF filling source slot ``i`` (-1 if the
        source joint has no counterpart), and ``tgt_from_src[j]`` is the source
        DOF driving target slot ``j`` (-1 if the target joint is surplus).
    """
    src = isaaclab_dof_names(SOURCE_ROBOT)
    tgt = isaaclab_dof_names(TARGET_ROBOT)
    tgt_lookup = {_canonical(n): j for j, n in enumerate(tgt)}
    src_lookup = {_canonical(n): i for i, n in enumerate(src)}
    src_from_tgt = [tgt_lookup.get(_canonical(n), -1) for n in src]
    tgt_from_src = [src_lookup.get(_canonical(n), -1) for n in tgt]
    return src, tgt, src_from_tgt, tgt_from_src


SOURCE_DOF_NAMES, TARGET_DOF_NAMES, SRC_FROM_TGT, TGT_FROM_SRC = build_joint_maps()
NUM_SOURCE_DOF = len(SOURCE_DOF_NAMES)
NUM_TARGET_DOF = len(TARGET_DOF_NAMES)
#: Source DOF slots with no target counterpart; padded, never driven by the robot.
SOURCE_ONLY_DOF = [i for i, t in enumerate(SRC_FROM_TGT) if t < 0]
#: Target DOF slots with no source counterpart; hold their default pose.
TARGET_ONLY_DOF = [j for j, s in enumerate(TGT_FROM_SRC) if s < 0]
MATCHED_PAIRS = [(j, i) for i, j in enumerate(SRC_FROM_TGT) if j >= 0]  # (target, source)


# =============================================================================
# Scatter / gather primitives
# =============================================================================
def scatter_dof(x: torch.Tensor, src_from_tgt: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """Scatter target-robot joint values into the source robot's joint layout.

    Args:
        x: ``(..., num_target_dof)``.
        src_from_tgt: ``(num_source_dof,)`` long tensor; -1 marks a padded slot.
        fill: value written into padded slots.

    Returns:
        ``(..., num_source_dof)``.
    """
    idx = src_from_tgt.clamp(min=0).to(x.device)
    out = x.index_select(-1, idx)
    pad = (src_from_tgt < 0).to(x.device)
    if pad.any():
        out = out.masked_fill(pad.expand_as(out), fill)
    return out


def gather_dof(a: torch.Tensor, tgt_from_src: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """Gather a source-layout action back onto the target robot's joints.

    Inverse of :func:`scatter_dof` on the matched joints. Target joints with no
    source counterpart receive ``fill`` (0 = hold default pose).
    """
    idx = tgt_from_src.clamp(min=0).to(a.device)
    out = a.index_select(-1, idx)
    drop = (tgt_from_src < 0).to(a.device)
    if drop.any():
        out = out.masked_fill(drop.expand_as(out), fill)
    return out


# =============================================================================
# Observation-group layouts
# =============================================================================
@dataclass(frozen=True)
class TermSpec:
    """One observation term's slice within a concatenated group tensor."""

    name: str
    start: int
    end: int
    dims: tuple[int, ...]

    @property
    def size(self) -> int:
        return self.end - self.start


def build_layout(names, dims_map) -> tuple[TermSpec, ...]:
    """Build ordered term slices from a group's term names and per-term dims."""
    specs, offset = [], 0
    for name in names:
        dims = tuple(dims_map[name])
        flat = int(np.prod(dims)) if dims else 1
        specs.append(TermSpec(name, offset, offset + flat, dims))
        offset += flat
    return tuple(specs)


def layout_from_observation_manager(observation_manager, group: str) -> tuple[TermSpec, ...]:
    """Read a group's term layout straight from the IsaacLab ObservationManager.

    ``env.config["obs"]["group_obs_names"]`` is only populated for non-concatenated
    groups, so the policy and critic layouts have to come from here.
    """
    names = list(observation_manager.active_terms[group])
    dims = [tuple(d) for d in observation_manager.group_obs_term_dim[group]]
    return build_layout(names, dict(zip(names, dims, strict=True)))


#: Terms whose last axis is one value per DOF, stacked over history frames.
_PER_DOF_HISTORY_TERMS = ("joint_pos", "joint_vel", "actions")
#: Terms laid out as ``[pos frames | vel frames]``, each frame one value per DOF.
_POS_VEL_TERMS = ("command_multi_future", "command_multi_future_nonflat")


def _align_pos_vel_block(x: torch.Tensor, src_from_tgt: torch.Tensor, fill: float) -> torch.Tensor:
    """Align a ``[pos frames | vel frames]`` reference block.

    The flat vector is ``[p_f0 ... p_fF | v_f0 ... v_fF]`` (``commands.py`` builds
    it with ``cat([joint_pos_multi_future, joint_vel_multi_future])``). It must be
    viewed as ``(..., 2, F, D)``. Viewing it as ``(..., F, 2, D)`` -- which is what
    the ``_nonflat`` reshape in ``observations.py`` superficially suggests --
    silently interleaves position and velocity into garbage.
    """
    lead = x.shape[:-1]
    flat = x.shape[-1]
    d = NUM_TARGET_DOF
    assert flat % (2 * d) == 0, f"{flat} not a multiple of 2*{d}"
    frames = flat // (2 * d)
    y = x.reshape(*lead, 2, frames, d)
    y = scatter_dof(y, src_from_tgt, fill)
    return y.reshape(*lead, 2 * frames * NUM_SOURCE_DOF)


def _align_history_block(x: torch.Tensor, src_from_tgt: torch.Tensor, fill: float) -> torch.Tensor:
    """Align a history-stacked per-DOF term.

    IsaacLab flattens history time-major -- ``CircularBuffer.buffer`` is
    ``(batch, max_len, dim)`` and is reshaped to ``(batch, -1)`` -- so viewing as
    ``(..., H, D)`` recovers the frames.
    """
    lead = x.shape[:-1]
    flat = x.shape[-1]
    assert flat % NUM_TARGET_DOF == 0, f"{flat} not a multiple of {NUM_TARGET_DOF}"
    hist = flat // NUM_TARGET_DOF
    y = x.reshape(*lead, hist, NUM_TARGET_DOF)
    y = scatter_dof(y, src_from_tgt, fill)
    return y.reshape(*lead, hist * NUM_SOURCE_DOF)


def align_concat_group(
    x: torch.Tensor,
    layout: tuple[TermSpec, ...],
    src_from_tgt: torch.Tensor,
    fill: float = 0.0,
) -> torch.Tensor:
    """Align every DOF-shaped term inside a concatenated observation group.

    Non-DOF terms (gravity, base velocities, tracked-body positions/orientations,
    VR points) pass through untouched -- both robots track the same 14 bodies, so
    those are already semantically 1:1.
    """
    pieces = []
    for spec in layout:
        chunk = x[..., spec.start : spec.end]
        if spec.name in _PER_DOF_HISTORY_TERMS:
            chunk = _align_history_block(chunk, src_from_tgt, fill)
        elif spec.name in _POS_VEL_TERMS:
            chunk = _align_pos_vel_block(chunk, src_from_tgt, fill)
        pieces.append(chunk)
    return torch.cat(pieces, dim=-1)


def aligned_group_dim(layout: tuple[TermSpec, ...]) -> int:
    """Total width of a group after alignment."""
    total = 0
    for spec in layout:
        if spec.name in _PER_DOF_HISTORY_TERMS:
            total += spec.size // NUM_TARGET_DOF * NUM_SOURCE_DOF
        elif spec.name in _POS_VEL_TERMS:
            total += spec.size // NUM_TARGET_DOF * NUM_SOURCE_DOF
        else:
            total += spec.size
    return total


def aligned_term_spans(layout: tuple[TermSpec, ...]) -> dict[str, tuple[int, int]]:
    """Map term name -> (start, n_blocks) inside the *aligned* group vector.

    Each block is ``NUM_SOURCE_DOF`` wide for DOF-shaped terms, so a per-joint
    correction can be broadcast across a term's history/future frames.
    """
    spans, offset = {}, 0
    for spec in layout:
        if spec.name in _PER_DOF_HISTORY_TERMS or spec.name in _POS_VEL_TERMS:
            blocks = spec.size // NUM_TARGET_DOF
            spans[spec.name] = (offset, blocks)
            offset += blocks * NUM_SOURCE_DOF
        else:
            offset += spec.size
    return spans


def add_pose_offset(
    x: torch.Tensor,
    spans: dict[str, tuple[int, int]],
    joint_pos_offset: torch.Tensor,
    action_offset: torch.Tensor,
) -> torch.Tensor:
    """Shift the aligned obs into the source robot's default-pose frame.

    ``joint_pos`` and ``actions`` are the only affine-dependent channels:
    ``joint_vel`` is relative to a zero default velocity, and the reference
    motion is absolute, so neither needs a shift.
    """
    out = x.clone()
    for name, delta in (("joint_pos", joint_pos_offset), ("actions", action_offset)):
        if name not in spans:
            continue
        start, blocks = spans[name]
        seg = out[..., start : start + blocks * NUM_SOURCE_DOF]
        lead = seg.shape[:-1]
        out[..., start : start + blocks * NUM_SOURCE_DOF] = (
            seg.reshape(*lead, blocks, NUM_SOURCE_DOF) + delta.to(seg.device)
        ).reshape(*lead, blocks * NUM_SOURCE_DOF)
    return out


def padded_columns(layout: tuple[TermSpec, ...], src_from_tgt) -> list[int]:
    """Indices in the *aligned* group vector that correspond to padded DOF slots.

    Used to zero the critic's padded columns after normalization (equivalent to
    mean-filling before it, but without needing the running statistics).
    """
    src_only = {i for i, t in enumerate(list(src_from_tgt)) if t < 0}
    cols, offset = [], 0
    for spec in layout:
        if spec.name in _PER_DOF_HISTORY_TERMS:
            blocks = spec.size // NUM_TARGET_DOF
        elif spec.name in _POS_VEL_TERMS:
            blocks = spec.size // NUM_TARGET_DOF
        else:
            offset += spec.size
            continue
        for b in range(blocks):
            for i in src_only:
                cols.append(offset + b * NUM_SOURCE_DOF + i)
        offset += blocks * NUM_SOURCE_DOF
    return cols


# =============================================================================
# Tokenizer group
# =============================================================================
def _resolve_regex_dict(spec, joint_names, default=0.0):
    """Resolve a regex-keyed IsaacLab config dict to a per-joint list."""
    import re

    out = []
    for joint in joint_names:
        value = default
        for pattern, val in spec.items():
            if re.fullmatch(pattern, joint):
                value = float(val)
                break
        out.append(value)
    return out


def _robot_pose_cfg(robot: str):
    """Return ``(articulation_cfg, action_scale_dict)`` for a robot by name.

    Imported lazily and per robot so this module stays importable outside a
    running Isaac Sim app (the R1/G1 config modules pull in ``isaaclab``).
    """
    if robot == "g1":
        from gear_sonic.envs.manager_env.robots.g1 import (
            G1_CYLINDER_MODEL_12_DEX_CFG,
            G1_MODEL_12_ACTION_SCALE,
        )

        return G1_CYLINDER_MODEL_12_DEX_CFG, G1_MODEL_12_ACTION_SCALE
    if robot == "r1":
        from gear_sonic.envs.manager_env.robots.r1 import R1_ACTION_SCALE_ANY2ANY, R1_CFG

        return R1_CFG, R1_ACTION_SCALE_ANY2ANY
    raise KeyError(f"no pose config registered for robot {robot!r}")


def default_pose_offset():
    """Return (offset_src, action_scale_src): the source/target default-pose gap.

    ``Phi`` is a purely linear map, but the quantities it maps are not in the same
    affine frame on the two robots:

    * proprioception is ``joint_pos_rel`` = ``q - q_default``
    * the action term applies ``q_cmd = q_default + scale * a`` (use_default_offset)
    * the reference motion is **absolute** joint angles

    So the relation the source policy learned is anchored to the *source's* default
    pose, while the target env defines both proprioception and actions against the
    *target's*. R1's standing pose differs from G1's by up to 0.2 rad on
    shoulder_roll, 0.157 on ankle_pitch and 0.121 on knee -- balance-critical
    joints -- which the source policy reads as a pose error that is not physically
    there.

    ``offset_src[i] = q_default_target(matched joint) - q_default_source[i]``, in
    source DOF order, zero on unmatched slots. Applied as:

        joint_pos observation  += offset            (perceive in source frame)
        actions observation    += offset / scale    (its own past action, source frame)
        emitted action         -= offset / scale    (command in target frame)

    Returns source-ordered tensors; the action-side correction is gathered to
    target order by the caller.
    """
    src_cfg, scale_cfg = _robot_pose_cfg(SOURCE_ROBOT)
    tgt_cfg, _ = _robot_pose_cfg(TARGET_ROBOT)

    src_default = _resolve_regex_dict(src_cfg.init_state.joint_pos, SOURCE_DOF_NAMES)
    tgt_default = _resolve_regex_dict(tgt_cfg.init_state.joint_pos, TARGET_DOF_NAMES)
    scale_src = _resolve_regex_dict(scale_cfg, SOURCE_DOF_NAMES, default=1.0)

    offset = [0.0] * NUM_SOURCE_DOF
    for tgt, src in MATCHED_PAIRS:
        offset[src] = tgt_default[tgt] - src_default[src]
    return torch.tensor(offset), torch.tensor(scale_src)


def build_wrist_map(source_joint_idx, target_joint_idx) -> list[int]:
    """Map the target's SMPL wrist-joint selection onto the source's.

    ``joint_pos_multi_future_wrist_for_smpl`` selects a handful of DOF by index.
    G1 selects 6 (both wrists' roll/pitch/yaw); R1 has only 2 (roll). Returns, for
    each source slot, the target slot feeding it, or -1.
    """
    tgt_pos = {t: q for q, t in enumerate(list(target_joint_idx))}
    out = []
    for s in list(source_joint_idx):
        t = SRC_FROM_TGT[s]
        out.append(tgt_pos.get(t, -1) if t >= 0 else -1)
    return out


#: Tokenizer term holding the SMPL wrist-joint selection.
_WRIST_TERM = "joint_pos_multi_future_wrist_for_smpl"


def align_tokenizer_group(
    x: torch.Tensor,
    tgt_layout: tuple[TermSpec, ...],
    src_from_tgt: torch.Tensor,
    wrist_map: torch.Tensor,
) -> torch.Tensor:
    """Align the flat tokenizer group from target layout into source layout.

    Only two of its terms are DOF-shaped; everything else (VR points, SMPL joint
    positions, anchor orientations, encoder index) passes through untouched.
    """
    pieces = []
    for spec in tgt_layout:
        chunk = x[..., spec.start : spec.end]
        if spec.name in _POS_VEL_TERMS:
            chunk = _align_pos_vel_block(chunk, src_from_tgt, 0.0)
        elif spec.name == _WRIST_TERM:
            lead = chunk.shape[:-1]
            n_tgt = spec.dims[-1]
            frames = spec.size // n_tgt
            y = chunk.reshape(*lead, frames, n_tgt)
            idx = wrist_map.clamp(min=0).to(y.device)
            y = y.index_select(-1, idx)
            drop = (wrist_map < 0).to(y.device)
            if drop.any():
                y = y.masked_fill(drop.expand_as(y), 0.0)
            chunk = y.reshape(*lead, frames * wrist_map.numel())
        pieces.append(chunk)
    return torch.cat(pieces, dim=-1)


# =============================================================================
# Config g1-ification
# =============================================================================
def group_term_layout(env_config):
    """Fetch the per-term policy/critic layout, with a pointed error if absent.

    This key is populated at runtime from the live IsaacLab ObservationManager by
    :func:`gear_sonic.utils.obs_utils.populate_env_obs_config`. It is never
    present in a checkpoint's ``config.yaml``, which is saved unresolved and
    before the env exists -- so a missing key means an entry point skipped that
    call, not that the checkpoint is bad. Without this the failure surfaces as a
    bare ``omegaconf ConfigAttributeError`` whose cause is unguessable from the
    traceback, since the writer lives in a different file from the reader.
    """
    if "group_term_layout" not in env_config.obs:
        raise RuntimeError(
            "env_config.obs.group_term_layout is missing. Any2Any needs the live "
            "per-term policy/critic observation layout, which is built from the "
            "IsaacLab ObservationManager by "
            "gear_sonic.utils.obs_utils.populate_env_obs_config(env) and is never "
            "stored in a checkpoint's config.yaml. Call it after creating the env "
            "and before instantiating the actor/critic."
        )
    return env_config.obs.group_term_layout


def g1ify_env_config(env_config, num_source_wrist_slots: int = 6):
    """Return a copy of the target env_config presenting *source* dimensions.

    ``UniversalTokenModule`` and ``BaseModule`` size every Linear from these
    numbers, so handing them a source-shaped config is what makes the pretrained
    checkpoint load with zero mismatch. The env itself is untouched -- it keeps
    producing real 26-DOF observations, which ``forward`` aligns.
    """
    cfg = copy.deepcopy(env_config)
    tok_dims = cfg.obs.group_obs_dims["tokenizer"]
    for name in list(tok_dims.keys()):
        dims = tuple(tok_dims[name])
        if name in _POS_VEL_TERMS:
            # (frames, 2 * target_dof) -> (frames, 2 * source_dof)
            tok_dims[name] = (dims[0], 2 * NUM_SOURCE_DOF)
        elif name == _WRIST_TERM:
            tok_dims[name] = (dims[0], num_source_wrist_slots)

    tok_total = int(sum(int(np.prod(tuple(d))) for d in tok_dims.values()))
    cfg.obs.obs_dims["tokenizer"] = tok_total
    cfg.robot.algo_obs_dim_dict["tokenizer"] = tok_total

    layouts = group_term_layout(cfg)
    for group, key in (("policy", "actor_obs"), ("critic", "critic_obs")):
        layout = build_layout([n for n, _ in layouts[group]], {n: tuple(d) for n, d in layouts[group]})
        aligned = aligned_group_dim(layout)
        cfg.obs.obs_dims[key] = aligned
        cfg.robot.algo_obs_dim_dict[key] = aligned

    cfg.robot.actions_dim = NUM_SOURCE_DOF
    return cfg


def g1ify_obs_dim_dict(obs_dim_dict, src_cfg):
    """Overlay the source-shaped observation widths onto a target obs_dim_dict.

    ``Actor``/``Critic`` resolve ``obs_dim_dict`` from the *target* env_config and
    forward it to the backbone, where it wins over the g1-ified env_config.
    """
    out = copy.deepcopy(obs_dim_dict)
    for key in ("actor_obs", "critic_obs", "tokenizer"):
        if key in out:
            out[key] = src_cfg.robot.algo_obs_dim_dict[key]
    return out


def target_layouts(env_config):
    """Extract the target robot's policy/critic term layouts from env_config."""
    layouts = group_term_layout(env_config)
    return {
        group: build_layout(
            [n for n, _ in layouts[group]], {n: tuple(d) for n, d in layouts[group]}
        )
        for group in ("policy", "critic")
    }


def tokenizer_layout(env_config) -> tuple[TermSpec, ...]:
    """Target-robot tokenizer term layout."""
    return build_layout(
        list(env_config.obs.group_obs_names["tokenizer"]),
        {k: tuple(v) for k, v in env_config.obs.group_obs_dims["tokenizer"].items()},
    )


# =============================================================================
# LoRA
# =============================================================================
class LoRALinear(nn.Linear):
    """``nn.Linear`` with an additive low-rank update ``W + (alpha/r) B A``.

    Subclasses ``nn.Linear`` (rather than wrapping it) so ``state_dict`` keys stay
    ``...weight`` / ``...bias`` and the pretrained checkpoint still loads; the
    adapter only *adds* ``lora_A`` / ``lora_B``. ``B`` is zero-initialised, so a
    freshly injected model is bit-for-bit identical to the original.
    """

    @classmethod
    def wrap(cls, linear: nn.Linear, r: int, alpha: float) -> LoRALinear:
        m = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        m.weight = linear.weight  # share the pretrained tensor, no copy
        m.bias = linear.bias
        m.r = r
        m.scaling = alpha / r
        fac = {"device": linear.weight.device, "dtype": linear.weight.dtype}
        m.lora_A = nn.Parameter(torch.empty(r, linear.in_features, **fac))
        m.lora_B = nn.Parameter(torch.zeros(linear.out_features, r, **fac))
        nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5))
        m.weight.requires_grad_(False)
        if m.bias is not None:
            m.bias.requires_grad_(False)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        return base + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


def inject_lora(
    sequential: nn.Sequential, r: int, alpha: float, skip_endpoints: bool = False
) -> int:
    """Replace ``nn.Linear`` layers in a flat Sequential with LoRA versions.

    ``skip_endpoints`` leaves the first and last Linear untouched. The paper's
    best injection scope (Fig. 7, S7) ticks *Critic Backbone* but not *Critic In.*
    or *Critic Out.* -- those belong to S9, which scored worse. In a plain MLP
    critic the first Linear is the input projection and the last is the value
    head, so skipping them reproduces S7.
    """
    linears = [i for i, m in enumerate(sequential) if isinstance(m, nn.Linear) and not isinstance(m, LoRALinear)]
    if skip_endpoints and len(linears) > 2:
        linears = linears[1:-1]
    for i in linears:
        sequential[i] = LoRALinear.wrap(sequential[i], r, alpha)
    return len(linears)


def apply_any2any_lora(
    policy,
    value_model,
    r: int = 16,
    alpha: float = 32.0,
    train_std: bool = False,
    adapt_critic: bool = True,
    actor_decoder_full: bool = False,
    critic_full: bool = False,
    full_finetune: bool = False,
) -> dict:
    """Inject LoRA into the dynamics-sensitive modules and freeze everything else.

    Scope follows the paper's best ablation (S7) and its Sonic-specific recipe:
    the action decoder ``g1_dyn`` -- whose first Linear *is* the proprioception +
    token input projection and whose last *is* the action output projection -- and
    the critic backbone. The reference-motion encoders, the kinematic decoder
    ``g1_kin`` and the (parameter-free) FSQ quantizer stay frozen, preserving the
    source policy's motion prior.
    """
    if full_finetune:
        # Paper's "Full FT (w align)" baseline: every parameter trainable from the
        # source weights, no LoRA, nothing frozen -- including the reference
        # encoders and g1_kin. The kinematic alignment still applies; only the
        # dynamics-adaptation constraint is removed. This is the widest possible
        # adaptation channel and therefore the last configuration in which
        # transfer could still work.
        for p in policy.parameters():
            p.requires_grad_(True)
        for p in value_model.parameters():
            p.requires_grad_(True)
        if not train_std:
            for attr in ("std", "log_std"):
                q = getattr(policy, attr, None)
                if isinstance(q, nn.Parameter):
                    q.requires_grad_(False)
        trainable = sum(p.numel() for m in (policy, value_model) for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for m in (policy, value_model) for p in m.parameters())
        return {
            "g1_dyn_linears_wrapped": 0,
            "critic_linears_wrapped": 0,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_fraction": trainable / max(total, 1),
        }

    for p in policy.parameters():
        p.requires_grad_(False)

    if actor_decoder_full:
        # Widen the adaptation channel: train the whole action decoder directly
        # instead of a rank-r update. Encoders, g1_kin and FSQ stay frozen, so the
        # reference-encoding path is still the pretrained one.
        for p in policy.actor_module.decoders["g1_dyn"].parameters():
            p.requires_grad_(True)
        n_dyn = 0
    else:
        n_dyn = inject_lora(policy.actor_module.decoders["g1_dyn"].module, r, alpha)

    if critic_full:
        for p in value_model.parameters():
            p.requires_grad_(True)
        n_critic = 0
    elif adapt_critic:
        for p in value_model.parameters():
            p.requires_grad_(False)
        # S7: critic *backbone* only -- not its input projection or value head.
        n_critic = inject_lora(value_model.critic_module.module, r, alpha, skip_endpoints=True)
    else:
        # Fresh critic: train it outright. LoRA exists to protect a behavioural
        # prior, which the *actor* has and the critic does not -- the critic is
        # discarded at deployment and its target (reward-to-go under the target
        # robot's dynamics, for a different policy) has changed completely. A
        # rank-16 update on five middle layers is a severe and unjustified
        # constraint on a function that has to be relearned.
        for p in value_model.parameters():
            p.requires_grad_(True)
        n_critic = 0

    # Exploration std stays FROZEN by default, matching the paper's "freeze
    # everything but LoRA".
    #
    # Leaving it trainable looks harmless but reliably destroys the run: with the
    # backbone frozen, std is almost the only parameter PPO can move, so the
    # entropy bonus (entropy_coef 0.01) inflates it with nothing pushing back --
    # the mean action cannot improve fast enough to earn the return that would
    # justify lower noise. Measured on R1: std climbed 0.05 -> 0.18 over 1000
    # iterations while mean episode length fell 40 -> 9 and reward went 2.55 -> 0.01,
    # after peaking at iteration ~100. Freezing std also makes entropy_coef a
    # no-op here, since a Gaussian's entropy depends only on its std.
    if train_std:
        for attr in ("std", "log_std"):
            p = getattr(policy, attr, None)
            if isinstance(p, nn.Parameter):
                p.requires_grad_(True)

    trainable = sum(p.numel() for m in (policy, value_model) for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for m in (policy, value_model) for p in m.parameters())
    return {
        "g1_dyn_linears_wrapped": n_dyn,
        "critic_linears_wrapped": n_critic,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_fraction": trainable / max(total, 1),
    }


# =============================================================================
# Aligned policy / critic
# =============================================================================
class Any2AnyUniversalTokenModule:
    """Mixin factory namespace -- see :func:`make_any2any_backbone`."""


def _lazy_bases():
    from gear_sonic.trl.modules.actor_critic_modules import Critic
    from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule

    return UniversalTokenModule, Critic


def _build_backbone_cls():
    UniversalTokenModule, _ = _lazy_bases()

    class _Any2AnyBackbone(UniversalTokenModule):
        """Frozen source-embodiment backbone driven by target-embodiment observations.

        Scatters the target robot's joint-space observations into the source
        layout before the pretrained network, and gathers the source policy's
        action back onto the target's actuated joints afterwards.
        """

        def __init__(
            self,
            env_config,
            algo_config,
            *args,
            source_wrist_joint_idx=(23, 24, 25, 26, 27, 28),
            target_wrist_joint_idx=(24, 25),
            correct_default_pose=True,
            obs_dim_dict=None,
            **kwargs,
        ):
            self._tgt_layouts = target_layouts(env_config)
            self._tgt_tokenizer_layout = tokenizer_layout(env_config)
            src_cfg = g1ify_env_config(env_config, num_source_wrist_slots=len(source_wrist_joint_idx))
            # ``Actor.__init__`` resolves obs_dim_dict from the *target* config and
            # passes it down, which would otherwise size the decoder's
            # proprioception input projection at 840 instead of the source's 930.
            if obs_dim_dict is not None:
                obs_dim_dict = g1ify_obs_dim_dict(obs_dim_dict, src_cfg)
            super().__init__(src_cfg, algo_config, *args, obs_dim_dict=obs_dim_dict, **kwargs)
            self.register_buffer("_src_from_tgt", torch.tensor(SRC_FROM_TGT, dtype=torch.long), persistent=False)
            self.register_buffer("_tgt_from_src", torch.tensor(TGT_FROM_SRC, dtype=torch.long), persistent=False)
            self.register_buffer(
                "_wrist_map",
                torch.tensor(build_wrist_map(source_wrist_joint_idx, target_wrist_joint_idx), dtype=torch.long),
                persistent=False,
            )

            # Default-pose frame correction (see `default_pose_offset`).
            self._correct_default_pose = bool(correct_default_pose)
            off_src, scale_src = default_pose_offset()
            if not self._correct_default_pose:
                off_src = torch.zeros_like(off_src)
            self._actor_spans = aligned_term_spans(self._tgt_layouts["policy"])
            self.register_buffer("_pose_off_src", off_src, persistent=False)
            self.register_buffer("_act_off_src", off_src / scale_src, persistent=False)
            # Action side is emitted in target order, so gather the same correction.
            self.register_buffer(
                "_act_off_tgt",
                gather_dof(off_src / scale_src, torch.tensor(TGT_FROM_SRC, dtype=torch.long)),
                persistent=False,
            )

        def align_inputs(self, input_data):
            """Return a new dict with actor_obs and tokenizer in source layout."""
            aligned = dict(input_data)  # never mutate the rollout TensorDict
            actor = align_concat_group(
                input_data["actor_obs"], self._tgt_layouts["policy"], self._src_from_tgt, 0.0
            )
            if self._correct_default_pose:
                actor = add_pose_offset(
                    actor, self._actor_spans, self._pose_off_src, self._act_off_src
                )
            aligned["actor_obs"] = actor
            aligned["tokenizer"] = align_tokenizer_group(
                input_data["tokenizer"], self._tgt_tokenizer_layout, self._src_from_tgt, self._wrist_map
            )
            if os.environ.get("SONIC_FWD_DEBUG"):
                self._dbg_in = {
                    "00_raw_actor_obs": input_data["actor_obs"].detach().clone(),
                    "00_raw_tokenizer": input_data["tokenizer"].detach().clone(),
                    "01_aligned_actor_obs": actor.detach().clone(),
                    "01_aligned_tokenizer": aligned["tokenizer"].detach().clone(),
                }
            return aligned

        def _to_target_action(self, action_src):
            """Gather to target joints and re-anchor to the target's default pose."""
            a = gather_dof(action_src, self._tgt_from_src)
            if self._correct_default_pose:
                a = a - self._act_off_tgt.to(a.device)
            return a

        def forward(self, input_data, *args, **kwargs):
            out = super().forward(self.align_inputs(input_data), *args, **kwargs)
            if isinstance(out, dict):
                return {**out, "action_mean": self._to_target_action(out["action_mean"])}
            return self._to_target_action(out)

        def forward_with_external_tokens(self, *args, **kwargs):
            raise NotImplementedError(
                "Any2Any alignment does not cover the external-token path; it is unused in PPO."
            )

    return _Any2AnyBackbone


def _build_critic_cls():
    _, Critic = _lazy_bases()

    class _Any2AnyCritic(Critic):
        """Critic that aligns target observations before its frozen normalizer.

        Alignment must precede normalization because the pretrained
        ``running_mean_std`` is 1645-dim source-shaped. Its Welford ``count`` is
        ~7e10 in the release checkpoint, so it is effectively frozen at source
        statistics; the padded columns are therefore zeroed *after* normalization,
        which is exactly equivalent to mean-filling before it but needs no access
        to the running statistics.
        """

        def __init__(
            self, env_config, algo_config, backbone, obs_dim_dict=None,
            correct_default_pose=True, **kwargs,
        ):
            self._tgt_layout = target_layouts(env_config)["critic"]
            src_cfg = g1ify_env_config(env_config)
            if obs_dim_dict is None:
                obs_dim_dict = src_cfg.robot.algo_obs_dim_dict
            else:
                obs_dim_dict = g1ify_obs_dim_dict(obs_dim_dict, src_cfg)
            super().__init__(src_cfg, algo_config, backbone, obs_dim_dict, **kwargs)
            self.register_buffer("_src_from_tgt", torch.tensor(SRC_FROM_TGT, dtype=torch.long), persistent=False)
            pad = torch.zeros(src_cfg.robot.algo_obs_dim_dict["critic_obs"], dtype=torch.bool)
            pad[padded_columns(self._tgt_layout, SRC_FROM_TGT)] = True
            self.register_buffer("_pad_mask", pad, persistent=False)

            # Same default-pose frame correction as the actor; the pretrained
            # critic was fitted on source-frame proprioception too.
            self._correct_default_pose = bool(correct_default_pose)
            off_src, scale_src = default_pose_offset()
            if not self._correct_default_pose:
                off_src = torch.zeros_like(off_src)
            self._critic_spans = aligned_term_spans(self._tgt_layout)
            self.register_buffer("_pose_off_src", off_src, persistent=False)
            self.register_buffer("_act_off_src", off_src / scale_src, persistent=False)

        def evaluate(self, obs_dict, **kwargs):
            obs_dict = dict(obs_dict)
            x = align_concat_group(obs_dict["critic_obs"], self._tgt_layout, self._src_from_tgt, 0.0)
            if self._correct_default_pose:
                x = add_pose_offset(x, self._critic_spans, self._pose_off_src, self._act_off_src)
            if self.running_mean_std is not None:
                if self.use_batch_norm:
                    x = self.running_mean_std(x)
                else:
                    with torch.no_grad():
                        x = self.running_mean_std(x)
                x = x.masked_fill(self._pad_mask, 0.0)
            obs_dict["critic_obs"] = x
            return self.critic(obs_dict, **kwargs)

    return _Any2AnyCritic


def Any2AnyBackbone(*args, **kwargs):  # noqa: N802 - hydra _target_ entry point
    """Construct the aligned source-embodiment backbone."""
    return _build_backbone_cls()(*args, **kwargs)


def Any2AnyCritic(*args, **kwargs):  # noqa: N802 - hydra _target_ entry point
    """Construct the aligned critic."""
    return _build_critic_cls()(*args, **kwargs)


# =============================================================================
# Post-training setup
# =============================================================================
def remap_std(
    std_param,
    source_std,
    head_std: float = 1e-3,
    copy_source_std: bool = False,
) -> None:
    """Set the target policy's per-joint exploration std.

    The Gaussian is target-shaped (one entry per actuated target joint), so the
    source's 29-vector cannot be loaded directly.

    ``copy_source_std`` is **off** by default, and that matters. The source
    checkpoint's std is its *converged* exploration level (~0.34 mean for the G1
    release), roughly 7x the ``init_noise_std`` a from-scratch run starts at.
    Transplanting it makes early rollouts on the target robot far jitterier, and
    the jitter-penalty rewards (``action_rate_l2``, ``feet_acc``) grow with the
    square of the noise -- measured at ~12x worse ``feet_acc`` and ~200x worse
    ``action_rate_l2`` than the from-scratch baseline, swamping the genuinely
    better tracking terms. Leaving std at ``init_noise_std`` keeps exploration
    identical to the baseline, so any difference is attributable to the
    transferred prior rather than to a noise-level change.

    Either way the surplus target joints get ``head_std`` so they hold still.
    """
    with torch.no_grad():
        if copy_source_std:
            for tgt, src in MATCHED_PAIRS:
                std_param[tgt] = source_std[src]
        for tgt in TARGET_ONLY_DOF:
            std_param[tgt] = head_std


def setup_any2any(cfg, policy, value_model, device) -> dict:
    """Load the source checkpoint into the aligned model, then adapt it.

    Order matters: construct -> load -> inject LoRA -> freeze, all *before* the
    trainer builds its optimizer over ``requires_grad`` parameters.
    """
    from loguru import logger

    # Released checkpoints pickle a class that moved in TRL 0.28. ppo_trainer
    # installs the compatibility shim at import time, but it has not necessarily
    # been imported yet at this point in startup.
    import gear_sonic.trl.trainer.ppo_trainer  # noqa: F401

    ckpt = torch.load(cfg.path, map_location=device, weights_only=False)
    psd = dict(ckpt[cfg.get("policy_state_dict_key", "policy_state_dict")])
    vsd = dict(ckpt[cfg.get("value_state_dict_key", "value_state_dict")])

    # Two kinds of checkpoint land here:
    #   * a SOURCE checkpoint (the pretrained G1 release) -- no adapters yet, so
    #     load first and inject LoRA afterwards.
    #   * an ANY2ANY checkpoint from a previous run of this pipeline -- it already
    #     contains lora_A/lora_B and a target-shaped `std`, so the adapters must
    #     exist *before* loading or every lora_* key comes back "unexpected".
    # A fresh critic is randomly initialised and fully trainable: skip its
    # checkpoint load, its LoRA, and the normalizer reset. It also takes the
    # target robot's native critic_obs (no g1-ification, no zero-padded columns),
    # since there is no pretrained critic whose input layout must be matched.
    fresh_critic = bool(cfg.get("fresh_critic", False))

    resuming = any("lora_" in k for k in psd) or any("lora_" in k for k in vsd)
    if resuming:
        stats = apply_any2any_lora(
            policy,
            value_model,
            r=cfg.get("lora_rank", 16),
            alpha=cfg.get("lora_alpha", 32.0),
            train_std=cfg.get("train_std", False),
        )
        missing, unexpected = policy.load_state_dict(psd, strict=False)
        v_missing, v_unexpected = value_model.load_state_dict(vsd, strict=False)
        if missing or unexpected or v_missing or v_unexpected:
            raise RuntimeError(
                f"Any2Any resume mismatch -- policy missing={missing[:6]} unexpected={unexpected[:6]}; "
                f"critic missing={v_missing[:6]} unexpected={v_unexpected[:6]}."
            )
        logger.info(
            f"Any2Any: RESUMED from an adapted checkpoint ({len(psd)} policy + {len(vsd)} critic "
            f"tensors, LoRA included); std and normalizer taken from the checkpoint as-is"
        )
        return stats

    # `std` is source-shaped; remap it rather than loading it.
    source_std = psd.pop("std", None)

    missing, unexpected = policy.load_state_dict(psd, strict=False)
    missing = [m for m in missing if m != "std" and "lora_" not in m]
    if missing or unexpected:
        raise RuntimeError(
            f"Any2Any policy load mismatch -- missing={missing[:8]} unexpected={unexpected[:8]}. "
            "The aligned model must match the source checkpoint exactly."
        )
    if fresh_critic:
        logger.info(
            f"Any2Any: loaded {len(psd)} policy tensors, zero mismatch; "
            f"critic left at RANDOM INIT and fully trainable ({len(vsd)} source critic tensors ignored)"
        )
    else:
        v_missing, v_unexpected = value_model.load_state_dict(vsd, strict=False)
        v_missing = [m for m in v_missing if "lora_" not in m]
        if v_missing or v_unexpected:
            raise RuntimeError(
                f"Any2Any critic load mismatch -- missing={v_missing[:8]} unexpected={v_unexpected[:8]}."
            )
        logger.info(f"Any2Any: loaded {len(psd)} policy + {len(vsd)} critic tensors, zero mismatch")

    if source_std is not None and hasattr(policy, "std"):
        copy_std = bool(cfg.get("copy_source_std", False))
        remap_std(
            policy.std,
            source_std.to(policy.std.device),
            head_std=cfg.get("head_std", 1e-3),
            copy_source_std=copy_std,
        )
        logger.info(
            f"Any2Any: exploration std "
            f"{'copied from source on ' + str(len(MATCHED_PAIRS)) + ' matched joints' if copy_std else 'left at init_noise_std'}"
            f", head joints pinned to {cfg.get('head_std', 1e-3)}; "
            f"mean std now {policy.std.mean().item():.3f}"
        )

    # The released checkpoint's critic normalizer has a Welford count of ~7e10,
    # so it would never adapt to the target robot's observation statistics.
    rms = getattr(value_model, "running_mean_std", None)
    reset_count = cfg.get("normalizer_reset_count", 1e6)
    if fresh_critic:
        rms = None  # a fresh normalizer already starts empty and adapts to the target
    if rms is not None and reset_count:
        old = float(rms.count)
        with torch.no_grad():
            rms.count.fill_(float(reset_count))
        logger.info(f"Any2Any: critic normalizer count {old:.3e} -> {float(rms.count):.3e}")

    stats = apply_any2any_lora(
        policy,
        value_model,
        r=cfg.get("lora_rank", 16),
        alpha=cfg.get("lora_alpha", 32.0),
        train_std=cfg.get("train_std", False),
        # `adapt_critic: false` keeps the pretrained critic loaded but leaves it
        # entirely untrainable (no LoRA, no gradients). One optimizer learning
        # rate drives both parameter groups, so trainability is the only way to
        # hold the critic fixed while the actor adapts -- which is what isolates
        # an actor-side defect from a critic-side one.
        adapt_critic=bool(cfg.get("adapt_critic", True)) and not fresh_critic,
        full_finetune=bool(cfg.get("full_finetune", False)),
        actor_decoder_full=bool(cfg.get("actor_decoder_full", False)),
        critic_full=bool(cfg.get("critic_full", False)) or fresh_critic,
    )
    logger.info(
        f"Any2Any: LoRA r={cfg.get('lora_rank', 16)} on "
        f"{stats['g1_dyn_linears_wrapped']} action-decoder + "
        f"{stats['critic_linears_wrapped']} critic linears; trainable "
        f"{stats['trainable_params']:,}/{stats['total_params']:,} "
        f"({100 * stats['trainable_fraction']:.2f}%)"
    )
    return stats
