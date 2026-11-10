#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201, DOC
"""Quality gate for a directory of R1 motion_lib PKLs.

Checks what a malformed conversion would actually get wrong, before a training
run spends GPU hours discovering it:

* **schema** -- the six keys SONIC reads, with the right dtypes and shapes, all
  finite, consistent frame counts, unit-norm ``root_rot``.
* **consistency** -- ``pose_aa[:, 1:]`` really is ``dof_axis * dof`` with the
  axes read from the MJCF, and ``pose_aa[:, 0]`` really is ``root_rot`` as a
  rotation vector. These are the two places a wrong joint order or a dropped
  body would show up as a silently plausible dataset.
* **limits** -- every DOF inside the MJCF ``range`` for its joint. The ranges
  come from the MJCF SONIC trains against, not from whatever model the
  retargeter used, so a retargeter running a different envelope is caught here.
* **coverage** -- ``metadata.pkl`` agrees with the clips on disk, and (with
  ``--expected``) every clip that should exist does.

``--fk-sample`` additionally runs SONIC's own ``Humanoid_Batch`` FK over N clips
and reports ankle and root heights, which is a cheap read on whether the
motions stand on the floor at all.

Exit status is non-zero when any clip fails, so it can gate a pipeline.

Usage:
    python gear_sonic/data_process/validate_r1_motion_lib.py \
        --input /mnt/fast/soma_r1_sr_sonic --fk-sample 40
"""

import argparse
from pathlib import Path
import random
import xml.etree.ElementTree as ElementTree

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_INPUT = Path("/mnt/fast/soma_r1_sr_sonic")
DEFAULT_MJCF = Path("gear_sonic/data/assets/robot_description/mjcf/r1.xml")

NUM_SMPL_JOINTS = 24
LIMIT_TOLERANCE_RAD = 1e-4
# pose_aa is float32 and a product of two float32 arrays; 1e-5 rad is 6e-4 deg.
CONSISTENCY_TOLERANCE = 1e-5
ANKLE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def load_mjcf_joints(mjcf_path: Path):
    """Return (axes (N,3), names, lower (N,), upper (N,)) in MJCF tree order."""
    worldbody = ElementTree.parse(mjcf_path).getroot().find("worldbody")
    axes, names, lower, upper = [], [], [], []
    for joint in worldbody.iter("joint"):
        if joint.get("type") == "free":
            continue
        axes.append([float(v) for v in joint.get("axis").split()])
        names.append(joint.get("name"))
        lo, hi = (float(v) for v in joint.get("range").split())
        lower.append(lo)
        upper.append(hi)
    return (
        np.asarray(axes, dtype=np.float64),
        names,
        np.asarray(lower),
        np.asarray(upper),
    )


def check_clip(path: Path, dof_axis: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    """Return (name, num_frames, problems, stats) for one PKL."""
    data = joblib.load(path)
    if len(data) != 1:
        return path.stem, 0, [f"{len(data)} top-level keys, expected 1"], {}
    name, e = next(iter(data.items()))
    problems = []
    if name != path.stem:
        problems.append(f"key {name!r} does not match filename {path.stem!r}")

    num_dof = len(dof_axis)
    required = {
        "root_trans_offset": (3,),
        "pose_aa": (num_dof + 1, 3),
        "dof": (num_dof,),
        "root_rot": (4,),
        "smpl_joints": (NUM_SMPL_JOINTS, 3),
    }
    for key in (*required, "fps"):
        if key not in e:
            problems.append(f"missing key {key!r}")
    if problems:
        return name, 0, problems, {}

    num_frames = e["dof"].shape[0]
    for key, tail in required.items():
        arr = e[key]
        if arr.shape != (num_frames, *tail):
            problems.append(f"{key} shape {arr.shape}, expected {(num_frames, *tail)}")
        if arr.dtype != np.float32:
            problems.append(f"{key} dtype {arr.dtype}, expected float32")
        if not np.isfinite(arr).all():
            problems.append(f"{key} has non-finite values")
    if problems:
        return name, num_frames, problems, {}
    if num_frames < 2:
        problems.append(f"{num_frames} frame(s)")

    dof, pose_aa, root_rot = e["dof"], e["pose_aa"], e["root_rot"]

    quat_err = float(np.abs(np.linalg.norm(root_rot, axis=1) - 1.0).max())
    if quat_err > 1e-5:
        problems.append(f"root_rot off unit norm by {quat_err:.2e}")

    over = float(np.maximum(lower - dof, dof - upper).max())
    if over > LIMIT_TOLERANCE_RAD:
        worst = int(np.argmax(np.maximum(lower - dof, dof - upper).max(axis=0)))
        problems.append(f"DOF {worst} outside the MJCF range by {np.rad2deg(over):.3f} deg")

    joint_err = float(np.abs(pose_aa[:, 1:] - dof_axis[None] * dof[:, :, None]).max())
    if joint_err > CONSISTENCY_TOLERANCE:
        problems.append(f"pose_aa joints disagree with dof_axis * dof by {joint_err:.2e}")

    root_err = float(
        np.abs(pose_aa[:, 0] - Rotation.from_quat(root_rot).as_rotvec()).max()
    )
    if root_err > CONSISTENCY_TOLERANCE:
        problems.append(f"pose_aa root disagrees with root_rot by {root_err:.2e}")

    stats = {
        "fps": float(e["fps"]),
        "max_limit_overshoot_deg": float(np.rad2deg(max(over, 0.0))),
        "max_dof_step_deg": float(np.rad2deg(np.abs(np.diff(dof, axis=0)).max()))
        if num_frames > 1
        else 0.0,
        "root_z_min_m": float(e["root_trans_offset"][:, 2].min()),
        "root_z_max_m": float(e["root_trans_offset"][:, 2].max()),
    }
    return name, num_frames, problems, stats


def fk_heights(paths: list[Path], mjcf: Path):
    """Ankle and root heights from SONIC's own FK, for a sample of clips."""
    import omegaconf
    import torch

    from gear_sonic.utils.motion_lib import torch_humanoid_batch

    cfg = omegaconf.OmegaConf.create(
        {
            "asset": {
                "assetRoot": str(mjcf.parent) + "/",
                "assetFileName": mjcf.name,
                "urdfFileName": "",
            },
            "extend_config": [],
        }
    )
    fk = torch_humanoid_batch.Humanoid_Batch(cfg)
    body_names = [str(n) for n in fk.body_names]
    ankles = [body_names.index(n) for n in ANKLE_BODIES]

    rows = []
    for path in paths:
        e = next(iter(joblib.load(path).values()))
        fps = int(e["fps"])
        out = fk.fk_batch(
            torch.from_numpy(e["pose_aa"]).float()[None],
            torch.from_numpy(e["root_trans_offset"]).float()[None],
            fps=fps,
            target_fps=fps,
            interpolate_data=False,
        )
        pos = out["global_translation"][0].detach().cpu().numpy()
        rows.append(
            {
                "clip": path.stem,
                "min_ankle_z_m": float(pos[:, ankles, 2].min()),
                "root_z_min_m": float(pos[:, 0, 2].min()),
                "root_z_max_m": float(pos[:, 0, 2].max()),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    ap.add_argument("--fk-sample", type=int, default=0, help="run SONIC FK over N clips")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expected", type=Path, default=None, help="file of expected clip ids")
    args = ap.parse_args()

    dof_axis, joint_names, lower, upper = load_mjcf_joints(args.mjcf)
    print(f"{args.mjcf}: {len(joint_names)} DOF")

    paths = sorted(p for p in args.input.glob("*.pkl") if p.name != "metadata.pkl")
    if not paths:
        raise SystemExit(f"no clip PKLs under {args.input}")
    print(f"{len(paths)} clips under {args.input}")

    failures, stats, frames = [], [], {}
    for path in paths:
        try:
            name, num_frames, problems, row = check_clip(path, dof_axis, lower, upper)
        except Exception as exc:  # noqa: BLE001
            failures.append((path.stem, f"unreadable: {exc}"))
            continue
        frames[name] = num_frames
        if row:
            stats.append(row)
        for problem in problems:
            failures.append((name, problem))

    meta_path = args.input / "metadata.pkl"
    if not meta_path.exists():
        failures.append(("metadata.pkl", "missing"))
    else:
        metadata = joblib.load(meta_path)
        for name in sorted(set(frames) - set(metadata)):
            failures.append((name, "clip absent from metadata.pkl"))
        for name in sorted(set(metadata) - set(frames)):
            failures.append((name, "metadata.pkl lists a clip with no PKL"))
        for name in sorted(set(metadata) & set(frames)):
            if metadata[name]["length"] != frames[name]:
                failures.append(
                    (name, f"metadata length {metadata[name]['length']} != {frames[name]} frames")
                )

    if args.expected is not None:
        expected = {line.strip() for line in args.expected.read_text().split("\n") if line.strip()}
        for name in sorted(expected - set(frames)):
            failures.append((name, "expected clip is missing"))

    total_frames = sum(frames.values())
    fps_values = sorted({row["fps"] for row in stats})
    print("\n--- schema and consistency ---")
    print(f"  clips             : {len(frames)}")
    print(f"  frames            : {total_frames}")
    print(f"  fps               : {', '.join(str(v) for v in fps_values)}")
    print(f"  worst limit over  : {max((r['max_limit_overshoot_deg'] for r in stats), default=0):.4f} deg")
    print(f"  worst DOF step    : {max((r['max_dof_step_deg'] for r in stats), default=0):.1f} deg/frame")
    print(f"  root height range : {min((r['root_z_min_m'] for r in stats), default=0):.3f} - "
          f"{max((r['root_z_max_m'] for r in stats), default=0):.3f} m")

    if args.fk_sample:
        sample = random.Random(args.seed).sample(paths, min(args.fk_sample, len(paths)))
        rows = fk_heights(sorted(sample), args.mjcf)
        print(f"\n--- SONIC FK ({len(rows)} clips) ---")
        print(f"  lowest ankle origin : {min(r['min_ankle_z_m'] for r in rows) * 100:.2f} cm")
        print(f"  root height range   : {min(r['root_z_min_m'] for r in rows):.3f} - "
              f"{max(r['root_z_max_m'] for r in rows):.3f} m")

    print(f"\n{len(failures)} failure(s)")
    for name, why in failures[:20]:
        print(f"    {name}: {why}")
    if len(failures) > 20:
        print(f"    ... and {len(failures) - 20} more")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
