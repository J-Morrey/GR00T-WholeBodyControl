# Any2Any implementation audit — `gear_sonic/trl/modules/any2any.py`

Reviewed against *Any2Any: Efficient Cross-Embodiment Transfer for Humanoid Whole-Body
Tracking* (arXiv 2605.23733v3), the SONIC training stack in this repo, and the installed
IsaacLab (`/home/mindgoblin/shared_repos/IsaacLab-HARL`).

Every claim below is marked:

| Mark | Meaning |
|---|---|
| ✅ | Verified correct against source/paper. I read the code that proves it. |
| ⚠️ | Correct today, but resting on an unasserted assumption that will break silently. |
| ❌ | Wrong, or a substantive deviation from the paper. |
| 🔎 | Not a correctness bug in `any2any.py`, but a mechanism that plausibly causes the collapse. |

---

## 0. Executive summary

**Stage 1 (kinematic alignment) is, as far as I can verify, correct.** The joint maps, the
scatter/gather, the pos|vel block reshape, the history reshape, the default-pose affine
correction and the wrist selection all check out — including two places where the obvious
implementation would have been wrong and the code does the right thing. I found no defect
in the alignment that could explain a collapse.

**Stage 2 (dynamics adaptation) reproduces the paper's S7 injection scope faithfully.** LoRA
on all seven Linears of `g1_dyn` (= actor backbone + proprioception-in + action-out) and on
the middle five of the critic (= critic backbone, not in/out). Encoders, `g1_kin` and FSQ
frozen. That is exactly Fig. 7/S7.

**The collapse is not in `any2any.py`'s maths. It is caused by the frozen exploration `std`.**
This is now supported by checkpoint data from your two runs (§3.0), not just by reading code.
Ranked:

1. ❌ **Exploration `std` is frozen at `init_noise_std` = 0.05, but this task wants σ ≈ 0.46.**
   Your successful from-scratch run self-regulates from 0.05 up to 0.456 and sits there.
   The Any2Any run is pinned 9× below that with no way to move. Since KL ∝ 1/σ², the same
   `Δμ` produces ~80× the KL, which saturates the LR controller at its **floor** — where
   your logs show it sitting at 4/4 checkpoints. PPO therefore has no braking authority
   left, and the policy drifts monotonically off the prior. §3.0, §3.3, §2.12
2. ❌ `normalizer_reset_count: 1e6` moves the critic's input distribution under a *frozen*
   critic input-projection and value head. The released count of 7e10 is a feature, not a
   bug. §2.13
3. 🔎 `max_grad_norm: 0.1` was calibrated against a ~42 M-parameter gradient. With ~2 M
   trainable LoRA parameters the global norm is far smaller, so clipping stops attenuating.
   Secondary, but it compounds #1. §3.2
4. ❌ **You are running a larger SONIC than the paper did.** §3.2 scopes its Sonic
   experiments to "Robot Motion Encoder, FSQ bottleneck, and dynamics decoder" — one encoder,
   one decoder. This repo runs three encoders, two decoders and five aux losses. With SMPL
   disabled, **half of every rollout routes through the frozen teleop encoder**, whose inputs
   are the least-aligned in the whole pipeline. §2.16
5. ⚠️ The G1→G1 "identity control" is not actually a controlled experiment: it swaps the
   motion corpus for a 1% subset. §4.1
6. ⚠️ S7 was selected on the Oli-WBT (MLP/Transformer) backbones, **not on Sonic**. Mapping
   its rows onto SONIC's module graph is our interpretation, not a paper result. §2.11

> **Status note (Sept 9).** Sections 3 and 5 below are kept as the historical record of how
> the diagnosis evolved. **Section 8 supersedes them.** The std finding in 3.3 stands and was
> a large real gain; the *residual* failure has since been traced, with a seeded null control,
> to the PPO update having no step-size decay (8.3) rather than to a bad critic or noisy
> advantages. Explained variance is 0.988 and the advantages are provably informative.

**A hypothesis I put first in the previous revision of this document is now refuted** — I
predicted the LR would ramp to `adaptive_lr_max`; the checkpoints show it pinned at
`adaptive_lr_min`, i.e. the exact opposite saturation. See §3.1 for the postmortem; the
corrected mechanism is stronger and points at the same fix.

**The good news, and it is substantial: the transfer works.** At iteration **50** the Any2Any
run reaches mean episode reward **4.61**, versus **4.28** for the from-scratch run at
iteration **2600**. That is the paper's headline ~50× sample-efficiency claim, reproduced.
Your problem is not that transfer fails — it is that a good transferred policy is then
destroyed by an exploration-noise setting. §3.0

**Run Test 1 in §5 (`copy_source_std: true`, ideally with `init_noise_std` raised).** One
line, and it targets the mechanism the data actually supports.

---

## 1. What the paper actually specifies

Pulling the load-bearing sentences so the comparison is concrete.

**Kinematic alignment (§3.3.1, p5–6).**
- Level 1: rearrange the target's observation *components* into the source's layout.
- Level 2: `Φ_r = J_r D_r⁻¹ S_r`, `Φ_r⁺ = S_rᵀ D_r J_r⁻¹`, with `Φ_r⁺ Φ_r = I_{N_r}` on
  matched joints (Eq. 4–5).
  - `S_r ∈ {0,1}^{T×N_r}` sparse scattering matrix from a partial injection
    `π_r: target joints → source joints`. Unmatched source rows are zero-padded; surplus
    target joints are dropped columns.
  - `D_r` decouples an inclined hip-pitch axis (Eq. 3). **Table 1 lists Unitree G1 as
    "Incl. hip: –", i.e. `D = I` for G1.**
  - `J_r` corrects closed-chain (parallelogram) actuation. **Table 1 lists G1 as
    "Closed-ch.: –".**
- Observation side: `q̃_r = Φ_r q_r`. Action side: `a_r = Φ_r⁺ ã`.

**Dynamics adaptation (§3.3.2, p7).**
- `W' = W + BA`, rank `k ≪ min(d_in,d_out)`, only `{A,B}` trained, `θ_S` frozen.
- "we inject LoRA into the proprioceptive input projection and the actor and critic linear
  layers, while keeping the reference-motion encoder frozen."
- Sonic-specific (p10): "LoRA modules are inserted into the actor dynamics decoder and the
  critic network, while the FSQ module and other pretrained components remain frozen."
- Best ablation S7 (Fig. 7): Actor{Backbone, Prop. In, Out} + Critic{Backbone}. Notably
  **not** Actor Ref. In, and **not** Critic In./Out. (that is S9, which scored worse).

**Training procedure (§3.4, p8) — the sentence that matters most here:**

> "we keep the action space, observations, reward formulation, **PPO hyperparameters**,
> reference-motion sampling, and domain randomization protocol **identical** to those of the
> corresponding source pretraining, only the components introduced by our kinematic
> alignment and dynamics adaptation differ."

`θ_S` is "its full parameter set". The Gaussian's `std` is part of `θ_S`. Under the paper's
formulation it should be **the checkpoint's converged std, frozen** — not `init_noise_std`.
The critic's running normalizer is likewise part of the pretrained state.

---

## 2. Section-by-section audit of `any2any.py`

### 2.1 Joint-name ground truth (L45–147) — ✅ with one ⚠️

`build_joint_maps()` derives every index from joint *names* read out of the MJCFs, never
from a hardcoded index. I verified the result by hand:

- G1 DOF order (from `G1_ISAACLAB_JOINTS[1:]` through the MJCF body→joint map):
  indices 23–28 are `left/right_wrist_roll/pitch/yaw` — matches the config comment.
- R1 DOF order: indices 24–25 are `left/right_wrist_roll` — matches.
- `SOURCE_ONLY_DOF` = {`waist_pitch`, `left/right_wrist_pitch`, `left/right_wrist_yaw`} (5).
- `TARGET_ONLY_DOF` = {`head_pitch`, `head_yaw`} (2).
- `MATCHED_PAIRS` = 24. 29 = 24 + 5 ✅, 26 = 24 + 2 ✅.

I checked both MJCFs for near-miss names that a canonical-name match would silently drop
(e.g. a `waist_yaw` vs `torso_yaw` mismatch). There are none — all 24 pairs share exact
canonical names. The `torso_link ← waist_pitch_joint` irregularity that would break a naive
`<link> → <joint>` string rule is correctly handled by reading the MJCF (L90–106), and I
confirmed every non-root body in both files has exactly one non-free joint.

**⚠️ The one unasserted assumption in the whole file that I would fix first:**
`isaaclab_dof_names()` (L109–113) assumes **IsaacLab's DOF ordering equals the body ordering
minus the root**. That is true here — both `*_ISAACLAB_JOINTS` lists are breadth-first body
orders and every body carries exactly one DOF, so the two orders coincide — but nothing
checks it at runtime. If it were ever false you would get a silent 29-element permutation of
every joint-space observation, which is precisely the kind of failure that looks like "the
policy starts okay and then degrades".

```python
# add to Any2AnyBackbone.__init__ or setup_any2any
robot_dofs = list(env.scene["robot"].joint_names)
assert robot_dofs == TARGET_DOF_NAMES, f"DOF order mismatch:\n{robot_dofs}\n{TARGET_DOF_NAMES}"
```

Three lines, and it converts the single most dangerous silent assumption in the file into a
startup error. **Note this cannot explain the G1→G1 identity failure** (identity permutation
either way), but it must be ruled out before you trust the R1 result.

### 2.2 `scatter_dof` / `gather_dof` (L152–182) — ✅

`scatter_dof` is `S_r` applied to a row vector: `index_select` on the last axis with
`src_from_tgt`, then `masked_fill` on the padded slots. `gather_dof` is `S_rᵀ`. On the 24
matched joints `gather(scatter(x)) == x`, so `Φ⁺Φ = I_{N_r}` (Eq. 4) holds exactly. Since
`D = I` and `J = I` are correct for this pair per Table 1, `Φ_r` collapsing to `S_r` is a
faithful instantiation, not a shortcut. The module docstring justifies both, correctly.

### 2.3 `_align_history_block` (L249–262) — ✅ (verified against IsaacLab source)

The code assumes history is flattened **time-major**, i.e. `(H, D)` row-major. I verified
this non-circularly:

- `isaaclab/utils/buffers/circular_buffer.py:80-90` — `CircularBuffer.buffer` returns shape
  `(batch_size, max_length, ...)`.
- `isaaclab/managers/observation_manager.py:424` — `circular_buffer.buffer.reshape(num_envs, -1)`.

So `reshape(*lead, hist, NUM_TARGET_DOF)` recovers the frames correctly. ✅

(Worth flagging: a sub-agent I dispatched "proved" this by quoting the docstring in
`any2any.py` itself, which is circular. I re-derived it from the IsaacLab source directly.
The conclusion holds.)

### 2.4 `_align_pos_vel_block` (L230–246) — ✅ and this is the subtle one

The code views the flat `command_multi_future` chunk as `(..., 2, F, D)` rather than the
`(..., F, 2, D)` that the `_nonflat` naming superficially suggests. **This is correct**, and
I verified the reason:

- `commands.py:903` — `command_multi_future = cat([joint_pos_multi_future, joint_vel_multi_future], dim=1)`
- `commands.py:1685-1693` / `1719-1727` — each of those is `(num_envs, F*D)`.

So the flat layout really is `[p_f0…p_fF | v_f0…v_fF]` ⇒ `(2, F, D)`. ✅

The `_nonflat` tokenizer variant is `command_multi_future.reshape(B, F, -1)`
(`observations.py:585`) — which *does* interleave pos/vel across frames — but that is the
scrambling the pretrained G1 encoder learned on. Aligning the flat vector as `(2,F,D)` and
then letting `parse_tokenizer_obs` re-view it as `(F, 2*D_src)` reproduces exactly what the
source encoder saw. The docstring's warning is right and the implementation matches it.

Arithmetic cross-check that everything in §2.3–2.4 is consistent:
- G1 policy: `3·10 + 3·10 + 29·10·3 = 930` ✅ (matches the docstring)
- R1 policy: `3·10 + 3·10 + 26·10·3 = 840` ✅
- G1 critic: `2·10·29 + 29·10·3 + (3+6+14·3+14·6+30+30) = 580+870+195 = 1645` ✅
- R1 critic: `2·10·26 + 26·10·3 + 195 = 520+780+195 = 1495` ✅

All four numbers land exactly on the docstring's claims, which is strong evidence the term
inventory and the per-DOF classification are complete for these two groups.

### 2.5 `align_concat_group` / `aligned_group_dim` / `aligned_term_spans` (L265–315) — ✅ logic, ⚠️ no guard

The three functions agree with each other on block accounting (`blocks = size // D_tgt`,
aligned width `blocks · D_src`), which is what makes `add_pose_offset` and `padded_columns`
index correctly into the aligned vector. I traced all three; they are consistent.

**⚠️ Missing guard.** `align_concat_group` never checks that the layout covers the input:

```python
for spec in layout:
    chunk = x[..., spec.start : spec.end]   # short layout -> trailing obs silently dropped
    ...
return torch.cat(pieces, dim=-1)
```

If `env.config["obs"]["group_term_layout"]` were ever short or mis-ordered relative to what
the `ObservationManager` actually concatenates, this silently truncates or permutes the
observation with no error. Today it is fine — `train_agent_trl.py:386-398` builds the layout
from `obs_manager.active_terms` / `group_obs_term_dim` with `strict=True`, i.e. the same
source and order the manager concatenates in — but add:

```python
assert layout[-1].end == x.shape[-1], f"layout covers {layout[-1].end}, got {x.shape[-1]}"
```

### 2.6 `add_pose_offset` / `default_pose_offset` (L318–441) — ✅ algebra, ⚠️ fragile premise

I re-derived all three corrections independently and they are all right:

| Channel | Env convention | Source expects | Correction | Code |
|---|---|---|---|---|
| `joint_pos` obs | `q − q_def^T` | `q − q_def^S` | `+ off` | L331 ✅ |
| `actions` obs | `a_T` | `a_S = a_T + off/s` | `+ off/s` | L331 ✅ |
| emitted action | needs `a_T` | produces `a_S` | `− off/s` | L815 ✅ |

with `off = q_def^T − q_def^S`. Derivation: `q_def^T + s·a_T = q_def^S + s·a_S`
⇒ `a_T = a_S − off/s`. ✅ Signs and directions all match.

Not applying an offset to `joint_vel` (relative to a zero default) or to the reference motion
(absolute joint angles, `motion_lib.get_dof_pos`) is also correct. ✅

**⚠️ The premise is `scale_target == scale_source` on matched joints.** `_act_off_src` and
`_act_off_tgt` both divide by `scale_src`. That only holds because `sonic_r1_any2any.yaml`
sets `robot.type: r1_any2any`, which remaps R1's action scale to G1's per-joint values. Run
the same alignment against `type: r1` (flat 0.4 rad/unit) and every action-side correction is
silently wrong by the scale ratio. Assert it:

```python
scale_tgt = _resolve_regex_dict(tgt_scale_cfg, TARGET_DOF_NAMES, default=1.0)
for tgt, src in MATCHED_PAIRS:
    assert abs(scale_tgt[tgt] - scale_src[src]) < 1e-9, f"action-scale mismatch at {TARGET_DOF_NAMES[tgt]}"
```

**⚠️ Secondary.** `_resolve_regex_dict(..., default=0.0)` silently returns 0 for any joint
the `init_state.joint_pos` regex dict does not match. For G1 and R1 the unmatched joints
genuinely do default to 0, so the result is right — but a robot config that expressed
defaults differently would produce a wrong offset with no warning. Log the resolved vector
once at startup.

### 2.7 `padded_columns` (L343–363) — ✅

Offsets are accumulated identically to `aligned_term_spans`, and the `src_only` set is only
used to collect indices so its iteration order is irrelevant. Correct.

### 2.8 Tokenizer alignment + `build_wrist_map` (L444–491) — ✅

I hand-checked the wrist map. `source_joint_idx=[23..28]` are G1's six wrist DOFs;
`target_joint_idx=[24,25]` are R1's two wrist rolls. `SRC_FROM_TGT[23]=24 → 0`,
`SRC_FROM_TGT[24]=25 → 1`, and 25–28 have no counterpart → `-1`. So
`wrist_map = [0, 1, -1, -1, -1, -1]` ✅, and the `masked_fill` zeroes the four dropped
slots. Correct.

### 2.9 `command_multi_future_lower_body` — ⚠️ correct by coincidence

This is the one per-DOF tokenizer term that is **not** aligned. It is not in
`_POS_VEL_TERMS`, not `_WRIST_TERM`, and `g1ify_env_config` does not resize it, so it passes
through untouched into the frozen teleop encoder.

It works anyway, for a reason nobody wrote down:

- `commands.py:177-179` — `lower_joint_isaaclab_indices = [isaaclab_to_mujoco_dof[i] for i in range(12)]`,
  i.e. the first **twelve MuJoCo** DOFs.
- I dumped the joint order from both MJCFs. The first twelve non-free joints of
  `g1_29dof_rev_1_0.xml` and `r1.xml` are byte-for-byte the same twelve leg joints in the
  same order (`left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch,
  left_ankle_roll,` then the right leg).

So the term has identical width (`2·F·12 = 240`) and identical semantics on both robots —
which is also why the teleop encoder loads without a shape mismatch. **This is a coincidence
of the two URDFs, not a property the code enforces.** It will break silently on the next
embodiment (e.g. any robot whose MJCF puts the waist before the second leg). Add it to
`_POS_VEL_TERMS`-style handling with an explicit 12-joint name check, or at minimum assert:

```python
assert [_canonical(n) for n in mujoco_dof_names(SOURCE_ROBOT)[:12]] == \
       [_canonical(n) for n in mujoco_dof_names(TARGET_ROBOT)[:12]]
```

### 2.10 `g1ify_env_config` / `g1ify_obs_dim_dict` (L497–540) — ✅

This is the mechanism that makes the checkpoint load with zero shape mismatch, and it is
right. `UniversalTokenModule` sizes every Linear from `env_config.obs.group_obs_dims` /
`obs_dim_dict`, and both are overridden to source widths on a `deepcopy`, so the real env is
untouched. `cfg.robot.actions_dim = NUM_SOURCE_DOF` sizes the `g1_dyn` output at 29 while
`Actor.__init__` keeps reading the *target* config for `self.num_actions = 26`, so the
Gaussian stays target-shaped. That split is exactly what you want and it is easy to get
wrong. ✅

The comment at L768-770 about `Actor.__init__` resolving `obs_dim_dict` from the target
config and overriding the g1-ified `env_config` is accurate — I confirmed it at
`actor_critic_modules.py:75-92`.

### 2.11 `LoRALinear` / `inject_lora` / `apply_any2any_lora` (L565–724) — ✅ vs paper

**Faithful to S7.** I confirmed the module shapes:

- `g1_dyn` is a `BaseModule` whose `.module` is a flat `nn.Sequential` built by
  `_build_mlp_layer` (`base_module.py:288-303`) with `hidden_dims=[2048,2048,1024,1024,512,512]`
  ⇒ Linears at indices 0,2,4,6,8,10,12 — **seven**. `inject_lora` wraps all seven. Since
  `g1_dyn.inputs = ["token_flattened", "proprioception"]`, its first Linear **is** the
  proprioception input projection and its last **is** the action output projection. So
  "all seven" = Actor{Backbone + Prop. In + Out} — exactly S7's actor row. ✅
- Critic: same shape, `skip_endpoints=True` ⇒ indices 2,4,6,8,10 = **five**, i.e. critic
  backbone without In./Out. — exactly S7's critic row (S9 is the one that adds In./Out. and
  scores worse). ✅
- Encoders, `g1_kin` and the parameter-free FSQ stay frozen. ✅ Matches both p7 ("keeping the
  reference-motion encoder frozen") and p10 ("the FSQ module and other pretrained components
  remain frozen").

`LoRALinear` subclassing `nn.Linear` so state-dict keys stay `weight`/`bias` is a good call —
it is what lets the source checkpoint load unmodified. `lora_B` zero-init ⇒ the freshly
injected model is bit-identical to the source. ✅ Freeze order is right: the global
`requires_grad_(False)` happens *before* the `lora_A`/`lora_B` Parameters are created, so
they come out trainable. ✅ And `setup_any2any` runs before the trainer is constructed
(`train_agent_trl.py:465-468` vs `479`), so HF's `create_optimizer` — which filters on
`p.requires_grad` — captures exactly the LoRA params. ✅

**Important caveat on "faithful to S7": S7 was never validated on the Sonic backbone.**
Fig. 6 and Fig. 7 are both captioned "on OliWBT2Luna" — i.e. the ablation that selected S7
was run on the authors' own MLP and Transformer Oli-WBT policies, where "actor backbone" is a
plain trunk, not an FSQ-bottlenecked encoder/decoder. Mapping S7's rows onto SONIC's module
graph (g1_dyn's first Linear = "Prop. In.", its last = "Out.", the middle = "Backbone") is a
*reasonable interpretation*, but it is our interpretation, not a result the paper reports for
this architecture.

The only Sonic-specific instruction the paper gives is p10: LoRA into "the actor dynamics
decoder and **the critic network**" — the whole critic, not the backbone-only S7 reading the
code implements. Given the critic must be re-fit for a new embodiment *and* a new motion
corpus, the wider reading is the safer one here; see §3.4. You already have `critic_full` and
`fresh_critic` flags for it.

**Minor (not a bug):** `LoRALinear.wrap` calls `cls(in_features, out_features, ...)`, which
allocates and immediately discards a full weight matrix per wrapped layer (~30 MB transient
across the twelve wraps). Harmless, but `torch.nn.utils.skip_init` avoids it.

### 2.12 `remap_std` (L903–933) and `copy_source_std: false` — ❌ deviation

This is my third-ranked suspect and it is a clear departure from the paper.

With `copy_source_std: false` and no `TARGET_ONLY_DOF` (the G1→G1 case), `remap_std` is a
**complete no-op**: `source_std` is popped off the state dict at L987 and thrown away, and
`policy.std` stays at `init_noise_std = 0.05`. Then `apply_any2any_lora` freezes it.

So the run executes the source policy's mean function at a noise level it never operated at
(the G1 release converged near ~0.34), with no mechanism to change it — the entropy bonus is
a no-op on a frozen Gaussian, as the code comment at L699-709 correctly observes.

Why this matters beyond "different exploration":

- The policy-gradient w.r.t. `μ` scales as `1/σ²`: `∂ log-ratio/∂μ = (a−μ)/σ²`. At
  `σ=0.05` instead of `0.34` the raw gradient into the LoRA adapters is ~46× larger. The
  KL controller is supposed to absorb that by shrinking the LR — but see §3.1, where it is
  doing the opposite.
- The paper freezes `θ_S`, and `std ∈ θ_S`. Under Eq. 1 the correct behaviour is
  "checkpoint's std, frozen", i.e. `copy_source_std: true`.

**Your own counter-evidence does not apply to the identity control.** The docstring argues
against copying because the jitter penalties (`action_rate_l2`, `feet_acc`) blew up ~200×/12×.
That is an *action-scale* interaction and it is specific to R1. On G1→G1, the source policy,
action scale, reward weights and robot are all identical to what produced `std≈0.34`, so
copying it must be benign — the release checkpoint demonstrably trains stably at that noise
level. **If the G1→G1 control still collapses with `copy_source_std: true`, std is
exonerated; if it stops collapsing, you have your answer.** Cheap, decisive.

### 2.13 `normalizer_reset_count: 1e6` (L1025–1035) — ❌ deviation, and worse than it looks

```python
rms.count.fill_(float(reset_count))   # 7e10 -> 1e6
```

The released checkpoint's Welford count of ~7e10 is **not an accident to be worked around**.
It is what makes the pretrained normalizer effectively static, so that the critic — whose
input projection and value head this code deliberately **freezes** — keeps seeing the input
distribution it was fitted on. Resetting the count re-enables adaptation on the *inputs* of a
head that cannot adapt in response. That combination is strictly worse than either extreme.

And the reset is far more aggressive than `1e6` suggests, because
`RunningMeanStd.forward` updates its statistics on **every** call while `self.training`
(`running_mean_std.py`, `if self.training and not self.frozen`), including inside the PPO
update loop:

- rollout: 24 calls × 4096 envs ≈ 98 k samples
- update: `num_learning_epochs=5` × `num_mini_batches=4` = 20 calls, each over the full
  `4096×24/4` mini-batch ≈ 25 k ⇒ ≈ 492 k
- ⇒ **the same rollout is folded into the normalizer 21×**, ~590 k samples per iteration
  against a count of 1e6.

So the mean/var move substantially within the first two or three iterations. On G1→G1 the
statistics should be *similar* — but not identical, because the control run uses a different
motion corpus (§4.1) and a different exploration std (§2.12). "Similar but drifting, under a
frozen value head" is a good recipe for advantages that start fine and rot.

Recommended: `normalizer_reset_count: 0` (keep the pretrained normalizer static, matching the
paper's "everything else frozen"), **or** free the critic (`critic_full: true`) if you want
the normalizer to move. Not both halfway.

### 2.14 `Any2AnyBackbone` / `Any2AnyCritic` (L741–897) — ✅

- `align_inputs` copies the TensorDict rather than mutating the rollout buffer. ✅
- `forward` handles both the plain-tensor return (rollout, `is_training=False`) and the dict
  return (`compute_aux_loss=True` during the update). ✅ I traced both paths through
  `Actor.rollout` → `update_distribution(is_training=False)` and `Actor.act` →
  `update_distribution(is_training=True)`.
- Setting `self._tgt_layouts` before `super().__init__()` is legal — `nn.Module.__setattr__`
  only raises for Parameters/Modules/Tensors before init, and a tuple falls through to
  `object.__setattr__`. ✅
- Critic: aligning **before** normalization is required (the pretrained `running_mean_std` is
  1645-dim source-shaped) and the code does it. Zeroing padded columns *after* normalization
  is equivalent to mean-filling before it. ✅ Both correct, and both easy to get backwards.
- `_pad_mask` broadcasting over `(B, S, 1645)` works. ✅

**Minor:** `forward_with_external_tokens` raises `NotImplementedError`. Fine for PPO, but it
breaks any eval/deployment path that uses the token-bypass mode.

### 2.15 `setup_any2any` (L935–1060) — ✅ ordering, ❌ two policies

Load order (construct → load → inject → freeze → optimizer) is correct and the resume branch
correctly injects LoRA *before* loading so the adapter keys are not "unexpected". ✅ The
strict-ish mismatch checks are good practice. ✅

The two deviations are §2.12 and §2.13.

### 2.16 ❌ You are running a *larger* SONIC than the paper did

§3.2's Sonic paragraph is precise about scope: "we employ its **Robot Motion Encoder, FSQ
bottleneck, and dynamics decoder** modules for training and evaluation." That is a
three-module enumeration: one encoder, the quantizer, one decoder.

What this repo instantiates under `all_mlp_v1` is considerably more:

| Module | Paper's Sonic scope | This repo's config |
|---|---|---|
| Robot motion encoder (`g1`) | ✓ | ✓ |
| Teleop encoder | not mentioned | ✓ active |
| SMPL encoder | not mentioned | ✓ active |
| FSQ bottleneck | ✓ | ✓ |
| Dynamics decoder (`g1_dyn`) | ✓ | ✓ |
| Kinematic decoder (`g1_kin`) | not mentioned | ✓ active |
| Aux losses (recon + 4 latent) | not mentioned | ✓ 5 of them |

This is not a modified architecture on the paper's side — it is a *reduced* one. And the
difference has teeth, because the extra paths are all frozen and all fed
cross-embodiment data:

- **Encoder routing.** With `smpl_motion_file: dummy` the SMPL branch is correctly disabled
  (verified, see §4.1), so `encoder_sample_probs` renormalises to **g1 0.5 / teleop 0.5**.
  Half of every rollout therefore drives the frozen prior through the **teleop** encoder,
  whose inputs are `command_multi_future_lower_body` — the one per-DOF term the alignment
  does not touch (§2.9) — plus `vr_3point_local_target/orn_target`, which are computed from
  R1's substantially different arm and torso geometry (hand-in-wrist offset 0.068 m vs G1's
  0.18 m; `waist_yaw_link`+0.37 vs `torso_link`+0.35). Those 3-point targets are *nominally*
  the same semantic quantity, but their distribution on R1 is not the one the frozen teleop
  encoder was fitted on, and nothing in Stage 1 corrects it.
- **Aux losses.** `{g1_recon: 0.01, g1_smpl_latent: 1.0, g1_teleop_latent: 1.0,
  teleop_smpl_latent: 1.0, reencoded_smpl_g1_latent: 1.0}` are added to `loss` by
  `TRLAuxLossPPOTrainer._compute_loss`, but every one flows only through encoders, FSQ and
  `g1_kin` — **all frozen** — so their gradient contribution is exactly zero. They buy three
  extra encoder passes per micro-batch (`universal_token_modules.py:987`, `1005`, `1022`) and
  make the logged `loss` incomparable to the from-scratch run.

**Recommendation — match the paper's scope.** `UniversalTokenModule` already supports this
(`active_encoders` / `active_decoders`, L96-97, L275, L364, L414-422), and non-active decoders
stay in the `ModuleDict` so the checkpoint still loads with zero mismatch:

```yaml
algo.config.actor.backbone.active_encoders: ["g1"]
algo.config.actor.backbone.active_decoders: ["g1_dyn"]
algo.config.compute_aux_loss: false
manager_env.commands.motion.encoder_sample_probs: {g1: 1.0}
```

This is worth running as its own arm. It removes the 50 % of rollout that currently goes
through the least-aligned input path, and it is what the paper actually describes.

### 2.17 `eval_agent_trl.py` LoRA injection — minor

`apply_any2any_lora(...)` is called with hardcoded defaults, so `adapt_critic=True`. If a
training run used `fresh_critic: true` or `critic_full: true`, the eval-time model gets
critic LoRA keys the checkpoint does not have and the strict load at L481 fails. Pass the
same flags through from `config.algo.config.any2any`.

---

## 3. Why it collapses — now with data

### 3.0 What the checkpoints actually show

`report_to: none`, so there are no local tfevents/CSV logs — but every checkpoint pickles
`args.learning_rate`, `state.rewbuffer`, `state.lenbuffer` and the policy `std`, and the
Any2Any checkpoints carry the LoRA factors. Extracted from:

- `logs_rl/TRL_R1_Track/.../sonic_r1_scratch_full-20260903_145050` (13 ckpts, every 200 it)
- `logs_rl/TRL_R1_Any2Any/.../sonic_r1_any2any_poseoffset_s7-20260902_162350` (4 ckpts, every 50 it)

Plot: `any2any_lr_diagnosis.png`.

| iter | scratch lr | scratch σ | scratch rew | | iter | a2a lr | a2a σ | a2a rew | LoRA drift |
|---|---|---|---|---|---|---|---|---|---|
| 200 | 5.93e-5 | 0.164 | 1.09 | | 50 | **1.00e-5** | **0.0500** | **4.61** | 0.44 % |
| 600 | 3.38e-5 | 0.341 | 1.52 | | 100 | **1.00e-5** | **0.0500** | 4.34 | 0.61 % |
| 1000 | 3.38e-5 | 0.440 | 1.87 | | 150 | **1.00e-5** | **0.0500** | 4.96 | 0.74 % |
| 1600 | 1.50e-5 | 0.451 | 2.77 | | 200 | **1.00e-5** | **0.0500** | 3.15 | 0.88 % |
| 2600 | 1.50e-5 | **0.456** | **4.28** | | | | | | |

(σ for Any2Any is the value on the 24 matched joints; the 2 head joints are pinned at
`head_std=1e-3`, giving the 0.04623 mean. LoRA drift = `‖(α/r)BA‖_F / ‖W‖_F`, max over the
seven `g1_dyn` Linears.)

Four things fall out immediately:

1. **The transfer works.** Any2Any at iteration 50 (reward 4.61) already exceeds from-scratch
   at iteration 2600 (reward 4.28). The pretrained G1 prior transfers to R1 through your
   alignment. This is the paper's central claim, reproduced at ~50× sample efficiency.
2. **The Any2Any LR is pinned at `adaptive_lr_min = 1e-5` at every single checkpoint**, and
   never once ticks upward. From-scratch oscillates 1.5e-5 ↔ 5.9e-5, i.e. its KL regularly
   falls *below* `desired_kl/2` and the controller pushes the LR back up. Any2Any's never
   does. It started at 2e-5, so it can only have reached the floor via the
   `kl_mean > 2·desired_kl` branch. **The controller is saturated in the "slow down" direction
   and has no authority left.**
3. **From-scratch drives σ from 0.05 to 0.456** — up against `std_clamp_max: 0.5` — and stays
   there for 1500 iterations while reward doubles. That is the task telling you what σ it
   wants. **Any2Any is frozen at 0.05, 9× smaller.**
4. **The LoRA drift is monotone and doubling** (0.44 % → 0.88 % in 150 iterations) with no
   flattening. The adapter is not converging on a correction; it is being pushed in one
   direction, every update, by a trust region it cannot satisfy.

Since `KL = Σᵢ Δμᵢ²/(2σᵢ²)`, running at σ = 0.05 instead of 0.456 inflates the measured KL by
**(0.456/0.05)² ≈ 83×** for an identical change in the action mean. `desired_kl = 0.01` is
calibrated for σ ≈ 0.45. At σ = 0.05 the same policy change registers as KL ≈ 0.83 — far past
the `2·desired_kl` threshold — so the controller floors the LR and *still* cannot bring the
measured KL into range, because the LR floor is `1e-5` and the real problem is not the step
size, it is the denominator.

Meanwhile the *gradient* is inflated by the same factor: `∂ log-ratio/∂μ = (a−μ)/σ²`. So the
run gets the worst of both worlds — hypersensitive ratios and near-zero exploration, which
also starves the advantage estimate of the signal it would need to actually improve.

**Mechanism, end to end:** frozen σ = 0.05 → every update measures a huge KL → LR floors at
1e-5 → PPO's only remaining brake is gone → each update still moves the policy further than
the trust region allows → LoRA drifts monotonically off the pretrained weights → the prior
that made iteration 50 good is progressively destroyed. That is exactly the "starts
promising, then deteriorates" signature.

### 3.1 ❌ REFUTED: my earlier "LR ramps to the ceiling" hypothesis

The previous revision of this document argued that zero-init LoRA reports KL ≈ 0, causing
`_adjust_learning_rate_based_on_kl` to fire `lr *= 1.5` on every minibatch and pin the LR at
`adaptive_lr_max = 2e-4`. **The data refutes this.** The LR is at `adaptive_lr_min`, the
opposite saturation, at all four checkpoints.

Postmortem, because the error is instructive: the reasoning was right about the *first few*
minibatches (`lora_B = 0` genuinely does give KL ≈ 2.9e-4 from the `+1e-5` inside the log) but
wrong about how long that lasts. With σ = 0.05 the KL leaves the dead zone almost
immediately, and from then on the σ effect dominates completely. I had the two effects in the
right file and the wrong order of magnitude. The corrected story points at the same root
cause I had ranked third, so the fix is unchanged — but the ranking and the recommended first
experiment both change.

A residual, much smaller version of the effect may still exist in the first handful of
minibatches. It is not worth acting on unless Test 1 fails.

### 3.1b Formerly the primary suspect — retained for reference only

`ppo_trainer.py:2197-2209`:

```python
if kl_mean > self.desired_kl * 2.0:      new_lr = max(adaptive_lr_min, lr / 1.5)
elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                         new_lr = min(adaptive_lr_max, lr * 1.5)
self.args.learning_rate = new_lr
for param_group in optimizer.param_groups: param_group["lr"] = new_lr
```

with `actor_learning_rate: 2e-5`, `desired_kl: 0.01`, `adaptive_lr_max: 2e-4`
(`ppo_im_phc.yaml`), and `learning_rate: ${algo.config.actor_learning_rate}`
(`algo/trl/ppo.yaml`) — **one LR for every parameter group**, `lr_scheduler_type: constant`.

Now trace the first iteration of an Any2Any run:

1. `lora_B = 0` ⇒ the model is bit-identical to the source ⇒ `mu_batch == mb_old_mu` and
   `sigma_batch == mb_old_sigma` **exactly**.
2. The KL expression at L1374-1380 is not exactly zero because of the `+1e-5` inside the log:
   `kl ≈ 29 · log(1+1e-5) ≈ 2.9e-4`. That is `> 0.0` and `< desired_kl/2 = 0.005`.
   ⇒ **the `lr *= 1.5` branch fires.**
3. There are `5 epochs × 4 mini-batches = 20` updates per iteration, and the branch fires on
   *every* one while the adapter is still small. `2e-5 × 1.5⁶ = 2.28e-4 > 2e-4`, so
   **the LR is pinned at the ceiling within the first ~6 mini-batches of iteration 1.**
4. `self.args.learning_rate` is persistent trainer state — it stays at 2e-4 across
   iterations for as long as the measured KL stays under 0.005.
5. Because LoRA's update is `ΔW = (α/r)·(ΔB·A)` and `B` starts at zero, the *first* steps are
   second-order small — so the KL stays in the dead zone for a while, the LR stays at the
   ceiling, and then `B` grows and the effect compounds. By the time KL finally exceeds 0.02,
   you are running at **10× the LR the source policy was trained under**, and the controller
   needs ~11 consecutive mini-batches at `/1.5` just to get back to 2e-5.

**From-scratch never experiences this.** A randomly initialised policy produces a nonzero KL
on the very first update, so the LR settles wherever KL≈0.01 and the ceiling is never
reached. The free ride to `adaptive_lr_max` is created *specifically* by the zero-init
adapter — i.e. by Any2Any.

This also explains a puzzle in your own notes. `sonic_r1_any2any.yaml:40-47` records that
raising `actor_learning_rate` 2e-5→1e-4 made things worse and that you reverted it. Reverting
the *floor* changes nothing, because the controller climbs to `adaptive_lr_max` regardless.
The knob that matters is the **ceiling**.

**You can test this without running anything.** `metrics["lr"]` is logged every iteration
(`ppo_trainer.py:1895`). Pull the `lr` curve for a failing Any2Any run and the matching
from-scratch run. If the Any2Any `lr` sits at 2e-4 through the "promising" phase and only
starts oscillating once reward turns over, this is confirmed.

Fixes, in order of preference:
- `adaptive_lr_max: 2e-5` for Any2Any runs (the paper's "identical PPO hyperparameters"
  arguably means the *effective* LR should not exceed the source's).
- or `desired_kl: null` to disable the schedule entirely for the adaptation phase.
- or gate the `lr *= 1.5` branch on a floor: `elif desired_kl/2 > kl_mean > 1e-3:`, so the
  numerical-noise KL of a no-op adapter does not count as "too conservative".

### 3.2 🔎 `max_grad_norm: 0.1` is calibrated for a 42 M-parameter gradient

`sonic_release.yaml:83` overrides `max_grad_norm: 0.1` (down from the algo default of 1.0),
and `_gradient_clipping` clips the **global** norm over `model.parameters()`
(`ppo_trainer.py:2109-2112`). Frozen parameters have `grad is None` and are skipped, so:

- from-scratch / full FT: the norm is taken over ~42 M parameters, and 0.1 is a hard,
  constantly-active constraint that scales *everything* down.
- Any2Any: the norm is taken over ~2 M LoRA parameters (the paper's own figure is 5.26 %
  trainable). The global norm is far smaller, so the clip fires much less often and the LoRA
  parameters receive close to their full `lr × g` step.

Stacked on §3.1's 10× LR, the effective step size per trainable parameter is orders of
magnitude larger than anything the source policy ever took. This is consistent with your
recorded observation that raising `max_grad_norm` 0.1→1.0 made the decay worse — you were
loosening a constraint that was already barely binding.

If you fix §3.1, this becomes much less important. If you want a belt-and-braces version,
clip the actor and critic parameter groups separately, or scale `max_grad_norm` by
`sqrt(n_trainable / n_total)`.

### 3.3 ❌ Frozen `std` at 0.05 — **the root cause**

Mechanism in §3.0; code in §2.12. Three additional points now that the data is in.

**Your own successful baseline refutes the argument for freezing it.** The docstring at
`any2any.py:903-925` rejects `copy_source_std` because it measured "~12× worse `feet_acc` and
~200× worse `action_rate_l2` than the from-scratch baseline". But **the from-scratch run
converges at σ = 0.456** — 34 % *more* exploration noise than the source checkpoint's ~0.34
that the docstring declines to copy. R1 at high action noise is demonstrably not a problem;
it is what the successful run does. The jitter comparison appears to have been made against
an early-training scratch run still sitting at σ ≈ 0.05–0.16 (it is only at 0.164 by
iteration 200), which is not a like-for-like comparison. Against the converged baseline the
sign should reverse: σ = 0.34 is *smoother* than σ = 0.456.

**The earlier `train_std: true` experiment failed for a reason that also disappears.** The
docstring records std climbing only 0.05 → 0.18 over 1000 iterations. Compare from-scratch:
0.05 → 0.164 in **200** iterations, 0.44 by 1000. The Any2Any run's σ climbed 5× slower
because `std` shares the single global LR, which was **pinned at the 1e-5 floor** — floored
precisely *because* σ was small. That is a trap: small σ → huge KL → floored LR → σ cannot
grow → σ stays small. Starting from the source's 0.34 breaks the loop before it closes.

**Recommendation, in order:**
1. `copy_source_std: true` — paper-faithful (`std ∈ θ_S`, frozen at the checkpoint value
   ≈ 0.34). Should be sufficient: it takes the KL inflation from ~83× down to ~1.5×.
2. If it under-performs, additionally raise `init_noise_std` toward 0.45 and/or set
   `train_std: true`. This is a deviation from "freeze θ_S", but a justified one: the target
   embodiment's optimum σ is measurably higher than the source's, and that difference is
   itself part of the dynamics gap Any2Any is supposed to absorb.
3. Keep `head_std: 1e-3` for the two unmatched head joints. I checked that these contribute
   exactly zero to both the KL and the log-ratio — `gather_dof(..., fill=0.0)` makes their
   mean a hard constant 0 — so they are harmless as configured.

### 3.4 ❌/🔎 The critic cannot follow

Three things are simultaneously true in the current configuration:

1. The critic's **input projection and value head are frozen** (S7, `skip_endpoints=True`).
2. Its **input normalizer is deliberately un-frozen** (`normalizer_reset_count: 1e6`, §2.13).
3. Its **target distribution has changed** — different exploration std, different motion
   corpus, and (for R1) different dynamics and reward-point bodies.

A rank-16 update on five middle layers cannot rescale a value function whose output range has
moved, because the final Linear is frozen. Advantage normalization
(`ppo_trainer.py:2170-2182`) removes the global mean and scale, which hides the problem for a
while — which is consistent with "starts promising, then deteriorates".

The paper's own Sonic-specific text (p10) says "the critic network", not "the critic
backbone". For a transfer where the critic has to be substantially re-fit, I would run
`critic_full: true` (or `fresh_critic: true`) as a control. Your code already supports both,
and the reasoning in the `adapt_critic=False` comment at L688-694 is sound — the critic has no
behavioural prior worth protecting.

### 3.5 🔎 Adaptive sampling + adaptive termination make the metric non-stationary

Both configs inherit `adp_samp_failure_rate_max_over_mean: 200` (vs. the repo default of 50,
`commands/terms/motion.yaml:24`) and `terminations: tracking/base_adaptive_strict_ori_foot_xyz`,
plus the `im_resample` callback. As the sampler concentrates probability mass on the clips the
policy fails (up to 200× the mean weight), **mean episode length and mean return fall even if
the weights never change.** In a from-scratch run the policy improves alongside; a
LoRA-constrained policy at 0.2 % trainable parameters may simply not keep up.

This is the reason Test 0 in §5 matters so much: it distinguishes "our optimizer is destroying
the policy" from "the metric drifts down on its own".

---

## 4. Experimental-design problems

### 4.1 The G1→G1 "control" is not controlled

`sonic_g1_any2any.yaml` is meant to isolate Stage 2 by making Stage 1 the identity. But
relative to `sonic_release` it also changes:

- `motion_file: /mnt/fast/g1_1pct_sonic` — 1 % of the pretraining corpus, and your own
  comment notes only 1024 of the 1422 clips are resident.
- `smpl_motion_file: dummy` — **correction to an earlier revision of this document:** I
  claimed this feeds the frozen SMPL encoder zeros for ~1/3 of environments. **That is
  wrong, and the repo handles it correctly.** `smpl_motion_file: "dummy"` leaves
  `smpl_data_keys` an empty set (`motion_lib_base.py:250-254`), so `motion_has_smpl` is
  all-False (`motion_lib_base.py:1551-1554`) and every env falls into the
  `encoder_sample_probs_no_smpl` branch (`commands.py:2896-2921`), which zeroes the SMPL
  probability and renormalises. No env ever routes through SMPL. The real consequence is
  different: the split becomes **g1 0.5 / teleop 0.5**, so half the rollout drives the
  teleop encoder — see §2.16.
- `copy_source_std: false` — 0.05 instead of the converged ~0.34.
- `normalizer_reset_count: 1e6`.

Any of those alone could produce a downward drift. The comment argues the data reduction "can
only understate the control's stability, which is the safe direction" — that is true for
concluding *stability*, but you observed *collapse*, and for that direction the confound is
not safe at all.

**Make the control actually identical**: `sonic_release`'s `motion_file` and
`smpl_motion_file`, `copy_source_std: true`, `normalizer_reset_count: 0`. Then the only
differences from a plain fine-tune are the LoRA injection and the freeze pattern, which is
what you set out to test.

---

## 5. Diagnostic ladder

Ordered by information-per-GPU-hour. Each test changes exactly one thing.

**Test 1 - restore the source std. Run this first.**
`copy_source_std: true` on the R1 Any2Any config. One line. Targets 3.0/3.3 directly.
**Success criterion:** the logged `lr` should stop sitting on `1e-5` and start oscillating
the way from-scratch does. That is the direct readout of "the KL controller has authority
again". Then watch reward past iteration 200, where the current run turns over.

**Test 1b - if Test 1 helps but plateaus.** Raise `init_noise_std: 0.45` (where from-scratch
converges) and/or `train_std: true`. See 3.3.

**Test 2 - stop moving the normalizer.** `normalizer_reset_count: 0`. Tests 2.13/3.4.
Independent of Test 1.

**Test 3 - free the critic.** `critic_full: true`. Tests 3.4.

**Test 4 - honest control.** Full motion corpus + real SMPL file for the G1->G1 identity run.
Removes 4.1. Only worth it if Tests 1-3 leave residual decay.

**Test 5 - zero learning rate (fallback).** If Tests 1-3 all fail, set `actor_learning_rate: 0`
and `desired_kl: null` so nothing can update. If reward *still* decays, the drift is
environmental (adaptive sampling at 200x max-over-mean, the 1 % corpus, dummy SMPL) rather
than an optimisation failure, and 4.1 becomes the whole story.

You already have config flags for 1, 2 and 3, so all three are one-line edits.

**Instrumentation.** Correction to an earlier revision: I wrote that these runs "persist no
metric history at all". That is wrong about wandb - `opt/wandb` sets `use_wandb: True`
globally and `callbacks/wandb` is in the base defaults, so `WandbCallback.on_log` has been
mirroring the full metric dict to wandb all along. What was missing is a *local* copy:
`algo/trl/ppo.yaml` sets `report_to: none`, so no tfevents/CSV are written to disk, which is
why this analysis had to be reconstructed from checkpoint pickles at 50-200 iteration
resolution instead of read straight off a run directory.

Fixed in `sonic_r1_any2any.yaml` as of this run: `report_to: tensorboard` plus
`logging_dir: ${experiment_dir}/tensorboard`. wandb is unaffected and in fact picks the
tfevents up automatically, since `train_agent_trl.py:214-224` calls
`wandb.init(sync_tensorboard=True)`.

`self.approxkl_stats` is already registered (`ppo_trainer.py:1085`) and surfaces through
`_get_train_metrics`, so approxkl / `val/ratio` / `val/ratio_var` are in both backends
already - worth watching them directly rather than inferring KL from the LR, as I had to.

**Stale comment worth fixing** (`universal_token_modules.py:418`): "Keep all decoders in
ModuleDict (for checkpoint compat), but only iterate active ones" does not match the code at
L363-365, which `continue`s *before* instantiation. Same for `active_encoders` at L274-276.
Either kwarg therefore drops modules from the state dict and makes `setup_any2any` raise on
unexpected keys - which is why this run routes encoders by zeroed sampling probability
instead.

## 6. Asserts worth adding regardless of the outcome

These are all cheap, and each one converts a silent-wrong-answer failure into a startup error.
None of them can explain the identity-control collapse, but all of them must hold before the
G1→R1 negative result means anything.

```python
# 1. IsaacLab DOF order really is body order minus root  (§2.1) -- highest value
assert list(env.scene["robot"].joint_names) == TARGET_DOF_NAMES

# 2. the layout covers the whole observation  (§2.5)
assert layout[-1].end == x.shape[-1]

# 3. target action scale matches source on every matched joint  (§2.6)
assert all(abs(scale_tgt[t] - scale_src[s]) < 1e-9 for t, s in MATCHED_PAIRS)

# 4. the unaligned lower-body term is semantically identical on both robots  (§2.9)
assert mujoco_dof_names(SOURCE_ROBOT)[:12] == mujoco_dof_names(TARGET_ROBOT)[:12]

# 5. identity-mode sanity: the wrapped backbone is bit-identical to the plain one
#    when SOURCE == TARGET. Run once in CI.
assert torch.equal(Any2AnyBackbone(cfg)(obs), UniversalTokenModule(cfg)(obs))
```

---

## 7. Bottom line

The paper-fidelity of the implementation is high. Stage 1 is correct - including two reshape
decisions (`(2,F,D)` vs `(F,2,D)`, and time-major history) that were non-obvious and that the
code gets right for the right reasons. Stage 2 reproduces S7's injection scope exactly.

**And it works.** Iteration 50 of the Any2Any run beats iteration 2600 of the from-scratch
run. The G1 prior transfers to R1 through your alignment, at roughly the sample efficiency
the paper claims. Whatever else is true, that result is real and it is the hard part.

The failure is not in the alignment and not in the LoRA. It is one line of configuration:
**the exploration std is frozen at `init_noise_std` = 0.05, while this task demonstrably
wants sigma ~ 0.46 - your own from-scratch run drives it there and holds it.** Because
KL is proportional to 1/sigma^2, that 9x deficit inflates every measured KL by ~83x, which
floors the adaptive learning rate at `adaptive_lr_min` (confirmed at 4/4 checkpoints) and
strips PPO of its only trust-region brake. The adapter then drifts monotonically off the
pretrained weights - visible as ||BA||/||W|| doubling over 150 iterations with no sign of
converging - until the prior that made iteration 50 good is gone.

The docstring argument for freezing std is contradicted by the very baseline it cites: that
from-scratch run converges at sigma = 0.456, *more* noise than the source checkpoint's 0.34
that the code declines to copy.

Set `copy_source_std: true` and watch whether `lr` comes off the floor. That is Test 1, and
it is one line.

---

## 8. Experimental campaign (Sept 4-9) - what is now settled

Supersedes the mechanism proposed in 3.0/3.3. That section's *std* finding stands and was a
large real gain; its *diagnosis of the residual failure* has since been superseded twice.

### 8.1 Arm-by-arm results  (R1, 4096 envs, seed 0, 1% AMASS subset)

| arm | encoders | sigma | critic | peak reward | at iter | notes |
|---|---|---|---|---|---|---|
| poseoffset_s7 | 3 | 0.05 | pretrained | ~4.96 | ~150 | LR floored from iter 1 |
| freshcritic (old) | 3 | 0.05 | fresh | 4.28 | 150 | decayed to 0.22 by 800 |
| g1only_nostd | 1 | 0.05 | pretrained | 9.89 | 38 | LR floored from iter 1 |
| stdfix_g1only | 1 | 0.34 | pretrained | 11.69 | 788 | diverged past ~4600 |
| g1only_std_freshcritic | 1 | 0.34 | fresh | 9.85 | 1190 | slowest decay |
| diag_baseline | 1 | 0.34 | pretrained | 11.69 | 788 | exact replica of stdfix (seed 0) |
| diag_shuffleadv | 1 | 0.34 | pretrained | 5.53 | 13 | NULL CONTROL, collapsed to -6.6 |

Attribution: the **encoder scope** roughly doubles peak height (4.96 -> 9.89 at matched
sigma); the **std** buys durability (peak at iter 788 vs 38, far slower decay). Roughly
orthogonal contributions. A fresh critic neither helps nor hurts much once sigma is right.

### 8.2 Hypotheses eliminated, with the evidence

- **Curriculum confound** - REFUTED. `adp_samp/prob_max_over_uniform` peaks at 33.4 near iter
  800 then *falls* to 17.7 by 6000 while reward collapses; `effective_num_bins` recovers
  1995 -> 2400. The task gets *easier* during the decay.
- **Penalties eating the tracking reward** - REFUTED. Per *step* (Episode_Reward is an
  episode sum over a constant, reward_manager.py:120), total tracking reward is flat:
  8.81e-3 at iter 193 vs 8.57e-3 at iter 5211, a 3% change. The prior's skill stays intact.
- **Trust-region blowup** - REFUTED for the sigma=0.34 arms. approxkl steady at 0.0161,
  ratio 0.999, ratio_var 1e-6 for 2000 iterations.
- **Bad critic / uninformative advantages** - REFUTED. `critic/explained_variance` rises to
  **0.988** and never degrades. The shuffled-advantage null control collapses to reward -6.6
  / length 3.0 / time_out 0.000 versus +5.3 / 68 / 0.168 for the identical seeded baseline,
  so the advantages carry substantial real signal.
- **LoRA drift is a random walk** - PARTIALLY REFUTED. ||(alpha/r)BA|| does grow as t^0.48
  (measured over two decades), but the shuffle control proves the gradient is informative, so
  this is not free diffusion - it is a forced walk with a restoring force.

### 8.3 The remaining mechanism: the update has no off switch

```
desired_kl = 0.01  ->  controller hold band = [0.005, 0.020]
iterations with KL inside the hold band : 2000 / 2000  (100.0%)
iterations with KL > 0.02 (lr down)     : 0
iterations with KL < 0.005 (lr up)      : 0
KL: min 0.0116  mean 0.0161  max 0.0175   |  cumulative ~640 nats over 2000 iters
grad/clip_active_frac = 1.00 at every one of 2000 iterations
   (raw grad norm 1.6-4.0 vs max_grad_norm = 0.1, so every step is renormalised)
```

Two independent mechanisms enforce a step-size **floor**: the KL controller's dead band
never fires (0.0161 sits inside it), and the gradient clip is saturated so magnitude
information is discarded and every step has constant norm. Nothing in this configuration
can tell PPO to stop moving once the policy is good.

Corroborating signature: `grad/norm_mean` **grows 1.6 -> 4.0 while performance declines** -
a restoring force working harder as the policy is dragged further from the good solution. It
holds the line for ~2000 iterations, loses ground slowly, and past ~4600 loses entirely
(approxkl 10^3-10^4, |dmu| ~ 11.6 action units, reward -17.8).

This is specific to *adapting an already-good policy*. From-scratch training wants constant
motion, which is why the same hyperparameters are fine there and fatal here - and why the
paper's curves converge while ours plateau then decay.

### 8.4 Instrumentation added (ppo_trainer.py)

- `critic/explained_variance`, `critic/return_std`, `critic/raw_advantage_{mean,std,absmean}`,
  `critic/adv_to_return_ratio` - captured in `_compute_returns` **before** normalisation. The
  pre-existing `val/advantage_*` metrics are post-normalisation and therefore (0,1) by
  construction, i.e. uninformative.
- `grad/norm_mean`, `grad/norm_max`, `grad/clip_active_frac` - the norm was computed in
  `_gradient_clipping` and discarded.
- `algo.config.shuffle_advantages` - null control that permutes advantages within each
  micro-batch over the flattened batch*seq axis.
- **Fixed a silent metric loss**: `process_ep_infos` returns 0-dim tensors and `log()` passed
  the *raw* dict to the callbacks, so HF's TensorBoardCallback rejected every one - 480k
  dropped-scalar warnings per run, costing all 12 per-term `Episode_Reward` and all 5
  `Episode_Termination` curves. wandb coerced them, so wandb had the data all along and
  tensorboard did not. `log()` now passes the sanitised dict. Tag count 28 -> 124.

---

## 9. ROOT CAUSE: the measured KL is numerical noise, not policy movement

### 9.1 The frozen-policy test

Run `diag_lr0_frozen`: `actor_learning_rate=0`, `desired_kl=null` (controller disabled), so
**no parameter is ever updated**. With `SONIC_KL_DEBUG=1`:

```
[KL_DBG 1] kl=0.0113 = sigma_part 0.0003 + mu_part 0.0111
           |dmu| mean=6.155e-03 max=3.539e-01
           sigma new=0.33753 old=0.33753 ratio=1.00000
           mu(1024, 24, 26) old_mu(1024, 24, 26)
```

Identical weights, identical stored observations - and the action mean recomputed during the
PPO update differs from the one recorded during the rollout by **6.2e-3 on average and 0.354
at the extreme**, yielding KL = 0.0113 between a policy and *itself*.

### 9.2 What this invalidates

**Every KL number in every run in this document is dominated by this artifact.**

- `desired_kl = 0.01` is **below the noise floor of 0.0113**. The controller can never be
  satisfied. With the 2x dead band [0.005, 0.020] the floor sits inside it, so the controller
  holds forever and the policy is moved at a fixed rate indefinitely -- the "no off switch"
  behaviour of 8.3 is a *consequence* of this, not an independent bug.
- Lowering `desired_kl` to 0.002 (band [0.001, 0.004]) puts the noise floor *above* the upper
  edge, so the LR is slammed to `adaptive_lr_min` on iteration 1 and pinned there. Observed
  exactly: lr 2e-5 -> 1e-7 immediately, and **KL did not change** (0.0161 -> 0.0139) despite a
  200x LR reduction. A KL that is independent of the learning rate is not caused by learning.
- The sigma=0.05 arms are the same artifact scaled: KL ~ 1/sigma^2, so the floor at
  sigma=0.05 predicts 0.0111 x (0.34/0.05)^2 = **0.51**. Measured in `g1only_nostd`:
  **0.60-0.79**. So the "KL is 60-76x desired_kl" reading in 3.0 was *also* numerical noise,
  not policy divergence. The sigma finding stands (it is a real and large improvement) but the
  mechanism given in 3.0 was wrong.
- The PPO **importance ratio is corrupted**: `ratio = exp(new_logp - old_logp)` with a
  spurious |dmu| of up to 0.354 at sigma=0.34 (a z-shift of ~1.0). Part of the steady
  `policy/clipfrac ~ 0.08` is samples clipped because of arithmetic, not policy change.

### 9.3 ROOT CAUSE (verified): rollout runs fp32, the PPO update runs bf16

`trl.PPOConfig` defaults **`bf16=True`** - it is never set anywhere in this repo's configs.
`HF TrainingArguments.__post_init__` then exports `ACCELERATE_MIXED_PRECISION=bf16`, so the
`Accelerator` constructed at `train_agent_trl.py:184` comes up with `mixed_precision="bf16"`,
and `accelerator.prepare(model)` wraps the **prepared model's** `forward` in
`torch.autocast(bfloat16)`.

The two sides of PPO then reach the policy by different routes:

| | call route | precision |
|---|---|---|
| rollout | `unwrap_model_for_generation(self.model, ...)` -> unwrapped module | **fp32** |
| PPO update | `model.forward(modes=[...], ...)` on the prepared model | **bf16** |

`unwrap_model_for_generation` unwraps *past* accelerate's autocast wrapper, so the policy that
generates the data is not the policy that PPO evaluates against it.

Stage-by-stage diff of the two call paths, frozen policy (`actor_learning_rate=0`, so no
parameter ever changes), identical stored observations:

```
00_raw_actor_obs      0.000e+00      03d_w0_checksum     0.000e+00   (bit-identical weights)
01_aligned_actor_obs  0.000e+00      03c_enc_id          0.000e+00   (same module object)
03a_enc_input         0.000e+00      03g_autocast_on     1.000e+00   <-- ON vs OFF
03e_layer0_out        4.842e-02  (100% of elements differ)
04_latent_g1          1.058e-02  (100%)
05_tokens_g1          6.250e-02  (0.71%)   <-- exactly one FSQ bin step (2/32)
08_action_src         3.539e-01  (100%)
09_action_tgt         3.539e-01  mean 6.155e-03
```

bf16 carries 8 mantissa bits (~4e-3 relative). That perturbs the frozen encoder's latent by
~1e-2, which is enough to flip the FSQ bin for 0.71% of token elements - and because FSQ is a
step function, a flipped bin is a *discrete* O(0.3) jump in the decoded action. Hence the
signature we kept seeing: small mean (6e-3) with a 50x heavier max (0.354).

**Verification.** Same run with `bf16: false`:

```
every stage diff = 0.000e+00 exactly, including 09_action_tgt
KL (frozen policy) : 0.0113 -> 0.0003     |dmu| : 6.2e-3 -> 6.3e-7
```

and the residual 0.0003 is exactly the `26 * log(1 + 1e-5)` epsilon term in the KL
expression, i.e. not policy movement at all.

**Corrected attribution.** An earlier revision of this section blamed TF32 + shape-dependent
GEMM kernels. That was wrong and was falsified directly: `SONIC_DISABLE_TF32=1` left `|dmu|`
at 6.167e-3 versus 6.155e-3 with TF32 on. Also excluded by measurement, each exactly zero:
train-vs-eval mode, batch shape (B,S) vs (BS,1), the `is_training` flag, observation
mutation, `std` mutation, forward determinism, and module/parameter identity (69/69 shared
tensors, max param diff 0).

**Fix applied**: `bf16: false` in `gear_sonic/config/algo/trl/ppo.yaml`. The speed-preserving
alternative is to keep bf16 and wrap the rollout forward in the same autocast context so both
sides agree.

### 9.4 Why this defeats Any2Any specifically

From-scratch training tolerates it: the real policy change per update dwarfs a 6e-3
perturbation, so the noise is a small additive term. Adapting an already-good policy does
not: the intended update is *smaller* than the noise floor, so the trust region, the LR
controller and the importance ratio are all driven by arithmetic rather than by learning.
That is why the same hyperparameters converge from scratch and plateau-then-decay here, and
why the paper's curves look nothing like ours.
