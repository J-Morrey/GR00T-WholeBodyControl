# R1 sim2real TODO

What is left between the current R1 training runs and a SONIC policy that we can put on
real Unitree R1 hardware without breaking it.

Scope: the R1 target only. G1 is the source embodiment and its deployment path already
works; nothing here asks for G1 changes except where R1 can reuse G1 code.

Status marks:

| Mark | Meaning |
|---|---|
| ✅ | Done and verified. |
| 🔵 | In flight right now. |
| 🔴 | Blocking — no hardware attempt should happen without it. |
| 🟡 | Needed for a *good* policy, not for a safe first attempt. |
| ⚪ | Worth doing, not on the critical path. |
| ❓ | Open question; needs a measurement or a decision before it can be scheduled. |

**Scheduling note.** Items in §2 (domain randomization) change the training distribution, so
they have to be decided *before* the final long run — retrofitting them means retraining.
Items in §1 and §4 are code that can be written while that run trains. Plan accordingly:
settle §2 first even though §1 is what actually blocks hardware.

---

## 0. Already done

- ✅ R1 robot config with Unitree's own gains, armature, effort limits and actuator grouping
  (`gear_sonic/envs/manager_env/robots/r1.py`), cross-checked against `r1_constants.py` and
  the gains Unitree's deploy stack writes to hardware.
- ✅ `feet_acc` reward rescaled to the new ankle stiffness so fixing the actuator model did
  not silently retune the reward (`config/exp/manager/universal_token/all_modes/sonic_r1.yaml:62`).
- ✅ MJCF inertials refreshed from the URDF; all 27 merged bodies now agree to 1e-6, totals
  match at 29.13565 kg. Affects the MuJoCo path only — `from_mjcf` reads name/pos/quat/range
  and nothing else (`gear_sonic/utils/motion_lib/torch_humanoid_batch.py:257`).
- ✅ ONNX export in the R1's own frame. The Any2Any encode/decode entry points are
  G1-shaped; the alignment-aware exporters fold `align_inputs` and `_to_target_action` into
  the graph so the exported files take R1 observations and emit 26 R1 actions
  (`gear_sonic/utils/inference_helpers.py`). Verified against PyTorch at 1.4e-6 absolute.
  Contract, also dumped per-export as `any2any_layout.yaml`:

  ```
  encoder  [encoder_index(1) | tokenizer_target(1661)]            -> encoded_tokens(64)
  decoder  [encoded_tokens(64) | proprioception_target(840)]      -> action(26)
  policy   [encoder_index(1) | tokenizer(1661) | prop(840)]       -> action(26)
  ```

  Output is the deterministic action mean, not a sample. Head DOFs (target indices 8, 13)
  are driven by no source joint and come out exactly 0.0 — i.e. held at default pose.

---

## 1. Blocking: the deployment path does not exist for R1

Everything below the policy is G1-only today. This is the largest body of remaining work and
none of it is research — it is plumbing, and it is where sim2real attempts usually die.

### 1.1 🔴 MuJoCo sim2sim for R1

There is no R1 sim2sim path at all:

- `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py:56` raises for anything but
  `g1`/`h1`/`go2`.
- `gear_sonic/utils/mujoco_sim/configs.py:317` hardcodes the single WBC config,
  `wbc_configs/g1_29dof_sonic_model12.yaml`.
- `gear_sonic/scripts/run_sim_loop.py:46` calls `instantiate_g1_robot_model()`; only
  `gear_sonic/data/robot_model/instantiation/g1.py` exists.
- `gear_sonic/data/robot_model/model_data/` contains `g1/` only.

What R1 needs, taking the G1 config as the template:

- [ ] A sim2sim scene XML (`model_data/r1/scene_26dof.xml`). The training MJCF
      (`data/assets/robot_description/mjcf/r1.xml`) is kinematics + actuators only — no
      `<sensor>` block, no floor, no keyframe — so it is a starting point, not the scene.
- [ ] `wbc_configs/r1_26dof_sonic.yaml`: `NUM_MOTORS: 26`, `MOTOR2JOINT`/`JOINT2MOTOR`,
      `MOTOR_KP`/`MOTOR_KD`, `DEFAULT_MOTOR_ANGLES`.
- [ ] An `r1` branch in the bridge and `instantiate_r1_robot_model()`.

**This is the highest-value item in the document.** Sim2sim in MuJoCo with the exported ONNX
catches observation ordering, action scale, default-pose offset and gain-convention errors at
zero risk, and those four account for most first-day hardware failures. Do it before anything
touches a robot.

### 1.2 🔴 R1 deploy runtime

`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref` is G1-specific throughout —
`policy_parameters.hpp` carries `std::array<..., 29>` tables for action scale, kp, kd and
default angles, plus the G1 IsaacLab↔MuJoCo permutation at lines 100/103.

Two further complications beyond "change 29 to 26":

- The app was written against the **old** ONNX contract. The Any2Any export has a different
  one (leading `encoder_index` slot; 26-dim target-frame action). The input packing and
  output handling both change.
- It runs TensorRT. A Python deploy on `unitree_sdk2py` + onnxruntime would be far faster to
  get working and is probably the right first target, accepting the latency cost, with the
  C++ port deferred until the policy is actually worth deploying hard.

- [ ] Decide: port the C++ app, or write a Python R1 deploy first. (Recommend Python first.)
- [ ] Whichever: generate the constants from `any2any_layout.yaml` rather than hand-copying
      them. The layout dump exists precisely so these tables cannot drift.

### 1.3 🔴 Ankle parallel linkage

R1's ankle is a parallel A/B mechanism. The URDF carries the full linkage
(`left_ankle_A_joint`, `left_ankle_A_rod_joint`, `left_ankle_constraint_A_joint`, and the B
equivalents), but IsaacLab's URDF importer merges every one of those links into
`*_knee_link` / `*_ankle_roll_link` as fixed bodies — visible in the importer warnings at
startup. So **training models the ankle as a serial pitch/roll pair** with the linkage's mass
lumped into the parent.

Consequences:

- The policy emits ankle pitch/roll. Hardware wants A/B motor commands. Something must do
  the mapping, and the torque limits are coupled through the linkage Jacobian, not per-DOF.
- The sim does not see the linkage's coupled position limits, its reflected inertia, or its
  singular configurations.

Good news: the G1 deploy app already has the concept —
`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/robot_parameters.hpp:77` defines
`Mode::PR` (series) vs `Mode::AB` (parallel). R1's own linkage geometry needs filling in.

- [ ] Extract R1's pitch/roll ↔ A/B mapping and its Jacobian from the URDF geometry.
- [ ] Decide whether firmware accepts PR mode for R1 (if it does, most of this disappears).
- [ ] ❓ Measure whether the serial approximation's reachable ankle set exceeds the real
      linkage's. If it does, reference motions can command ankle poses the robot cannot
      reach, and that has to be clamped somewhere.

### 1.4 🔴 Gain and convention parity audit, sim ↔ hardware

A classic instant-failure source, and there is already a smell: the G1 deploy app writes
`2.0 * STIFFNESS_5020` for ankles (`policy_parameters.hpp:148`) while sim uses the raw
constant. Whether that factor is a G1 parallel-ankle convention, a deploy-side correction, or
something else needs to be known before R1 inherits or omits it.

- [ ] Confirm the kp/kd the R1 firmware applies equals what IsaacLab applies
      (`r1.py:233-312`), including units and any per-mode scaling.
- [ ] Confirm `default_angles` on hardware equals the sim default pose, joint for joint. The
      Any2Any policy already carries a default-pose offset correction between G1 and R1
      frames; a *third* disagreement with hardware would be silent and large.
- [ ] Confirm control rate: sim is 50 Hz (`config/manager_env/base_env.yaml:30,32` —
      `decimation: 4`, `sim_dt: 0.005`). Verify the deploy loop runs at the same rate.
- [ ] Confirm joint sign conventions and zero offsets after hardware calibration.

### 1.5 🔴 Safety layer

Nothing in the R1 path has any of this:

- [ ] Startup FSM: zero-torque → damping → ramp to default pose → policy. Never start the
      policy from an arbitrary pose.
- [ ] Abort to damping on: tracking error blowup, base tilt beyond threshold, joint-limit
      violation, torque saturation, DDS timeout, operator E-stop.
- [ ] Output clamps applied *after* the policy and independent of it: joint position limits,
      joint velocity limits, torque limits.
- [ ] Action-rate limit, so a single bad inference cannot step-command the full action range.
- [ ] A watchdog on inference time.

---

## 2. Domain randomization and training fidelity

*(This is the user's item 2, expanded with what the audit turned up.)*

### What exists today

Composed in `config/manager_env/events/tracking/level0_4.yaml`, five terms:

| Term | Mode | Range | Note |
|---|---|---|---|
| `physics_material` | startup | static 0.3–1.6, dynamic 0.3–1.2, restitution 0–0.5 | good coverage |
| `add_joint_default_pos` | startup | ±0.01 rad | models encoder zero error |
| `base_com` | startup | x ±0.025, y/z ±0.05 m | **waist link only** |
| `randomize_rigid_body_mass` | startup | scale 0.8–2.5 | **wrists + waist only** |
| `push_robot` | interval 4–6 s | ±0.5 m/s lin, ±0.52/0.78 rad/s ang | fine |

Observation noise *is* configured and is reasonable
(`config/manager_env/observations/policy/local_dir_hist.yaml`): gravity dir ±0.05, base ang
vel ±0.2, joint pos ±0.01, joint vel ±0.5, all uniform additive, plus ±0.05 on the tokenizer
orientation/SMPL terms. Terrain is already rough trimesh for R1, not flat
(`sonic_r1.yaml:104`).

So the starting point is better than "nothing", but the gaps below are the ones that matter
for hardware.

### 2.1 🔴 Actuator gain randomization

The single most commonly cited DR term for humanoid sim2real, and it is **not used**.
IsaacLab provides it ready-made: `randomize_actuator_gains` at
`IsaacLab-HARL/source/isaaclab/isaaclab/envs/mdp/events.py:541`.

- [ ] Add a `terms/randomize_actuator_gains.yaml`, scale kp and kd independently, startup
      mode, something like 0.85–1.15 to start. Widen only with evidence.

Note this interacts with §1.4: if the hardware gain convention turns out to differ by a fixed
factor, gain randomization wide enough to span it buys real robustness.

### 2.2 🔴 Joint friction and armature randomization

Also available and unused: `randomize_joint_parameters` at the same file, line 652 — it
covers joint friction coefficient, armature, and joint limits.

- [ ] Randomize joint friction. The sim currently has effectively none; Unitree's own MJCF
      carries `damping=0.05`, `frictionloss=0.2` per joint (noted at `r1.py:165`) which the
      IsaacLab config does not reproduce.
- [ ] Randomize armature ±10–20% around the configured `ARMATURE`.

### 2.3 🟡 Whole-body mass and COM

Current mass randomization touches only `.*_wrist_roll_link|waist_yaw_link`, and COM only the
waist (`sonic_r1.yaml:86-93`). That is a payload model, not a model-error model.

- [ ] Add a whole-body mass term, ±10–20% on `.*`, in addition to the payload term.
- [ ] Extend COM randomization beyond the waist — at minimum the thigh/shank links, which is
      where mass-distribution error most affects balance.

### 2.4 🔴 Actuation and sensing latency

There is **no delay model anywhere in training** — a grep for latency/delay across
`gear_sonic/` finds only camera and VLA inference code, nothing in the env.

Real stack: sensor read → DDS → policy → DDS → motor, typically 1–2 control steps at 50 Hz
plus jitter. A policy trained at zero delay learns gains it cannot sustain with delay, and
this shows up as exactly the high-frequency ankle chatter that destroys hardware runs.

- [ ] Implement an observation/action delay buffer with a randomized per-episode delay of
      0–2 control steps. This needs new code; IsaacLab does not provide it as an event term.
- [ ] ❓ Measure the real loop's end-to-end delay on R1 hardware before fixing the range.

### 2.5 🟡 IMU model

Current noise is i.i.d. uniform, resampled every step. Real IMUs have **persistent** error:
per-boot bias, slow drift, and a mounting rotation that is never exactly nominal.

- [ ] Add a per-episode constant bias on `gravity_dir` and `base_ang_vel` on top of the
      existing per-step noise.
- [ ] Add a small random mounting rotation (~1–2°) applied to the IMU frame per episode.
- [ ] ❓ Decide which physical IMU the R1 deploy reads. The URDF has `imu_in_pelvis_link`,
      and the bridge also publishes a torso IMU on `rt/secondary_imu`. The policy's
      observations are in the pelvis root frame; if deploy reads the torso IMU, the frames
      do not match and that difference must be compensated explicitly.

### 2.6 🟡 Motor saturation model

Actuators are `ImplicitActuatorCfg` with `effort_limit_sim`. Hardware additionally has
velocity-dependent torque limits (the motor's torque-speed curve), stiction, backlash and
thermal derating.

- [ ] At minimum, add a torque penalty or soft limit well inside the hardware limit so the
      policy keeps headroom rather than learning to live at saturation.
- [ ] ⚪ Consider a torque-speed envelope. Diminishing returns versus the items above.

### 2.7 ⚪ Other

- [ ] Gravity magnitude randomization (`randomize_physics_scene_gravity`, same events file,
      line 498) — cheap, modest value.
- [ ] Ground-height / slope variation beyond what the trimesh generator already does.

---

## 3. Policy quality

### 3.1 🟡 Train with all three encoders

*(The user's item 1.)*

The current R1 Any2Any run routes 100% of environments through the `g1` robot-motion encoder
(`sonic_r1_any2any.yaml`, `encoder_sample_probs: g1 1.0 / teleop 0.0 / smpl 0.0`).

**The hypothesis — that nothing upstream of the motion latent is modified, so all three
encoders should produce interchangeable latents — is worth testing but is not safe to assume
here**, for two concrete reasons already documented in that config:

1. The teleop encoder's `command_multi_future_lower_body` is the one per-DOF term the
   Any2Any Stage-1 alignment does not touch, so it reaches the frozen encoder unaligned.
2. The VR 3-point targets are computed from R1's geometry, which differs substantially from
   G1's — the hand-in-wrist offset is 0.068 m on R1 against G1's 0.18 m. The teleop encoder
   is frozen and was trained on G1's numbers.

So teleop latents are *not* a priori comparable, and the SMPL branch is separately disabled
because `smpl_motion_file: dummy` (`sonic_r1.yaml:149`).

The test the hypothesis deserves, in order:

- [ ] On a trained checkpoint, evaluate each encoder at 100% and compare tracking error. This
      is cheap and directly answers the question.
- [ ] Measure latent distance between encoders on identical motions. If g1 and teleop latents
      already agree, items 1 and 2 above are harmless in practice and this gets much easier.
- [ ] If they do not agree: align `command_multi_future_lower_body` in Stage 1, and recompute
      the VR 3-point offsets so the teleop encoder sees G1-comparable geometry.
- [ ] Source real SMPL motion data to enable the smpl branch at all.
- [ ] Only then consider raising the probabilities in training.

Worth noting the upside the user identified is real: more encoders means more usable data and
a policy that can be driven by teleop at deployment, not just by retargeted clips.

### 3.2 🔵 Exploration std

The frozen-`std` collapse is diagnosed in `ANY2ANY_AUDIT.md` §3.0 and `copy_source_std: true`
is the fix currently under test. Success criterion, from the config: `lr` comes off its
`1e-5` floor and starts oscillating the way the from-scratch run's does.

- [ ] Confirm from the in-flight runs, then fix the setting and stop carrying the comparison.

### 3.3 🟡 Reference-motion feasibility for R1

R1 is 1.2 m and materially weaker than the G1 these motions were retargeted from. Motions that
saturate R1's joint limits or exceed its velocity/torque capability are not trackable, and
training on them teaches the policy to fail.

- [ ] Measure, over the corpus: fraction of frames at joint limits per joint; per-joint
      velocity distribution against motor limits; estimated torque against effort limits.
- [ ] Filter or re-retarget rather than asking the policy to track the infeasible.

### 3.4 🟡 Corpus scale

The cluster run is capped at 2048 clips per rank by **host** RAM, not GPU — bowser has ~61 GB
against 2×98 GB VRAM, and every clip is deserialized into RAM and passed through an
`mp.Manager` queue, so it exists roughly twice at peak (`slurm_scripts/train_r1.slurm:14-21`).
8192 clips was killed at the first resample.

- [ ] A streaming or memory-mapped motion loader would unlock the full corpus. This is the
      main thing standing between us and training on all the data we have.

---

## 4. Validation and tooling

### 4.1 🟡 ONNX parity regression test

The alignment-aware export was verified once, by hand. It should not be possible for it to
rot silently.

- [ ] Install `onnxruntime` — it is in neither conda env, which is why verification used
      `onnx.reference` (correct, but slow and not what deployment runs).
- [ ] Add a test: export a small checkpoint, assert ONNX output matches torch within
      tolerance, assert head slots are exactly zero, assert dims match `any2any_layout.yaml`.

### 4.2 🟡 R1 evaluation suite

- [ ] A held-out motion set never seen in training.
- [ ] Per-joint tracking error, foot slip, contact force, torque headroom, base drift.
- [ ] The same metrics computed in MuJoCo sim2sim, so sim-vs-sim divergence is quantified
      *before* sim-vs-real.

### 4.3 🔴 Hardware pre-flight checklist

To be written once §1 lands, but at minimum: gantry/harness first; joint-by-joint motion
check at low gains; default-pose match verified physically; single-motion replay before any
closed-loop policy; E-stop tested before the robot leaves the ground.

---

## 5. Suggested order

1. **Decide and land §2.1, §2.2, §2.4** (gains, joint friction, latency). These change the
   training distribution, so they must precede the final long run. §2.1 and §2.2 are
   config-only — IsaacLab already has the functions. §2.4 needs code.
2. **Build §1.1 (MuJoCo sim2sim) while that run trains.** Highest value per hour of work in
   this document.
3. **§1.4 (gain/convention parity) and §1.3 (ankle linkage)** — both are desk work that can
   happen in parallel and both are cheap to get wrong silently.
4. **§1.5 (safety layer) and §1.2 (deploy runtime)**, Python first.
5. **§3.1 (encoder study)** on whatever checkpoint exists by then — it is an evaluation, not a
   training change, so it does not gate the above.
6. **§4.3, then hardware.**

---

## 6. Open questions

- ❓ Does R1 firmware accept series (pitch/roll) ankle commands, or only A/B? (§1.3)
- ❓ Which IMU does the R1 deploy read, and in what frame? (§2.5)
- ❓ What is the real end-to-end control-loop delay on R1? (§2.4)
- ❓ Is the `2.0 *` ankle stiffness factor in the G1 deploy app a parallel-ankle convention
  that R1 inherits? (§1.4)
- ❓ Do the g1 and teleop encoders actually produce comparable latents on R1 geometry? (§3.1)
- ❓ Should `anti_shake_ang_vel` be rescaled for the new ankle damping the way `feet_acc` was?
  (Carried over, still unanswered.)
