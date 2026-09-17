#!/bin/bash
#SBATCH --job-name=r1-npz-schema
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --nodelist=zac
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/r1-npz-schema-%j.out

# The dataset at $RAW is .npz, but with keys (cost/fps/human_joints/qpos) that do
# NOT match what convert_soma_npz_to_motion_lib.py expects (joint_names/joint_pos
# /fps). It came from a different retargeting pipeline than the 1% sample the
# converter was written against, so the converter cannot run as-is.
#
# The dangerous part is joint ORDER: the converter's only safety check is
# `data["joint_names"] != mjcf_order` (convert_soma_npz_to_motion_lib.py:86),
# and this data carries no joint names at all. Feeding it in with a silently
# wrong permutation would train perfectly happily on garbage references.
#
# So this job answers, without assuming anything:
#   * is qpos laid out as [trans(3), quat(4), dof(26)] like joint_pos was?
#   * does dof, read in MJCF actuator order, respect the MJCF joint limits?
#     (a permutation control shows that test actually discriminates)
#   * what is human_joints -- the SMPL reference, or something else?
#   * is there any sidecar file naming the joint order?
# Read-only; writes nothing.

set -euo pipefail
source "$HOME/repos/GR00T-WholeBodyControl/gear_sonic/slurm_scripts/env.sh"
cd "$REPO"

echo "RAW: $RAW"
echo

python - <<'PY'
import glob, json, os, random
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

raw = Path(os.environ["RAW"])
files = sorted(glob.glob(str(raw / "**" / "*.npz"), recursive=True))
print(f"{len(files)} npz clips\n")

# ---- 1. full schema of one clip -------------------------------------------
d = np.load(files[0], allow_pickle=True)
print(f"=== schema: {Path(files[0]).name} ===")
for k in sorted(d.files):
    a = np.asarray(d[k])
    print(f"  {k:16s} shape={str(a.shape):18s} dtype={a.dtype}")
    if a.dtype.kind in "OUS" and a.size <= 40:
        print(f"      values: {list(a.reshape(-1))[:8]}")
    elif a.size and a.dtype.kind in "fiu":
        print(f"      min={np.nanmin(a):+.4f} max={np.nanmax(a):+.4f}")

# ---- 2. MJCF reference order ----------------------------------------------
mjcf = Path("gear_sonic/data/assets/robot_description/mjcf/r1.xml")
wb = ET.parse(str(mjcf)).getroot().find("worldbody")
joints = [j for j in wb.iter("joint") if j.get("type") != "free"]
names = [j.get("name") for j in joints]
lo, hi = np.array([[float(v) for v in j.get("range").split()] for j in joints]).T
ndof = len(names)
print(f"\nMJCF actuator order: {ndof} joints")

# ---- 3. qpos layout -------------------------------------------------------
qpos = np.asarray(d["qpos"])
print(f"\n=== qpos layout ===")
print(f"  shape {qpos.shape};  expected width {7 + ndof} for [trans(3), quat(4), dof({ndof})]")
if qpos.shape[1] != 7 + ndof:
    print(f"  !! WIDTH MISMATCH -- the [3,4,26] assumption is wrong, stop and inspect")
else:
    q = qpos[:, 3:7]
    print(f"  cols 3:7 norms: min={np.linalg.norm(q,axis=1).min():.4f} "
          f"max={np.linalg.norm(q,axis=1).max():.4f}  (1.0 => it is a quaternion)")
    print(f"  col 2 (root z): min={qpos[:,2].min():+.3f} max={qpos[:,2].max():+.3f} "
          f"(expect ~0.3-1.3 m for a standing humanoid)")

# ---- 4. joint order: MJCF limit violations, with a permutation control ----
# If dof is in MJCF order, nearly every sample sits inside its joint's range.
# If it is permuted, values land in the wrong joint's range and violations
# explode. The shuffled control proves the test has power rather than passing
# trivially because the limits are wide.
print(f"\n=== joint-order test: MJCF limit violations over {min(len(files),400)} clips ===")
rng = random.Random(0)
sample = rng.sample(files, min(len(files), 400))
tot = 0
viol = np.zeros(ndof, dtype=np.int64)
viol_shuf = 0
perm = np.array(rng.sample(range(ndof), ndof))
frames_total = 0
for f in sample:
    dd = np.load(f, allow_pickle=True)
    qq = np.asarray(dd["qpos"])
    frames_total += qq.shape[0]
    if qq.shape[1] != 7 + ndof:
        continue
    dof = qq[:, 7:]
    tol = 1e-3
    viol += ((dof < lo - tol) | (dof > hi + tol)).sum(axis=0)
    ds = dof[:, perm]
    viol_shuf += int(((ds < lo - tol) | (ds > hi + tol)).sum())
    tot += dof.shape[0]

pct = 100.0 * viol.sum() / max(tot * ndof, 1)
pct_s = 100.0 * viol_shuf / max(tot * ndof, 1)
print(f"  as-is (assume MJCF order): {viol.sum():,} / {tot*ndof:,} samples out of range = {pct:.3f}%")
print(f"  shuffled control:          {viol_shuf:,} / {tot*ndof:,} = {pct_s:.3f}%")
print(f"\n  VERDICT: ", end="")
if pct < 0.5 and pct_s > 5 * max(pct, 0.05):
    print("consistent with MJCF actuator order (control confirms the test discriminates)")
elif pct_s <= 5 * max(pct, 0.05):
    print("INCONCLUSIVE -- limits too permissive to distinguish order; do not rely on this")
else:
    print("NOT MJCF order -- violations too high. Joint order must be established another way")
print("\n  worst joints as-is:")
for i in np.argsort(-viol)[:8]:
    if viol[i]:
        print(f"    {viol[i]:>9,}  {names[i]:32s} range=[{lo[i]:+.4f},{hi[i]:+.4f}]")
if not viol.any():
    print("    (none -- every sample within limits)")

# ---- 5. what is human_joints ---------------------------------------------
if "human_joints" in d.files:
    hj = np.asarray(d["human_joints"])
    print(f"\n=== human_joints ===")
    print(f"  shape {hj.shape}  dtype {hj.dtype}")
    print(f"  frames match qpos: {hj.shape[0] == qpos.shape[0]}")
    if hj.ndim == 3 and hj.shape[2] == 3:
        print(f"  -> (T, {hj.shape[1]}, 3) joint POSITIONS"
              f"{'; 24 = SMPL body joint count' if hj.shape[1]==24 else ''}")
        print(f"     smpl_joints in the PKL is currently written as zeros "
              f"(convert:112) -- this could populate it instead")

if "cost" in d.files:
    c = np.asarray(d["cost"])
    print(f"\n=== cost ===  shape {c.shape}  "
          f"min={np.nanmin(c):.5f} max={np.nanmax(c):.5f}  (retargeting residual;"
          f" usable as a quality filter)")

# ---- 6. sidecar metadata -------------------------------------------------
print(f"\n=== sidecar files near the dataset ===")
hits = []
for p in [raw, raw.parent]:
    for pat in ("*.json", "*.yaml", "*.yml", "*.txt", "*.md", "*.csv"):
        hits += glob.glob(str(p / pat))
print("  " + ("\n  ".join(sorted(set(hits))) if hits else "none found"))

# ---- 7. size projection --------------------------------------------------
mean_frames = frames_total / max(len(sample), 1)
est = 745 * mean_frames * len(files)      # 745 B/frame measured from the 1% PKLs
print(f"\n=== projection ===")
print(f"  mean frames/clip {mean_frames:.0f} over {len(sample)} sampled")
print(f"  converted PKL ~{est/1e9:.0f} GB; + filtered copy ~{est*1.85/1e9:.0f} GB peak")
PY

echo
df -h "$(dirname "$FILTERED")"
