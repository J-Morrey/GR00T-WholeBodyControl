#!/bin/bash
#SBATCH --job-name=r1-inspect
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=logs/r1-inspect-%j.out

# Reports what the retargeted dataset at $RAW actually is, so 01_prepare_data
# takes the right branch and so a schema mismatch surfaces here rather than 40
# minutes into a GPU job. Run this first; it writes nothing.

set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO"

echo "python: $(which python)"
echo "RAW:    $RAW"
echo

python - <<'PY'
import glob, os, sys
from collections import Counter
from pathlib import Path

raw = Path(os.environ["RAW"])
if not raw.is_dir():
    sys.exit(f"RAW does not exist or is not a directory: {raw}")

npz = glob.glob(str(raw / "**" / "*.npz"), recursive=True)
pkl = [f for f in glob.glob(str(raw / "**" / "*.pkl"), recursive=True)
       if os.path.basename(f) != "metadata.pkl"]
flat_pkl = [f for f in pkl if Path(f).parent == raw]

print(f"clips found: {len(npz)} .npz, {len(pkl)} .pkl ({len(flat_pkl)} of them flat in RAW)")
print(f"metadata.pkl at top level: {(raw / 'metadata.pkl').exists()}")
subdirs = [d for d in raw.iterdir() if d.is_dir()]
print(f"subdirectories: {len(subdirs)}" + (f"  e.g. {[d.name for d in subdirs[:5]]}" if subdirs else ""))

if npz and not pkl:
    layout = "npz"
elif flat_pkl:
    layout = "flat-pkl"
elif pkl:
    layout = "nested-pkl"
else:
    sys.exit("no .npz or .pkl clips found -- check RAW")
print(f"\n==> LAYOUT: {layout}")

# Schema check on one clip. motion_lib_base.py:1763-1778 requires
# root_trans_offset / pose_aa / fps; R1 is 26 DOF so pose_aa is (T, 27, 3).
sample = (pkl or npz)[0]
print(f"\nsample clip: {sample}")
if layout == "npz":
    import numpy as np
    d = np.load(sample, allow_pickle=True)
    print(f"  keys: {sorted(d.files)}")
    if "joint_pos" in d:
        print(f"  joint_pos: {d['joint_pos'].shape}   (expect (T, 33) = 3 xyz + 4 quat + 26 dof)")
    if "joint_names" in d:
        print(f"  joint_names: {len(d['joint_names'])}  (expect 26)")
else:
    import joblib
    obj = joblib.load(sample)
    if isinstance(obj, dict) and "pose_aa" in obj:
        clip = obj            # clip dict stored bare
    elif isinstance(obj, dict):
        clip, *extra = obj.values()   # the usual {clip_name: clip_dict} wrapper
        if extra:
            print(f"  NOTE: pkl holds {len(extra) + 1} clips, not a single-key dict")
    else:
        sys.exit(f"  FATAL: unexpected pkl payload type {type(obj)}")
    print(f"  keys: {sorted(clip.keys())}")
    for k in ("root_trans_offset", "pose_aa", "dof", "root_rot"):
        if k in clip:
            print(f"  {k}: {getattr(clip[k], 'shape', type(clip[k]))}")
    print(f"  fps: {clip.get('fps', 'MISSING -> motion_lib defaults to 30.0')}")
    missing = {"root_trans_offset", "pose_aa"} - set(clip)
    if missing:
        sys.exit(f"\nFATAL: required keys missing: {sorted(missing)}")

# How many clips the filter will drop, without copying anything.
if layout != "npz":
    from filter_r1_retargeted import load_default_filter_keywords
    from filter_and_copy_bones_data import should_filter_out
    kw = load_default_filter_keywords()
    hits = Counter()
    for f in pkl:
        p = Path(f)
        if should_filter_out(f"{p.parent.name}/{p.name}", kw):
            hits[next(k for k in kw if k.lower() in f"{p.parent.name}/{p.name}".lower())] += 1
    drop = sum(hits.values())
    print(f"\nfilter preview: keep {len(pkl) - drop}, drop {drop} "
          f"({100.0 * drop / max(len(pkl), 1):.2f}%) using {len(kw)} keywords")
    for k, n in hits.most_common(10):
        print(f"    {n:>7}  {k}")
PY

echo
echo "Disk free on the prep/train target filesystem:"
df -h "$(dirname "$FILTERED")"
