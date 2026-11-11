#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201, DOC
"""Build one array task's symlink farm of BVH clips still needing retargeting.

soma-retargeter's ``bvh_to_csv_converter.py`` takes a single ``import_folder``,
rglobs it for ``*.bvh``, and names each output
``export_folder / <path relative to import_folder>.csv``. It has no notion of a
shard and no notion of resuming. Both fall out of giving each array task its own
directory of symlinks:

* **sharding** -- clip *i* of the corpus goes to task ``i % count``.
* **resume** -- a clip whose CSV already exists is simply not linked, so a
  re-submitted array picks up exactly where the last one died. Nothing is
  recomputed and nothing is overwritten.

Because the symlinks keep the ``<session>/<clip>.bvh`` layout, the CSVs land as
``<session>/<clip>.csv`` -- 142k files spread over a few hundred directories
rather than one directory with 142k entries in it.

Clips are ordered by size before striding, so every shard gets a comparable mix
of long and short motions. Retargeting cost is roughly linear in frame count, and
a contiguous split would hand one task all the longest clips.

Usage (from 11_retarget_r1.slurm):
    python make_retarget_shard.py --bvh-root "$BVH_ROOT" --csv-root "$R1_CSV" \
        --shard-dir "$SHARD_ROOT/$SLURM_ARRAY_TASK_ID" \
        --index "$SLURM_ARRAY_TASK_ID" --count "$RETARGET_SHARDS"
"""

import argparse
from pathlib import Path
import shutil
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bvh-root", type=Path, required=True)
    ap.add_argument("--csv-root", type=Path, required=True)
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    args = ap.parse_args()

    if not 0 <= args.index < args.count:
        raise SystemExit(f"--index {args.index} outside 0..{args.count - 1}")
    if not args.bvh_root.is_dir():
        raise SystemExit(f"BVH root not found: {args.bvh_root}")

    print(f"scanning {args.bvh_root} ...", flush=True)
    clips = sorted(args.bvh_root.rglob("*.bvh"))
    if not clips:
        raise SystemExit(f"no .bvh files under {args.bvh_root}")
    # Sort by size so the stride below spreads long motions evenly.
    clips.sort(key=lambda p: p.stat().st_size, reverse=True)
    mine = clips[args.index :: args.count]

    todo = []
    for src in mine:
        dst = (args.csv_root / src.relative_to(args.bvh_root)).with_suffix(".csv")
        # Size check as well as existence: a CSV from a job killed mid-write is
        # truncated or empty, and silently keeping it would punch a hole in the
        # dataset that only turns up at training time.
        if dst.is_file() and dst.stat().st_size > 0:
            continue
        todo.append(src)

    # Rebuilt from scratch every run: a farm left over from an earlier, wider
    # array would otherwise hand this task clips that are not its own.
    if args.shard_dir.exists():
        shutil.rmtree(args.shard_dir)
    for src in todo:
        link = args.shard_dir / src.relative_to(args.bvh_root)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(src.resolve())

    done = len(mine) - len(todo)
    total_gb = sum(p.stat().st_size for p in todo) / 1e9
    print(f"corpus          : {len(clips)} clips")
    print(f"shard {args.index}/{args.count}   : {len(mine)} clips")
    print(f"already done    : {done}")
    print(f"to retarget     : {len(todo)} clips, {total_gb:.1f} GB of BVH")
    print(f"linked into     : {args.shard_dir}")
    if not todo:
        print("nothing to do for this shard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
