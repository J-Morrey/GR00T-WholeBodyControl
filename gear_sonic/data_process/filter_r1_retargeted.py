#!/usr/bin/env python3
"""Filter an already-retargeted flat clip directory with the Bones-SEED rules.

``filter_and_copy_bones_data.py`` filters the *source* Bones-SEED corpus before
retargeting, and it expects the ``bones_xxx/`` nested layout. The R1 clips were
retargeted from an already-flattened 1% sample, so they never went through that
step -- they sit in a single flat directory of ``<clip>.pkl`` plus a
``metadata.pkl`` lookup dict.

This applies the *same* exclusion rules to that flat layout so the R1 training
set matches the G1 one. To guarantee parity rather than approximate it:

* ``should_filter_out`` is imported directly from the G1 script, so the matching
  logic is literally shared code.
* the keyword list is read out of that script's ``--filter-keywords`` argparse
  default via ``ast``, so it tracks any future edit instead of drifting.

``metadata.pkl`` is never filtered (matching the G1 script, which special-cases
it); it is rewritten containing only the surviving clips. That is cosmetic --
``motion_lib_base.py:378`` builds the clip list by globbing ``*.pkl`` and
excluding ``metadata.pkl``, so the dict is a lookup table, not the index -- but
it keeps the two in sync.

Usage:
    python gear_sonic/data_process/filter_r1_retargeted.py \
        --source /mnt/fast/soma_r1_1pct_sonic \
        --dest   /mnt/fast/r1_1_pct_filtered
"""

import argparse
import ast
import os.path as osp
from pathlib import Path
import shutil

from filter_and_copy_bones_data import should_filter_out

_G1_SCRIPT = Path(__file__).with_name("filter_and_copy_bones_data.py")


def load_default_filter_keywords(script: Path = _G1_SCRIPT) -> list[str]:
    """Read the ``--filter-keywords`` argparse default out of the G1 script."""
    tree = ast.parse(script.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--filter-keywords"
        ):
            for kw in node.keywords:
                if kw.arg == "default":
                    return [str(k) for k in ast.literal_eval(kw.value)]
    raise RuntimeError(f"--filter-keywords default not found in {script}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--add-keywords", nargs="+", default=None)
    args = p.parse_args()

    src, dest = Path(args.source), Path(args.dest)
    keywords = load_default_filter_keywords()
    if args.add_keywords:
        keywords.extend(args.add_keywords)
    print(f"{len(keywords)} filter keywords loaded from {_G1_SCRIPT.name}")

    pkls = sorted(src.glob("*.pkl"))
    kept, dropped = [], []
    for f in pkls:
        if f.name == "metadata.pkl":
            continue
        # Match the G1 convention of testing "<parent>/<base>".
        name_to_check = f"{osp.basename(f.parent)}/{f.name}"
        (dropped if should_filter_out(name_to_check, keywords) else kept).append(f)

    total = len(kept) + len(dropped)
    pct = 100.0 * len(dropped) / max(total, 1)
    print(f"\n{total} clips: keep {len(kept)}, drop {len(dropped)}  ({pct:.2f}% filtered out)")
    print("\nfirst 15 dropped:")
    for f in dropped[:15]:
        hit = next(k for k in keywords if k.lower() in f.name.lower())
        print(f"    {f.name}   [{hit}]")
    if len(dropped) > 15:
        print(f"    ... and {len(dropped) - 15} more")

    if args.dry_run:
        print("\nDRY RUN - nothing written")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    for f in kept:
        shutil.copy2(f, dest / f.name)

    meta_src = src / "metadata.pkl"
    if meta_src.exists():
        import joblib

        meta = joblib.load(meta_src)
        keep_stems = {f.stem for f in kept}
        if isinstance(meta, dict):
            filtered = {k: v for k, v in meta.items() if k in keep_stems}
            print(f"metadata.pkl: {len(meta)} -> {len(filtered)} entries")
            joblib.dump(filtered, dest / "metadata.pkl")
        else:
            shutil.copy2(meta_src, dest / "metadata.pkl")

    n_out = len(list(dest.glob("*.pkl")))
    print(f"\nwrote {n_out} files ({len(kept)} clips + metadata) to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
