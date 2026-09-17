#!/bin/bash
#SBATCH --job-name=r1-prep-data
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/r1-prep-%j.out

# Converts (if needed) and filters the retargeted R1 dataset into $FILTERED,
# which is what 02_train_any2any.slurm reads. CPU only.
#
# Branches on the layout that 00_inspect_data.slurm reports:
#   npz         -> convert_soma_npz_to_motion_lib.py, then filter the result
#   flat-pkl    -> filter_r1_retargeted.py    (single dir of <clip>.pkl)
#   nested-pkl  -> filter_and_copy_bones_data.py (bones_xxx/ subdirs, parallel)
#
# Re-running is safe but not incremental: it refuses to overwrite a populated
# $FILTERED so a half-finished earlier attempt can't be mistaken for a complete
# dataset. Delete the directory to redo it.

set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO"

echo "python: $(which python)"
echo "RAW -> FILTERED:  $RAW -> $FILTERED"
echo "cpus: ${SLURM_CPUS_PER_TASK:-1}"
echo

if [ -d "$FILTERED" ] && [ -n "$(ls -A "$FILTERED" 2>/dev/null)" ]; then
    echo "FATAL: $FILTERED already exists and is non-empty."
    echo "       Remove it to regenerate, or point FILTERED elsewhere in env.sh."
    exit 1
fi

# --- detect layout ----------------------------------------------------------
first_npz=$(find "$RAW" -name '*.npz' -print -quit)
first_flat_pkl=$(find "$RAW" -maxdepth 1 -name '*.pkl' ! -name 'metadata.pkl' -print -quit)
first_any_pkl=$(find "$RAW" -name '*.pkl' ! -name 'metadata.pkl' -print -quit)

if [ -n "$first_npz" ] && [ -z "$first_any_pkl" ]; then
    LAYOUT=npz
elif [ -n "$first_flat_pkl" ]; then
    LAYOUT=flat-pkl
elif [ -n "$first_any_pkl" ]; then
    LAYOUT=nested-pkl
else
    echo "FATAL: no .npz or .pkl clips under $RAW"; exit 1
fi
echo "detected layout: $LAYOUT"
echo

# --- step 1: convert, only if the source is raw SOMA .npz -------------------
if [ "$LAYOUT" = npz ]; then
    echo "=== converting .npz -> SONIC .pkl ==="
    if [ -d "$STAGE" ] && [ -n "$(ls -A "$STAGE" 2>/dev/null)" ]; then
        echo "reusing existing conversion at $STAGE"
    else
        # --mjcf is passed explicitly rather than relying on DEFAULT_MJCF
        # (convert_soma_npz_to_motion_lib.py:51), which is a repo-relative path.
        # The script validates joint order against this MJCF's actuators and
        # errors out on a mismatch, so it doubles as a retargeting sanity check.
        python gear_sonic/data_process/convert_soma_npz_to_motion_lib.py \
            --input "$RAW" \
            --output "$STAGE" \
            --mjcf gear_sonic/data/assets/robot_description/mjcf/r1.xml \
            --num_workers "${SLURM_CPUS_PER_TASK:-16}"
    fi
    FILTER_SRC=$STAGE
    # The converter writes one flat directory plus metadata.pkl.
    FILTER_MODE=flat
else
    FILTER_SRC=$RAW
    [ "$LAYOUT" = flat-pkl ] && FILTER_MODE=flat || FILTER_MODE=nested
fi

# --- step 2: filter ---------------------------------------------------------
echo
echo "=== filtering ($FILTER_MODE) ==="
if [ "$FILTER_MODE" = flat ]; then
    python gear_sonic/data_process/filter_r1_retargeted.py \
        --source "$FILTER_SRC" \
        --dest   "$FILTERED"
else
    # The nested path is the same code G1 used, parallelised over bones_xxx/
    # subdirs. It preserves the subdirectory structure, which is fine --
    # motion_lib_base.py:378 globs **/*.pkl recursively.
    python gear_sonic/data_process/filter_and_copy_bones_data.py \
        --source  "$FILTER_SRC" \
        --dest    "$FILTERED" \
        --workers "${SLURM_CPUS_PER_TASK:-16}"
fi

# --- verify -----------------------------------------------------------------
echo
n=$(find "$FILTERED" -name '*.pkl' ! -name 'metadata.pkl' | wc -l)
echo "=== done: $n clips in $FILTERED ==="
du -sh "$FILTERED"
if [ "$n" -eq 0 ]; then
    echo "FATAL: filter produced no clips -- training would fail on an empty motion lib"
    exit 1
fi
