#!/bin/bash
#SBATCH --job-name=r1-motionlib
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --nodelist=zac
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/r1-motionlib-%j.out

# Stage 3 of 3. R1 CSVs -> motion_lib PKLs -> the filtered training set.
# CPU only; 02_train_any2any.slurm reads $FILTERED afterwards.
#
# Three steps, each gating the next:
#   1. convert  $R1_CSV -> $R1_PKL   (compressed, one PKL per clip)
#   2. validate $R1_PKL              (schema, DOF limits, pose_aa consistency)
#   3. filter   $R1_PKL -> $FILTERED (the keyword list G1 SONIC was trained on)
#
# Step 3 is what makes this the G1-comparable training set. The retargeting
# deliberately covers all of BONES-SEED, so the unfiltered $R1_PKL stays on
# disk for anything else that wants the motions the G1 could not perform.
#
# Re-running is safe: it refuses to overwrite a populated $FILTERED, and the
# conversion simply rewrites $R1_PKL.

set -euo pipefail
REQUIRE_RAW=0 source "$HOME/repos/GR00T-WholeBodyControl/gear_sonic/slurm_scripts/env.sh"
cd "$REPO"

echo "node:    $(hostname)"
echo "csv:     $R1_CSV"
echo "pkl:     $R1_PKL"
echo "train:   $FILTERED"
echo "cpus:    ${SLURM_CPUS_PER_TASK:-1}"
echo

[ -d "$R1_CSV" ] || { echo "FATAL: $R1_CSV missing -- run 11_retarget_r1.slurm"; exit 1; }
n_csv=$(find "$R1_CSV" -name '*.csv' | wc -l)
[ "$n_csv" -gt 0 ] || { echo "FATAL: no CSVs under $R1_CSV"; exit 1; }
echo "$n_csv retargeted clips"

if [ -d "$FILTERED" ] && [ -n "$(ls -A "$FILTERED" 2>/dev/null)" ]; then
    echo "FATAL: $FILTERED already exists and is non-empty."
    echo "       Remove it to regenerate, or point FILTERED elsewhere in env.sh."
    exit 1
fi

# --- step 1: convert --------------------------------------------------------
# No resampling: entries are stored at the CSVs' native 120 fps and motion_lib
# interpolates to its own target_fps (50) at load. --compress is what keeps the
# full corpus near 30 GB instead of 86 GB in home.
echo
echo "=== converting to motion_lib ($(date -Is)) ==="
python gear_sonic/data_process/convert_soma_r1_csv_to_motion_lib.py \
    --input "$R1_CSV" \
    --output "$R1_PKL" \
    --mjcf gear_sonic/data/assets/robot_description/mjcf/r1.xml \
    --source-fps "${SOURCE_FPS:-120}" \
    --compress "$PKL_COMPRESS" \
    --num_workers "${SLURM_CPUS_PER_TASK:-16}"

# --- step 2: validate -------------------------------------------------------
# Exits non-zero on any bad clip, so a corrupt conversion never reaches
# training. --fk-sample runs SONIC's own FK over a sample as a sanity read on
# whether the motions stand on the floor.
echo
echo "=== validating ($(date -Is)) ==="
python gear_sonic/data_process/validate_r1_motion_lib.py \
    --input "$R1_PKL" \
    --mjcf gear_sonic/data/assets/robot_description/mjcf/r1.xml \
    --fk-sample 100

# --- step 3: filter ---------------------------------------------------------
# The nested <session>/<clip>.pkl layout is the one filter_and_copy_bones_data.py
# was written for, and it is the same code G1 used -- so the exclusion rules are
# literally shared rather than reimplemented.
echo
echo "=== filtering to the G1 SONIC training set ($(date -Is)) ==="
python gear_sonic/data_process/filter_and_copy_bones_data.py \
    --source "$R1_PKL" \
    --dest   "$FILTERED" \
    --workers "${SLURM_CPUS_PER_TASK:-16}"

# filter_and_copy_bones_data.py walks subdirectories and skips metadata.pkl, so
# $FILTERED comes out without one. Without it the motion library opens all
# ~130k clips at startup just to read their frame counts, because adaptive
# sampling needs the lengths. Copying the unfiltered one is enough: metadata is
# allowed to be a superset of what is on disk and the extra keys are ignored
# (motion_lib_base.py:411).
cp "$R1_PKL/metadata.pkl" "$FILTERED/metadata.pkl"
echo "copied metadata.pkl ($(python -c "import joblib,sys;print(len(joblib.load(sys.argv[1])))" "$FILTERED/metadata.pkl") entries)"

echo
n=$(find "$FILTERED" -name '*.pkl' ! -name 'metadata.pkl' | wc -l)
echo "=== done: $n clips in $FILTERED ($(date -Is)) ==="
du -sh "$R1_PKL" "$FILTERED"
[ "$n" -gt 0 ] || { echo "FATAL: filter produced no clips"; exit 1; }
echo
echo "Next: sbatch gear_sonic/slurm_scripts/02_train_any2any.slurm"
