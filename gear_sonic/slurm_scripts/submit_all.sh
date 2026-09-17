#!/bin/bash
# Chains the prep and training jobs so training starts only if prep succeeded.
# Run 00_inspect_data.slurm yourself first and read its output -- this script
# assumes you have already confirmed the dataset layout looks right.
#
#   ./submit_all.sh                    # prep, then train
#   ./submit_all.sh --train-only       # skip prep ($FILTERED already built)

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../../logs

if [ "${1:-}" = "--train-only" ]; then
    shift
    sbatch 02_train_any2any.slurm "$@"
else
    prep=$(sbatch --parsable 01_prepare_data.slurm)
    echo "prep job:  $prep"
    # afterok: training is never submitted if prep exits non-zero, so a failed
    # filter can't leave a GPU job training on a partial dataset.
    train=$(sbatch --parsable --dependency=afterok:"$prep" 02_train_any2any.slurm "$@")
    echo "train job: $train  (waits on $prep)"
fi
