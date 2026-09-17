#!/bin/bash
# Chains the prep and training jobs so training starts only if prep succeeded.
# Run 00_inspect_data.slurm yourself first and read its output -- this script
# assumes you have already confirmed the dataset layout looks right.
#
#   ./submit_all.sh                    # prep, then train
#   ./submit_all.sh --train-only       # skip prep ($FILTERED already built)
#
# Both jobs carry `#SBATCH --nodelist=zac`; nothing here overrides that.

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)

# The scripts' `#SBATCH --output=logs/...` paths are relative to the cwd at
# submission time, not to the script's own location. Submitting from the repo
# root is what makes them land in <repo>/logs/ -- from anywhere else Slurm
# fails to open the output file and the job's stdout is silently lost.
cd "$HERE/../.."
mkdir -p logs

if [ "${1:-}" = "--train-only" ]; then
    shift
    sbatch "$HERE/02_train_any2any.slurm" "$@"
else
    prep=$(sbatch --parsable "$HERE/01_prepare_data.slurm")
    echo "prep job:  $prep"
    # afterok: training is never submitted if prep exits non-zero, so a failed
    # filter can't leave a GPU job training on a partial dataset.
    train=$(sbatch --parsable --dependency=afterok:"$prep" "$HERE/02_train_any2any.slurm" "$@")
    echo "train job: $train  (waits on $prep)"
fi
