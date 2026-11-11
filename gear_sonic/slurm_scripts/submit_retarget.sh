#!/usr/bin/env bash
# Submits the whole R1 pipeline as one dependency chain and returns immediately.
#
#   ./submit_retarget.sh                   # extract -> retarget -> motion_lib
#   ./submit_retarget.sh --with-training    # ... -> train, all in one chain
#   ./submit_retarget.sh --skip-extract     # BVH already unpacked on scratch
#   ./submit_retarget.sh --retarget-only    # stop after the CSVs
#
# This is a submitter, not a job. A single master job holding an allocation
# while it waits for its own children would occupy a GPU for the hours that
# stages 1 and 3 spend on CPU work, and a crash in it would strand everything.
# Slurm dependencies do the same sequencing for free, and each stage asks for
# only the resources it needs.
#
# Every link is afterok, so a failed stage stops the chain rather than letting
# the next one run on a partial dataset. Submit it and log out; `squeue -u
# $USER` shows where it has got to.
#
# --nodelist and --array are passed on the command line, which overrides the
# #SBATCH directives inside the job files. That makes $CLUSTER_NODE and
# $RETARGET_SHARDS in env.sh the single source of truth for both.

set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

# REQUIRE_RAW=0: $RAW is a training input that does not exist yet, and the
# guard in env.sh would otherwise refuse to source on the login node.
REQUIRE_RAW=0 source "$HERE/env.sh"

# As in submit_all.sh: #SBATCH --output paths are relative to the cwd at
# submission time, so submit from the repo root or Slurm cannot open them.
cd "$HERE/../.."
mkdir -p logs

skip_extract=0
retarget_only=0
with_training=0
for arg in "$@"; do
    case "$arg" in
        --skip-extract)  skip_extract=1 ;;
        --retarget-only) retarget_only=1 ;;
        --with-training) with_training=1 ;;
        *) echo "unknown option: $arg" >&2; exit 1 ;;
    esac
done

if [ "$RETARGET_SHARDS" -lt 1 ]; then
    echo "FATAL: RETARGET_SHARDS must be >= 1 (got $RETARGET_SHARDS)" >&2
    exit 1
fi
ARRAY="0-$((RETARGET_SHARDS - 1))"
NODE="--nodelist=$CLUSTER_NODE"

echo "node:     $CLUSTER_NODE"
echo "shards:   $RETARGET_SHARDS  (--array=$ARRAY)"
echo "tarball:  $SEED_ROOT/soma_uniform.tar.gz"
echo "bvh:      $BVH_ROOT"
echo "csv:      $R1_CSV"
echo "train on: $FILTERED"
echo

dep=""
if [ "$skip_extract" -eq 0 ]; then
    # shellcheck disable=SC2086
    extract=$(sbatch --parsable $NODE "$HERE/10_extract_bvh.slurm")
    echo "  extract     $extract"
    dep="--dependency=afterok:$extract"
fi

# afterok on an array waits for every task, so one timed-out shard stops the
# chain instead of converting a dataset with holes in it. Re-submitting the
# array picks up only the clips that are still missing.
# shellcheck disable=SC2086
retarget=$(sbatch --parsable $dep $NODE --array="$ARRAY" "$HERE/11_retarget_r1.slurm")
echo "  retarget    $retarget  (array $ARRAY)"

if [ "$retarget_only" -eq 1 ]; then
    echo
    echo "stopping after retargeting, as asked"
    exit 0
fi

# shellcheck disable=SC2086
convert=$(sbatch --parsable --dependency=afterok:"$retarget" $NODE \
    "$HERE/12_convert_motion_lib.slurm")
echo "  motion_lib  $convert"

if [ "$with_training" -eq 1 ]; then
    # shellcheck disable=SC2086
    train=$(sbatch --parsable --dependency=afterok:"$convert" $NODE \
        "$HERE/02_train_any2any.slurm")
    echo "  train       $train"
    echo
    echo "NOTE: training runs for days, not overnight, and resumes the previous"
    echo "      checkpoint because auto_load_latest=true. Change EXPERIMENT_NAME"
    echo "      in env.sh first if this should be a new run on the new data."
fi

echo
echo "watch with:  squeue -u $USER"
echo "logs in:     $(pwd)/logs/"
