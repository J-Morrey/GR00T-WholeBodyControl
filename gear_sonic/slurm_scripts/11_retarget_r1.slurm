#!/bin/bash
#SBATCH --job-name=r1-retarget
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --nodelist=zac
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
# Per array task, not for the array as a whole: Slurm treats each task as its
# own job, so N concurrent shards reserve N x this. Measured peak RSS is 1.5 GB
# on the worst batch in a 1298-clip sample (the 10 largest clips, 195 MB of
# BVH, which the size-descending sort makes batch 1). RSS tracks the batch's
# BVH bytes at roughly 0.46 GB + 5 MB per MB, so even a corpus tail four times
# heavier lands near 4.6 GB. 12 G is generous; it was 48 G, which reserved
# memory other people's jobs on the node could have used. If a pathological
# clip ever does OOM, lower batch_size in the generated config rather than
# raising this.
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/r1-retarget-%A_%a.out
# This --array is only the fallback for a bare `sbatch`. submit_retarget.sh
# passes --array on the command line, derived from RETARGET_SHARDS in env.sh,
# and a command-line value overrides the directive. The two must agree: the
# script checks and refuses if they do not, because a mismatch silently drops
# clips rather than failing.

# Stage 2 of 3. BONES-SEED BVH -> Unitree R1 joint CSVs, in $R1_CSV under HOME.
#
# The whole corpus is ~142k clips at ~0.88 s/clip on one 4090: about 35
# GPU-hours. The array divides that -- 8 shards is ~4.4 h each, inside the 12 h
# limit with room for a slow node. `scontrol show node zac | grep Gres` shows
# how many GPUs are actually there; asking for more shards than GPUs just
# queues them, which is fine but buys nothing.
#
# This runs soma-retargeter, NOT gear_sonic: different repo, different
# environment (uv, not conda). env.sh is still sourced, for the paths.
#
# Re-running is incremental. make_retarget_shard.py links only the clips whose
# CSV is missing or zero-length, so a timed-out or preempted array can simply be
# re-submitted and will pick up where it stopped. Nothing is recomputed.

set -euo pipefail
REQUIRE_RAW=0 source "$HOME/repos/GR00T-WholeBodyControl/gear_sonic/slurm_scripts/env.sh"

SHARD=${SLURM_ARRAY_TASK_ID:-0}
NSHARD=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD_DIR=$SHARD_ROOT/$SHARD
CONFIG=$SHARD_ROOT/config_$SHARD.json

echo "node:    $(hostname)"
echo "shard:   $SHARD of $NSHARD"
echo "bvh:     $BVH_ROOT"
echo "csv:     $R1_CSV"
echo "soma:    $SOMA_REPO"
echo "python:  $SOMA_PYTHON"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# --- pre-flight -------------------------------------------------------------
if [ "$NSHARD" != "$RETARGET_SHARDS" ]; then
    echo "FATAL: --array width $NSHARD != RETARGET_SHARDS $RETARGET_SHARDS."
    echo "       They partition the corpus together; a mismatch would leave"
    echo "       clips unassigned. Fix one of them and resubmit."
    exit 1
fi
[ -d "$BVH_ROOT" ] || { echo "FATAL: $BVH_ROOT missing -- run 10_extract_bvh.slurm"; exit 1; }
[ -d "$SOMA_REPO" ] || { echo "FATAL: soma-retargeter not found at $SOMA_REPO"; exit 1; }
# The R1 target is not in upstream soma-retargeter; it comes from the
# unitree-r1-target branch. Without it the run fails ~40 minutes in, when the
# first batch asks the registry for a target that was never registered.
[ -f "$SOMA_REPO/assets/robotics/unitree/r1/manifest.json" ] || {
    echo "FATAL: $SOMA_REPO has no unitree_r1 target."
    echo "       Check out the unitree-r1-target branch there."
    exit 1
}
# Check that the interpreter can actually import the package, not merely that
# some binary is on PATH. soma-retargeter needs python >= 3.12 and a working
# newton/warp, neither of which `command -v` would notice. $SOMA_PYTHON is left
# unquoted on purpose: it may be a wrapper like `uv run python` that has to
# word-split.
# shellcheck disable=SC2086
# newton and warp rather than soma_retargeter alone: run from $SOMA_REPO the
# package imports off the cwd whatever the interpreter, so importing it proves
# nothing. The dependencies are what distinguish a prepared environment.
if ! ( cd "$SOMA_REPO" && $SOMA_PYTHON -c "import newton, warp, soma_retargeter" ) >/dev/null 2>&1; then
    echo "FATAL: SOMA_PYTHON ('$SOMA_PYTHON') cannot import soma_retargeter in $SOMA_REPO."
    echo "       env_isaaclab is too old for it (needs python >= 3.12), so it"
    echo "       needs an interpreter of its own. Either install uv:"
    echo "         curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "       or make an environment and point SOMA_PYTHON at it in env.sh:"
    echo "         conda create -y --prefix \$HOME/envs/soma python=3.12"
    echo "         conda run -p \$HOME/envs/soma pip install -e $SOMA_REPO"
    echo "         export SOMA_PYTHON=\$HOME/envs/soma/bin/python"
    exit 1
fi

mkdir -p "$R1_CSV" "$SHARD_ROOT"

# --- this shard's work ------------------------------------------------------
echo "=== building shard $SHARD ==="
python "$HOME/repos/GR00T-WholeBodyControl/gear_sonic/slurm_scripts/make_retarget_shard.py" \
    --bvh-root "$BVH_ROOT" --csv-root "$R1_CSV" \
    --shard-dir "$SHARD_DIR" --index "$SHARD" --count "$NSHARD"

todo=$(find "$SHARD_DIR" -name '*.bvh' 2>/dev/null | wc -l)
if [ "$todo" -eq 0 ]; then
    echo "shard $SHARD is already complete"
    exit 0
fi

# --- retarget ---------------------------------------------------------------
# Written per task: import_folder is this shard's symlink farm, and the
# converter has no CLI for it. export_folder is shared across the array --
# clip paths are unique, so tasks never collide.
#
# extra_robot_paths is relative to the cwd the converter runs from, which is
# why this cds into $SOMA_REPO rather than passing an absolute path.
cat > "$CONFIG" <<JSON
{
    "import_folder": "$SHARD_DIR",
    "export_folder": "$R1_CSV",
    "batch_size": 10,
    "retargeter": "Newton",
    "retarget_source": "soma",
    "retarget_target": "unitree_r1",
    "retarget_source_facing_direction": "Mujoco",
    "extra_robot_paths": ["assets/robotics"]
}
JSON

cd "$SOMA_REPO"
echo
echo "=== retargeting $todo clips ($(date -Is)) ==="
# shellcheck disable=SC2086
srun $SOMA_PYTHON app/bvh_to_csv_converter.py --config "$CONFIG"

# --- report -----------------------------------------------------------------
# Counted from the shard rather than from $R1_CSV: the other array tasks are
# writing into the same tree, so a global count here would be meaningless.
done_now=0
while IFS= read -r link; do
    rel=${link#"$SHARD_DIR"/}
    [ -s "$R1_CSV/${rel%.bvh}.csv" ] && done_now=$((done_now + 1))
done < <(find "$SHARD_DIR" -name '*.bvh')

echo
echo "=== shard $SHARD: $done_now of $todo clips produced a CSV ($(date -Is)) ==="
if [ "$done_now" -lt "$todo" ]; then
    echo "WARNING: $((todo - done_now)) clips produced nothing."
    echo "         Re-submitting this array retries exactly those."
    exit 1
fi
