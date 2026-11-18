#!/bin/bash
#SBATCH --job-name=r1-resume
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --nodelist=bowser
#SBATCH --gres=gpu:2
#SBATCH --mem=58G
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/r1-resume-%j.out
#
# Resume an existing R1 Any2Any run with the impossible clips filtered out.
#
# Differences from train_r1.slurm:
#   * +motion_filter=r1_impossible -- drops the 173 clips the policy cannot
#     do (see that config's header for the criterion and the measurement).
#   * max_grad_norm 2.0 -> 1.0.
#   * resume=true with an explicit checkpoint. resume_training() sets
#     experiment_dir to the checkpoint's own directory, so this continues the
#     original run in place: same tensorboard file, same checkpoint series, and
#     the same wandb run (meta.yaml in that directory carries its id).
#
# Pass CKPT=<path to last.pt>. Removing clips changes the bin count, so the
# adaptive-sampling history does not survive (motion_lib_base.py:2455) -- the
# curriculum restarts uniform while the policy, optimizer and step count carry
# over.

NGPU=${NGPU:-2}
NUM_ENVS=${NUM_ENVS:-8192}
ITERS=${ITERS:-100000}
CLIPS=${CLIPS:-2048}          # per process; host RAM bound, see train_r1.slurm
GRAD_NORM=${GRAD_NORM:-1.0}
CKPT=${CKPT:?set CKPT to the last.pt of the run to resume}

REPO=$HOME/repos/GR00T-WholeBodyControl
MOTIONS=/scratch/$USER/seed/r1/filtered
LOGS=/scratch/$USER/seed/logs_rl

source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
set -eo pipefail
[ "${CONDA_DEFAULT_ENV:-}" = env_isaaclab ] || { echo "conda activate failed"; exit 1; }
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }

ulimit -n "$(ulimit -Hn)"

export OMNI_KIT_ACCEPT_EULA=YES
export TMPDIR=/scratch/$USER/tmp
export OMNI_CACHE_ROOT=/scratch/$USER/ov_cache
mkdir -p "$TMPDIR" "$OMNI_CACHE_ROOT" "$LOGS" logs

cd "$REPO"
echo "resuming from $CKPT"
echo "commit: $(git rev-parse --short HEAD) on $(git branch --show-current)"

exec accelerate launch --num_processes=$NGPU \
    gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_r1_any2any \
    +motion_filter=r1_impossible \
    experiment_name=sonic_r1_any2any_full \
    base_dir="$LOGS" \
    num_envs=$NUM_ENVS \
    headless=True \
    +resume=true \
    +checkpoint="$CKPT" \
    ++algo.config.max_grad_norm=$GRAD_NORM \
    ++algo.config.num_learning_iterations=$ITERS \
    ++manager_env.commands.motion.debug_vis=false \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file="$MOTIONS" \
    ++manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load=$CLIPS \
    '++wandb.wandb_dir=${experiment_dir}/.wandb' \
    "$@"
