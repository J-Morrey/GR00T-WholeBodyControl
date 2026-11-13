#!/bin/bash
#SBATCH --job-name=r1-train
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --nodelist=bowser
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/r1-train-%j.out

# 1, not 2. Two ranks load and transplant the checkpoint fine, then die in
# accelerator.wait_for_everyone() with ncclUnhandledCudaError / Cuda failure
# 'invalid argument'. NCCL_P2P_DISABLE=1 does not help. Isaac Sim reports its
# GPU as active index 0 in BOTH ranks, so rank 1's barrier(device_ids=[1])
# likely names a device its context does not have. Unproven; NGPU=2 still
# works from the submit line if you want to retry it.
NGPU=${NGPU:-1}
NUM_ENVS=${NUM_ENVS:-4096}
ITERS=${ITERS:-100000}
CLIPS=${CLIPS:-8192}
RUN=sonic_r1_any2any_full

REPO=$HOME/repos/GR00T-WholeBodyControl
MOTIONS=/scratch/$USER/seed/r1/filtered
LOGS=/scratch/$USER/seed/logs_rl

# Sourced before `set -e`: conda.sh warns when its shell functions were
# inherited through --export=ALL, and that warning is fatal under strict mode.
source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
set -eo pipefail
[ "${CONDA_DEFAULT_ENV:-}" = env_isaaclab ] || { echo "conda activate failed"; exit 1; }

ulimit -n "$(ulimit -Hn)"                 # motion loaders pass one fd per tensor

export OMNI_KIT_ACCEPT_EULA=YES          # batch jobs have no stdin to prompt on
export TMPDIR=/scratch/$USER/tmp         # /tmp/isaaclab belongs to another user
export OMNI_CACHE_ROOT=/scratch/$USER/ov_cache
mkdir -p "$TMPDIR" "$OMNI_CACHE_ROOT" "$LOGS" logs

cd "$REPO"

exec accelerate launch --num_processes=$NGPU \
    gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_r1_any2any \
    experiment_name="$RUN" \
    base_dir="$LOGS" \
    num_envs=$NUM_ENVS \
    headless=True \
    auto_load_latest=true \
    ++algo.config.max_grad_norm=2.0 \
    ++algo.config.num_learning_iterations=$ITERS \
    ++manager_env.commands.motion.debug_vis=false \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file="$MOTIONS" \
    ++manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load=$CLIPS \
    '++wandb.wandb_dir=${experiment_dir}/.wandb' \
    "$@"
