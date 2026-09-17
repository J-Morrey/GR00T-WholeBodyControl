#!/bin/bash
#SBATCH --job-name=r1-any2any
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/r1-any2any-%j.out
# TODO: add --partition=<gpu partition> (and/or --nodelist=) for your cluster;
# the example script pinned --nodelist=zac. `sinfo -o "%P %G %N"` lists them.

# G1 -> R1 transfer learning on the full filtered retargeted dataset.
#
# Hyperparameters are deliberately NOT set here. sonic_r1_any2any.yaml inherits
# PPO settings from the source pretraining on purpose, and its comments record
# two separate runs that were wrecked by overriding actor_learning_rate and
# max_grad_norm from a job script. Only paths and scale are set below.
#
# Extra Hydra overrides can be appended on the command line. A 5-iteration
# smoke test that exercises the checkpoint load, the G1<->R1 joint mapping and
# the motion-lib glob without burning an allocation:
#   sbatch 02_train_any2any.slurm num_envs=64 algo.config.num_learning_iterations=5

set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO"

# Isaac Sim writes its shader/extension cache here and fetches missing pieces
# over the network on a cold start. Keep it off a quota'd home directory, and
# warm it once somewhere with internet access before relying on a compute node.
export OMNI_CACHE_ROOT=/scratch/$USER/ov_cache
mkdir -p "$OMNI_CACHE_ROOT" "$RUN_LOGS" logs

echo "python:   $(which python)"
echo "node:     $(hostname)"
echo "gpu:      ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "motions:  $FILTERED"
echo "logs:     $RUN_LOGS"
echo

# Fail before Isaac Sim spends ~2 minutes initialising if an input is missing.
[ -f sonic_release/last.pt ] || { echo "FATAL: sonic_release/last.pt missing -- run download_from_hf.py --training --no-smpl"; exit 1; }
[ -d "$FILTERED" ] || { echo "FATAL: $FILTERED missing -- run 01_prepare_data.slurm"; exit 1; }
n=$(find "$FILTERED" -name '*.pkl' ! -name 'metadata.pkl' | wc -l)
[ "$n" -gt 0 ] || { echo "FATAL: no clips in $FILTERED"; exit 1; }
echo "$n motion clips available"
echo

srun python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_r1_any2any \
    experiment_name="$EXPERIMENT_NAME" \
    base_dir="$RUN_LOGS" \
    num_envs=4096 \
    headless=True \
    auto_load_latest=true \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file="$FILTERED" \
    "$@"

# auto_load_latest resolves experiment_dir back to the newest
# "<base_dir>/<project>/<experiment_name>-*/last.pt" (train_agent_trl.py:81-89)
# and prints "starting fresh" when there is none -- so this same script both
# starts the run and resumes it after a crash, as long as EXPERIMENT_NAME is
# unchanged. Change EXPERIMENT_NAME in env.sh to begin a genuinely new run.
