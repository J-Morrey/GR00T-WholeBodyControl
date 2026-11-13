#!/bin/bash
#SBATCH --job-name=r1-any2any-2gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --nodelist=bowser
#SBATCH --gres=gpu:2
#SBATCH --mem=56G
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/r1-any2any-2gpu-%j.out
#SBATCH --open-mode=append

# G1 -> R1 any2any transfer on both of bowser's GPUs, for 100k PPO updates.
#
# Submit with (logs/ must exist BEFORE submitting -- Slurm opens the output file
# itself and will not create the directory):
#
#     cd ~/repos/GR00T-WholeBodyControl/gear_sonic/slurm_scripts
#     mkdir -p logs
#     sbatch 03_train_any2any_2gpu.slurm
#
# Smoke test first. It exercises the checkpoint load, the G1<->R1 joint mapping,
# the motion-lib glob and both ranks' DDP handshake in a couple of minutes:
#
#     sbatch --time=00:30:00 03_train_any2any_2gpu.slurm \
#         num_envs=64 ++algo.config.num_learning_iterations=5
#
# Everything after the script name is appended to the Hydra command line, and
# Hydra lets a later override win over an earlier one -- verified, not assumed --
# so appending num_envs= or any other key already set below simply replaces it.
#
# Monitoring: wandb, project TRL_R1_Any2Any, run name "$EXPERIMENT_NAME_2GPU"
# plus the launch timestamp. Only rank 0 opens the run (train_agent_trl.py:206
# gates on accelerator.is_main_process), so two GPUs still produce one run, not
# two. The run id is written to meta.yaml in the experiment dir and read back on
# resume, so re-submitting after a crash continues the same wandb run rather
# than starting a second one. Metrics arrive through WandbCallback.on_log with
# an explicit step=global_step; sync_tensorboard is deliberately off, because
# having both paths live made every metric land twice and advanced wandb's
# internal _step at 2x the iteration count.

set -euo pipefail

# REQUIRE_RAW=0 because $RAW (/scratch/$USER/seed/retarget) is the OmniRetarget
# layout that 01_prepare_data.slurm consumes. The soma-retargeter pipeline
# (jobs 10-12) writes $FILTERED directly and never creates $RAW, so the guard at
# the bottom of env.sh would abort this job over a directory it does not need.
# $FILTERED is checked for real below.
REQUIRE_RAW=0 source "$HOME/repos/GR00T-WholeBodyControl/gear_sonic/slurm_scripts/env.sh"
cd "$REPO"

# Isaac Sim's shader/extension cache. Both ranks share it, and on a cold cache
# they will race to populate it and fetch the missing pieces over the network.
# Warm it once with the smoke test above before starting a two-week run.
export OMNI_CACHE_ROOT=/scratch/$USER/ov_cache
mkdir -p "$OMNI_CACHE_ROOT" "$RUN_LOGS" logs

# Accept NVIDIA's Omniverse EULA without a prompt. isaacsim/kit/kit_app.py's
# check_eula() takes y/yes/1 here, and otherwise calls input() -- which in a
# batch job has no stdin, so it reads EOF and the rank dies with "Unable to
# bootstrap inner kit kernel". That is what killed job 1487: the torch
# ChildFailedError underneath it was only the launcher reporting the corpse.
#
# The same function writes site-packages/isaacsim/kit/EULA_ACCEPTED after an
# interactive acceptance, which is why a workstation that has been used by hand
# never prompts and a fresh cluster env always does. If you would rather accept
# it yourself once and drop this line, that is equivalent:
#     srun --nodelist=bowser --pty python -c "import isaacsim"
export OMNI_KIT_ACCEPT_EULA=YES

# Keep temp files out of the shared /tmp. IsaacLab's configure_logging() opens
# os.path.join(tempfile.gettempdir(), "isaaclab", "logs") and, because
# makedirs(exist_ok=True) happily succeeds on a directory someone else owns, the
# failure surfaces only when it tries to create the file inside:
#
#     PermissionError: [Errno 13] Permission denied:
#     '/tmp/isaaclab/logs/isaaclab_2026-11-12_17-15-00.log'
#
# which is what killed job 1488. bowser is shared, so /tmp/isaaclab belongs to
# whoever ran IsaacLab on it first. tempfile.gettempdir() honours TMPDIR, so
# pointing that at our own node-local scratch moves the whole tree somewhere we
# certainly own -- no IsaacLab patch needed.
#
# Not per-job: one directory per user is enough to fix the collision, and two
# jobs of the same user sharing it is harmless (log filenames are timestamped,
# and sharing the AppLauncher lock just serialises their startups briefly).
# A per-job directory would also leak, since the script ends in exec and so can
# never run a cleanup trap.
export TMPDIR=/scratch/$USER/tmp
mkdir -p "$TMPDIR"

# Assert it took. tempfile.gettempdir() only honours TMPDIR if the directory
# exists AND is writable, and falls back to /tmp silently otherwise -- so a full
# or unwritable scratch would put us straight back in the shared directory with
# no hint in the log until the PermissionError. Checked rather than trusted,
# since the failure it guards against is exactly the one that looks like it
# cannot be happening.
RESOLVED_TMP=$(python -c "import tempfile; print(tempfile.gettempdir())")
if [ "$RESOLVED_TMP" != "$TMPDIR" ]; then
    echo "FATAL: TMPDIR is $TMPDIR but python resolves tempdir to $RESOLVED_TMP."
    echo "       It is missing, unwritable, or the filesystem is full:"
    df -h "$TMPDIR" 2>&1 | sed 's/^/         /'
    exit 1
fi

# Each rank converts the robot URDF into its own USD directory under $TMPDIR
# every run (robots/usd_cache.py explains why it is per-run and not cached), so
# without this they accumulate for as long as scratch is never cleared -- which
# /tmp used to handle for us. A week is far longer than any run needs them.
find "$TMPDIR/IsaacLab" -maxdepth 1 -name 'usd_*' -type d -mtime +7 \
     -exec rm -rf {} + 2>/dev/null || true

echo "=============================================================="
echo "node:     $(hostname)"
echo "python:   $(which python)"
echo "job:      ${SLURM_JOB_ID:-interactive}"
echo "cpus:     ${SLURM_CPUS_PER_TASK:-?}"
echo "gpus:     ${CUDA_VISIBLE_DEVICES:-unset}  (requesting $TRAIN_GPUS processes)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/          /'
echo "logs:     $RUN_LOGS/TRL_R1_Any2Any"
echo "run:      $EXPERIMENT_NAME_2GPU"
echo "=============================================================="
echo

# --- pre-flight -------------------------------------------------------------
# Every check here fails in under a second. Isaac Sim takes ~2 minutes to
# initialise and a two-week allocation is not something to spend discovering a
# missing file.

[ -f sonic_release/last.pt ] || {
    echo "FATAL: sonic_release/last.pt missing."
    echo "       run: python gear_sonic/data_process/download_from_hf.py --training --no-smpl"
    exit 1
}

# Prefer the node-local copy, fall back to the durable one in home. Staging is
# optional -- an unstaged run is correct, just slow -- so this reports which it
# picked rather than insisting.
if [ -d "${TRAIN_MOTIONS:-}" ] && [ -n "$(ls -A "$TRAIN_MOTIONS" 2>/dev/null)" ]; then
    MOTIONS=$TRAIN_MOTIONS
    echo "motions:  $MOTIONS  (node-local copy)"
else
    MOTIONS=$FILTERED
    echo "motions:  $MOTIONS  (shared home -- NOT staged)"
    echo "          Startup globs every clip and each resample rereads"
    echo "          MOTION_BUDGET of them, both over the network. Stage it with:"
    echo "            rsync -a --info=progress2 \"$FILTERED/\" \"${TRAIN_MOTIONS:-/scratch/\$USER/seed/r1/filtered}/\""
fi

[ -d "$MOTIONS" ] || {
    echo "FATAL: $MOTIONS missing -- run 12_convert_motion_lib.slurm (or 01_prepare_data.slurm)."
    exit 1
}

NCLIPS=$(find "$MOTIONS" -name '*.pkl' ! -name 'metadata.pkl' | wc -l)
[ "$NCLIPS" -gt 0 ] || { echo "FATAL: no clips in $MOTIONS"; exit 1; }

# A warning, not a failure. The loader works without metadata.pkl -- every
# sampled clip is opened from its own file either way
# (motion_lib_base.py:1758: `if "path" in curr_file: joblib.load(...)`), and the
# metadata only merges extra per-clip fields that the load then overwrites. But
# its absence is a good sign that a hand-staged copy caught only *.pkl and may
# have missed other things too.
[ -f "$MOTIONS/metadata.pkl" ] || {
    echo "WARNING: no metadata.pkl in $MOTIONS."
    echo "         Not fatal, but if you staged by hand, copy the whole"
    echo "         directory rather than matching a pattern:"
    echo "           rsync -a \"$FILTERED/\" \"$MOTIONS/\""
}

# wandb. use_wandb is true via opt/wandb and WandbCallback is in base.yaml's
# defaults, so logging is on without anything being set here -- but only if this
# node can authenticate. train_agent_trl.py:214 calls wandb.init() unguarded on
# the main process, so a node with no credentials or no outbound route turns
# into a retry loop two minutes into Isaac Sim startup instead of a clear error.
#
# Exit status only, never printing the key.
if [ "${WANDB_MODE:-online}" != "disabled" ]; then
    if python -c "import wandb,sys; sys.exit(0 if wandb.api.api_key else 1)" 2>/dev/null; then
        echo "wandb:    authenticated, logging online to project TRL_R1_Any2Any"
    else
        echo "FATAL: wandb has no API key on $(hostname)."
        echo "       fix it with ONE of:"
        echo "         wandb login                     # writes ~/.netrc, do this once on the node"
        echo "         export WANDB_API_KEY=...        # in your submit environment"
        echo "       or run without it:"
        echo "         WANDB_MODE=offline sbatch 03_train_any2any_2gpu.slurm"
        echo "           -> buffers to \$experiment_dir/.wandb; 'wandb sync' it later"
        echo "         WANDB_MODE=disabled sbatch 03_train_any2any_2gpu.slurm"
        echo "           -> tensorboard only, under \$experiment_dir/tensorboard"
        echo
        echo "       NOTE: the wandb_tags: 'online'/'offline'/'disabled' key in"
        echo "       config/opt/wandb.yaml is dead -- no Python reads it. WANDB_MODE"
        echo "       is the only thing that actually switches mode."
        exit 1
    fi
else
    echo "wandb:    DISABLED by WANDB_MODE -- metrics go to tensorboard only"
fi

# Assert the allocation actually matches what accelerate is about to launch.
# --gres=gpu:2 on a node whose GPUs are named something other than the generic
# "gpu" resource silently yields fewer, and rank 1 would then land on rank 0's
# card and OOM a day into the run rather than now.
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$NGPU" -lt "$TRAIN_GPUS" ]; then
    echo "FATAL: TRAIN_GPUS=$TRAIN_GPUS but only $NGPU GPU(s) visible."
    echo "       check:  scontrol show node $CLUSTER_NODE | grep -E 'Gres|AllocTRES'"
    exit 1
fi

# Motion budget vs. VRAM. ~15.5 GB is Isaac Sim with 4096 envs plus the network;
# ~0.65 MB per clip is measured on the 1% set (see MOTION_BUDGET in env.sh).
#
# --id=0 rather than querying every GPU and taking the first line with `head`.
# That pipeline killed job 1486 on bowser, silently: `head -1` closes the pipe
# after one line, the second GPU's row gives nvidia-smi a SIGPIPE, it exits 141,
# `set -o pipefail` promotes that to the pipeline's status and `set -e` aborts
# the script -- and bash prints nothing at all when it does. The log simply
# stopped after the wandb line. It cannot happen on one GPU, where nvidia-smi
# has written everything before head exits, which is why job 02 never hit it.
VRAM_MB=$(nvidia-smi --id=0 --query-gpu=memory.total --format=csv,noheader,nounits)
BUDGET=$MOTION_BUDGET
if [ "$BUDGET" -gt "$NCLIPS" ]; then BUDGET=$NCLIPS; fi
EST_MB=$(( 15500 + BUDGET * 65 / 100 ))
echo "$NCLIPS clips available; loading $BUDGET at a time"
if [ "$BUDGET" -lt "$NCLIPS" ]; then
    echo "  -> ImResampleCallback redraws the subset every 250 iterations"
    # Measured on the 1% set: 1298 clips in ~18 s with 24 jobs, so ~72 clips/s.
    RELOAD_S=$(( BUDGET / 72 ))
    if [ "$RELOAD_S" -gt 120 ]; then
        echo
        echo "WARNING: that redraw reloads $BUDGET clips, about $(( RELOAD_S / 60 )) min, every 250"
        echo "         iterations. This is the expensive middle: a budget just under the"
        echo "         corpus pays a full load repeatedly without ever gaining the"
        echo "         load-all shortcut. Go to one end instead --"
        echo "           MOTION_BUDGET >= $NCLIPS  loads once, never redraws (needs"
        echo "                                       ~$(( 15500 + NCLIPS * 65 / 100 )) MB of ${VRAM_MB} MB)"
        echo "           MOTION_BUDGET ~= 8192       redraws in ~2 min, more churn but cheap"
        echo "         Continuing in 20 s; Ctrl-C or scancel to change it."
        sleep 20
    fi
else
    echo "  -> whole corpus fits; resampling is a no-op (all_motions_loaded)"
fi
echo "  estimated VRAM: ~${EST_MB} MB of ${VRAM_MB} MB per GPU"
if [ "$EST_MB" -gt "$VRAM_MB" ]; then
    echo
    echo "FATAL: that does not fit. Lower MOTION_BUDGET (or TRAIN_NUM_ENVS) in env.sh."
    exit 1
fi
echo

# --- launch -----------------------------------------------------------------
# No `srun`. accelerate forks the ranks itself, and wrapping it in a single
# --ntasks=1 step only adds a layer of CPU-affinity binding between Slurm and
# those forks. Single-node multi-GPU is accelerate's job; Slurm's job is the
# allocation.
#
# No --mixed_precision either, deliberately. algo/trl/ppo.yaml sets bf16: false
# and documents at length why: with bf16 the rollout policy and the policy PPO
# evaluates during the update are not bit-identical, which puts a KL of ~0.0113
# between the policy and itself -- above desired_kl, so the adaptive LR
# controller steers on numerical noise. HF TrainingArguments derives
# ACCELERATE_MIXED_PRECISION from that flag; passing --mixed_precision here
# would override it from the command line and silently undo the fix.
#
# max_grad_norm IS overridden, and this is the one place where that is correct.
# The warning in 02_train_any2any.slurm is about an earlier attempt that raised
# actor_learning_rate 2e-5 -> 1e-4 AND max_grad_norm 0.1 -> 1.0 together, making
# each update ~50x more aggressive; that run peaked around iteration 200 and
# decayed. 2.0 with the learning rate left alone is a different change, and it
# is the configuration of sonic_r1_any2any_weekend_filtered_gn20 -- the best run
# to date, and the one the current 1% job has been climbing on for two days
# (reward 13.15 -> 21.91, episode length 146.7 -> 231.8 over iterations
# 750-1100). "gn20" is max_grad_norm=2.0, not 0.20.
#
# Nothing else is overridden. sonic_r1_any2any.yaml inherits the source
# pretraining's PPO settings on purpose and records two runs wrecked by job
# scripts second-guessing them.

exec accelerate launch \
    --multi_gpu \
    --num_processes="$TRAIN_GPUS" \
    --num_machines=1 \
    --mixed_precision=no \
    gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_r1_any2any \
    experiment_name="$EXPERIMENT_NAME_2GPU" \
    base_dir="$RUN_LOGS" \
    num_envs="$TRAIN_NUM_ENVS" \
    headless=True \
    auto_load_latest=true \
    ++algo.config.max_grad_norm="$TRAIN_MAX_GRAD_NORM" \
    ++algo.config.num_learning_iterations="$TRAIN_ITERS" \
    ++manager_env.commands.motion.debug_vis=false \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file="$MOTIONS" \
    ++manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load="$BUDGET" \
    '++wandb.wandb_dir=${experiment_dir}/.wandb' \
    "$@"

# The single quotes above are load-bearing: ${experiment_dir} is a Hydra
# interpolation, not a shell variable, and must reach Hydra unexpanded.
#
# It moves wandb's staging directory off opt/wandb.yaml's default of /tmp, whose
# stated reason is "so that it will automatically be cleaned up". That is the
# wrong trade for a fourteen-day job on a shared node: wandb buffers every
# logged step there before syncing, and a /tmp that is small, full, or swept by
# the node's cleaner mid-run takes the run's history with it. Under
# $experiment_dir it sits on the same scratch as the checkpoints, and resuming
# picks up the resumed run's directory because auto_load_latest rewrites
# experiment_dir first.

# Resuming: auto_load_latest resolves experiment_dir back to the newest
# "$RUN_LOGS/TRL_R1_Any2Any/$EXPERIMENT_NAME_2GPU-*/last.pt"
# (train_agent_trl.py:81-89) and prints "starting fresh" when there is none, so
# re-running this script both starts the run and continues it. last.pt is
# rewritten every 50 iterations (ModelSaveCallback.save_last_frequency), so a
# node failure or a hit wall-clock costs at most 50 updates. meta.yaml carries
# the wandb id, so the resumed run reports into the same wandb run rather than
# starting a second one.
#
# Checkpoint disk: model_step_*.pt every 2000 iterations at ~469 MB = ~23 GB
# over 100k updates, which is why $RUN_LOGS points at scratch and not the repo.
