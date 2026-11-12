# Shared config for the R1 SLURM scripts. Sourced by every job; not submitted itself.
#
# EDIT THE PATHS IN THIS FILE ONLY -- the job scripts read everything from here.
#
# NOTE: every job is pinned to node zac via `#SBATCH --nodelist=zac`, because
# /scratch is node-local and only zac's copy holds the dataset. Do not relax
# that without first replicating $RAW to the other node's scratch.

# --- conda ------------------------------------------------------------------
# Do not simplify this back to `source conda.sh && conda activate env_isaaclab`.
# That form produced, and the job died on:
#   conda.sh: line 52: pop_var_context: head of shell_variables not a function context
#
# Line 52 is a `local` declaration inside conda's own `conda()` shell function.
# `local` only fails that way when bash's function-context stack is already
# inconsistent, which happens when a conda shell function is inherited as an
# exported function (BASH_FUNC_conda%%) and then redefined by re-sourcing
# conda.sh. The submitting shell had env_isaaclab active and Slurm's default
# --export=ALL copies that environment -- functions included -- into the job.
#
# Two separate things then have to be fixed:
#   1. clear the inherited conda state so conda.sh initialises from scratch
#   2. run conda's machinery without `set -euo pipefail`, which it is not
#      written to tolerate, and without `&&`, which made the warning fatal
# Snapshot the strict flags so they can be put back afterwards.
#
# Do NOT snapshot with `_strict=$(set +o)` and `eval "$_strict"`. Bash disables
# errexit inside command substitution, so that snapshot always reads back as
# `set +o errexit` and restoring it silently switches `set -e` OFF for the whole
# rest of the job -- the opposite of the intent, and invisible until something
# fails without aborting. Read `$-` in the current shell instead.
#
# (`$-` also has no letter for pipefail, hence the separate probe. And note that
# for the same reason you cannot verify errexit from inside `$(...)` either.)
_restore=
case $- in *e*) _restore="$_restore -e";; esac
case $- in *u*) _restore="$_restore -u";; esac
if set -o | grep -qE '^pipefail[[:space:]]+on'; then _restore="$_restore -o pipefail"; fi

set +eu +o pipefail
unset -f conda __conda_activate __conda_reactivate __conda_hashr 2>/dev/null
unset CONDA_SHLVL CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE
source /home/software/miniforge3/etc/profile.d/conda.sh
conda activate env_isaaclab
# Deliberately unquoted: the flags must word-split back into separate args.
# shellcheck disable=SC2086
[ -n "$_restore" ] && set $_restore
unset _restore

# Assert rather than trust: the warning above is noise on some conda versions
# and fatal on others, so check the outcome instead of the exit status.
if [ "${CONDA_DEFAULT_ENV:-}" != "env_isaaclab" ]; then
    echo "FATAL: env_isaaclab not active (CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-unset}')"
    return 1 2>/dev/null || exit 1
fi

# Repo checkout. Everything runs with this as cwd: the Hydra config defaults
# (assetRoot, base_dir) and convert_soma_npz_to_motion_lib.py's DEFAULT_MJCF are
# all relative paths, so running from anywhere else silently breaks them.
export REPO=$HOME/repos/GR00T-WholeBodyControl

# filter_r1_retargeted.py does a bare `from filter_and_copy_bones_data import ...`
# (filter_r1_retargeted.py:36), so its own directory has to be importable. Setting
# this rather than cd-ing in keeps cwd at REPO for the reasons above.
export PYTHONPATH=$REPO/gear_sonic/data_process:${PYTHONPATH:-}

# --- data locations ---------------------------------------------------------
# RAW: the retargeted full BONES-SEED dataset as it currently sits on the HPC.
# Read-only as far as these scripts are concerned.
export RAW=/scratch/$USER/seed/retarget

# STAGE: only used if RAW turns out to be .npz and needs conversion to SONIC
# .pkl. Ignored otherwise.
export STAGE=/scratch/$USER/seed/r1_sonic_pkl

# FILTERED: what training actually reads. Produced by 01_prepare_data.slurm.
# In HOME, not scratch, so the dataset is readable from any node rather than
# only zac. Was /scratch/$USER/seed/r1_filtered when the data lived on scratch.
export FILTERED=$HOME/seed/r1/filtered

# --- retargeting (jobs 10-12) -----------------------------------------------
# These three produce $FILTERED from raw BONES-SEED. Skip them entirely if you
# already have retargeted clips.

# Where soma_uniform.tar.gz lives -- either the HF tarballs or an
# already-extracted tree; 10_extract_bvh.slurm detects which. This does NOT
# have to be on the same filesystem as BVH_ROOT: home is the normal place for
# it after moving the corpus between nodes, since only the 45 GB tarball has to
# cross the network and the 295 GB it unpacks to stays on node-local scratch.
export SEED_ROOT=$HOME/seed

# Extracted soma_uniform BVH. The full corpus is ~295 GB unpacked (from a 45 GB
# .tar.gz -- BVH is plain text and compresses hard), so this stays on scratch
# and never goes near home.
export BVH_ROOT=/scratch/$USER/seed/soma_uniform/bvh

# Per-array-task symlink farms. Symlinks only, so a few MB.
export SHARD_ROOT=/scratch/$USER/seed/shards

# Retargeted output, in HOME for cross-node access. ~45 GB of CSV and ~30 GB of
# compressed PKLs for the full corpus -- check your home quota before starting.
export R1_CSV=$HOME/seed/r1/csv
export R1_PKL=$HOME/seed/r1/motion_lib

# soma-retargeter checkout. Must be on a branch carrying the unitree_r1 target
# (assets/robotics/unitree/r1/) -- see the README section on getting it here.
export SOMA_REPO=$HOME/repos/soma-retargeter

# How to run python inside $SOMA_REPO. It cannot be this env: soma-retargeter
# needs python >= 3.12 and env_isaaclab is older, so it has an interpreter of
# its own.
#
# The default assumes uv, which is how the repo is set up upstream and which
# fetches a 3.12 for itself if the system has none. If uv is not available,
# point this at the interpreter of any environment where soma-retargeter is
# installed, e.g.
#     export SOMA_PYTHON=$HOME/envs/soma/bin/python
export SOMA_PYTHON="${SOMA_PYTHON:-uv run python}"

# A user-level `uv` installs here, and a Slurm batch script does not source
# .bashrc. Jobs normally inherit PATH through --export=ALL, but only if it was
# right in the submitting shell -- adding it here means the jobs do not depend
# on that.
case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

# The node everything runs on. The job scripts still carry `#SBATCH
# --nodelist=zac`, but submit_retarget.sh passes --nodelist on the command
# line, which overrides the directive -- so this one variable moves the whole
# pipeline to another node. Set it to whichever node holds the data, because
# /scratch is node-local.
export CLUSTER_NODE=bowser

# Width of the retargeting array. Size this by the TIME LIMIT, not by the GPU
# count. The corpus is ~35 GPU-hours on a 4090 and ~23 on a Blackwell, and
# 11_retarget_r1.slurm asks for 12 h, so one task must cover at most a twelfth
# of the work however many GPUs are free:
#
#     shards    per task, 4090    per task, Blackwell
#        1          34.8 h              23.2 h        over the limit
#        2          17.4 h              11.6 h        over / no margin
#        4           8.7 h               5.8 h        ok
#        8           4.3 h               2.9 h        ok
#
# Shards are not capped by GPUs. Slurm runs as many at once as there are free
# GPUs and queues the rest, so an array wider than the node adapts on its own:
# it uses a second GPU the moment someone else's job releases it, and a shard
# that times out loses only its own remainder because make_retarget_shard.py
# resumes. Oversharding is therefore the safe direction.
#
# submit_retarget.sh derives --array from this, so there is no second place to
# keep in sync. `scontrol show node $CLUSTER_NODE | grep -E 'Gres|AllocTRES'`
# shows how many GPUs exist and how many are already taken.
export RETARGET_SHARDS=8

# joblib compression for the motion_lib PKLs. 3 cuts 86 GB to 30 GB at corpus
# scale; 0 is faster to load. Mostly it is squeezing the all-zero smpl_joints.
export PKL_COMPRESS=3

# --- training output --------------------------------------------------------
# Hydra's base_dir defaults to ./logs_rl (base.yaml:30), i.e. inside the repo.
# Do not leave it there: the trainer writes a 469 MB model_step_*.pt every 500
# steps and keeps them all, so a long run is tens of GB. Point it at scratch.
export RUN_LOGS=/scratch/$USER/seed/logs_rl

export EXPERIMENT_NAME=sonic_r1_any2any_full_filtered

# Guard against a job that somehow landed off zac: /scratch exists on every
# node but is empty elsewhere, so without this the failure mode is a confusing
# "0 clips" or an empty-glob motion lib rather than an obvious wrong-node error.
# Jobs 10-12 run before $RAW exists and source this with REQUIRE_RAW=0.
if [ "${REQUIRE_RAW:-1}" = 1 ] && [ ! -d "$RAW" ]; then
    echo "FATAL: $RAW not found on $(hostname)."
    echo "       /scratch is node-local; this job must run on zac."
    # `return` so that sourcing this file by hand on the login node (where
    # zac's scratch is genuinely absent) complains instead of killing the shell.
    return 1 2>/dev/null || exit 1
fi

# --- multi-GPU training (job 03) --------------------------------------------
# Everything below is read only by 03_train_any2any_2gpu.slurm. Job 02 (single
# GPU) is unaffected.

# Number of GPUs, i.e. accelerate processes. Each one launches its own Isaac Sim
# and its own copy of the motion library, so this multiplies both VRAM and host
# RAM -- it is not a way to fit a bigger run on one card.
export TRAIN_GPUS=2

# Envs PER PROCESS, not in total. The upstream docs keep num_envs=4096 whether
# they run on 1 GPU or 8 (docs/source/user_guide/training.md:125), so
# TRAIN_GPUS=2 means 8192 envs overall and twice the rollout per PPO update.
export TRAIN_NUM_ENVS=4096

# Clips held on the GPU at once. motion_lib_base.py:962 warns that leaving this
# unset loads the whole corpus and "can OOM if the dataset is too large", which
# the full BONES-SEED set certainly is. Below the corpus size, ImResampleCallback
# draws a fresh failure-weighted subset every 250 iterations
# (config/callbacks/im_resample.yaml), so the policy still sees everything --
# just not all at once. At or above the corpus size the loader takes the
# load-all branch (motion_lib_base.py:966) and resampling becomes a no-op, so
# any value is safe; it only ever costs memory.
#
# Sizing, measured on the running 1% job (4090, 4096 envs, 1298 clips = 471,417
# frames at 50 fps, 16.4 GB of 24 GB used):
#
#     Isaac Sim + 4096 envs + network      ~15.5 GB   fixed
#     motion library                        ~0.85 GB  ~1.8 KB/frame
#
# so roughly 0.65 MB per clip at the corpus's average 363 frames. Budgets that
# leave ~2 GB spare:
#
#     VRAM    clips
#     24 GB    8192     (~5.3 GB of motions)
#     48 GB   40000
#     96 GB   120000    the default -- bowser's two RTX PRO 6000 Blackwells
#                       report 97887 MiB each, which fits the whole corpus
#
# Prefer a budget AT OR ABOVE the corpus size to one just under it. The two
# branches cost very different amounts: at or above, the loader takes the
# load-all path once and resampling is a no-op, so the (slow) load is paid a
# single time at startup. Below, every 250 iterations pays it again -- and at
# this budget that is tens of minutes of reload per resample, which would
# dominate the run. A small budget is cheap to redraw and a large one is cheap
# because it never redraws; the expensive place is in between.
#
# The script prints the detected VRAM and refuses to start if the budget does
# not fit, so raising this cannot silently OOM a run days in.
export MOTION_BUDGET=120000

# PPO updates. 100000 is already ppo_im_phc.yaml's default; it is set here
# explicitly so the run length is visible in one place rather than inherited.
# Note these are UPDATES, not samples: at TRAIN_GPUS=2 each one consumes twice
# the rollout a single-GPU iteration does, so this is not 100k of the iterations
# plotted in the 1% run's wandb history.
export TRAIN_ITERS=100000

# The one hyperparameter this pipeline overrides, and the reason is empirical,
# not theoretical -- see the long comment in 03_train_any2any_2gpu.slurm before
# changing it.
export TRAIN_MAX_GRAD_NORM=2.0

# Kept distinct from $EXPERIMENT_NAME (job 02) so the two never resume into each
# other's directory. auto_load_latest globs on this name, so re-submitting the
# same script continues the run; change the name to start a genuinely new one.
export EXPERIMENT_NAME_2GPU=sonic_r1_any2any_full_gn20_2gpu
