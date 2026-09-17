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
export FILTERED=/scratch/$USER/seed/r1_filtered

# --- training output --------------------------------------------------------
# Hydra's base_dir defaults to ./logs_rl (base.yaml:30), i.e. inside the repo.
# Do not leave it there: the trainer writes a 469 MB model_step_*.pt every 500
# steps and keeps them all, so a long run is tens of GB. Point it at scratch.
export RUN_LOGS=/scratch/$USER/seed/logs_rl

export EXPERIMENT_NAME=sonic_r1_any2any_full_filtered

# Guard against a job that somehow landed off zac: /scratch exists on every
# node but is empty elsewhere, so without this the failure mode is a confusing
# "0 clips" or an empty-glob motion lib rather than an obvious wrong-node error.
if [ ! -d "$RAW" ]; then
    echo "FATAL: $RAW not found on $(hostname)."
    echo "       /scratch is node-local; this job must run on zac."
    # `return` so that sourcing this file by hand on the login node (where
    # zac's scratch is genuinely absent) complains instead of killing the shell.
    return 1 2>/dev/null || exit 1
fi
