# Shared config for the R1 SLURM scripts. Sourced by every job; not submitted itself.
#
# EDIT THE PATHS IN THIS FILE ONLY -- the job scripts read everything from here.
#
# NOTE: every job is pinned to node zac via `#SBATCH --nodelist=zac`, because
# /scratch is node-local and only zac's copy holds the dataset. Do not relax
# that without first replicating $RAW to the other node's scratch.

source /home/software/miniforge3/etc/profile.d/conda.sh && conda activate env_isaaclab

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
