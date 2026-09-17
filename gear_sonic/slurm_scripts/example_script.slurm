#!/bin/bash
#SBATCH --job-name=cat-scratch-data
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --nodelist=zac
#SBATCH --time=1:00:00
#SBATCH --output=logs/cat-data-%j.out

source /home/software/miniforge3/etc/profile.d/conda.sh && conda activate env_isaaclab

DATA=/scratch/$USER/seed

which python

ls $DATA


