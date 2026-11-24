#!/bin/bash
#SBATCH --job-name=initialSNN_80bins
#SBATCH --partition=earth-4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00

# --- Settings variables for the filename ---
BINS=80
BATCH=16
MODEL="hpc_initial"

# This line dynamically names the log file
# It will look like: train_hpc_initial_bins80_batch16.log
#SBATCH --output=train_%j_${MODEL}_bins${BINS}_batch${BATCH}.log
This tells Slurm where your micromamba lives
export MAMBA_ROOT_PREFIX=/cfs/earth/scratch/matheron/micromamba
eval "$($MAMBA_ROOT_PREFIX/bin/micromamba shell hook -s bash)"

# Load your environment
# Note: Some clusters require 'eval "$(micromamba shell hook --shell bash)"' 
# before activating if it's a fresh shell.
micromamba activate speck__hpc1

# Memory fix for A100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting training with Bins: $BINS, Batch: $BATCH"
echo "Job ID: $SLURM_JOB_ID"

# Run the training
python train_curv.py