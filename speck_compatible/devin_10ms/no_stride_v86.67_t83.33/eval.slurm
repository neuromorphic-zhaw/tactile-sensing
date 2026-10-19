#!/bin/bash
#SBATCH --job-name=eval_no
#SBATCH --partition=earth-5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH --output=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin_10ms/no_stride_RUNNING_v83.33/eval_%j.out
#SBATCH --error=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin_10ms/no_stride_RUNNING_v83.33/eval_%j.err
export MAMBA_EXE='/cfs/earth/scratch/matheron/bin/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1
cd /cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin_10ms/no_stride_RUNNING_v83.33/run_1069301
python -u ../eval_ckpt.py
