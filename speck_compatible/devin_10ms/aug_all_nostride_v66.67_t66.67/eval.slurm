#!/bin/bash
#SBATCH --job-name=eval_aug
#SBATCH --partition=earth-5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH --output=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin_10ms/aug_all_nostride_RUNNING_v66.67/eval_%j.out
#SBATCH --error=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin_10ms/aug_all_nostride_RUNNING_v66.67/eval_%j.err
export MAMBA_EXE='/cfs/earth/scratch/matheron/bin/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1
cd /cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin_10ms/aug_all_nostride_RUNNING_v66.67/run_1069321
python -u ../eval_ckpt.py
