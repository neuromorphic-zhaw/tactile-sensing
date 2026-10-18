#!/bin/bash
#SBATCH --job-name=eval_wclstr
#SBATCH --partition=earth-5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin_10ms/wclamp_stretch/eval_%j.out
#SBATCH --error=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin_10ms/wclamp_stretch/eval_%j.err
export MAMBA_EXE='/cfs/earth/scratch/matheron/bin/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1
cd /cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin_10ms/wclamp_stretch/run_1069328
python -u ../eval_ckpt.py
