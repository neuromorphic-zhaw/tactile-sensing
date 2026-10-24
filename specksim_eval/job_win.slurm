#!/bin/bash
#SBATCH --job-name=ss_win
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=/cfs/earth/scratch/matheron/tactile-sensing/specksim_eval/win_%j.out
#SBATCH --error=/cfs/earth/scratch/matheron/tactile-sensing/specksim_eval/win_%j.err
export MAMBA_EXE="/cfs/earth/scratch/matheron/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1
export OMP_NUM_THREADS=16
cd /cfs/earth/scratch/matheron/tactile-sensing/demo
python -u eval_specksim_events.py --model "$CKPT" --split val --window_events "$W"
