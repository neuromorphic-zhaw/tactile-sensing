#!/bin/bash
#SBATCH --job-name=ss_ssm_l8
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --output=/cfs/earth/scratch/matheron/tactile-sensing/specksim_eval/ssml8_%j.out
#SBATCH --error=/cfs/earth/scratch/matheron/tactile-sensing/specksim_eval/ssml8_%j.err

export MAMBA_EXE="/cfs/earth/scratch/matheron/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1
export OMP_NUM_THREADS=16

cd /cfs/earth/scratch/matheron/tactile-sensing/demo
# l8 models: eval auto-detects feat_dim=64 from ckpt and monitors the
# last-conv IAF (layer 7). CKPT/SPLIT/WM/MAXEV passed via --export.
python -u eval_specksim_ssm.py --model "$CKPT" --split "$SPLIT" --win_mode "$WM" --max_events "$MAXEV"
