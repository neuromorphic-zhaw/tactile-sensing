#!/bin/bash
#SBATCH --job-name=ss_winft
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/cfs/earth/scratch/matheron/tactile-sensing/phase1_curvature/specksim_eval/win_%j.out
#SBATCH --error=/cfs/earth/scratch/matheron/tactile-sensing/phase1_curvature/specksim_eval/win_%j.err
export MAMBA_EXE="/cfs/earth/scratch/matheron/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1
export OMP_NUM_THREADS=32
cd /cfs/earth/scratch/matheron/tactile-sensing/phase1_curvature/demo
# windowed majority vote — the champion recipe (200K-event windows).
# CKPT/SPLIT/W/MAXEV passed via --export.
python -u eval_specksim_events.py --model "$CKPT" --split "$SPLIT" --window_events "$W" --max_events "$MAXEV"
