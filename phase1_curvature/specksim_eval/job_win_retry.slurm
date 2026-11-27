#!/bin/bash
#SBATCH --job-name=ss_win
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/cfs/earth/scratch/matheron/tactile-sensing/specksim_eval/win_%j.out
#SBATCH --error=/cfs/earth/scratch/matheron/tactile-sensing/specksim_eval/win_%j.err
export MAMBA_EXE="/cfs/earth/scratch/matheron/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1
export OMP_NUM_THREADS=32
cd /cfs/earth/scratch/matheron/tactile-sensing/demo
# retry of 1074714 (timed out after 4 clips on evT166): --max_events caps
# the monster clips where a per-window run stalled; walltime 10h -> 24h,
# 16 -> 32 threads. W + CKPT passed via --export.
python -u eval_specksim_events.py --model "$CKPT" --split val --window_events "$W" --max_events 400000
