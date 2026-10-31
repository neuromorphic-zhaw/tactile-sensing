#!/bin/bash
#SBATCH --job-name=ss_ssm_l8t
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/cfs/earth/scratch/matheron/tactile-sensing/specksim_eval/ssml8_%j.out
#SBATCH --error=/cfs/earth/scratch/matheron/tactile-sensing/specksim_eval/ssml8_%j.err

export MAMBA_EXE="/cfs/earth/scratch/matheron/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1
export OMP_NUM_THREADS=32

cd /cfs/earth/scratch/matheron/tactile-sensing/demo
# t10ms l8 model: time-mode windows (40ms) = ~333 windows/clip -> 24h/32cpu.
# 250K cap: wclamp backbone bad_alloc'd at 500K in prior ss_ssm runs.
python -u eval_specksim_ssm.py --model "$CKPT" --split "$SPLIT" --win_mode "$WM" --max_events "$MAXEV"
