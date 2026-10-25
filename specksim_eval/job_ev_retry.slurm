#!/bin/bash
#SBATCH --job-name=ss_ev
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --output=/cfs/earth/scratch/matheron/tactile-sensing/specksim_eval/ev_%j.out
#SBATCH --error=/cfs/earth/scratch/matheron/tactile-sensing/specksim_eval/ev_%j.err

export MAMBA_EXE="/cfs/earth/scratch/matheron/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1
export OMP_NUM_THREADS=16

cd /cfs/earth/scratch/matheron/tactile-sensing/demo
# --max_events 500K: the 1M-event clips blew the C++ inter-layer queues
# (std::bad_alloc, jobs 1074682/83/84/86 + hang 1074720). Truncated clips
# are labeled in the log. CKPT passed via --export.
python -u eval_specksim_events.py --model "$CKPT" --split val --max_events 500000
