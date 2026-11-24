#!/bin/bash
#SBATCH --job-name=quant_eval
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/cfs/earth/scratch/matheron/tactile-sensing/quant_eval/run_%j.out
#SBATCH --error=/cfs/earth/scratch/matheron/tactile-sensing/quant_eval/run_%j.err

export MAMBA_EXE="/cfs/earth/scratch/matheron/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1
export OMP_NUM_THREADS=8

cd /cfs/earth/scratch/matheron/tactile-sensing/demo
python -u eval_quantized.py \
  --model ../deploy_checkpoints/wclamp_v93.33_t86.67.pt \
  --data /cfs/earth/scratch/matheron/tactile-sensing/data/dataset_curv_padded_10ms \
  --match_train
