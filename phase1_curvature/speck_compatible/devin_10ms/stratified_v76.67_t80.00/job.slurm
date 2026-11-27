#!/bin/bash
#SBATCH --job-name=stratified
#SBATCH --partition=earth-5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin/stratified/run_%j.out
#SBATCH --error=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin/stratified/run_%j.err

export MAMBA_EXE='/cfs/earth/scratch/matheron/bin/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1
python -c "import sinabs; print('sinabs loaded')" || exit 1
rm -rf ~/.cache/triton/* /tmp/torchinductor_${USER}* ~/.cache/torch/kernels/*
ipcrm -a 2>/dev/null || true
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${SLURM_JOB_ID}"
mkdir -p $TORCH_EXTENSIONS_DIR
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

VD=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin/stratified
RUN_DIR=$VD/run_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"
cp "$VD/job.slurm" "$RUN_DIR/"
cp "$VD/train.py" "$RUN_DIR/"
cd "$RUN_DIR"
echo "Run dir: $RUN_DIR"
python -u "$VD/train.py"
cp "$VD/run_${SLURM_JOB_ID}.out" "$RUN_DIR/" 2>/dev/null || true
rm -rf /tmp/torchinductor_${USER}* ~/.cache/triton/* $TORCH_EXTENSIONS_DIR
