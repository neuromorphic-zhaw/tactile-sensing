#!/bin/bash
#SBATCH --job-name=2ms_s4_best
#SBATCH --partition=earth-4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin_2ms/s4_best/run_%j.out
#SBATCH --error=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin_2ms/s4_best/run_%j.err

export MAMBA_EXE='/cfs/earth/scratch/matheron/bin/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1
python -c "import sinabs; print('sinabs loaded')" || exit 1
rm -rf ~/.cache/triton/* /tmp/torchinductor_${USER}* ~/.cache/torch/kernels/*
ipcrm -a 2>/dev/null || true
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${SLURM_JOB_ID}"
mkdir -p $TORCH_EXTENSIONS_DIR
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

VD=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin_2ms/s4_best
RUN_DIR=$VD/run_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"
cp "$VD/job.slurm" "$RUN_DIR/"
cp "$VD/train.py" "$RUN_DIR/"
cd "$RUN_DIR"
echo "Run dir: $RUN_DIR"
python -u "$VD/train.py"
# move slurm logs inside the run folder so the variant dir stays clean
mv "$VD/run_${SLURM_JOB_ID}.out" "$VD/run_${SLURM_JOB_ID}.err" "$RUN_DIR/" 2>/dev/null || true
rm -rf /tmp/torchinductor_${USER}* ~/.cache/triton/* $TORCH_EXTENSIONS_DIR
