#!/bin/bash
#SBATCH --job-name=speck-fitting
#SBATCH --partition=earth-5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin/%x_%j.out
#SBATCH --error=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matheron@students.zhaw.ch

# --- 1. Load Environment ---
export MAMBA_EXE='/cfs/earth/scratch/matheron/bin/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1

python -c "import sinabs; print('sinabs loaded')" || exit 1

# cache wipe (same as elevate run)
rm -rf ~/.cache/triton/* /tmp/torchinductor_${USER}* ~/.cache/torch/kernels/*
ipcrm -a 2>/dev/null || true
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${SLURM_JOB_ID}"
mkdir -p $TORCH_EXTENSIONS_DIR
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- 2. Per-run output folder ---
# Everything this run produces (model .pt, confusion matrices, logs, scripts)
# lives in speck_compatible/devin/<jobname>_<jobid>/ — old experiments untouched.
SLURM_SRC=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/10MS-SPECK/10ms-4striding/train_fitting.slurm
SCRIPT_DIR=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/devin
RUN_ROOT=$SCRIPT_DIR
RUN_DIR=$RUN_ROOT/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

# snapshot the exact scripts this run used
cp "$SLURM_SRC" "$RUN_DIR/job.slurm"
cp "$SCRIPT_DIR/train_speck_fitting.py" "$RUN_DIR/"

cd "$RUN_DIR"
echo "Run dir: $RUN_DIR"

echo "Fitting-backbone training started on $(hostname) at $(date)"
python -u "$SCRIPT_DIR/train_speck_fitting.py"
echo "Training finished at $(date)"

# pull the slurm logs into the run folder so it is self-contained
cp "$RUN_ROOT/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$RUN_DIR/" 2>/dev/null || true
cp "$RUN_ROOT/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "$RUN_DIR/" 2>/dev/null || true

# post-run cleanup
rm -rf /tmp/torchinductor_${USER}* ~/.cache/triton/* $TORCH_EXTENSIONS_DIR
nvidia-smi --gpu-reset -i ${SLURM_STEP_GPUS:-0} 2>/dev/null || true
