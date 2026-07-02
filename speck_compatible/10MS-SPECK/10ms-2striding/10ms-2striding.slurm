#!/bin/bash
#SBATCH --job-name=10ms-2striding
#SBATCH --partition=earth-4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
# 🎯 FORCED ABSOLUTE LOG DESTINATIONS:
#SBATCH --output=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/10MS-SPECK/10ms-2striding/train_main_%j.out
#SBATCH --error=/cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/10MS-SPECK/10ms-2striding/train_main_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matheron@students.zhaw.ch

# --- 1. Load Environment ---
export MAMBA_EXE='/cfs/earth/scratch/matheron/bin/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1

python -c "import sinabs; print('✅ Sinabs loaded successfully')" || exit 1

# 🧹 ========================================================
# 🔥 AGGRESSIVE PRE-RUN CACHE WIPEOUT (Blank Slate Protocol)
# ========================================================
echo "🧹 Wiping all previous session artifacts from the cluster node..."

# 1. Clear out pre-compiled Triton and TorchInductor GPU kernels
rm -rf ~/.cache/triton/*
rm -rf /tmp/torchinductor_${USER}*
rm -rf ~/.cache/torch/kernels/*

# 2. Clear out any dead shared memory blocks left behind by PyTorch DataLoader workers
ipcrm -a 2>/dev/null || true

# 3. Force isolation for compiler extensions for this specific job ID
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${SLURM_JOB_ID}"
mkdir -p $TORCH_EXTENSIONS_DIR

# Hardware Memory Tuning for SNN Unrolling
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "✅ Node disk and OS memory caches cleared cleanly."
# ========================================================

# --- 2. Execution ---
cd /cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/10MS-SPECK/10ms-2striding

echo "🚀 Baseline Training started on $(hostname) at $(date)"
python train_10ms_2striding.py

echo "✅ Training finished at $(date)"

# 🧹 ========================================================
# 🛑 POST-RUN CLEANUP & SYSTEM RESET
# ========================================================
echo "🧹 Post-run cleanup: Erasing transient compiler artifacts..."
rm -rf /tmp/torchinductor_${USER}*
rm -rf ~/.cache/triton/*
rm -rf $TORCH_EXTENSIONS_DIR

# Reset active context on your assigned GPU to ensure 0MB leakage
nvidia-smi --gpu-reset -i ${SLURM_STEP_GPUS:-0} 2>/dev/null || true
echo "✅ All post-session caches wiped cleanly."
# ========================================================

# 🎯 FORCED ABSOLUTE PATH FOR EMAIL REDIRECTION COPIER:
mail -s "Slurm Job Results: $SLURM_JOB_NAME ($SLURM_JOB_ID)" matheron@students.zhaw.ch < /cfs/earth/scratch/matheron/tactile-sensing/speck_compatible/10MS-SPECK/10ms-2striding/train_main_${SLURM_JOB_ID}.out