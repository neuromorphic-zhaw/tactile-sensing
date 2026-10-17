#!/bin/bash
#SBATCH --job-name=snn_10ms
#SBATCH --partition=earth-4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --mail-type=END,FAIL          # Send email when job ends or fails
#SBATCH --mail-user=matheron@students.zhaw.ch  # <--- CHANGE THIS TO YOUR EMAIL

# --- 1. Load Environment ---
# Using the path you just confirmed
export MAMBA_EXE='/cfs/earth/scratch/matheron/bin/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1

# Verification check: ensures 'tonic' or 'sinabs' is actually visible
python -c "import sinabs; print('✅ Sinabs loaded successfully')" || exit 1

# --- 2. Execution ---
# Navigate to your script's folder
cd /cfs/earth/scratch/matheron/tactile-sensing/4_layer/83refine/

echo "🚀 Training started on $(hostname) at $(date)"
echo "📟 Using GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Run your updated 10ms script
python 83refine.py

echo "✅ Training finished at $(date)"

# --- Send the log file to your email ---
mail -s "Slurm Job Results: $SLURM_JOB_NAME ($SLURM_JOB_ID)" matheron@students.zhaw.ch < train_${SLURM_JOB_ID}.out