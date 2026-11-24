#!/bin/bash
#SBATCH --job-name=speck_convert
#SBATCH --partition=earth-1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4             # Reduced from 16 (plenty for conversion)
#SBATCH --mem=16G                     # Reduced from 64G
#SBATCH --time=00:30:00               # Reduced to 1 hour (conversion takes ~5 mins)
#SBATCH --output=convert_%j.out
#SBATCH --error=convert_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matheron@students.zhaw.ch

# --- 1. Load Environment ---
# Using the exact logic that worked for your training
export MAMBA_EXE='/cfs/earth/scratch/matheron/bin/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1

# Verification check
python -c "import sinabs; print('✅ Sinabs loaded successfully')" || exit 1

# --- 2. Execution ---
# Navigate to the folder where your conversion script and .pt file are
cd /cfs/earth/scratch/matheron/tactile-sensing/convert2speck/

echo "🚀 Conversion started on $(hostname) at $(date)"

# Run the conversion script
python convert_to_speck.py

echo "✅ Conversion finished at $(date)"

# --- Send the results via email ---
mail -s "Speck Conversion Result: $SLURM_JOB_ID" matheron@students.zhaw.ch < convert_${SLURM_JOB_ID}.out