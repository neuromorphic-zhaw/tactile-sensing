#!/bin/bash
#SBATCH --job-name=ss_tex
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=8:00:00

# Specksim event-level eval launcher — one job per run dir.
# Everything lands inside the run dir (self-contained):
#   specksim_eval.out/err logs + specksim_test.json results
# Env: RUN_DIR (required), SPLIT (default test), LIMIT (default 0=all)

export MAMBA_EXE='/cfs/earth/scratch/matheron/bin/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1
python -c "import samna; print('samna loaded')" || exit 1

VD=/cfs/earth/scratch/matheron/tactile-sensing/phase2_texture_dot_ab
cd "$RUN_DIR"
echo "specksim eval: $RUN_DIR split=${SPLIT:-test}"
python -u "$VD/eval_specksim_tex.py" \
    --model "$RUN_DIR/best.pt" \
    --split ${SPLIT:-test} \
    ${LIMIT:+--limit $LIMIT}
