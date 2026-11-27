#!/bin/bash
# sweep_noise.sh — noise-augmentation experiment, ON TOP of the baseline evT166_s2
# recipe (baseline results already exist in runs/{dot,nodot}_evT166_s2).
#
# Question: does noise-augmented training change the dot-vs-nodot picture?
# Each variant isolates one noise type (or all combined); every variant runs
# both gel conditions for the paired A/B. All other knobs = baseline.
#
# Usage: bash sweep_noise.sh

VD=/cfs/earth/scratch/matheron/tactile-sensing/phase2_texture_dot_ab
mkdir -p "$VD/runs"

VARIANTS=(
    "evT166_s2_drop30   NOISE_DROP=0.30"                                   # 3x event dropout
    "evT166_s2_addp1    NOISE_ADD=0.01"                                    # 1% hot-pixel noise
    "evT166_s2_shift8   NOISE_SHIFT=8"                                     # +-8px spatial jitter
    "evT166_s2_allnoise NOISE_DROP=0.30 NOISE_ADD=0.01 NOISE_SHIFT=8"      # combined
)

for line in "${VARIANTS[@]}"; do
    tag=$(echo "$line" | awk '{print $1}')
    extra=$(echo "$line" | cut -d' ' -f2-)
    for cond in dot nodot; do
        export COND=$cond
        eval "export $extra"
        RUN_DIR="$VD/runs/${cond}_${tag}"
        mkdir -p "$RUN_DIR"
        sbatch --job-name="tex_${cond}_${tag}" \
               --chdir="$RUN_DIR" \
               --output="$RUN_DIR/train.out" \
               --error="$RUN_DIR/train.err" \
               --export=ALL,COND,TAG=$tag,DATA,EPOCHS,BS,LR,WD,SEED,TSTRIDE,PRETRAINED,NOISE_DROP,NOISE_ADD,NOISE_SHIFT \
               "$VD/job_tex.slurm"
    done
done
