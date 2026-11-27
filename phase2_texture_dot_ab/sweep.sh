#!/bin/bash
# sweep.sh — submit a table of train_tex variants × both gel conditions.
#
# Each VARIANT line:  TAG  [extra KEY=VAL ...]
# KEYs map to job_tex.slurm env vars: DATA EPOCHS BS LR WD SEED TSTRIDE
# PRETRAINED PARTITION. COND is handled by the loop (dot + nodot always
# run the same variant — required for the paired A/B).
#
# Usage:  bash sweep.sh            # submit all
#         bash sweep.sh dot        # one condition only
# Edit the VARIANTS block to add/remove runs.

VD=/cfs/earth/scratch/matheron/tactile-sensing/phase2_texture_dot_ab
ONLY=${1:-both}
mkdir -p "$VD/runs"   # sbatch --output redirect needs this to exist

VARIANTS=(
    # resolution x compute grid — count-bin family prepped 11-27
    # (evT166/332/664, all slice=touch; see data/phase2_texture_dot_ab/README.md)
    "evT166_s2"                                                        # baseline = stage-1 champion recipe
    "evT166_s4 TSTRIDE=4"                                              # same data, half compute
    "evT332_s2 DATA=$VD/../data/phase2_texture_dot_ab/dataset_tex_evT332/\$COND"                     # finer res, 2x compute
    "evT332_s4 TSTRIDE=4 DATA=$VD/../data/phase2_texture_dot_ab/dataset_tex_evT332/\$COND"           # finer res, same compute as baseline
    "evT664_s4 TSTRIDE=4 DATA=$VD/../data/phase2_texture_dot_ab/dataset_tex_evT664/\$COND"           # resolution probe, 2x compute
    # "evT166_s1 TSTRIDE=1"                                            # no stride — slowest
    # "evT166_s2_lr1e4 LR=1e-4"
    # "evT166_s2_ft PRETRAINED=<ckpt> LR=5e-5 EPOCHS=50"               # warm-start
)

for line in "${VARIANTS[@]}"; do
    tag=${line%% *}
    extra=${line#* }
    [ "$extra" = "$tag" ] && extra=""
    for cond in dot nodot; do
        [ "$ONLY" != "both" ] && [ "$cond" != "$ONLY" ] && continue
        export COND=$cond
        eval "export $extra"   # expand $COND inside DATA= etc.
        sbatch --job-name="tex_${cond}_${tag}" \
               --export=ALL,COND,TAG=$tag,DATA,EPOCHS,BS,LR,WD,SEED,TSTRIDE,PRETRAINED \
               "$VD/job_tex.slurm"
    done
done
