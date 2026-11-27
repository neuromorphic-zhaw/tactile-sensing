#!/bin/bash
# sweep_specksim.sh — submit specksim event-level evals for every run dir
# that has best.pt + metrics.json (i.e. finished training).
# Usage: bash sweep_specksim.sh [run_substr_filter]

VD=/cfs/earth/scratch/matheron/tactile-sensing/phase2_texture_dot_ab
FILTER=${1:-}

for d in "$VD"/runs/*/; do
    name=$(basename "$d")
    [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]] && continue
    [ -f "$d/best.pt" ] && [ -f "$d/metrics.json" ] || { echo "skip $name (incomplete)"; continue; }
    [ -f "$d/specksim_test.json" ] && { echo "skip $name (already evaluated)"; continue; }
    sbatch --job-name="ss_${name}" \
           --chdir="$d" \
           --output="$d/specksim_eval.out" \
           --error="$d/specksim_eval.err" \
           --export=ALL,RUN_DIR="$d" \
           "$VD/job_specksim.slurm"
done
