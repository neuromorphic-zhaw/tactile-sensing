#!/bin/bash
#SBATCH --job-name=specksim_sanity
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/cfs/earth/scratch/matheron/tactile-sensing/specksim_sanity/run_%j.out
#SBATCH --error=/cfs/earth/scratch/matheron/tactile-sensing/specksim_sanity/run_%j.err

export MAMBA_EXE="/cfs/earth/scratch/matheron/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1
export OMP_NUM_THREADS=8

cd /cfs/earth/scratch/matheron/tactile-sensing/demo
BIN=/cfs/earth/scratch/matheron/tactile-sensing/data/time_bins/dataset_curv_padded_10ms
CKPT=../deploy_checkpoints/wclamp_v93.33_t86.67.pt

echo "=== A) BINNED PLAYBACK flat_01 (raw events flipped it to curved) ==="
echo "    expect pred=0 if specksim faithfully replays the quantized eval"
python -u eval_specksim_events.py --model $CKPT --binned_data $BIN \
  --clip 2026-03-17_flat_01.npz --max_events 0

echo "=== B) BINNED PLAYBACK flat_05 (raw events kept it flat) ==="
python -u eval_specksim_events.py --model $CKPT --binned_data $BIN \
  --clip 2026-03-17_flat_05.npz --max_events 0

echo "=== C) RAW flat_01 with INVERTED polarity ==="
python -u eval_specksim_events.py --model $CKPT --invert_p \
  --clip 2026-03-17_flat_01.npz --max_events 0
