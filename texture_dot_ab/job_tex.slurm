#!/bin/bash
#SBATCH --job-name=tex
#SBATCH --partition=earth-4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/cfs/earth/scratch/matheron/tactile-sensing/texture_dot_ab/runs/%x_%j.out
#SBATCH --error=/cfs/earth/scratch/matheron/tactile-sensing/texture_dot_ab/runs/%x_%j.err

# Parameterized trainer launcher for texture_dot_ab sweeps.
# Override via sbatch --export / --chdir / --job-name / --partition.
# Env vars (all have sane defaults):
#   COND       dot|nodot        condition (default: dot)
#   TAG        string           variant tag -> run dir runs/${COND}_${TAG}
#   DATA       path             npz dir (default: ../data/dataset_tex_evT166/$COND)
#   EPOCHS     int   (100)   LR float (2e-4)   BS int (10)
#   WD         float (1e-4)  SEED int (42)     TSTRIDE int (2)
#   PRETRAINED path             warm-start ckpt (default: none)

export MAMBA_EXE='/cfs/earth/scratch/matheron/bin/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate speck__hpc1
python -c "import sinabs; print('sinabs loaded')" || exit 1
rm -rf ~/.cache/triton/* /tmp/torchinductor_${USER}* ~/.cache/torch/kernels/*
ipcrm -a 2>/dev/null || true
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${SLURM_JOB_ID}"
mkdir -p $TORCH_EXTENSIONS_DIR
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

VD=/cfs/earth/scratch/matheron/tactile-sensing/texture_dot_ab
COND=${COND:-dot}
TAG=${TAG:-evT166_s2}
DATA=${DATA:-$VD/../data/dataset_tex_evT166/$COND}
RUN_DIR=$VD/runs/${COND}_${TAG}
mkdir -p "$RUN_DIR"
cp "$VD/job_tex.slurm" "$RUN_DIR/job.slurm"
cp "$VD/train_tex.py" "$RUN_DIR/train.py"
cd "$RUN_DIR"
echo "Run dir: $RUN_DIR  |  cond=$COND tag=$TAG data=$DATA"

python -u "$VD/train_tex.py" \
    --data "$DATA" \
    --out "$RUN_DIR" \
    --epochs ${EPOCHS:-100} \
    --batch ${BS:-10} \
    --lr ${LR:-2e-4} \
    --weight_decay ${WD:-1e-4} \
    --seed ${SEED:-42} \
    --time_stride ${TSTRIDE:-2} \
    ${PRETRAINED:+--pretrained "$PRETRAINED"}
