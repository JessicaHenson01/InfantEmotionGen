#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SG2_DIR="${STYLEGAN2_DIR:-$ROOT/vendor/stylegan2-ada-pytorch}"
DATASET_ZIP="${DATASET_ZIP:-$ROOT/data/baby_samples_gan_256.zip}"
OUTDIR="${OUTDIR:-$ROOT/outputs_mps}"

BATCH="${BATCH:-1}"
KIMG="${KIMG:-1}"
SNAP="${SNAP:-1}"
WORKERS="${WORKERS:-1}"
METRICS="${METRICS:-none}"
RESUME="${RESUME:-ffhq256}"
SEED="${SEED:-0}"

if [[ ! -f "$DATASET_ZIP" ]]; then
  echo "Dataset not found: $DATASET_ZIP" >&2
  echo "Run bash scripts/prepare_dataset_256.sh first." >&2
  exit 1
fi

if [[ ! -f "$SG2_DIR/train.py" ]]; then
  echo "StyleGAN2-ADA not found: $SG2_DIR" >&2
  exit 1
fi

python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
if not torch.backends.mps.is_available():
    raise SystemExit(
        "MPS is unavailable in this environment. The MPS run cannot start."
    )
PY

mkdir -p "$OUTDIR"

# Unsupported MPS operations may execute on CPU. This can be considerably slower,
# but allows a wider range of ordinary PyTorch operations to run.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export STYLEGAN_DEVICE="mps"

cd "$SG2_DIR"
python train.py \
  --outdir="$OUTDIR" \
  --data="$DATASET_ZIP" \
  --cond=1 \
  --gpus=1 \
  --cfg=auto \
  --batch="$BATCH" \
  --aug=ada \
  --mirror=1 \
  --resume="$RESUME" \
  --snap="$SNAP" \
  --kimg="$KIMG" \
  --workers="$WORKERS" \
  --metrics="$METRICS" \
  --seed="$SEED" \
  --fp32=1 \
  --nobench=1
