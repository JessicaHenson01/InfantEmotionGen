#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SG2_DIR="${STYLEGAN2_DIR:-$ROOT/vendor/stylegan2-ada-pytorch}"
DATASET_ZIP="${DATASET_ZIP:-$ROOT/data/baby_samples_gan.zip}"
OUTDIR="${OUTDIR:-$ROOT/outputs}"

BATCH="${BATCH:-16}"
KIMG="${KIMG:-500}"
SNAP="${SNAP:-25}"
WORKERS="${WORKERS:-2}"
METRICS="${METRICS:-fid50k_full}"
RESUME="${RESUME:-ffhq512}"
SEED="${SEED:-0}"

if [[ ! -f "$DATASET_ZIP" ]]; then
  echo "Dataset not found: $DATASET_ZIP" >&2
  echo "Run python scripts/download_dataset.py first." >&2
  exit 1
fi

if [[ ! -f "$SG2_DIR/train.py" ]]; then
  echo "StyleGAN2-ADA not found: $SG2_DIR" >&2
  echo "Run bash scripts/bootstrap_stylegan2.sh first." >&2
  exit 1
fi

mkdir -p "$OUTDIR"

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
  --seed="$SEED"
