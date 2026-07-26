#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SG2_DIR="${STYLEGAN2_DIR:-$ROOT/vendor/stylegan2-ada-pytorch}"
DATASET_ZIP="${DATASET_ZIP:-$ROOT/data/baby_samples_gan.zip}"
OUTDIR="${OUTDIR:-$ROOT/outputs}"

if [[ ! -f "$DATASET_ZIP" ]]; then
  echo "Dataset not found: $DATASET_ZIP" >&2
  exit 1
fi

if [[ ! -f "$SG2_DIR/train.py" ]]; then
  echo "StyleGAN2-ADA not found: $SG2_DIR" >&2
  exit 1
fi

cd "$SG2_DIR"
python train.py \
  --outdir="$OUTDIR" \
  --data="$DATASET_ZIP" \
  --cond=1 \
  --gpus=1 \
  --cfg=auto \
  --batch="${BATCH:-4}" \
  --aug=ada \
  --mirror=1 \
  --resume="${RESUME:-ffhq512}" \
  --snap=1 \
  --kimg=1 \
  --workers="${WORKERS:-2}" \
  --metrics=none \
  --dry-run
