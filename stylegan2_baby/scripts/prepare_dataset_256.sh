#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SG2_DIR="${STYLEGAN2_DIR:-$ROOT/vendor/stylegan2-ada-pytorch}"
SOURCE="${SOURCE:-$ROOT/data/baby_samples_gan.zip}"
DEST="${DEST:-$ROOT/data/baby_samples_gan_256.zip}"

if [[ ! -f "$SG2_DIR/dataset_tool.py" ]]; then
  echo "Missing StyleGAN2 checkout. Run scripts/bootstrap_stylegan2.sh first." >&2
  exit 1
fi

if [[ ! -f "$SOURCE" ]]; then
  echo "Missing source dataset: $SOURCE" >&2
  exit 1
fi

if [[ -e "$DEST" ]]; then
  echo "Destination already exists: $DEST"
  echo "Remove it first to rebuild it."
  exit 1
fi

cd "$SG2_DIR"
python dataset_tool.py \
  --source="$SOURCE" \
  --dest="$DEST" \
  --width=256 \
  --height=256

echo "Prepared 256x256 dataset: $DEST"
