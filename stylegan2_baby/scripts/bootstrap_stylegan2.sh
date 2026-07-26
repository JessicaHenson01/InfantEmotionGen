#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${STYLEGAN2_DIR:-$ROOT/vendor/stylegan2-ada-pytorch}"

python -m pip install -r "$ROOT/requirements.txt"

if [[ ! -d "$VENDOR_DIR/.git" ]]; then
  mkdir -p "$(dirname "$VENDOR_DIR")"
  git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git "$VENDOR_DIR"
else
  echo "StyleGAN2-ADA checkout already exists: $VENDOR_DIR"
fi

echo "StyleGAN2-ADA ready: $VENDOR_DIR"
echo "For strict reproducibility, record and pin:"
git -C "$VENDOR_DIR" rev-parse HEAD
