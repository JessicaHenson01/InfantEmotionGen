#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${STYLEGAN2_DIR:-$ROOT/vendor/stylegan2-ada-pytorch}"

if [[ ! -d "$VENDOR_DIR" ]]; then
  echo "Missing StyleGAN2 checkout. Run scripts/bootstrap_stylegan2.sh first." >&2
  exit 1
fi

applied=0

if [[ -f "$ROOT/patches/training_loop.py" ]]; then
  cp "$ROOT/patches/training_loop.py" "$VENDOR_DIR/training/training_loop.py"
  echo "Applied patches/training_loop.py"
  applied=1
fi

if [[ -f "$ROOT/patches/misc.py" ]]; then
  cp "$ROOT/patches/misc.py" "$VENDOR_DIR/torch_utils/misc.py"
  echo "Applied patches/misc.py"
  applied=1
fi

if [[ "$applied" -eq 0 ]]; then
  echo "No partner patch files were found."
  echo "The official implementation remains unchanged."
fi
