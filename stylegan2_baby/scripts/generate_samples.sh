#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SG2_DIR="${STYLEGAN2_DIR:-$ROOT/vendor/stylegan2-ada-pytorch}"
NETWORK="${NETWORK:-}"
CLASS_IDX="${CLASS_IDX:-0}"
SEEDS="${SEEDS:-0-15}"
GENERATED_DIR="${GENERATED_DIR:-$ROOT/outputs/generated}"
TRUNC="${TRUNC:-0.7}"

if [[ -z "$NETWORK" ]]; then
  echo "Set NETWORK to a network-snapshot-*.pkl file." >&2
  exit 1
fi

mkdir -p "$GENERATED_DIR"
cd "$SG2_DIR"

python generate.py \
  --outdir="$GENERATED_DIR" \
  --network="$NETWORK" \
  --class="$CLASS_IDX" \
  --seeds="$SEEDS" \
  --trunc="$TRUNC"
