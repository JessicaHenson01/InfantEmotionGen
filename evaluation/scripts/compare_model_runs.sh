#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python "${SCRIPT_DIR}/compare_results.py" \
  --run SDXLPrimary \
    "${REPO_ROOT}/evaluation/results/sdxl_primary/fid.json" \
    "${REPO_ROOT}/evaluation/results/sdxl_primary/clip.json" \
    "${REPO_ROOT}/evaluation/results/sdxl_primary/fer.json" \
  --run SD35Medium \
    "${REPO_ROOT}/evaluation/results/sd35_medium/fid.json" \
    "${REPO_ROOT}/evaluation/results/sd35_medium/clip.json" \
    "${REPO_ROOT}/evaluation/results/sd35_medium/fer.json" \
  --csv "${REPO_ROOT}/evaluation/results/comparison.csv" \
  --markdown "${REPO_ROOT}/evaluation/results/comparison.md"
