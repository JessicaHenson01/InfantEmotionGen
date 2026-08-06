#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REPO_ID="${REPO_ID:-InfantEmotionGen/baby_samples_gan}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/evaluation/data}"
TEST_FRACTION="${TEST_FRACTION:-0.20}"
SEED="${SEED:-42}"

exec python "${SCRIPT_DIR}/populate_real_eval_folders.py" \
  --repo-id "${REPO_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  --test-fraction "${TEST_FRACTION}" \
  --seed "${SEED}" \
  "$@"
