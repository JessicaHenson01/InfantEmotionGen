#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REPO_ID="${REPO_ID:-InfantEmotionGen/InfantEmotionGen_Dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/evaluation/data}"

exec python "${SCRIPT_DIR}/populate_external_reference.py" \
  --repo-id "${REPO_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  "$@"
