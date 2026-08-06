#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_NAME" >&2
  exit 2
fi

RUN_NAME="$1"
REAL_DIR="${REAL_DIR:-${REPO_ROOT}/evaluation/data/real_reference}"
GENERATED_DIR="${GENERATED_DIR:-${REPO_ROOT}/evaluation/generated/${RUN_NAME}}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/evaluation/results/${RUN_NAME}}"
CLIP_PROMPTS="${CLIP_PROMPTS:-${REPO_ROOT}/evaluation/configs/clip_prompts.smoke.json}"
FER_MAPPING="${FER_MAPPING:-${REPO_ROOT}/evaluation/configs/fer_mapping.smoke.json}"
FER_MODEL_ID="${FER_MODEL_ID:-trpakov/vit-face-expression}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-16}"
FER_BATCH_SIZE="${FER_BATCH_SIZE:-16}"

mkdir -p "${RESULTS_DIR}"

python "${SCRIPT_DIR}/evaluate_fid.py" \
  --real "${REAL_DIR}" \
  --generated "${GENERATED_DIR}" \
  --output "${RESULTS_DIR}/fid.json" \
  --batch-size "${BATCH_SIZE}"

python "${SCRIPT_DIR}/evaluate_clip.py" \
  --generated "${GENERATED_DIR}" \
  --prompts "${CLIP_PROMPTS}" \
  --output "${RESULTS_DIR}/clip.json" \
  --device "${DEVICE}" \
  --batch-size "${CLIP_BATCH_SIZE}"

python "${SCRIPT_DIR}/evaluate_fer.py" \
  --generated "${GENERATED_DIR}" \
  --model-id "${FER_MODEL_ID}" \
  --mapping "${FER_MAPPING}" \
  --output "${RESULTS_DIR}/fer.json" \
  --device "${DEVICE}" \
  --batch-size "${FER_BATCH_SIZE}"

echo "Evaluation outputs saved under: ${RESULTS_DIR}"
