#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/evaluation/data}"
REAL_DIR="${REAL_DIR:-${DATA_ROOT}/real_reference}"
TEST_DIR="${TEST_DIR:-${DATA_ROOT}/pipeline_test}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/evaluation/results/pipeline_test}"
CLIP_PROMPTS="${CLIP_PROMPTS:-${REPO_ROOT}/evaluation/configs/clip_prompts.smoke.json}"
FER_MAPPING="${FER_MAPPING:-${REPO_ROOT}/evaluation/configs/fer_mapping.smoke.json}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/evaluation/.cache}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-16}"
FER_BATCH_SIZE="${FER_BATCH_SIZE:-16}"
FID_FEATURE="${FID_FEATURE:-2048}"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${CACHE_ROOT}/torch" "${CACHE_ROOT}/huggingface" "${CACHE_ROOT}/matplotlib"

export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_ROOT}/matplotlib}"

"${SCRIPT_DIR}/populate_real_eval_folders.sh" --validate-only

python "${SCRIPT_DIR}/evaluate_fid.py" \
  --real "${REAL_DIR}" \
  --generated "${TEST_DIR}" \
  --output "${RESULTS_DIR}/fid.json" \
  --batch-size "${BATCH_SIZE}" \
  --feature "${FID_FEATURE}"

python "${SCRIPT_DIR}/evaluate_clip.py" \
  --generated "${TEST_DIR}" \
  --prompts "${CLIP_PROMPTS}" \
  --output "${RESULTS_DIR}/clip.json" \
  --device "${DEVICE}" \
  --batch-size "${CLIP_BATCH_SIZE}"

if [[ -n "${FER_MODEL_ID:-}" ]]; then
  python "${SCRIPT_DIR}/evaluate_fer.py" \
    --generated "${TEST_DIR}" \
    --model-id "${FER_MODEL_ID}" \
    --mapping "${FER_MAPPING}" \
    --output "${RESULTS_DIR}/fer.json" \
    --device "${DEVICE}" \
    --batch-size "${FER_BATCH_SIZE}"
else
  cat >&2 <<EOF
Skipping FER smoke test because FER_MODEL_ID is not set.
Run again with:
  FER_MODEL_ID=YOUR_FER_MODEL_OR_LOCAL_PATH evaluation/scripts/run_smoke_tests.sh
EOF
fi

echo "Smoke-test outputs saved under: ${RESULTS_DIR}"
