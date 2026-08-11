#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MIN_FREE_GB="${MIN_FREE_GB:-60}"
DEVICE="${DEVICE:-auto}"
DTYPE_SDXL="${DTYPE_SDXL:-auto}"
DTYPE_SD35="${DTYPE_SD35:-float16}"
RUN_SDXL="${RUN_SDXL:-1}"
RUN_SD35="${RUN_SD35:-1}"
REAL_DIR="${REAL_DIR:-${REPO_ROOT}/evaluation/data/external_reference}"
export REAL_DIR

free_kb="$(df -Pk "${REPO_ROOT}" | awk 'NR == 2 {print $4}')"
free_gb="$((free_kb / 1024 / 1024))"

if (( free_gb < MIN_FREE_GB )); then
  cat >&2 <<EOF
ERROR: Only ${free_gb}GiB free under ${REPO_ROOT}.
Full SDXL + SD3.5 evaluation is large. Free more disk or rerun with a lower
MIN_FREE_GB if you intentionally want to continue.

Example:
  MIN_FREE_GB=25 ${0}
EOF
  exit 1
fi

generation_overrides=()
if [[ -n "${NUM_IMAGES:-}" ]]; then
  generation_overrides+=(--num-images "${NUM_IMAGES}")
fi
if [[ -n "${HEIGHT:-}" ]]; then
  generation_overrides+=(--height "${HEIGHT}")
fi
if [[ -n "${WIDTH:-}" ]]; then
  generation_overrides+=(--width "${WIDTH}")
fi
if [[ -n "${NUM_INFERENCE_STEPS:-}" ]]; then
  generation_overrides+=(--num-inference-steps "${NUM_INFERENCE_STEPS}")
fi
if [[ -n "${GUIDANCE_SCALE:-}" ]]; then
  generation_overrides+=(--guidance-scale "${GUIDANCE_SCALE}")
fi
if [[ -n "${SEED:-}" ]]; then
  generation_overrides+=(--seed "${SEED}")
fi

echo "Free disk: ${free_gb}GiB"
echo "Reference images: ${REAL_DIR}"
echo "Generation overrides: ${generation_overrides[*]:-(protocol defaults)}"

if [[ "${RUN_SDXL}" == "1" ]]; then
  "${SCRIPT_DIR}/generate_model_run.sh" sdxl_primary \
    --device "${DEVICE}" \
    --dtype "${DTYPE_SDXL}" \
    --skip-existing \
    "${generation_overrides[@]}"

  "${SCRIPT_DIR}/evaluate_generated_run.sh" sdxl_primary
fi

if [[ "${RUN_SD35}" == "1" ]]; then
  "${SCRIPT_DIR}/generate_model_run.sh" sd35_medium \
    --device "${DEVICE}" \
    --dtype "${DTYPE_SD35}" \
    --disable-xet \
    --hf-transfer-workers 1 \
    --skip-existing \
    "${generation_overrides[@]}"

  "${SCRIPT_DIR}/evaluate_generated_run.sh" sd35_medium
fi

"${SCRIPT_DIR}/compare_model_runs.sh"

echo "Full evaluation comparison:"
cat "${REPO_ROOT}/evaluation/results/comparison.md"
