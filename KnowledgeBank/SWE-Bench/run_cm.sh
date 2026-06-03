#!/bin/bash

MODEL_NAME="${MODEL_NAME:-deepseek/deepseek-chat}"
SUBSET="${SUBSET:-verified}"
SPLIT="${SPLIT:-test}"
WORKERS="${WORKERS:-3}"
RESULTS_DIR="${RESULTS_DIR:-./results_memory_cm}"
MEMORY_ROOT="${MEMORY_ROOT:-./memory_v2}"
TOP_K_MEMORIES="${TOP_K_MEMORIES:-3}"
CONSOLIDATE_EVERY="${CONSOLIDATE_EVERY:-25}"
VERIFIER_MODE="${VERIFIER_MODE:-text}"
VERIFIER_MODEL="${VERIFIER_MODEL:-gemini-2.5-flash}"
VERIFIER_REPS="${VERIFIER_REPS:-1}"
VERIFIER_SUCCESS_THRESHOLD="${VERIFIER_SUCCESS_THRESHOLD:-0.70}"
VERIFIER_FAIL_THRESHOLD="${VERIFIER_FAIL_THRESHOLD:-0.35}"
WRITE_UNCERTAIN_MEMORY="${WRITE_UNCERTAIN_MEMORY:-false}"
SLICE="${SLICE:-}"
FILTER="${FILTER:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export MSWEA_MODEL_API_KEY="${MSWEA_MODEL_API_KEY:-${DEEPSEEK_API_KEY:-}}"
export PYTHONPATH="${REPO_ROOT}/third_party/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${MODEL_NAME}" == vertex_ai/* ]]; then
    VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
    VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-${GOOGLE_CLOUD_LOCATION:-us-central1}}"
    export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-true}"
    export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$VERTEXAI_PROJECT}"
    export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-$VERTEXAI_LOCATION}"
fi

cd "${REPO_ROOT}/third_party"

WRITE_UNCERTAIN_ARG=""
if [[ "${WRITE_UNCERTAIN_MEMORY}" == "true" ]]; then
    WRITE_UNCERTAIN_ARG="--write-uncertain-memory"
fi

"${PYTHON_BIN}" -m minisweagent.run.mini_extra swebench-cm \
    --model "${MODEL_NAME}" \
    --subset "${SUBSET}" \
    --split "${SPLIT}" \
    --workers "${WORKERS}" \
    --memory-root "${MEMORY_ROOT}" \
    --top-k-memories "${TOP_K_MEMORIES}" \
    --consolidate-every "${CONSOLIDATE_EVERY}" \
    --verifier-mode "${VERIFIER_MODE}" \
    --verifier-model "${VERIFIER_MODEL}" \
    --verifier-reps "${VERIFIER_REPS}" \
    --verifier-success-threshold "${VERIFIER_SUCCESS_THRESHOLD}" \
    --verifier-fail-threshold "${VERIFIER_FAIL_THRESHOLD}" \
    ${WRITE_UNCERTAIN_ARG:+${WRITE_UNCERTAIN_ARG}} \
    ${SLICE:+--slice "${SLICE}"} \
    ${FILTER:+--filter "${FILTER}"} \
    --output "${RESULTS_DIR}"
