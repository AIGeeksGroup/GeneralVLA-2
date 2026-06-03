#!/bin/bash

MODEL_NAME="${MODEL_NAME:-deepseek/deepseek-chat}"
SUBSET="${SUBSET:-verified}"
SPLIT="${SPLIT:-test}"
WORKERS="${WORKERS:-3}"
RESULTS_DIR="${RESULTS_DIR:-./results_memory_baseline}"
SLICE="${SLICE:-}"
FILTER="${FILTER:-}"
OPENAI_API_BASE="${OPENAI_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
POLICY_EMBEDDING_MODEL="${POLICY_EMBEDDING_MODEL:-text-embedding-v4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${REPO_ROOT}/third_party/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${MODEL_NAME,,}" == openai/qwen* ]] || [[ "${MODEL_NAME,,}" == *qwen3.5-flash* ]]; then
    export OPENAI_API_BASE
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENAI_API_BASE}}"
    export MSWEA_MODEL_API_KEY="${MSWEA_MODEL_API_KEY:-${OPENAI_API_KEY:-${QWEN_API_KEY:-${DASHSCOPE_API_KEY:-}}}}"
    export MSWEA_EMBEDDING_BACKEND="${MSWEA_EMBEDDING_BACKEND:-dashscope_qwen}"
    export MSWEA_EMBEDDING_API_BASE="${MSWEA_EMBEDDING_API_BASE:-${OPENAI_API_BASE}}"
    export MSWEA_EMBEDDING_API_KEY="${MSWEA_EMBEDDING_API_KEY:-${MSWEA_MODEL_API_KEY}}"
    export MSWEA_EMBEDDING_MODEL="${MSWEA_EMBEDDING_MODEL:-${POLICY_EMBEDDING_MODEL}}"
else
    export MSWEA_MODEL_API_KEY="${MSWEA_MODEL_API_KEY:-${DEEPSEEK_API_KEY:-}}"
fi

if [[ "${MODEL_NAME}" == vertex_ai/* ]]; then
    VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
    VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-${GOOGLE_CLOUD_LOCATION:-us-central1}}"
    export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-true}"
    export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$VERTEXAI_PROJECT}"
    export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-$VERTEXAI_LOCATION}"
fi

cd "${REPO_ROOT}/third_party"

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    "${PYTHON_BIN}" -m minisweagent.run.mini_extra swebench \
    --model "${MODEL_NAME}" \
    --subset "${SUBSET}" \
    --split "${SPLIT}" \
    --workers "${WORKERS}" \
    ${SLICE:+--slice "${SLICE}"} \
    ${FILTER:+--filter "${FILTER}"} \
    --output "${RESULTS_DIR}"
