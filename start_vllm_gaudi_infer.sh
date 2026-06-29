#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/test1/habanalabs-venv/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-235B-A22B}"
TP_SIZE="${VLLM_TP_SIZE:-8}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
PROMPT="${PROMPT:-日本語で短く自己紹介して}"

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

exec "$PYTHON_BIN" vllm_gaudi_infer.py \
  --model-id "$MODEL_ID" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --prompt "$PROMPT" \
  "$@"
