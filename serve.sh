#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-Coder-Next-FP8}"
ARGS=(serve "$MODEL"
  --host "${HOST:-0.0.0.0}"
  --port "${PORT:-8000}"
  --tensor-parallel-size "${TP_SIZE:-8}"
  --max-model-len "${MAX_MODEL_LEN:-262144}"
  --enable-auto-tool-choice
  --tool-call-parser "qwen3_coder")

test -z "${API_KEY:-}" || ARGS+=(--api-key "$API_KEY")
exec "${VLLM_BIN:-vllm}" "${ARGS[@]}"
