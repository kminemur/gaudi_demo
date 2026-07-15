#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-235B-A22B}"
ARGS=(serve "$MODEL"
  --host "${HOST:-0.0.0.0}"
  --port "${PORT:-8000}"
  --tensor-parallel-size "${TP_SIZE:-8}"
  --max-model-len "${MAX_MODEL_LEN:-4096}")

test -z "${API_KEY:-}" || ARGS+=(--api-key "$API_KEY")
exec "${VLLM_BIN:-vllm}" "${ARGS[@]}"
