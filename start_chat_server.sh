#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/test1/habanalabs-venv-optimum/bin/python}"
TP_SIZE="${CHAT_TENSOR_PARALLEL_SIZE:-8}"
HOST="${SERVER_HOST:-0.0.0.0}"
PORT="${SERVER_PORT:-8000}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-235B-A22B}"
LOG_DIR="${TORCHRUN_LOG_DIR:-$SCRIPT_DIR/logs/torchrun}"

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"
export CHAT_TENSOR_PARALLEL_SIZE="$TP_SIZE"
export MODEL_ID
export SERVER_HOST="$HOST"
export SERVER_PORT="$PORT"

mkdir -p "$LOG_DIR"

if [[ "${CHAT_MODEL_AUTO_PREPARE:-1}" == "1" ]]; then
  echo "Ensuring local model snapshot: $MODEL_ID"
  "$PYTHON_BIN" download_hf_models.py \
    --model-id "$MODEL_ID" \
    --ensure
fi

exec "$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --log-dir "$LOG_DIR" \
  --redirects 3 \
  --tee 3 \
  --nproc_per_node="$TP_SIZE" \
  chat_server.py \
  --host "$HOST" \
  --port "$PORT" \
  --model-id "$MODEL_ID" \
  --tensor-parallel-size "$TP_SIZE"
