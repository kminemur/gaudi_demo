#!/usr/bin/env bash
set -euo pipefail

URL="${URL:-http://localhost:8000}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-Next-FP8}"
HEADERS=(-H 'Content-Type: application/json')
test -z "${API_KEY:-}" || HEADERS+=(-H "Authorization: Bearer $API_KEY")

curl -sS "$URL/v1/chat/completions" "${HEADERS[@]}" -d "$(
  printf '{"model":"%s","messages":[{"role":"user","content":"%s"}]}' \
    "$MODEL" "${PROMPT:-Gaudiを一文で説明して}"
)"
printf '\n'
