#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DEPS="${DEPS:-$PWD/.deps}"

mkdir -p "$DEPS"
cd "$DEPS"

test -d vllm-gaudi || git clone https://github.com/vllm-project/vllm-gaudi
git -C vllm-gaudi fetch origin main vllm/last-good-commit-for-vllm-gaudi
git -C vllm-gaudi checkout main
git -C vllm-gaudi pull --ff-only

VLLM_COMMIT=$(git -C vllm-gaudi show origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT)
test -d vllm || git clone https://github.com/vllm-project/vllm
git -C vllm fetch origin "$VLLM_COMMIT"
git -C vllm checkout "$VLLM_COMMIT"

"$PYTHON" -m pip install -r <(sed '/^torch/d' vllm/requirements/build/cuda.txt)
(cd vllm && VLLM_TARGET_DEVICE=empty "$PYTHON" -m pip install --no-build-isolation -e .)
(cd vllm-gaudi && "$PYTHON" -m pip install -e .)

TORCH_VERSION=$("$PYTHON" -c 'import re, torch; print(re.match(r"\d+\.\d+\.\d+", torch.__version__).group())')
"$PYTHON" -m pip install --no-deps "torchaudio==$TORCH_VERSION" --extra-index-url https://download.pytorch.org/whl/cpu
