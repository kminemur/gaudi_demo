#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DEPS="${DEPS:-$PWD/.deps}"
VERSION="${VERSION:-v0.24.0}"

TORCH_VERSION=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON" -c '
import torch
assert torch.version.cuda is None, f"CUDA PyTorch {torch.__version__} is installed"
assert torch.__version__.startswith("2.11."), f"PyTorch 2.11 is required: {torch.__version__}"
import re
print(re.match(r"\d+\.\d+\.\d+", torch.__version__).group())
' || { echo "Use the Gaudi Software 1.24.1 Python environment." >&2; exit 1; })
"$PYTHON" -c 'import torch, habana_frameworks.torch; assert torch.hpu.is_available()'

CONSTRAINTS=$(mktemp)
trap 'rm -f "$CONSTRAINTS"' EXIT
TORCH_FULL_VERSION=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON" -c 'import torch; print(torch.__version__)')
printf 'torch==%s\n' "$TORCH_FULL_VERSION" > "$CONSTRAINTS"
export PIP_CONSTRAINT="$CONSTRAINTS"

mkdir -p "$DEPS"
cd "$DEPS"

test -d vllm-gaudi || git clone https://github.com/vllm-project/vllm-gaudi
git -C vllm-gaudi fetch --tags
git -C vllm-gaudi checkout "$VERSION"

test -d vllm || git clone https://github.com/vllm-project/vllm
git -C vllm fetch --tags
git -C vllm checkout "$VERSION"

"$PYTHON" -m pip install -r <(sed '/^torch/d' vllm/requirements/build/cuda.txt)
"$PYTHON" -m pip install -r vllm/requirements/common.txt
"$PYTHON" -m pip install -r vllm-gaudi/requirements.txt
(cd vllm && VLLM_TARGET_DEVICE=empty "$PYTHON" -m pip install --no-build-isolation --no-deps -e .)
(cd vllm-gaudi && "$PYTHON" -m pip install --no-deps -e .)

"$PYTHON" -m pip install --no-deps "torchaudio==$TORCH_VERSION" --extra-index-url https://download.pytorch.org/whl/cpu
"$PYTHON" -c 'import torch, habana_frameworks.torch; assert torch.version.cuda is None and torch.hpu.is_available()'
