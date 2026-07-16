#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DEPS="${DEPS:-$PWD/.deps}"

TORCH_VERSION=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON" -c '
import torch
assert torch.version.cuda is None, f"CUDA PyTorch {torch.__version__} is installed"
import re
print(re.match(r"\d+\.\d+\.\d+", torch.__version__).group())
' || { echo "Use an Intel Gaudi Python environment." >&2; exit 1; })
"$PYTHON" -c 'import torch, habana_frameworks.torch; assert torch.hpu.is_available()'

case "$TORCH_VERSION" in
  2.11.*) DEFAULT_VERSION=v0.24.0 ;;
  *)
    echo "Unsupported PyTorch: $TORCH_VERSION" >&2
    echo "Intel Gaudi Software 1.24.1 with PyTorch 2.11 is required." >&2
    exit 1
    ;;
esac
VERSION="${VERSION:-$DEFAULT_VERSION}"
echo "Using vLLM Gaudi $VERSION with PyTorch $TORCH_VERSION"

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
VLLM_COMMIT=$(tr -d '[:space:]' < vllm-gaudi/VLLM_STABLE_COMMIT)
test -n "$VLLM_COMMIT" || { echo "VLLM_STABLE_COMMIT is empty" >&2; exit 1; }

test -d vllm || git clone https://github.com/vllm-project/vllm
git -C vllm fetch origin "$VLLM_COMMIT"
git -C vllm checkout "$VLLM_COMMIT"

"$PYTHON" -m pip install -r <(sed '/^torch/d' vllm/requirements/build/cuda.txt)
"$PYTHON" -m pip install -r vllm/requirements/common.txt
"$PYTHON" -m pip install -r vllm-gaudi/requirements.txt
(cd vllm && VLLM_TARGET_DEVICE=empty "$PYTHON" -m pip install --no-build-isolation --no-deps -e .)
(cd vllm-gaudi && "$PYTHON" -m pip install --no-deps -e .)

"$PYTHON" -m pip install --no-deps "torchaudio==$TORCH_VERSION" --extra-index-url https://download.pytorch.org/whl/cpu
"$PYTHON" -c 'import torch, habana_frameworks.torch; assert torch.version.cuda is None and torch.hpu.is_available()'
