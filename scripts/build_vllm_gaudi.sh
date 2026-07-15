#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/test1/habanalabs-venv/bin/python}"
WORK_DIR="${VLLM_GAUDI_BUILD_DIR:-$PROJECT_ROOT/.deps}"
VLLM_GAUDI_REPO="${VLLM_GAUDI_REPO:-https://github.com/vllm-project/vllm-gaudi}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm}"
VLLM_GAUDI_REF="${VLLM_GAUDI_REF:-main}"
INSTALL_MODE="${INSTALL_MODE:-editable}"

PIP=("$PYTHON_BIN" -m pip)

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

clone_or_update() {
  local repo_url="$1"
  local dir_name="$2"
  local ref_name="$3"

  if [[ ! -d "$dir_name/.git" ]]; then
    git clone "$repo_url" "$dir_name"
  fi

  git -C "$dir_name" fetch --tags origin
  git -C "$dir_name" checkout "$ref_name"
  git -C "$dir_name" pull --ff-only origin "$ref_name" || true
}

clone_or_update "$VLLM_GAUDI_REPO" vllm-gaudi "$VLLM_GAUDI_REF"

VLLM_COMMIT_HASH="${VLLM_COMMIT_HASH:-}"
if [[ -z "$VLLM_COMMIT_HASH" ]]; then
  VLLM_COMMIT_HASH="$(git -C vllm-gaudi show "origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT" 2>/dev/null)"
fi
if [[ -z "$VLLM_COMMIT_HASH" ]]; then
  echo "Could not resolve VLLM_COMMIT_HASH from vllm-gaudi." >&2
  exit 1
fi

if [[ ! -d vllm/.git ]]; then
  git clone "$VLLM_REPO" vllm
fi
git -C vllm fetch --tags origin
git -C vllm checkout "$VLLM_COMMIT_HASH"

"${PIP[@]}" install --upgrade pip
"${PIP[@]}" install -r <(sed '/^torch/d' vllm/requirements/build/cuda.txt)

FILTERED_COMMON_REQUIREMENTS="$(mktemp)"
trap 'rm -f "$FILTERED_COMMON_REQUIREMENTS"' EXIT
sed \
  -e '/^torch[=<> ]/d' \
  -e '/^torchaudio[=<> ]/d' \
  -e '/^torchvision[=<> ]/d' \
  -e '/^compressed-tensors[=<> ]/d' \
  vllm/requirements/common.txt > "$FILTERED_COMMON_REQUIREMENTS"
"${PIP[@]}" install -r "$FILTERED_COMMON_REQUIREMENTS"
"${PIP[@]}" install --no-deps "compressed-tensors==0.17.0"

if [[ "$INSTALL_MODE" == "editable" ]]; then
  (cd vllm && VLLM_TARGET_DEVICE=empty "${PIP[@]}" install --no-build-isolation --no-deps -e .)
  (cd vllm-gaudi && "${PIP[@]}" install -r requirements.txt && "${PIP[@]}" install --no-deps -e .)
else
  (cd vllm && VLLM_TARGET_DEVICE=empty "${PIP[@]}" install --no-build-isolation --no-deps .)
  (cd vllm-gaudi && "${PIP[@]}" install -r requirements.txt && "${PIP[@]}" install --no-deps .)
fi

TORCH_VERSION="$("$PYTHON_BIN" - <<'PY'
import re
import torch
match = re.match(r"(\d+\.\d+\.\d+)", torch.__version__)
print(match.group(1) if match else torch.__version__.split("+", 1)[0])
PY
)"
"${PIP[@]}" install --no-deps "torchaudio==$TORCH_VERSION" --extra-index-url https://download.pytorch.org/whl/cpu

"$PYTHON_BIN" - <<'PY'
import importlib.metadata as metadata

for package in ("torch", "habana-torch-plugin", "vllm", "vllm-gaudi", "vllm_gaudi"):
    try:
        print(f"{package}: {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}: not installed")
PY

echo "vLLM Gaudi build/install complete."
