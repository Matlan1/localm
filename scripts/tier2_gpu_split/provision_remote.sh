#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Provisions a freshly rented Linux GPU box for the Tier 2 GPU-split gate
# (issues/issues.txt GPU-SPLIT-TESTING). Runs ON the rented box - uploaded and
# invoked by scripts/tier2_gpu_split/run_gate.py over ssh, or by hand for a
# manual dry run against an already-running instance.
#
# What this does NOT do: pick or download the "too big for one device, fits
# combined" model. That choice depends on the box's REAL per-device free VRAM,
# which is only known at test time - tests/test_gpu_split_real_hardware.py
# resolves and downloads it lazily via scripts/tier2_gpu_split/model_selection.py,
# exactly like tests/test_gpu_split_native_vulkan.py's tiny-model fixture.
#
# Usage: provision_remote.sh <vulkan|amd-rocm> <git-ref> [repo-url]
set -euo pipefail

BACKEND="${1:?usage: provision_remote.sh <vulkan|amd-rocm> <git-ref> [repo-url]}"
REF="${2:?usage: provision_remote.sh <vulkan|amd-rocm> <git-ref> [repo-url]}"
REPO_URL="${3:-https://github.com/Matlan1/localm.git}"
WORKDIR="${TIER2_WORKDIR:-$HOME/localm-tier2}"

log() { printf '[provision] %s\n' "$*"; }
die() { printf '[provision] FATAL: %s\n' "$*" >&2; exit 2; }

case "$BACKEND" in
  vulkan|amd-rocm) ;;
  *) die "unsupported backend '$BACKEND' (expected vulkan or amd-rocm)" ;;
esac

log "host: $(uname -a)"
log "cpu: $(nproc) cores, mem: $(free -h | awk '/Mem:/ {print $2}')"

log "cloning $REPO_URL @ $REF into $WORKDIR"
rm -rf "$WORKDIR"
if ! git clone --depth 1 --branch "$REF" "$REPO_URL" "$WORKDIR" 2>/tmp/tier2-clone.err; then
  log "branch/tag clone failed (ref is likely a raw commit SHA) - falling back to full fetch + checkout"
  cat /tmp/tier2-clone.err >&2 || true
  git clone "$REPO_URL" "$WORKDIR"
  (cd "$WORKDIR" && git fetch --depth 1 origin "$REF" && git checkout FETCH_HEAD)
fi
cd "$WORKDIR"
log "checked out $(git rev-parse HEAD): $(git log -1 --format=%s)"

log "installing uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

log "creating the project venv and installing dev+rag deps (matches CI's Install step)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,rag]"

# Test-harness-only dependency: the native GGUF/llama.cpp backend under test
# needs no torch at all. This is purely so list_gpus()/vram_capacity() (which
# this harness calls directly - see the plan's Context) can enumerate real
# CUDA/ROCm devices; discover.py has an nvidia-smi subprocess fallback but NO
# rocm-smi equivalent, so the amd-rocm run needs a real torch-ROCm build.
log "installing torch for GPU enumeration (test-harness dependency only)"
case "$BACKEND" in
  vulkan)
    # Standard PyPI Linux wheels bundle CUDA support; no special index needed
    # for on-demand NVIDIA instances (Lambda's Ubuntu image ships a matching
    # driver). Verified below, not assumed - a wrong guess here fails loudly.
    pip install torch
    ;;
  amd-rocm)
    ROCM_VER="$(rocm-smi --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)"
    if [ -z "$ROCM_VER" ] && [ -f /opt/rocm/.info/version ]; then
      ROCM_VER="$(cut -d. -f1,2 /opt/rocm/.info/version 2>/dev/null || true)"
    fi
    case "$ROCM_VER" in
      6.4*) IDX="rocm6.4" ;;
      6.3*) IDX="rocm6.3" ;;
      6.2*) IDX="rocm6.2" ;;
      6.1*) IDX="rocm6.1" ;;
      *)
        IDX="rocm6.2"
        log "WARNING: could not detect the box's ROCm version (rocm-smi and /opt/rocm/.info/version both came up empty) - GUESSING pytorch index '$IDX'. If the device-visibility check below shows 0 devices, this guess was wrong: run 'rocm-smi --version' by hand and re-install torch from the matching https://download.pytorch.org/whl/rocmX.Y index."
        ;;
    esac
    log "using pytorch ROCm index $IDX (detected ROCm version: ${ROCM_VER:-unknown, guessed})"
    pip install torch --index-url "https://download.pytorch.org/whl/$IDX"
    ;;
esac

log "verifying torch sees the real GPU(s)"
python3 -c "
import torch
print('torch version:', torch.__version__)
print('torch.cuda.is_available():', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  device {i}:', torch.cuda.get_device_name(i))
"

log "provisioning the native llama.cpp runtime (backend=$BACKEND)"
python3 -m localm setup-llama --backend "$BACKEND" --yes

log "pulling the pinned tiny smoke-test GGUF (same model tests/test_gpu_split_native_vulkan.py uses)"
python3 -m localm pull "bartowski/SmolLM2-135M-Instruct-GGUF:SmolLM2-135M-Instruct-Q4_K_M.gguf"

log "sanity-checking localm's own GPU enumeration (list_gpus() - the function the gate tests exercise directly)"
python3 -c "
from localm.discover import list_gpus
gpus = list_gpus()
print('list_gpus() ->', gpus)
if len(gpus) < 2:
    print('WARNING: fewer than 2 GPUs visible to list_gpus() - the split tests '
          'will report this and skip/fail informatively; this is not a passing '
          'setup for the gate.')
"

log "provisioning complete"
