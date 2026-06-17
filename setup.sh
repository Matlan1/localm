#!/usr/bin/env bash
# =============================================================================
#  localm setup - Linux / macOS.  Run after cloning:  bash setup.sh
#
#  Self-contained: creates a private .venv in THIS folder and keeps data here
#  (./home) or in ~/.localm. Nothing is installed globally; PATH is not changed.
#  Pass --yes for a non-interactive install with sensible defaults (used by the
#  one-click install.sh).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

YES=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) YES=1 ;;
  esac
done

say() { printf '%s\n' "$*"; }
ask() {  # ask "prompt" "default"  ->  echoes the answer (the default in --yes mode)
  local prompt="$1" def="$2" ans
  if [ "$YES" = 1 ]; then echo "$def"; return; fi
  read -r -p "$prompt" ans || ans=""
  echo "${ans:-$def}"
}

say ""
say "  localm setup - self-contained install in: $(pwd)"
say ""

# ---- uv is required ---------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  say "  [!] uv is not installed. Install it first:"
  say "      curl -LsSf https://astral.sh/uv/install.sh | sh"
  say "      then re-open your shell and run setup.sh again."
  exit 1
fi

# ---- detect GPU acceleration ------------------------------------------------
detect_gpu() {
  if command -v rocminfo >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1 || [ -d /opt/rocm ]; then
    echo rocm
  elif command -v nvidia-smi >/dev/null 2>&1; then
    echo cuda
  else
    echo cpu
  fi
}
GPU="$(detect_gpu)"
say "  Detected acceleration: $GPU"
if [ "$YES" != 1 ] && [ "$GPU" != cpu ]; then
  pick="$(ask "  Use $GPU acceleration? [Y/n] (n = CPU only): " Y)"
  case "$pick" in [Nn]*) GPU=cpu ;; esac
fi

# ---- create the venv --------------------------------------------------------
# An existing .venv is reused unless the user chooses to replace it, so a
# re-run never aborts mid-setup. uv refuses to clobber an existing environment
# (exits non-zero, which set -e would treat as fatal), so we branch explicitly.
is_our_venv() {  # a venv we created carries the marker / the localm console script
  [ -f .venv/.localm-venv ] || [ -x .venv/bin/localm ]
}
create_venv() {
  say ""
  say "  Creating .venv (Python 3.12) ..."
  uv venv --python 3.12 --clear .venv
  : > .venv/.localm-venv   # marker: this venv was created by localm setup
}

if [ -d .venv ]; then
  if is_our_venv; then
    say ""
    say "  An existing localm .venv was found in this folder."
    rep="$(ask "  Replace it and reinstall from scratch? [y/N]: " N)"
  else
    say ""
    say "  [!] A .venv exists here but does not look like a localm environment."
    say "      Replacing it deletes its current contents."
    rep="$(ask "  Replace this foreign .venv? [y/N]: " N)"
  fi
  case "$rep" in
    [Yy]*) create_venv ;;
    *)     say "  Keeping the existing .venv and continuing setup." ;;
  esac
else
  create_venv
fi

# ---- install localm (editable) ----------------------------------------------
say "  Installing localm into .venv ..."
uv pip install -p .venv -e ".[coder,voice]"

# ---- GPU stack (PyTorch + transformers for the HF backend) ------------------
case "$GPU" in
  rocm)
    say "  Installing PyTorch (ROCm) + transformers ..."
    uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2 \
      || say "  [!] ROCm torch install failed - install a matching torch+rocm manually (see docs/linux-setup.md)."
    uv pip install -p .venv "transformers[kernels]~=5.10" "accelerate>=1.0" "pillow>=10.0" || true
    ;;
  cuda)
    say "  Installing PyTorch (CUDA) + transformers ..."
    uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
      || say "  [!] CUDA torch install failed - install a matching torch+cuda manually (see docs/linux-setup.md)."
    uv pip install -p .venv "transformers[kernels]~=5.10" "accelerate>=1.0" "pillow>=10.0" || true
    ;;
  cpu)
    say "  CPU mode - skipping the GPU/torch stack (GGUF inference needs no torch)."
    ;;
esac

# ---- native llama.cpp runtime wheel (loader imports it) ---------------------
uv pip install -p .venv -e ./runtime >/dev/null 2>&1 || true

# ---- provision the native library -------------------------------------------
say ""
say "  Native llama.cpp library (libllama.so): no prebuilt is hosted for Linux."
say "  Build llama.cpp for your GPU (see docs/linux-setup.md), then copy it in."
buildpath="$(ask "  Path to a llama.cpp build dir to copy now (blank = skip): " "")"
if [ -n "$buildpath" ]; then
  .venv/bin/localm setup-llama --from "$buildpath" \
    || say "  [!] setup-llama failed - run it later:  .venv/bin/localm setup-llama --from <dir>"
else
  say "  Skipped. Provision later:  .venv/bin/localm setup-llama --from <build dir>"
fi

# ---- data directory ---------------------------------------------------------
say ""
say "  Where should localm keep its data (models, config, logs, images)?"
say "    [1] Inside this folder (./home) - portable, isolated per clone"
say "    [2] Shared per-user (~/.localm) - clones share models and settings"
dpick="$(ask "  Pick 1 or 2 [2]: " 2)"
if [ "$dpick" = 1 ]; then
  mkdir -p home; rm -f localm-home.cfg
  say "  Data directory: $(pwd)/home"
else
  rm -f localm-home.cfg
  [ -d home ] && rmdir home 2>/dev/null || true
  say "  Data directory: $HOME/.localm (shared)"
fi

# ---- application menu entry --------------------------------------------------
mk="$(ask "  Create an application menu entry? [Y/n]: " Y)"
case "$mk" in
  [Nn]*) say "  No desktop entry created." ;;
  *)
    apps="$HOME/.local/share/applications"
    mkdir -p "$apps"
    icon="$(pwd)/assets/localm.png"; [ -f "$icon" ] || icon="$(pwd)/assets/localm.ico"
    cat > "$apps/localm.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=localm
Comment=Local LLM - chat, coder, models, images
Exec=$(pwd)/localm-launcher.sh
Icon=$icon
Terminal=false
Categories=Utility;Development;Science;
EOF
    say "  Created $apps/localm.desktop"
    ;;
esac

say ""
say "  Which optional features (plugins) do you want? chat is always on."
.venv/bin/localm plugin setup \
  || say "  [!] Skipped - choose later with:  .venv/bin/localm plugin setup"

say ""
say "  Done. This clone is self-contained:"
say "    ./localm-launcher.sh    graphical launcher (GUI / chat / server / coder)"
say "    ./localm.sh <args>      the localm CLI, e.g.:  ./localm.sh gui"
say "    .venv/bin/localm ...    CLI directly"
say ""
say "  Note: the GUI launcher needs Tk (sudo apt install python3-tk, or your"
say "  distro's equivalent). The web GUI itself needs only a browser."
say ""
