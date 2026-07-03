#!/usr/bin/env bash
# =============================================================================
#  localm setup - Linux / macOS.  Run after cloning:  bash setup.sh
#
#  Self-contained: creates a private .venv (and optionally its Python runtime)
#  in THIS folder and keeps data here (./home, or a custom path you pick). Nothing
#  installed globally; your PATH is left unchanged unless you opt into the global
#  `localm` command.
#  Pass --yes for a non-interactive install with sensible defaults (used by the
#  one-click install.sh).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

YES=0; UNINSTALL=0; PURGE=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) YES=1 ;;
    --uninstall|--rollback) UNINSTALL=1 ;;
    --purge-data) PURGE=1 ;;
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
say "  LocaLM setup - self-contained install in: $(pwd)"
say ""

# ---- uninstall / rollback (report first, then remove) -----------------------
# Removes ONLY what this installer created: our .venv (marker-checked), the
# provisioned native binaries, the home.cfg, and the menu entry. Your data is
# kept unless --purge-data, and even then unsafe target paths are refused.
do_uninstall() {
  say "  Uninstall / rollback for this clone:"
  say "    $(pwd)"
  say ""
  local py=".venv/bin/python"
  [ -x "$py" ] || py="$(command -v python3 || command -v python || echo "")"
  local pflag=""
  if [ "$PURGE" = 1 ]; then pflag="--purge-data"; fi

  # The manifest module removes ONLY what install recorded (binaries, home.cfg,
  # shortcut, and the data dir only when WE created it and --purge-data is set,
  # with hard guards on the single rm -rf). It never derives or globs a path -
  # so an empty/relative/symlinked value cannot point a delete somewhere else.
  if [ -n "$py" ]; then
    say "  Planned removals (from the install manifest .localm-install.json):"
    "$py" -m localm.install_manifest uninstall --root . $pflag --dry-run \
      || say "  [!] No usable install manifest - only the marked ./.venv will be removed."
  else
    say "  [!] No Python found - only the marked ./.venv will be removed."
  fi
  say ""
  local ok; ok="$(ask "  Proceed? [y/N]: " N)"
  case "$ok" in [Yy]*) ;; *) say "  Aborted - nothing changed."; return 0 ;; esac

  if [ -n "$py" ]; then
    # --force: the dry-run above already showed any unrecorded items and the
    # at-your-own-risk warning; the user chose to continue. The catastrophic-path
    # guard inside the module still refuses root/$HOME/repo regardless.
    "$py" -m localm.install_manifest uninstall --root . $pflag --force || true
  fi
  # The manifest never deletes the running venv; remove it here, marker-checked.
  if [ -f .venv/.localm-venv ]; then
    if rm -rf .venv 2>/dev/null; then say "  Removed ./.venv"
    else say "  [!] Could not remove ./.venv (check permissions)."; fi
  fi
  say ""
  say "  Done. To reinstall:  bash setup.sh"
}

if [ "$UNINSTALL" = 1 ]; then
  do_uninstall
  exit 0
fi

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
  elif command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -Eiq 'intel.*(arc|dg2|xe)'; then
    echo intel
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

# ---- portable vs shared: where the Python runtime + downloads live ----------
# Portable pulls uv's managed Python AND its wheel cache INTO this folder, so the
# clone is truly self-contained (delete it and nothing is left behind) at the cost
# of a per-clone re-download. Shared reuses uv's per-user Python + cache. The UV_*
# vars are exported for THIS setup process only (not persisted / not global), so
# they never touch any other uv project. --python-preference only-managed forces
# the contained download instead of reusing a system Python.
say ""
say "  Keep localm's Python runtime + downloads inside this folder?"
say "    [1] Portable - everything in this folder (self-contained; re-downloads per clone)"
say "    [2] Shared   - reuse uv's per-user Python + cache (faster; lives in ~/.local)"
spick="$(ask "  Pick 1 or 2 [1]: " 1)"
CONTAINED=0; PYPREF=""
if [ "$spick" = 1 ]; then
  CONTAINED=1
  export UV_PYTHON_INSTALL_DIR="$(pwd)/.python"
  export UV_CACHE_DIR="$(pwd)/.cache"
  PYPREF="--python-preference only-managed"
  say "  Portable: Python under ./.python and downloads under ./.cache"
else
  say "  Shared: reusing uv's per-user Python + cache (outside this folder)."
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
  # PYPREF is empty (shared) or "--python-preference only-managed" (portable);
  # left unquoted on purpose so an empty value expands to no argument.
  # shellcheck disable=SC2086
  uv venv --python 3.12 $PYPREF --clear .venv
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
uv pip install -p .venv -e ".[coder,voice,monitor]"

# ---- native llama.cpp runtime wheel (loader imports it) ---------------------
# (The PyTorch/transformers stack is installed further down, AFTER the backend
# pick, so the HF torch variant can FOLLOW the chosen runtime - see SETUP-1.)
uv pip install -p .venv -e ./runtime >/dev/null 2>&1 || true

# ---- provision the native library (official llama.cpp prebuilt) -------------
# The RECOMMENDED backend comes from the SAME tested policy the Windows installer
# uses (`python -m localm.hwdetect` -> "<vendor> <backend>"), so the two installers
# can never drift: any GPU -> vulkan (runs on AMD/NVIDIA/Intel via the display
# driver, no vendor toolkit), Apple Silicon -> metal, no GPU -> cpu. CUDA/ROCm are
# offered below for peak vendor performance (they need that vendor's runtime).
# setup-llama fetches the matching upstream build, so a tester never compiles by hand.
REC="$(.venv/bin/python -m localm.hwdetect 2>/dev/null | awk '{print $2}')"
case "$REC" in
  vulkan|cuda|hip|cpu|metal|amd-rocm) ;;                      # a known backend from the policy
  *) case "$GPU" in cpu) REC=cpu ;; *) REC=vulkan ;; esac ;;  # fallback if the probe failed
esac
say ""
say "  Native inference runtime (llama.cpp). Recommended for your hardware: $REC"
say "    [1] $REC  (recommended)"
say "    [2] vulkan   - any GPU, no vendor toolkit"
say "    [3] cuda     - NVIDIA, peak performance (needs the CUDA runtime)"
say "    [4] hip      - AMD ROCm, peak performance (needs the ROCm runtime)"
say "    [5] cpu      - no GPU"
say "    [6] I will build / provide my own (skip the download)"
say "    (your pick is load-tested; a failure offers Vulkan, never a silent swap)"
bpick="$(ask "  Pick 1-6 [1]: " 1)"
case "$bpick" in
  2) BACKEND=vulkan ;; 3) BACKEND=cuda ;; 4) BACKEND=hip ;; 5) BACKEND=cpu ;;
  6) BACKEND=own ;;   *) BACKEND="$REC" ;;
esac
if [ "$BACKEND" = own ]; then
  buildpath="$(ask "  Path to a llama.cpp build dir to copy now (blank = skip): " "")"
  if [ -n "$buildpath" ]; then
    .venv/bin/localm setup-llama --from "$buildpath" || {
      # Mirror setup.bat: a declined/failed provision stops setup with a non-zero
      # exit instead of silently continuing into the torch/data-dir steps as if
      # the runtime were installed (NEW-CUDADECLINE).
      say "  [!] setup-llama failed - run later:  .venv/bin/localm setup-llama --from <dir>"
      exit 1
    }
  else
    say "  Skipped. Provision later:  .venv/bin/localm setup-llama --backend <vulkan|cuda|hip|cpu>"
  fi
else
  .venv/bin/localm setup-llama --backend "$BACKEND" || {
    # Mirror setup.bat's `if errorlevel 1 (... exit /b 1)`: stop on a declined or
    # failed runtime provision rather than falling through silently (NEW-CUDADECLINE).
    say "  [!] setup-llama failed - run later:  .venv/bin/localm setup-llama --backend $BACKEND"
    exit 1
  }
fi

# ---- PyTorch + transformers for the HuggingFace backend (FOLLOWS the backend) -
# PyTorch powers the HuggingFace/transformers backend; GGUF chat needs none of it.
# The variant FOLLOWS the llama.cpp BACKEND picked above (not just the detected
# GPU), so choosing the vendor-neutral 'vulkan' runtime does not drag in the ROCm
# stack (the SETUP-1 surprise). `hwdetect torch-args <backend>` resolves the exact
# wheel SOURCE for this hardware+OS (cuda/rocm/xpu/none), so setup.sh and setup.bat
# never drift and every card gets the correct packages.
TORCHSPEC="$(.venv/bin/python -m localm.hwdetect torch-args "$BACKEND" 2>/dev/null)"
if [ -n "$TORCHSPEC" ]; then
  say ""
  say "  Installing PyTorch + transformers for HuggingFace models ..."
  # TORCHSPEC is a multi-token pip arg list (e.g. "torch torchvision --index-url ...");
  # it is intentionally left unquoted so the words split into separate arguments.
  # shellcheck disable=SC2086
  uv pip install -p .venv $TORCHSPEC \
    || say "  [!] torch install failed - install a matching torch manually (see docs/gpu-setup.md)."
  uv pip install -p .venv "transformers[kernels]~=5.12" "tokenizers==0.22.2" "accelerate>=1.0" "pillow>=10.0" "soundfile>=0.12" || true
else
  say ""
  say "  Skipping the PyTorch/transformers stack (not needed for GGUF chat)."
  say "  You picked the '$BACKEND' runtime, so no vendor GPU torch was auto-installed."
  say "  For HuggingFace transformers models, add PyTorch later (see docs/gpu-setup.md):"
  say "    CPU (any machine): uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cpu"
  say "    NVIDIA CUDA:       uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cu126"
  say "    AMD ROCm (Linux):  uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2"
  say "    Intel Arc / XPU:   uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/xpu"
fi

# ---- data directory ---------------------------------------------------------
# Default is CONTAINED (./home): NO silent ~/.localm fallback. A shared / other
# location is an explicit Custom choice, recorded in localm-home.cfg.
say ""
say "  Where should localm keep its data (models, config, logs, images)?"
say "    [1] Portable (./home) - self-contained; delete this folder and it is all gone"
say "    [2] Custom path       - a folder you choose (e.g. a shared models drive)"
dpick="$(ask "  Pick 1 or 2 [1]: " 1)"
if [ "$dpick" = 2 ]; then
  # Custom path: ask, then confirm (re-ask until confirmed). In --yes mode there is
  # no prompt, so an unconfirmed path is never recorded - fall back to portable.
  CUSTOMHOME=""
  if [ "$YES" != 1 ]; then
    while : ; do
      CUSTOMHOME="$(ask "  Enter the data directory path (blank = portable ./home): " "")"
      if [ -z "$CUSTOMHOME" ]; then break; fi
      ok="$(ask "  Use '$CUSTOMHOME'? [Y/n]: " Y)"
      case "$ok" in [Nn]*) continue ;; *) break ;; esac
    done
  fi
  if [ -z "$CUSTOMHOME" ]; then
    say "  No path given - using the portable ./home."
    mkdir -p home; rm -f localm-home.cfg
    DATA_DIR="$(pwd)/home"; DATA_CREATED=1
  else
    printf '%s\n' "$CUSTOMHOME" > localm-home.cfg
    mkdir -p "$CUSTOMHOME"
    DATA_DIR="$CUSTOMHOME"; DATA_CREATED=1
    say "  Data directory: $CUSTOMHOME (recorded in localm-home.cfg)"
  fi
else
  mkdir -p home; rm -f localm-home.cfg
  DATA_DIR="$(pwd)/home"; DATA_CREATED=1        # we created ./home
  say "  Data directory: $DATA_DIR"
fi

# ---- build the native LocaLM launcher ---------------------------------------
# So a process monitor shows LocaLM, not python. It is a copy of the venv
# interpreter in .venv/bin/LocaLM, self-contained in this clone; if the copy
# cannot run standalone (non-relocatable interpreter) the menu entry below falls
# back to the venv python. `localm gui` always works; this never blocks install.
say ""
say "  Building the LocaLM app launcher ..."
.venv/bin/localm make-launcher --force \
  || say "  [!] Could not build the LocaLM launcher - 'localm gui' still works."

# ---- application menu entry --------------------------------------------------
SHORTCUT=""
mk="$(ask "  Create an application menu entry? [Y/n]: " Y)"
case "$mk" in
  [Nn]*) say "  No desktop entry created." ;;
  *)
    apps="$HOME/.local/share/applications"
    mkdir -p "$apps"
    SHORTCUT="$apps/localm.desktop"
    # Prefer the scalable SVG (the freedesktop-friendly format); fall back to
    # the .ico that ships for the Windows shortcut if it is ever missing.
    icon="$(pwd)/assets/localm.svg"; [ -f "$icon" ] || icon="$(pwd)/assets/localm.ico"
    # Launch the GUI directly as the branded LocaLM binary when it built and runs;
    # otherwise open the graphical launcher (the venv python).
    if [ -x "$(pwd)/.venv/bin/LocaLM" ]; then
      launch_exec="$(pwd)/.venv/bin/LocaLM -m localm gui"
    else
      launch_exec="$(pwd)/localm-launcher.sh"
    fi
    cat > "$apps/localm.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=LocaLM
Comment=Local LLM - chat, coder, models, images
Exec=$launch_exec
Icon=$icon
Terminal=false
Categories=Utility;Development;Science;
EOF
    say "  Created $apps/localm.desktop"
    ;;
esac

# ---- optional: make `localm` runnable from any terminal --------------------
# Symlinks `localm` into ~/.local/bin (already on PATH by convention; pip/pipx/uv
# use it) - never the venv bin dir, which would shadow your python/pip. Reversible
# by the uninstaller. Default No: the CLI already works via ./localm.sh or
# .venv/bin/localm.
PATH_DIR=""; CMD_SHIM=""; PATH_MOD=""
gmk="$(ask "  Make 'localm' runnable from any terminal? (symlink into ~/.local/bin) [y/N]: " N)"
case "$gmk" in
  [Yy]*)
    # globalcmd exit code: 0 = installed + PATH modified; 20 = installed but PATH
    # was already set (record the command but NOT --path-modified); other = failed
    # (record nothing). || gcrc=$? keeps set -e from aborting on the 20/failure code.
    gcrc=0
    .venv/bin/python -m localm.globalcmd install --root . || gcrc=$?
    if [ "$gcrc" = 0 ] || [ "$gcrc" = 20 ]; then
      PATH_DIR="$HOME/.local/bin"; CMD_SHIM="$HOME/.local/bin/localm"
      if [ "$gcrc" = 0 ]; then PATH_MOD="--path-modified"; fi
    fi
    ;;
esac

say ""
# `localm plugin setup` prints its own header (it states chat is always on), so
# this is just a section divider - do not repeat that line here.
say "  Optional features (plugins):"
.venv/bin/localm plugin setup \
  || say "  [!] Skipped - choose later with:  .venv/bin/localm plugin setup"

# ---- record what we installed (so uninstall removes ONLY what we created) ----
crd=""; if [ "${DATA_CREATED:-0}" = 1 ]; then crd="--data-created"; fi
rcflag=""; pydir=""; cachedir=""
if [ "${CONTAINED:-0}" = 1 ]; then
  rcflag="--runtime-contained"; pydir="$(pwd)/.python"; cachedir="$(pwd)/.cache"
fi
# $crd / $rcflag are flags or empty; left unquoted on purpose (empty -> no arg).
# shellcheck disable=SC2086
.venv/bin/python -m localm.install_manifest record --root . \
  --venv "$(pwd)/.venv" \
  --lib-dir "$(pwd)/runtime/localm_llama_runtime/lib" \
  --data-dir "${DATA_DIR:-}" $crd \
  --shortcut "${SHORTCUT:-}" \
  $rcflag --python-dir "$pydir" --cache-dir "$cachedir" \
  --path-dir "${PATH_DIR:-}" --command-shim "${CMD_SHIM:-}" ${PATH_MOD:-} \
  --stamp "$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "")" \
  >/dev/null 2>&1 || say "  [!] Could not record the install manifest (uninstall will be conservative)."

say ""
say "  Done. This clone is self-contained:"
say "    ./localm-launcher.sh    graphical launcher (GUI / chat / server / coder)"
say "    ./localm.sh <args>      the localm CLI, e.g.:  ./localm.sh gui"
say "    .venv/bin/localm ...    CLI directly"
say ""
say "  The GUI launcher uses Tk; localm's bundled Python includes it, so it"
say "  normally works out of the box. If the launcher ever reports Tk missing"
say "  (e.g. a system Python without it), install your distro's python3-tk."
say "  The web GUI itself needs only a browser."
say ""
