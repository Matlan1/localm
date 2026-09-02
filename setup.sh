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
export LOCALM_SETUP=1

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
  # Drop any TYPE-AHEAD before showing this question. Setup runs long steps
  # between questions (a uv download, a Python download, a venv build, a model
  # pull); anything typed while one of those runs sits in the terminal's input
  # queue and is delivered to the NEXT read - answering a question the user never
  # saw. One stray Enter silently accepts a default, and a double Enter accepts
  # two questions in a row. Every answer must be one the user actually gave to the
  # question in front of them.
  #
  # Only when stdin is a TERMINAL: piped input is deliberate (a script feeding
  # answers), and draining it would eat the caller's real answers. --yes returns
  # above and never reaches here.
  if [ -t 0 ]; then
    while read -r -t 0.05 -n 4096 _typeahead 2>/dev/null; do :; done
  fi
  read -r -p "$prompt" ans || ans=""
  echo "${ans:-$def}"
}
uv_manual_hint() {  # the manual uv-install command; used from two call sites below
  say "      curl -LsSf https://astral.sh/uv/install.sh | sh"
}
# heartbeat_start SECS "MESSAGE" / heartbeat_stop - print MESSAGE every SECS
# seconds while a following long, quiet command is still running (a uv
# download, a venv build, a torch install), so it never looks identical to a
# hung terminal. The command itself runs completely unchanged - same
# foreground process, same exit status, same output capture, same live
# progress if it has any; only the heartbeat runs in the background.
#
# It watches a flag file rather than killing a PID (killing a backgrounded
# subshell is not reliably instant on every shell, and even where `kill`
# succeeds it only kills the subshell itself - its `sleep` child can survive
# as an orphan holding the same stdout/stderr open). The loop polls the flag
# every 1s (never sleeping for the full SECS at a stretch) and only PRINTS
# once SECS worth of polls have passed - so the print cadence is unchanged,
# but heartbeat_stop is noticed within about a second, not up to SECS later.
# That matters beyond tidiness: an orphaned background writer keeps a PIPE's
# write end open, so `setup.sh | tee log.txt` would otherwise hang for the
# full SECS after setup itself finished, waiting on a heartbeat nobody reads
# from - measured live at the original SECS-granularity poll before this.
#
# Each call gets its OWN flag file (a counter suffix), not a shared/reused
# path: create_venv can call heartbeat_start more than once per run (retry
# after a certificate fallback), and a shared path lets a still-sleeping OLD
# loop see a just-recreated file disappear again and keep printing after the
# NEW step already finished. The EXIT trap is a safety net for any exit path
# that skips an explicit heartbeat_stop call, so a heartbeat never keeps
# printing after setup itself has ended.
HB_SEQ=0
HB_FLAG=""
trap '[ -n "$HB_FLAG" ] && { : > "$HB_FLAG" 2>/dev/null || true; }' EXIT
heartbeat_start() {
  local secs="$1" msg="$2"
  HB_SEQ=$((HB_SEQ + 1))
  HB_FLAG="${TMPDIR:-/tmp}/localm-setup-hb.$$.$HB_SEQ"
  rm -f "$HB_FLAG"
  (
    n=0
    while [ ! -e "$HB_FLAG" ]; do
      sleep 1
      n=$((n + 1))
      if [ "$n" -ge "$secs" ]; then
        n=0
        [ -e "$HB_FLAG" ] || say "$msg"
      fi
    done
  ) &
}
heartbeat_stop() {
  [ -n "$HB_FLAG" ] && : > "$HB_FLAG"
  return 0
}
offer_report() {  # offer_report "summary" "detail"
  # Offer to file a bug report for a setup failure via the standalone reporter
  # (report-issue.sh), which works even though setup did not finish (it needs no
  # working install). The caller still exits non-zero afterwards - reporting never
  # masks the failure ("we do not hide problems"). Skipped in --yes mode (no prompt).
  local here ans
  here="$(cd "$(dirname "$0")" && pwd)"
  [ -f "$here/report-issue.sh" ] || return 0
  [ "$YES" = 1 ] && return 0
  ans="$(ask "  Report this problem to the maintainer (no GitHub account needed)? [Y/n]: " Y)"
  case "$ans" in [Nn]*) return 0 ;; esac
  bash "$here/report-issue.sh" --summary "$1" --detail "$2" || true
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

# ---- point at the graphical installer ---------------------------------------
# Same install, same questions, in a window. Mentioned here rather than only in
# the README because the person who would rather not answer questions in a
# console is, by definition, already looking at one. Only offered when there is
# a display to open it on.
if [ -x "./setup-gui.sh" ] && { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ] || [ "$(uname -s)" = "Darwin" ]; }; then
  say ""
  say "  Prefer a window? Ctrl+C and run ./setup-gui.sh instead - it performs"
  say "  this same setup graphically. Otherwise, carry on here."
fi

# ---- portable vs shared: where localm's Python tooling lives ----------------
# Portable pulls uv ITSELF (when we have to install it below), its managed Python,
# and its wheel cache INTO this folder, so the clone is truly self-contained
# (delete it and nothing is left behind) at the cost of a per-clone re-download.
# Shared reuses (or installs) uv at its normal per-user location and reuses its
# per-user Python + cache. Asked BEFORE the uv bootstrap below so a Portable pick
# also confines uv's own binary to this folder, not just the runtime it manages -
# silently installing a tool into the user's home directory without ever asking
# where is exactly the kind of outside-the-root write AGENTS.md rule 4 forbids.
# The UV_* vars are exported for THIS setup process only (not persisted / not
# global), so they never touch any other uv project. --python-preference
# only-managed forces the contained download instead of reusing a system Python.
say ""
say "  Keep localm's Python tooling (uv itself, its runtime, and downloads) inside this folder?"
say "    [1] Portable - everything in this folder (self-contained; re-downloads per clone)"
say "    [2] Shared   - reuse/install uv at its normal per-user location (faster; lives in ~/.local)"
spick="$(ask "  Pick 1 or 2 [1]: " 1)"
CONTAINED=0; PYPREF=""; UVDIR=""
if [ "$spick" = 1 ]; then
  CONTAINED=1
  export UV_PYTHON_INSTALL_DIR="$(pwd)/.python"
  export UV_CACHE_DIR="$(pwd)/.cache"
  PYPREF="--python-preference only-managed"
  say "  Portable: uv, Python, and downloads all under this folder"
else
  say "  Shared: reusing/installing uv and its Python + cache (outside this folder)."
fi

# 1 = verify against the platform's NATIVE certificate store - the same trust a
# browser, or an IT-provisioned corporate/security-product proxy's injected
# root, already has, so a managed machine behind one verifies on the very first
# attempt with nothing ever shown (verified live: uv's own bundled-only default
# fails such a network with "invalid peer certificate: UnknownIssuer";
# --system-certs does not). Falls back to unset (uv's own bundled Mozilla root
# list) below ONLY on a certificate error - the one case native-store-first
# misses: a freshly-imaged system whose store has not yet cached a legitimate CA
# chain. Real env var (UV_SYSTEM_CERTS is uv's own documented override), so
# every uv call for the REST of this run - venv creation, localm, the runtime
# wheel, torch/transformers - inherits whichever choice wins (see create_venv
# below for the actual fallback).
export UV_SYSTEM_CERTS=1

# ---- uv is required; bootstrap it ourselves if it is missing ----------------
# uv (Astral's fast Python package manager) builds the venv and resolves the GPU
# wheels. Instead of dead-ending with "install it yourself", fetch it via Astral's
# official installer and make it callable in THIS process (the installer updates a
# shell profile, not the PATH of an already-running shell). In --yes mode this is
# automatic (like the one-click install.sh); interactively it asks first (default
# Yes) since it runs a script fetched from the network. A bootstrap failure is
# surfaced, not hidden (rule 5): we re-check that uv is callable and exit if not.
#
# Portable (CONTAINED=1) must not settle for whatever uv happens to already be on
# PATH - that could be a Shared install, a package manager, or a different clone
# entirely, and reusing it silently would break the "uv itself ... inside this
# folder" promise the user just picked two prompts ago. So Portable checks ONLY
# for its own confined copy at ./.uv; anything else on PATH is irrelevant to it and
# falls through to the same bootstrap a genuinely-missing uv would trigger. Shared
# keeps the original behaviour: any uv already on PATH is fine to reuse.
uv_present=0
if [ "$CONTAINED" = 1 ]; then
  if [ -x "./.uv/uv" ]; then
    export PATH="$(pwd)/.uv:$PATH"
    UVDIR="$(pwd)/.uv"
    uv_present=1
  fi
elif command -v uv >/dev/null 2>&1; then
  uv_present=1
fi
if [ "$uv_present" != 1 ]; then
  say "  [!] uv (the Python package manager localm builds on) is not installed."
  getuv="$(ask "  Install it now with Astral's official installer? [Y/n]: " Y)"
  case "$getuv" in
    [Nn]*)
      say "  Setup needs uv. Install it, then re-run setup.sh:"
      uv_manual_hint
      exit 1
      ;;
  esac
  say "  Installing uv ..."
  if [ "$CONTAINED" = 1 ]; then
    # Portable was picked: confine uv's OWN binary to this folder too, not just the
    # Python runtime it manages - UV_INSTALL_DIR is Astral's own documented
    # override for the installer's target dir.
    export UV_INSTALL_DIR="$(pwd)/.uv"
    UVDIR="$(pwd)/.uv"
    say "  Portable: installing uv itself under ./.uv"
  fi
  # || true so a curl/install failure does not trip set -e before our own check;
  # the re-check below decides honestly whether the bootstrap actually worked.
  curl -LsSf https://astral.sh/uv/install.sh | sh || true
  # uv lands in ~/.local/bin (older builds used ~/.cargo/bin) unless UV_INSTALL_DIR
  # was set above; the installer edits a shell profile, not this running shell, so
  # add it (when set) plus both defaults to PATH for the rest of setup. Guarded so
  # an unset UV_INSTALL_DIR never leaves an empty leading PATH entry (POSIX shells
  # treat that as the current directory).
  if [ -n "${UV_INSTALL_DIR:-}" ]; then
    export PATH="$UV_INSTALL_DIR:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  else
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  fi
  if ! command -v uv >/dev/null 2>&1; then
    say ""
    say "  [!] uv still is not callable after the install attempt."
    say "      Open a new shell (so the updated PATH applies) and run setup.sh again,"
    say "      or install uv manually first:"
    uv_manual_hint
    offer_report "localm setup could not install uv" "setup.sh tried Astral's installer but uv was still not callable afterwards."
    exit 1
  fi
fi

# ---- detect GPU acceleration ------------------------------------------------
# Checked in the SAME vendor priority as hwdetect.py's VENDORS ("nvidia", "amd",
# "intel"): nvidia-smi first, so leftover ROCm tooling on an NVIDIA box (a shared
# ML rig, a base image bundling both vendor stacks) never shadows the real GPU.
detect_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo cuda
  elif command -v rocminfo >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1 || [ -d /opt/rocm ]; then
    echo rocm
  elif command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -Eiq 'intel.*(arc|dg2|xe)'; then
    echo intel
  elif [ "$(uname -s 2>/dev/null)" = "Darwin" ] && [ "$(uname -m 2>/dev/null)" = "arm64" ]; then
    echo metal
  else
    echo cpu
  fi
}
GPU="$(detect_gpu)"
# Independent of $GPU, which the Y/n prompt below can downgrade to "cpu". See
# test_menu_shows_metal_even_when_gpu_was_downgraded_to_cpu.
IS_APPLE_SILICON=0
if [ "$(uname -s 2>/dev/null)" = "Darwin" ] && [ "$(uname -m 2>/dev/null)" = "arm64" ]; then
  IS_APPLE_SILICON=1
fi
# NOT presented as "Detected acceleration: $GPU" here on purpose: this crude
# pre-venv guess and the authoritative "Recommended for your hardware: $REC"
# line further down (sourced from `python -m localm.hwdetect`, once the venv
# exists) can legitimately disagree - leftover vendor tooling, a bare /opt/rocm
# on a non-AMD box, or a vendor+backend vocabulary mismatch (this prints
# "rocm", hwdetect may say "hip" or "vulkan" for the same box). Two lines
# claiming to answer the same question, able to contradict each other, is
# worse than one; $GPU's only remaining job is gating this Y/n prompt (and the
# fallback below if the hwdetect probe itself fails).
if [ "$YES" != 1 ] && [ "$GPU" != cpu ]; then
  pick="$(ask "  GPU acceleration looks available ($GPU). Use it? [Y/n] (n = CPU only): " Y)"
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
  # Captured, never shown live: the DEFAULT (native store, exported above)
  # already gets the common case right on the first try - a plain network and a
  # network behind an IT-provisioned proxy both verify immediately, so this
  # never shows the user anything beyond the line above. Capturing costs
  # nothing observable even when the rare fallback below IS needed: a rejected
  # TLS handshake fails in well under a second, and a real download still
  # completes normally, just without a live byte-progress readout. A periodic
  # heartbeat line covers the case that genuinely does take a while - a fresh
  # box downloading uv's managed Python - so that never looks like a hang.
  while : ; do
    # PYPREF is empty (shared) or "--python-preference only-managed" (portable);
    # left unquoted on purpose so an empty value expands to no argument. The
    # `if var=$(cmd); then` form (not a bare assignment) is required under
    # `set -e` - it is one of the contexts POSIX exempts from aborting on a
    # non-zero exit, so a failure here can be examined instead of killing setup.
    heartbeat_start 15 "  ... still creating the environment (this can take a few minutes on a slow connection)"
    # shellcheck disable=SC2086
    if errtext="$(uv venv --python 3.12 $PYPREF --clear .venv 2>&1)"; then
      heartbeat_stop
      : > .venv/.localm-venv   # marker: this venv was created by localm setup
      break
    fi
    heartbeat_stop
    # Failed silently so far. Only ever falls back once: if UV_SYSTEM_CERTS is
    # already unset, the fallback was already tried - show it for real below
    # instead of guessing again. Applies in --yes mode too (fully automatic, no
    # prompt needed) - only the GENERIC failure path below still aborts
    # immediately under --yes.
    if [ -n "${UV_SYSTEM_CERTS:-}" ] && printf '%s' "$errtext" | grep -qi certificate; then
      say "  [i] Your system's certificate store did not verify a required download"
      say "      (possibly a freshly-installed system that has not cached the real"
      say "      certificate yet). Falling back to uv's own verified certificate"
      say "      bundle ..."
      unset UV_SYSTEM_CERTS
      continue
    fi
    say ""
    say "$errtext"
    say ""
    say "  [!] Could not create the environment."
    say "      If a localm process is still running, it may be locking the directory."
    say "      Please close any open LocaLM launchers or servers."
    say ""
    if [ "$YES" = 1 ]; then
      say "  Setup aborted (--yes mode cannot wait for retry). Please stop processes and try again."
      exit 1
    fi
    retry="$(ask "  Try again? [Y/n]: " Y)"
    case "$retry" in [Nn]*) say "  Setup aborted."; exit 1 ;; esac
  done
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

# ---- browser tab or standalone app window? -----------------------------------
# Decides whether the `desktop` extra (pywebview) gets installed at all - a NEW
# dependency every fresh install would otherwise take on unasked (pythonnet on
# Windows; qtpy + PyQt6 + PyQt6-WebEngine on Linux - see pyproject.toml's
# desktop extra comment for why Linux gets Qt, not GTK, and why that is
# entirely pip-installable, no system package manager involved). Default
# stays Browser for exactly that reason (AGENTS.md rule 1/5: no surprise new
# deps, no silent behavior change). Runtime override without re-running
# setup: Settings -> Desktop app -> Default window mode (config key
# desktop_window_mode, "auto" - use it if installed - or "browser"). Leaving
# that key at its "auto" default here is deliberate: once the extra IS
# installed, "auto" already means "use it", so setup needs no config.json
# write of its own.
say ""
say "  Open localm's GUI as its own app window, or in your browser?"
say "    [1] Browser    - opens in your default browser (no extra install)"
say "    [2] App window - its own window, no browser tab (installs localm[desktop])"
wpick="$(ask "  Pick 1 or 2 [1]: " 1)"
EXTRAS="coder,voice,monitor"
if [ "$wpick" = 2 ]; then
  EXTRAS="${EXTRAS},desktop"
  if [ "$(uname -s)" = "Linux" ]; then
    # No apt/dnf/pacman step needed here on purpose - verified live (a real,
    # fully isolated venv, zero system packages, zero sudo) that pip alone
    # gets a working window through pywebview's Qt backend. The one
    # residual gap, also verified live rather than assumed: the underlying
    # Qt/XCB windowing layer can still want a couple of small base system
    # libraries (e.g. libxcb-cursor0) that no Python package provides -
    # commonly already present on a real desktop install, not on a
    # minimal/headless one.
    say "  [i] On Linux the app window needs a couple of small base Qt/X11"
    say "      runtime libraries (e.g. libxcb-cursor0) that are commonly"
    say "      already present on a desktop install. If the window fails to"
    say "      open, localm gui still works - just via the browser."
  fi
fi

# ---- install localm (editable) ----------------------------------------------
say "  Installing localm into .venv ..."
# Catch a hard install failure (set -e would otherwise abort silently) so we can
# offer a bug report before exiting - and still exit non-zero, never masking it.
# NO heartbeat here, deliberately. uv writes STRAIGHT TO THE TERMINAL at this
# site, so it already draws a live byte-progress readout - and it redraws that
# readout IN PLACE (cursor up N lines, rewrite). A second writer printing into
# the same terminal desynchronises the redraw: uv's next frame lands a line low,
# the previous frame is stranded on screen for good, and the heartbeat's own line
# is overwritten by the redraw that follows it. Reported live as garbled progress
# bars during the torch install. The heartbeat is ONLY correct where uv's output
# is CAPTURED and the terminal would otherwise be silent - the ONE such site is
# the create_venv retry loop above. Do not copy it back here.
uv pip install -p .venv -e ".[${EXTRAS}]" || {
  say "  [!] Installing localm failed - see the error above."
  offer_report "localm install failed during setup" "uv pip install -e .[${EXTRAS}] failed - see the error output above."
  exit 1
}

# Verify the CLI entry point actually landed. Reported live: on a WSL2 clone under
# a Windows-drive mount (/mnt/c, /mnt/d, ...) the install can report success while
# .venv/bin/localm itself is simply missing - every OTHER file (localcoder, the
# venv's own python symlink) present and correct, only this one absent. That mount
# is a 9p/DrvFs bridge where uv's own reflink fast-path already falls back to a
# plain copy (unsupported there); a transient hiccup during that fallback write is
# the leading suspect, and it does not reliably repeat back-to-back. Retry once;
# if it is STILL missing, say so loudly rather than silently reaching "Done,
# self-contained" with a broken CLI (do-not-hide-problems) - the rest of this
# script does not depend on the entry point, so this warns and continues rather
# than aborting the whole setup over it.
if [ ! -x .venv/bin/localm ]; then
  say "  [!] .venv/bin/localm did not get installed - retrying once ..."
  # || true: the retry itself failing must not abort setup here either - the
  # STILL-missing branch right below is what reports it (loudly, with the manual
  # fix), and the rest of this script does not depend on the entry point (see the
  # block comment above). Without this guard, a non-zero retry under
  # `set -euo pipefail` kills setup mid-way and that warning never prints.
  uv pip install -p .venv -e ".[${EXTRAS}]" || true
  if [ ! -x .venv/bin/localm ]; then
    say ""
    say "  [!!] .venv/bin/localm is STILL missing after a retry."
    say "       The CLI (.venv/bin/localm ...) and ./localm.sh will not work until"
    say "       this is fixed; the graphical launcher (./localm-launcher.sh) is"
    say "       unaffected."
    say "       Fix it yourself with:"
    say "         uv pip install -p .venv -e \".[${EXTRAS}]\" --reinstall"
    say "       Seen specifically on a WSL2 clone under a Windows-drive mount"
    say "       (/mnt/c, /mnt/d, ...) - cloning into a native Linux path (e.g."
    say "       ~/localm) instead avoids that filesystem bridge entirely."
    say ""
  fi
fi
# Decided once, reused below: setup-llama, make-launcher and plugin setup all
# invoke .venv/bin/localm directly, and a still-missing binary must not turn
# into a confusing mid-script exit or a false "still works" claim once it has
# already been warned about above.
LOCALM_BIN_OK=1
[ -x .venv/bin/localm ] || LOCALM_BIN_OK=0

# ---- native llama.cpp runtime wheel (loader imports it) ---------------------
# (The PyTorch/transformers stack is installed further down, AFTER the backend
# pick, so the HF torch variant can FOLLOW the chosen runtime - see SETUP-1.)
uv pip install -p .venv -e ./runtime >/dev/null 2>&1 || true

# ---- provision the native library (official llama.cpp prebuilt) -------------
# The RECOMMENDED backend comes from the SAME tested policy the Windows installer
# uses (`python -m localm.hwdetect` -> "<vendor> <backend>"), so the two installers
# can never drift: NVIDIA -> cuda (self-contained on both OSes), AMD -> hip when a
# system ROCm/HIP toolkit is detected present (else vulkan; gfx103X on Windows
# always gets the self-contained amd-rocm build regardless), Intel -> vulkan (no
# toolkit-presence probe for oneAPI yet), Apple Silicon -> metal, no GPU -> cpu.
# setup-llama fetches the matching upstream build, so a tester never compiles by hand.
REC="$(.venv/bin/python -m localm.hwdetect 2>/dev/null | awk '{print $2}')"
case "$REC" in
  vulkan|cuda|hip|cpu|metal|amd-rocm) ;;   # a known backend from the policy
  *) case "$GPU" in                        # fallback if the probe failed
       cpu)   REC=cpu ;;
       metal) REC=metal ;;   # see test_probe_failed_fallback_metal_gpu_stays_metal
       *)     REC=vulkan ;;
     esac ;;
esac
# The prompt above promises "n = CPU only", so honour it: hwdetect answers what
# this HARDWARE could use, which is a different question from what the user just
# said they WANT. Without this the next screen recommends a GPU backend and makes
# it the default, so pressing Enter downloads and load-tests a ~195 MB GPU
# runtime the user declined one question earlier. The menu still lists every
# backend, so this changes the DEFAULT, never the available choice.
if [ "$GPU" = cpu ]; then REC=cpu; fi
# [1] is a shortcut for whichever backend the policy recommended, so it is ALWAYS
# the same choice as one of the numbered entries below (vulkan on most GPUs, cpu
# with none, metal on Apple Silicon). Listing it twice with no relation shown
# reads as two different options that happen to share a name. Mark the twin
# instead of removing it: the numbering has to stay stable.
_same="   (same as [1])"
_m2=""; _m3=""; _m4=""; _m5=""; _m7=""
case "$REC" in
  vulkan) _m2="$_same" ;;
  cuda)   _m3="$_same" ;;
  hip)    _m4="$_same" ;;
  cpu)    _m5="$_same" ;;
  metal)  _m7="$_same" ;;
esac
say ""
say "  Native inference runtime (llama.cpp). Recommended for your hardware: $REC"
say "    [1] $REC  (recommended)"
say "    [2] vulkan   - any GPU, no vendor toolkit$_m2"
say "    [3] cuda     - NVIDIA, peak performance (needs the CUDA runtime)$_m3"
say "    [4] hip      - AMD ROCm, peak performance (needs the ROCm runtime)$_m4"
say "    [5] cpu      - no GPU$_m5"
say "    [6] I will build / provide my own (skip the download)"
# See test_menu_shows_metal_line_only_on_apple_silicon.
_pick_range="1-6"
if [ "$IS_APPLE_SILICON" = 1 ]; then
  say "    [7] metal    - Apple Silicon, native GPU acceleration$_m7"
  _pick_range="1-7"
fi
say "    (your pick is load-tested; a failure offers Vulkan, never a silent swap)"
bpick="$(ask "  Pick $_pick_range [1]: " 1)"
case "$bpick" in
  2) BACKEND=vulkan ;; 3) BACKEND=cuda ;; 4) BACKEND=hip ;; 5) BACKEND=cpu ;;
  6) BACKEND=own ;;    7) BACKEND=metal ;; *) BACKEND="$REC" ;;
esac
if [ "$LOCALM_BIN_OK" != 1 ]; then
  # .venv/bin/localm never got installed (warned above) - it is what setup-llama
  # runs through, so calling it here would just fail again, less clearly.
  say "  Skipped - .venv/bin/localm is missing (see the warning above)."
  say "  Provision later:  .venv/bin/localm setup-llama --backend <vulkan|cuda|hip|cpu>"
elif [ "$BACKEND" = own ]; then
  buildpath="$(ask "  Path to a llama.cpp build dir to copy now (blank = skip): " "")"
  if [ -n "$buildpath" ]; then
    .venv/bin/localm setup-llama --from "$buildpath" || {
      # Mirror setup.bat: a declined/failed provision stops setup with a non-zero
      # exit instead of silently continuing into the torch/data-dir steps as if
      # the runtime were installed (NEW-CUDADECLINE).
      say "  [!] setup-llama failed - run later:  .venv/bin/localm setup-llama --from <dir>"
      offer_report "localm setup-llama failed" "Provisioning the native llama.cpp runtime with --from failed during setup."
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
    offer_report "localm setup-llama failed" "Provisioning the native llama.cpp runtime (--backend $BACKEND) failed during setup."
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
  # NO heartbeat around the installs below, deliberately - see the base install
  # above for the mechanism. uv's output goes straight to the terminal here, so it
  # already shows live per-package byte progress (which is exactly what a
  # gigabyte-plus torch download needs), and a second writer would corrupt that
  # in-place redraw rather than reassure anyone.
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
  # NVIDIA CUDA's wheel line depends on GPU generation (Blackwell and newer need
  # a different index than older cards - see hwdetect.pytorch_index_url), so ask
  # localm's own detector for THIS machine's actual line instead of hardcoding
  # one that would silently install kernel-less torch on a Blackwell card.
  cudaspec="$(.venv/bin/python -m localm.hwdetect torch-args cuda 2>/dev/null)"
  say "    NVIDIA CUDA:       uv pip install -p .venv ${cudaspec:-torch torchvision --index-url https://download.pytorch.org/whl/cu126}"
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
if [ "$LOCALM_BIN_OK" = 1 ]; then
  say "  Building the LocaLM app launcher ..."
  .venv/bin/localm make-launcher --force \
    || say "  [!] Could not build the LocaLM launcher - 'localm gui' still works."
else
  say "  Skipping the LocaLM app launcher (.venv/bin/localm is missing)."
fi

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
if [ "$LOCALM_BIN_OK" = 1 ]; then
  .venv/bin/localm plugin setup \
    || say "  [!] Skipped - choose later with:  .venv/bin/localm plugin setup"
else
  say "  Skipped - .venv/bin/localm is missing; run later once fixed:"
  say "    .venv/bin/localm plugin setup"
fi

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
  $rcflag --python-dir "$pydir" --cache-dir "$cachedir" --uv-dir "${UVDIR:-}" \
  --path-dir "${PATH_DIR:-}" --command-shim "${CMD_SHIM:-}" ${PATH_MOD:-} \
  --stamp "$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "")" \
  >/dev/null 2>&1 || say "  [!] Could not record the install manifest (uninstall will be conservative)."

say ""
if [ "$LOCALM_BIN_OK" = 1 ]; then
  say "  Done. This clone is self-contained:"
  say "    ./localm-launcher.sh    graphical launcher (GUI / chat / server / coder)"
  say "    ./localm.sh <args>      the localm CLI, e.g.:  ./localm.sh gui"
  say "    .venv/bin/localm ...    CLI directly"
else
  say "  Done, with one open issue: .venv/bin/localm never got installed (see the"
  say "  warning above), so the CLI is not usable yet:"
  say "    ./localm-launcher.sh    graphical launcher - works now (does not need it)"
  say "    ./localm.sh <args>      will NOT work until it is fixed"
  say "    .venv/bin/localm ...    will NOT work until it is fixed"
  say "  Fix it with:  uv pip install -p .venv -e \".[${EXTRAS}]\" --reinstall"
fi
if [ -n "$SHORTCUT" ]; then
  say "    A LocaLM entry was also added to your application menu."
fi
say ""
say "  The GUI launcher uses Tk; localm's bundled Python includes it, so it"
say "  normally works out of the box. If the launcher ever reports Tk missing"
say "  (e.g. a system Python without it), install your distro's python3-tk."
say "  The web GUI itself needs only a browser."
say ""
