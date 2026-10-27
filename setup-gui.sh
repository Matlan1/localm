#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LocaLM graphical setup - the windowed alternative to setup.sh.
#
# Bootstraps uv (the only part that cannot be graphical: something has to
# provide a Python before a window can exist), then hands the whole install to
# installer/gui.py, which runs on uv's managed CPython and needs no other
# dependency - tkinter ships with it.
#
# setup.sh remains the console installer and is unchanged. This is the same
# install, asked for in a window.
set -euo pipefail
cd "$(dirname "$0")"

# The uv release this setup installs; its installer script runs only when its
# sha256 matches UV_INSTALLER_SHA256.
UV_INSTALLER_VERSION="0.13.6"
UV_INSTALLER_SHA256="f7a56e7afaec49a3bcbc3ac3cdbe917e7c6ce30e0d815308a12e531aa5ef23b7"
file_sha256() {  # file_sha256 FILE  ->  prints the hex sha256, or nothing when no hashing tool exists
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
    elif command -v openssl >/dev/null 2>&1; then openssl dgst -sha256 "$1" | sed 's/^.*= *//'
    fi
}
fetch_uv_installer() {  # fetch_uv_installer DEST  ->  0 only when DEST holds the checksum-verified installer
    local dest="$1" got
    if ! curl -fsSL -o "$dest" "https://github.com/astral-sh/uv/releases/download/${UV_INSTALLER_VERSION}/uv-installer.sh"; then
        echo "  [!] Could not download the uv ${UV_INSTALLER_VERSION} installer."
        return 1
    fi
    got="$(file_sha256 "$dest")"
    if [ -z "$got" ]; then
        echo "  [!] No sha256 tool (sha256sum, shasum or openssl) was found, so the uv installer cannot be verified and was not run."
        return 1
    fi
    if [ "$got" != "$UV_INSTALLER_SHA256" ]; then
        echo "  [!] The downloaded uv installer did not match its expected checksum and was not run."
        return 1
    fi
}

echo
echo "  LocaLM graphical setup"
echo

# ---- portable vs shared: where localm's Python tooling lives ----------------
# The question setup.sh asks, asked once, here. The window reads the answer from
# LOCALM_GUI_STORE (portable or shared), the uv in use from LOCALM_GUI_UV, and
# LOCALM_GUI_UV_INSTALLED=1 when uv was installed at its per-user location by
# this run. Portable uses only ./.uv/uv, installing it there when it is missing.
# Shared uses the uv on PATH, never ./.uv, and installs uv at its normal per-user
# location when there is none.
shared_uv() {  # prints the first uv on PATH that is not this folder's ./.uv copy
    local d dirs
    IFS=: read -r -a dirs <<< "$PATH"
    for d in "${dirs[@]}"; do
        [ -n "$d" ] || continue
        [ -x "$d/uv" ] && [ ! -d "$d/uv" ] || continue
        if [ "$(cd "$d" 2>/dev/null && pwd)" = "$PWD/.uv" ]; then continue; fi
        echo "$d/uv"
        return 0
    done
    return 0
}
ask_install_uv() {  # setup.sh's uv-install question; exits when the user declines
    local getuv=""
    echo "  [!] uv (the Python package manager localm builds on) is not installed."
    # SETUP-Q: uv-install
    read -r -p "  Install it now with Astral's official installer? [Y/n]: " getuv || getuv=""
    case "${getuv:-Y}" in
        [Nn]*)
            echo
            echo "  Nothing was installed. Run ./setup.sh for the console installer."
            exit 1
            ;;
    esac
}
run_uv_installer() {  # the pinned, checksum-verified Astral installer; exits on failure
    UVTMP="$(mktemp -d 2>/dev/null)" || { UVTMP="$(pwd)/.uv-installer-tmp"; mkdir -p "$UVTMP"; }
    if ! { fetch_uv_installer "$UVTMP/uv-installer.sh" && sh "$UVTMP/uv-installer.sh"; }; then
        rm -rf "$UVTMP"
        echo
        echo "  [!] Could not download or run Astral's uv installer."
        echo "      Use the console installer instead:  ./setup.sh"
        exit 1
    fi
    rm -rf "$UVTMP"
}

echo "  Keep localm's Python tooling (uv itself, its runtime, and downloads) inside this folder?"
echo "    [1] Portable - everything in this folder (self-contained; re-downloads per clone)"
echo "    [2] Shared   - reuse/install uv at its normal per-user location (faster; lives in ~/.local)"
STOREPICK=""
# SETUP-Q: tooling-store
read -r -p "  Pick 1 or 2 [1]: " STOREPICK || STOREPICK=""
STOREPICK="${STOREPICK:-1}"
UVEXE=""
unset LOCALM_GUI_UV_INSTALLED
if [ "$STOREPICK" = 1 ]; then
    export LOCALM_GUI_STORE=portable
    echo "  Portable: uv, Python, and downloads all under this folder"
    if [ ! -x "./.uv/uv" ]; then
        ask_install_uv
        echo "  Installing uv ..."
        # UV_UNMANAGED_INSTALL keeps it in ./.uv without editing shell startup
        # files or writing an install receipt under ~/.config/uv.
        export UV_INSTALL_DIR="$PWD/.uv"
        export UV_UNMANAGED_INSTALL="$PWD/.uv"
        echo "  Portable: installing uv itself under ./.uv"
        run_uv_installer
    fi
    if [ -x "./.uv/uv" ]; then UVEXE="$PWD/.uv/uv"; fi
else
    export LOCALM_GUI_STORE=shared
    echo "  Shared: reusing/installing uv and its Python + cache (outside this folder)."
    UVEXE="$(shared_uv)"
    if [ -z "$UVEXE" ]; then
        ask_install_uv
        echo "  Installing uv ..."
        unset UV_INSTALL_DIR UV_UNMANAGED_INSTALL
        run_uv_installer
        export LOCALM_GUI_UV_INSTALLED=1
        # The installer updates the shell profile, which this already running
        # shell does not see. Prepend both directories it may have used, in
        # setup.sh's own order, so the uv just installed is callable right now.
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        UVEXE="$(shared_uv)"
    fi
fi
echo

if [ -z "$UVEXE" ]; then
    echo
    echo "  [!] uv still is not callable, so the graphical setup cannot start."
    if [ "$LOCALM_GUI_STORE" = portable ]; then
        echo "      Looked for $PWD/.uv/uv (the installer did not leave one there)."
    fi
    echo "      Open a NEW terminal and run ./setup.sh instead."
    exit 1
fi

# ---- open the installer window ----------------------------------------------
# --no-project so uv never resolves this repo as its own project, and an explicit
# --python so the interpreter is the managed 3.12 the install targets.
#
# A headless box (no DISPLAY) has no window to show: say so and point at the
# console installer rather than failing with a tkinter traceback.
if [ "$(uname -s)" != "Darwin" ] && [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "  [!] No graphical display was found (DISPLAY/WAYLAND_DISPLAY are unset)."
    echo "      Use the console installer instead:  ./setup.sh"
    exit 1
fi

export LOCALM_GUI_UV="$UVEXE"
echo "  Opening the setup window ..."
# Portable keeps the interpreter this window runs on inside the folder, so the
# install reuses it rather than downloading a second copy.
if [ "$LOCALM_GUI_STORE" = portable ]; then
    export UV_PYTHON_INSTALL_DIR="$PWD/.python"
    export UV_CACHE_DIR="$PWD/.cache"
fi
export UV_SYSTEM_CERTS=1
rc=0
"$UVEXE" run --no-project --isolated --python 3.12 python installer/gui.py || rc=$?
# 42 and 43: an uninstall finished in the window; the Python runtime it ran on
# is removed now that it has closed. 43 and 44: something asked for was kept.
# 45: the uninstall did not finish.
if [ "$rc" = 42 ] || [ "$rc" = 43 ]; then
    echo "  Removing the last LocaLM folders ..."
    frc=0
    bash ./setup.sh --finish-uninstall || frc=$?
    if [ "$frc" != 0 ]; then exit "$frc"; fi
fi
if [ "$rc" = 43 ] || [ "$rc" = 44 ]; then
    echo
    echo "  [!] LocaLM was removed, but some things you asked to delete were not"
    echo "      deleted - the setup window listed them under REFUSED."
    exit 2
fi
if [ "$rc" = 42 ]; then
    exit 0
fi
if [ "$rc" = 45 ]; then
    echo
    echo "  [!] The uninstall did not finish - the setup window listed why. Close"
    echo "      any LocaLM window and run the setup again."
    exit 1
fi
if [ "$rc" != 0 ]; then
    echo
    echo "  [!] The setup window could not run."
    echo "      Use the console installer instead:  ./setup.sh"
    exit 1
fi
