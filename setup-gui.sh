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

echo
echo "  LocaLM graphical setup"
echo

# ---- locate uv --------------------------------------------------------------
UVEXE=""
if [ -x "./.uv/uv" ]; then
    UVEXE="$PWD/.uv/uv"
elif command -v uv >/dev/null 2>&1; then
    UVEXE="uv"
fi

if [ -z "$UVEXE" ]; then
    echo "  uv (the Python package manager LocaLM builds on) is not installed yet."
    echo "  It is a small download and is needed before any window can open."
    echo
    read -r -p "  Install it now? [Y/n]: " GETUV
    case "${GETUV:-Y}" in
        [Nn]*)
            echo
            echo "  Nothing was installed. Run ./setup.sh for the console installer."
            exit 1
            ;;
    esac
    echo "  Installing uv ..."
    export UV_INSTALL_DIR="$PWD/.uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer updates the shell profile, which this already running shell
    # does not see. Prepend every directory it may have used, in setup.sh's own
    # order, so the uv just installed is callable right now.
    export PATH="$PWD/.uv:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if [ -x "./.uv/uv" ]; then
        UVEXE="$PWD/.uv/uv"
    elif command -v uv >/dev/null 2>&1; then
        UVEXE="uv"
    fi
fi

if [ -z "$UVEXE" ]; then
    echo
    echo "  [!] uv still is not callable, so the graphical setup cannot start."
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

echo "  Opening the setup window ..."
# Name the uv that worked here, so the installer's own steps run that exact
# binary rather than searching for one again.
export LOCALM_UV="$UVEXE"
if ! "$UVEXE" run --no-project --python 3.12 python installer/gui.py; then
    echo
    echo "  [!] The setup window could not run."
    echo "      Use the console installer instead:  ./setup.sh"
    exit 1
fi
