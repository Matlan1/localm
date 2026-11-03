#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LocaLM graphical setup - the windowed alternative to setup.sh.
#
# Installs localm's own pinned uv (the part that cannot be graphical: a Python
# must exist before a window can), then hands the whole install to
# installer/gui.py, which runs on localm's own managed CPython (tkinter ships
# with it). setup.sh is the console installer; this is the same install, asked
# for in a window.
set -euo pipefail
# ---- isolation: no inherited tool settings; programs from the system folders --
LOCALM_SYSDIRS="/usr/bin:/bin:/usr/sbin:/sbin"
export PATH="$LOCALM_SYSDIRS"
for _var in $(compgen -e); do
    case "$_var" in UV_*|PIP_*) unset "$_var" ;; esac
done
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX LOCALM_HOME
export UV_NO_CONFIG=1 PIP_CONFIG_FILE=/dev/null PYTHONNOUSERSITE=1
cd "$(dirname "$0")"
ROOT="$(pwd)"
# Every temporary file of this run, the setup window's included, goes to
# ./.localm-setup-tmp, removed when this script exits.
STMP="$ROOT/.localm-setup-tmp"
if ! mkdir -p "$STMP"; then
    echo "  [!] Could not create $STMP for setup's temporary files."
    exit 1
fi
export TMPDIR="$STMP" TMP="$STMP" TEMP="$STMP"
trap 'rm -rf -- "$STMP" 2>/dev/null || true' EXIT
# Shared mode's store: localm's own uv, Python and download cache for every
# Shared localm install of this user.
case "${XDG_DATA_HOME:-}" in
    /*) SHARED_STORE="$XDG_DATA_HOME/localm/shared" ;;
    *)  SHARED_STORE="${HOME:+$HOME/.local/share/localm/shared}" ;;
esac

sys_prog() {  # sys_prog NAME  ->  prints NAME's full path in the fixed system folders; 1 when absent
    local d IFS=:
    for d in $LOCALM_SYSDIRS; do
        if [ -x "$d/$1" ] && [ ! -d "$d/$1" ]; then printf '%s\n' "$d/$1"; return 0; fi
    done
    return 1
}
download() {  # download URL DEST  ->  0 when DEST holds URL, fetched with the system curl, else wget
    local url="$1" dest="$2" dirs="$LOCALM_SYSDIRS" d
    # macOS: a download tool may also come from /usr/local/bin or /opt/homebrew/bin.
    if [ "$(uname -s 2>/dev/null)" = Darwin ]; then dirs="$dirs:/usr/local/bin:/opt/homebrew/bin"; fi
    local IFS=:
    for d in $dirs; do
        if [ -x "$d/curl" ] && [ ! -d "$d/curl" ]; then "$d/curl" -fsSL -o "$dest" "$url"; return; fi
    done
    for d in $dirs; do
        if [ -x "$d/wget" ] && [ ! -d "$d/wget" ]; then "$d/wget" -q -O "$dest" "$url"; return; fi
    done
    return 1
}
# The uv release this setup installs; its installer script runs only when its
# sha256 matches UV_INSTALLER_SHA256.
UV_INSTALLER_VERSION="0.13.6"
UV_INSTALLER_SHA256="f7a56e7afaec49a3bcbc3ac3cdbe917e7c6ce30e0d815308a12e531aa5ef23b7"
file_sha256() {  # file_sha256 FILE  ->  prints the hex sha256, or nothing when no hashing tool exists
    local t
    if t="$(sys_prog sha256sum)"; then "$t" "$1" | cut -d' ' -f1
    elif t="$(sys_prog shasum)"; then "$t" -a 256 "$1" | cut -d' ' -f1
    elif t="$(sys_prog openssl)"; then "$t" dgst -sha256 "$1" | sed 's/^.*= *//'
    fi
}
fetch_uv_installer() {  # fetch_uv_installer DEST  ->  0 only when DEST holds the checksum-verified installer
    local dest="$1" got
    if ! download "https://github.com/astral-sh/uv/releases/download/${UV_INSTALLER_VERSION}/uv-installer.sh" "$dest"; then
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
# LOCALM_GUI_STORE (portable or shared) and LOCALM_GUI_UV_INSTALLED=1 for Shared
# (the store's uv is one localm setup installed); it finds the store's uv
# itself. Both choices use only localm's own uv, the pinned release, from
# the store: ./.uv for Portable, $SHARED_STORE/uv for Shared. It is installed
# there when it is missing or reports another version.
uv_is_pinned() {  # uv_is_pinned UV  ->  true when UV runs and reports the pinned release
    local v
    [ -x "$1" ] || return 1
    v="$("$1" --version 2>/dev/null)" || return 1
    v="${v#uv }"; v="${v%% *}"
    [ "$v" = "$UV_INSTALLER_VERSION" ]
}
run_uv_installer() {  # the pinned, checksum-verified Astral installer into $UV_INSTALL_DIR; 1 on failure
    # The installer script and everything Astral's installer downloads (TMPDIR)
    # stay in ./.localm-setup-tmp/uv-installer.<pid>.
    UVTMP="$STMP/uv-installer.$$"
    if ! { mkdir -p "$UVTMP/tmp" && fetch_uv_installer "$UVTMP/uv-installer.sh" \
            && TMPDIR="$UVTMP/tmp" sh "$UVTMP/uv-installer.sh"; }; then
        rm -rf "$UVTMP"
        echo
        echo "  [!] Could not download or run Astral's uv installer."
        return 1
    fi
    rm -rf "$UVTMP"
}

echo "  Keep localm's Python tooling (uv itself, its runtime, and downloads) inside this folder?"
echo "    [1] Portable - everything in this folder (self-contained; re-downloads per clone)"
echo "    [2] Shared   - localm's own uv, Python and downloads in one localm folder in"
echo "                   your user profile, reused by every localm install (faster)"
STOREPICK=""
# SETUP-Q: tooling-store
read -r -p "  Pick 1 or 2 [1]: " STOREPICK || STOREPICK=""
STOREPICK="${STOREPICK:-1}"
unset LOCALM_GUI_UV_INSTALLED
if [ "$STOREPICK" = 1 ]; then
    export LOCALM_GUI_STORE=portable
    STORE_UV="$ROOT/.uv"; STORE_PY="$ROOT/.python"; STORE_CACHE="$ROOT/.cache"
    echo "  Portable: uv, Python, and downloads all under this folder"
else
    if [ -z "$SHARED_STORE" ]; then
        echo "  [!] Shared needs your home folder (HOME is not set). Pick Portable, or set HOME."
        exit 1
    fi
    export LOCALM_GUI_STORE=shared
    export LOCALM_GUI_UV_INSTALLED=1
    STORE_UV="$SHARED_STORE/uv"; STORE_PY="$SHARED_STORE/python"; STORE_CACHE="$SHARED_STORE/cache"
    echo "  Shared: localm's uv, Python and downloads in $SHARED_STORE"
fi
UVEXE="$STORE_UV/uv"
# uv keeps its credentials store in the store's uv folder, and puts no Python
# shim in a user folder and no Python registry entry.
export UV_CREDENTIALS_DIR="$STORE_UV/credentials"
export UV_PYTHON_INSTALL_BIN=0 UV_PYTHON_INSTALL_REGISTRY=0
if ! uv_is_pinned "$UVEXE"; then
    echo "  Installing localm's uv ${UV_INSTALLER_VERSION} into $STORE_UV ..."
    export UV_INSTALL_DIR="$STORE_UV"
    export UV_UNMANAGED_INSTALL="$STORE_UV"
    run_uv_installer || true
    unset UV_INSTALL_DIR UV_UNMANAGED_INSTALL
fi
echo

if ! uv_is_pinned "$UVEXE"; then
    echo
    echo "  [!] uv still is not callable, so the graphical setup cannot start."
    echo "      localm could not install its own uv ${UV_INSTALLER_VERSION} into $STORE_UV."
    echo "      Check the network connection and run ./setup-gui.sh again."
    exit 1
fi

# ---- open the installer window ----------------------------------------------
# --no-project so uv never resolves this repo as its own project; the window
# runs on localm's own managed Python 3.12 from the store (only-managed).
#
# A headless box (no DISPLAY) has no window to show: say so and point at the
# console installer rather than failing with a tkinter traceback.
if [ "$(uname -s)" != "Darwin" ] && [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "  [!] No graphical display was found (DISPLAY/WAYLAND_DISPLAY are unset)."
    echo "      Use the console installer instead:  ./setup.sh"
    exit 1
fi

echo "  Opening the setup window ..."
export UV_PYTHON_INSTALL_DIR="$STORE_PY"
export UV_CACHE_DIR="$STORE_CACHE"
export UV_PYTHON_PREFERENCE=only-managed
export UV_SYSTEM_CERTS=1
rc=0
"$UVEXE" run --no-project --isolated --python 3.12 --python-preference only-managed python installer/gui.py || rc=$?
# 42 and 43: an uninstall finished in the window; the Python runtime it ran on
# is removed now that it has closed. 43 and 44: something asked for was kept.
# 45: the uninstall did not finish.
if [ "$rc" = 42 ] || [ "$rc" = 43 ]; then
    echo "  Removing the last LocaLM folders ..."
    frc=0
    "$BASH" ./setup.sh --finish-uninstall || frc=$?
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
