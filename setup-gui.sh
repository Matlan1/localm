#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LocaLM graphical setup - the windowed alternative to setup.sh.
#
# Asks nothing in the terminal: puts uv into ./.uv (and uv puts Python 3.12 into
# ./.python and its downloads into ./.cache) when they are not there yet, then
# hands the whole install to installer/gui.py, which runs on that Python and
# needs no other dependency - tkinter ships with it. The window's first page
# asks Portable or Shared.
#
# setup.sh remains the console installer and is unchanged. This is the same
# install, asked for in a window.
set -euo pipefail
cd "$(dirname "$0")"

# The uv release this setup installs; its installer script runs only when its
# sha256 matches UV_INSTALLER_SHA256.
UV_INSTALLER_VERSION="0.13.9"
UV_INSTALLER_SHA256="ccc888115ac831bd9c30855c1d4abeae32fce679701d2aac8e72eb0beffb41c2"
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

# ---- the uv that opens the window --------------------------------------------
# The window reads LOCALM_GUI_STORE (the choice its first page starts on:
# portable or shared), LOCALM_GUI_UV (the uv in use), LOCALM_GUI_BOOTSTRAP
# (what this run put into the folder to open the window: uv, python, cache)
# and, once Shared was chosen, LOCALM_GUI_REPOINTED=1 and
# LOCALM_GUI_UV_INSTALLED=1 when the window installed that per-user uv.
# The folder's own ./.uv/uv is used when it is there; an earlier Shared install
# (no ./.uv, an install record) reuses the per-user uv; anything else installs
# uv into ./.uv without asking. ./.localm-gui-bootstrap names what a run of this
# file put into the folder and no install has taken over yet (the window removes
# it when a Portable install starts), so a later Shared choice removes those
# folders too.
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
run_uv_installer() {  # the pinned, checksum-verified Astral installer; exits on failure
    # The installer script and everything Astral's installer downloads (TMPDIR)
    # stay in ./.localm-setup-tmp, which is removed again when it is empty.
    UVTMP="$(pwd)/.localm-setup-tmp/uv-installer.$$"
    if ! { mkdir -p "$UVTMP/tmp" && fetch_uv_installer "$UVTMP/uv-installer.sh" \
            && TMPDIR="$UVTMP/tmp" sh "$UVTMP/uv-installer.sh"; }; then
        rm -rf "$UVTMP"
        rmdir ./.localm-setup-tmp 2>/dev/null || true
        echo
        echo "  [!] Could not download or run Astral's uv installer."
        echo "      Use the console installer instead:  ./setup.sh"
        exit 1
    fi
    rm -rf "$UVTMP"
    rmdir ./.localm-setup-tmp 2>/dev/null || true
}
bootstrap_add() {  # bootstrap_add NAME  ->  adds NAME to LOCALM_GUI_BOOTSTRAP once
    case " $LOCALM_GUI_BOOTSTRAP " in *" $1 "*) ;; *) LOCALM_GUI_BOOTSTRAP="${LOCALM_GUI_BOOTSTRAP:+$LOCALM_GUI_BOOTSTRAP }$1" ;; esac
}
write_bootstrap() {  # writes LOCALM_GUI_BOOTSTRAP to ./.localm-gui-bootstrap, or removes that file when empty
    if [ -n "$LOCALM_GUI_BOOTSTRAP" ]; then
        printf '%s\n' "$LOCALM_GUI_BOOTSTRAP" > ./.localm-gui-bootstrap 2>/dev/null || true
    else
        rm -f ./.localm-gui-bootstrap
    fi
}
remove_bootstrap() {  # removes ./.uv ./.python ./.cache as named in LOCALM_GUI_BOOTSTRAP; LEFTOVER and LOCALM_GUI_BOOTSTRAP then list what stayed
    local n stayed=""
    LEFTOVER=""
    for n in $LOCALM_GUI_BOOTSTRAP; do
        case "$n" in uv|python|cache) ;; *) continue ;; esac
        if [ -e "./.$n" ]; then
            chmod -R u+w "./.$n" 2>/dev/null || true
            rm -rf "./.$n" 2>/dev/null || true
        fi
        if [ -e "./.$n" ]; then LEFTOVER="$LEFTOVER ./.$n"; stayed="${stayed:+$stayed }$n"; fi
    done
    LOCALM_GUI_BOOTSTRAP="$stayed"
}
uv_not_callable() {  # says so and exits 1
    echo
    echo "  [!] uv still is not callable, so the graphical setup cannot start."
    if [ "$LOCALM_GUI_STORE" = portable ]; then
        echo "      Looked for $PWD/.uv/uv (the installer did not leave one there)."
    else
        echo "      Looked on PATH and in ~/.local/bin for the per-user uv."
    fi
    echo "      Open a NEW terminal and run ./setup.sh instead."
    exit 1
}

# A headless box (no DISPLAY) has no window to show: say so and point at the
# console installer, before uv is installed.
if [ "$(uname -s)" != "Darwin" ] && [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "  [!] No graphical display was found (DISPLAY/WAYLAND_DISPLAY are unset)."
    echo "      Use the console installer instead:  ./setup.sh"
    exit 1
fi

UVEXE=""
LOCALM_GUI_BOOTSTRAP=""
unset LOCALM_GUI_UV_INSTALLED LOCALM_GUI_REPOINTED
if [ -f ./.localm-gui-bootstrap ]; then
    for n in $(cat ./.localm-gui-bootstrap 2>/dev/null || true); do
        case "$n" in uv|python|cache) if [ -e "./.$n" ]; then bootstrap_add "$n"; fi ;; esac
    done
fi
export LOCALM_GUI_STORE=portable
if [ -x "./.uv/uv" ]; then
    UVEXE="$PWD/.uv/uv"
elif { [ -f ./.localm-install.json ] || [ -f ./.venv/.localm-venv ]; } && [ -n "$(shared_uv)" ]; then
    UVEXE="$(shared_uv)"
    export LOCALM_GUI_STORE=shared
else
    echo "  Installing uv into ./.uv ..."
    # UV_UNMANAGED_INSTALL keeps it in ./.uv without editing shell startup
    # files or writing an install receipt under ~/.config/uv.
    export UV_INSTALL_DIR="$PWD/.uv"
    export UV_UNMANAGED_INSTALL="$PWD/.uv"
    run_uv_installer
    unset UV_INSTALL_DIR UV_UNMANAGED_INSTALL
    if [ -x "./.uv/uv" ]; then
        UVEXE="$PWD/.uv/uv"
        bootstrap_add uv
    fi
fi
if [ -z "$UVEXE" ]; then uv_not_callable; fi

# ---- open the installer window ----------------------------------------------
# --no-project so uv never resolves this repo as its own project, and an explicit
# --python so the interpreter is the managed 3.12 the install targets. Portable
# keeps the interpreter this window runs on inside the folder, so the install
# reuses it rather than downloading a second copy. 46 and 47: Shared was chosen
# on the first page and the per-user uv is in place (47: the window installed
# it); what this run put in the folder is removed and the window opens again on
# that uv.
rc=0
while : ; do
    export LOCALM_GUI_UV="$UVEXE"
    echo "  Opening the setup window ..."
    if [ "$LOCALM_GUI_STORE" = portable ]; then
        if [ ! -d ./.python ]; then bootstrap_add python; fi
        if [ ! -d ./.cache ]; then bootstrap_add cache; fi
        export UV_PYTHON_INSTALL_DIR="$PWD/.python"
        export UV_CACHE_DIR="$PWD/.cache"
    fi
    write_bootstrap
    export LOCALM_GUI_BOOTSTRAP
    export UV_SYSTEM_CERTS=1
    rc=0
    "$UVEXE" run --no-project --isolated --python 3.12 python installer/gui.py || rc=$?
    if [ "$rc" != 46 ] && [ "$rc" != 47 ]; then break; fi
    if [ -n "${LOCALM_GUI_REPOINTED:-}" ]; then break; fi
    if [ "$rc" = 47 ]; then export LOCALM_GUI_UV_INSTALLED=1; fi
    export LOCALM_GUI_REPOINTED=1
    unset UV_PYTHON_INSTALL_DIR UV_CACHE_DIR
    export LOCALM_GUI_STORE=shared
    UVEXE="$(shared_uv)"
    if [ -z "$UVEXE" ]; then
        # Every directory the per-user install may have used, in setup.sh's own order.
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        if [ -n "${XDG_BIN_HOME:-}" ]; then export PATH="$XDG_BIN_HOME:$PATH"; fi
        UVEXE="$(shared_uv)"
    fi
    if [ -z "$UVEXE" ]; then uv_not_callable; fi
    remove_bootstrap
    write_bootstrap
    if [ -n "$LEFTOVER" ]; then
        echo "  [!] Could not remove$LEFTOVER from this folder."
        echo "      Setup put them there to open its window; the Shared choice does not use"
        echo "      them. Delete them by hand."
    fi
    echo "  Shared: using $UVEXE"
done
# 42 and 43: an uninstall finished in the window; the Python runtime it ran on
# is removed now that it has closed. 43 and 44: something asked for was kept.
# 45: the uninstall did not finish.
if [ "$rc" = 42 ] || [ "$rc" = 43 ]; then
    echo "  Removing the last LocaLM folders ..."
    rm -f ./.localm-gui-bootstrap
    frc=0
    bash ./setup.sh --finish-uninstall || frc=$?
    if [ "$frc" != 0 ]; then
        echo
        echo "  [!] Some LocaLM folders could not be removed - close any LocaLM window"
        echo "      and delete them by hand, or run  ./setup.sh --finish-uninstall  again."
        exit 1
    fi
fi
if [ "$rc" = 43 ] || [ "$rc" = 44 ]; then
    echo
    echo "  [!] LocaLM was removed, but some things you asked to delete were not"
    echo "      deleted - the setup window listed them under REFUSED."
    exit 2
fi
if [ "$rc" = 42 ]; then
    echo
    echo "  LocaLM was uninstalled."
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
    echo "  [!] The setup window could not run (exit $rc)."
    echo "      Use the console installer instead:  ./setup.sh"
    exit "$rc"
fi
