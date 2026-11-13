#!/usr/bin/env bash
# =============================================================================
#  One-click localm install (Linux / macOS). Clones the repo and runs setup.
#
#    curl -fsSL https://raw.githubusercontent.com/Matlan1/localm/master/install.sh | bash
#
#  uv (the Python toolchain localm uses) is installed by setup.sh itself, inside
#  the clone (the default Portable choice), so nothing is written to your shell
#  startup files. On Linux, setup also adds an application menu entry
#  (~/.local/share/applications/localm.desktop); the uninstaller removes it.
#
#  Override the destination or repo:
#    LOCALM_DIR=~/apps/localm  LOCALM_REPO=https://github.com/you/localm.git  \
#      bash install.sh
# =============================================================================
set -euo pipefail

REPO="${LOCALM_REPO:-https://github.com/Matlan1/localm.git}"
DEST="${LOCALM_DIR:-$HOME/localm}"

command -v git >/dev/null 2>&1 || { echo "git is required - install it first."; exit 1; }

if [ -d "$DEST/.git" ]; then
  echo "Updating existing clone at $DEST ..."
  git -C "$DEST" pull --ff-only || true
else
  echo "Cloning localm into $DEST ..."
  git clone --depth 1 "$REPO" "$DEST"
fi

cd "$DEST"
exec bash setup.sh --yes
