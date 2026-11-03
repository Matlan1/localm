#!/usr/bin/env bash
# =============================================================================
#  One-click localm install (Linux / macOS). Downloads localm and runs setup.
#
#    curl -fsSL https://raw.githubusercontent.com/Matlan1/localm/master/install.sh | bash
#
#  localm is downloaded as a source archive from GitHub with the system curl (or
#  wget) and unpacked with the system tar; no git is needed. uv (the Python
#  toolchain localm uses) is installed by setup.sh itself, inside the install
#  folder for the default Portable choice, so nothing is written to your home
#  folder or shell startup files unless you pick Shared.
#
#  A folder that already holds localm is not downloaded again: its setup runs.
#
#  Override the destination, the GitHub repository or its branch:
#    LOCALM_DIR=~/apps/localm  LOCALM_REPO=https://github.com/you/localm.git  \
#      LOCALM_BRANCH=master  bash install.sh
# =============================================================================
set -euo pipefail
LOCALM_SYSDIRS="/usr/bin:/bin:/usr/sbin:/sbin"
export PATH="$LOCALM_SYSDIRS"

REPO="${LOCALM_REPO:-https://github.com/Matlan1/localm.git}"
BRANCH="${LOCALM_BRANCH:-master}"
DEST="${LOCALM_DIR:-$HOME/localm}"

sys_prog() {  # sys_prog NAME  ->  prints NAME's full path in the fixed system folders; 1 when absent
  local d IFS=:
  for d in $LOCALM_SYSDIRS; do
    if [ -x "$d/$1" ] && [ ! -d "$d/$1" ]; then printf '%s\n' "$d/$1"; return 0; fi
  done
  return 1
}
download() {  # download URL DEST  ->  0 when DEST holds URL, fetched with the system curl, else wget
  local t
  if t="$(sys_prog curl)"; then "$t" -fsSL -o "$2" "$1"; return; fi
  if t="$(sys_prog wget)"; then "$t" -q -O "$2" "$1"; return; fi
  echo "Neither curl nor wget was found in $LOCALM_SYSDIRS, so localm cannot be downloaded."
  return 1
}

if [ -f "$DEST/setup.sh" ]; then
  echo "localm is already in $DEST; running its setup ..."
else
  case "$REPO" in
    https://github.com/*) ;;
    *) echo "LOCALM_REPO must be a https://github.com/ repository address (got: $REPO)."; exit 1 ;;
  esac
  if [ -e "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then
    echo "$DEST already exists and is not a localm folder. Set LOCALM_DIR to another folder."
    exit 1
  fi
  tar_bin="$(sys_prog tar)" || { echo "tar was not found in $LOCALM_SYSDIRS."; exit 1; }
  url="${REPO%.git}/archive/refs/heads/${BRANCH}.tar.gz"
  echo "Downloading localm into $DEST ..."
  mkdir -p "$DEST"
  archive="$DEST/.localm-download.tar.gz"
  if ! download "$url" "$archive"; then
    rm -f "$archive"
    echo "Could not download $url"
    exit 1
  fi
  if ! "$tar_bin" -xzf "$archive" -C "$DEST" --strip-components=1; then
    rm -f "$archive"
    echo "Could not unpack the downloaded archive into $DEST."
    exit 1
  fi
  rm -f "$archive"
fi

cd "$DEST"
exec "$BASH" setup.sh --yes
