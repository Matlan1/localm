#!/usr/bin/env bash
# =============================================================================
#  LocaLM - roll back a bad update, even when localm will not start (Linux / macOS).
#
#  Run:  bash rollback.sh
#  Restores the PREVIOUS localm version from the last update's backup. Use it when
#  an update left localm broken or misbehaving and `localm update --rollback` will
#  not run because the new build will not start.
#
#  Optional args pass straight through:  --yes  (skip the confirmation prompt)
# =============================================================================
# Inherited uv / pip / Python settings and LOCALM_HOME are cleared, and PATH is
# the fixed system folders.
LOCALM_SYSDIRS="/usr/bin:/bin:/usr/sbin:/sbin"
export PATH="$LOCALM_SYSDIRS"
for _var in $(compgen -e); do
  case "$_var" in UV_*|PIP_*) unset "$_var" ;; esac
done
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX LOCALM_HOME
export UV_NO_CONFIG=1 PIP_CONFIG_FILE=/dev/null PYTHONNOUSERSITE=1
case "$0" in */*) cd -- "${0%/*}" || exit 1 ;; esac

# The rollback script is stdlib-only and loads the restore helper by file path,
# so it runs on this clone's .venv Python, else on localm's own managed Python
# (./.python, or the Shared store); never on a Python from PATH.
case "${XDG_DATA_HOME:-}" in
  /*) SHARED_STORE="$XDG_DATA_HOME/localm/shared" ;;
  *)  SHARED_STORE="${HOME:+$HOME/.local/share/localm/shared}" ;;
esac
RBPY=""
for c in .venv/bin/python3 .venv/bin/python .python/*/bin/python3 \
    "${SHARED_STORE:-/nonexistent}"/python/*/bin/python3; do
  [ -x "$c" ] && { RBPY="$c"; break; }
done

if [ -n "$RBPY" ]; then
  exec "$RBPY" scripts/rollback_update.py "$@"
fi

echo "No localm Python was found to run the rollback (.venv is missing). Run setup" >&2
echo "first (bash setup.sh), or restore the previous version by hand:" >&2
echo "copy the contents of home/updates/backup back into $(pwd)." >&2
exit 1
