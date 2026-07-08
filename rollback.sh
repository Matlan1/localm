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
cd "$(dirname "$0")" || exit 1

# Prefer the clone's venv python; else any python3/python. The rollback script is
# stdlib-only and loads the restore helper by file path, so a venv that cannot
# import localm still rolls back fine.
RBPY=""
for c in ".venv/bin/python3" ".venv/bin/python"; do
  [ -x "$c" ] && { RBPY="$c"; break; }
done
if [ -z "$RBPY" ]; then
  for c in python3 python; do
    command -v "$c" >/dev/null 2>&1 && { RBPY="$c"; break; }
  done
fi

if [ -n "$RBPY" ]; then
  exec "$RBPY" scripts/rollback_update.py "$@"
fi

echo "No Python found to run the rollback. Restore the previous version by hand:" >&2
echo "copy the contents of home/updates/backup back into $(pwd)." >&2
exit 1
