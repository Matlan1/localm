#!/usr/bin/env bash
# localm CLI (Linux / macOS) - a passthrough to this clone's own venv.
# Examples:  ./localm.sh gui     ./localm.sh run <model>     ./localm.sh pull <spec>
# The current directory is left as it is, so a relative path argument means the
# folder it was typed in.
HERE="$(cd "$(dirname "$0")" && pwd)"
# An update that is not finished may replace this file: it is then started anew.
if [ -e "$HERE/.localm-update/job.json" ] && [ -z "${LOCALM_GATE:-}" ]; then
  bash "$HERE/scripts/update_gate.sh" || exit 1
  LOCALM_GATE=1 exec bash "$HERE/localm.sh" "$@"
fi
unset LOCALM_GATE
if [ ! -x "$HERE/.venv/bin/python" ]; then
  echo "No .venv found in $HERE. Run:  bash \"$HERE/setup.sh\"" >&2
  exit 1
fi
if [ ! -x "$HERE/.venv/bin/localm" ]; then
  echo "The .venv exists but its localm CLI entry point is missing (a known" >&2
  echo "setup.sh install quirk on some filesystems). Repair the install with:" >&2
  echo "  bash \"$HERE/setup.sh\"    and pick [1] Install again / repair" >&2
  exit 1
fi
exec "$HERE/.venv/bin/localm" "$@"
