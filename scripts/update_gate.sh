#!/usr/bin/env bash
# Waits for an update of LocaLM that is running, or puts the previous version back after
# one that was interrupted. The launchers and setup call this, while there is an
# unfinished update, before anything of LocaLM runs. Exit status 0: LocaLM may start;
# anything else: it may not, and the reason was shown.
# The Python is the one in .venv, then the one in .python, then the one .venv was made
# from; never one from PATH. With the argument "window" it says everything in a window
# (update_gate.pyw) where there is a desktop and no terminal.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN=localm/__init__.py
if [ "${1:-}" = window ] && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] \
    && [ ! -t 0 ] && [ ! -t 1 ] && [ ! -t 2 ]; then
  RUN=scripts/update_gate.pyw
fi
made_from="$(sed -n 's/^home *= *//p' "$ROOT/.venv/pyvenv.cfg" 2>/dev/null)"
for py in "$ROOT/.venv/bin/python" "$ROOT"/.python/*/bin/python3 \
          "${made_from:-$ROOT/.venv}/python3.12" "${made_from:-$ROOT/.venv}/python3"; do
  if [ -x "$py" ] \
      && "$py" -I -S -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >/dev/null 2>&1; then
    exec "$py" -I -S "$ROOT/$RUN"
  fi
done
echo "An update of LocaLM was interrupted and no Python is left in $ROOT" >&2
echo "that can undo it. Download LocaLM, unzip it over that folder, delete everything in" >&2
echo "its folder .localm-update except kept, which has files of your own, and run setup." >&2
echo "Your chats, settings and models stay." >&2
exit 4
