#!/usr/bin/env bash
# localm graphical launcher (Linux / macOS) - opens the tkinter launcher, which
# lets you pick a mode (GUI / chat / server / coder), a model, and options.
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ ! -x "$HERE/.venv/bin/python" ]; then
  echo "No .venv found in $HERE. Run:  bash \"$HERE/setup.sh\"" >&2
  exit 1
fi
if ! "$HERE/.venv/bin/python" -c "import tkinter" >/dev/null 2>&1; then
  echo "The graphical launcher needs Tk, which this Python does not have." >&2
  echo "Install it (Debian/Ubuntu: sudo apt install python3-tk; Fedora:" >&2
  echo "sudo dnf install python3-tkinter; macOS: brew install python-tk), or" >&2
  echo "start LocaLM without the launcher:  \"$HERE/localm.sh\" gui" >&2
  exit 1
fi
cd "$HERE" || exit 1
exec "$HERE/.venv/bin/python" "$HERE/launcher.pyw" "$@"
