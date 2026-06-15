#!/usr/bin/env bash
# localm graphical launcher (Linux / macOS) - opens the tkinter launcher, which
# lets you pick a mode (GUI / chat / server / coder), a model, and options.
# Needs Tk: `sudo apt install python3-tk` (or your distro's equivalent).
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "No .venv found. Run:  bash setup.sh" >&2
  exit 1
fi
exec .venv/bin/python launcher.pyw "$@"
