#!/usr/bin/env bash
# localm graphical launcher (Linux / macOS) - opens the tkinter launcher, which
# lets you pick a mode (GUI / chat / server / coder), a model, and options.
# It runs on this clone's own .venv Python, whose managed runtime includes Tk.
# Inherited uv / pip / Python settings and LOCALM_HOME are cleared first, so
# the install's own data folder (localm-home.cfg or ./home) is the one used.
for _var in $(compgen -e); do
  case "$_var" in UV_*|PIP_*) unset "$_var" ;; esac
done
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX LOCALM_HOME
export UV_NO_CONFIG=1 PIP_CONFIG_FILE=/dev/null PYTHONNOUSERSITE=1
case "$0" in */*) cd -- "${0%/*}" || exit 1 ;; esac
if [ ! -x .venv/bin/python ]; then
  echo "No .venv found. Run setup first:  bash setup.sh" >&2
  exit 1
fi
exec .venv/bin/python launcher.pyw "$@"
