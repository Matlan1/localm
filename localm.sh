#!/usr/bin/env bash
# localm CLI (Linux / macOS) - a passthrough to this clone's own venv.
# Examples:  ./localm.sh gui     ./localm.sh run <model>     ./localm.sh pull <spec>
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
if [ ! -x .venv/bin/localm ]; then
  echo "The .venv exists but its localm CLI entry point is missing. Run setup" >&2
  echo "again and pick [1] Install again / repair:  bash setup.sh" >&2
  exit 1
fi
exec .venv/bin/localm "$@"
