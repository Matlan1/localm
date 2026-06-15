#!/usr/bin/env bash
# localm CLI (Linux / macOS) - a passthrough to this clone's own venv.
# Examples:  ./localm.sh gui     ./localm.sh run <model>     ./localm.sh pull <spec>
cd "$(dirname "$0")"
if [ ! -x .venv/bin/localm ]; then
  echo "No .venv found. Run:  bash setup.sh" >&2
  exit 1
fi
exec .venv/bin/localm "$@"
