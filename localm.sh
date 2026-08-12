#!/usr/bin/env bash
# localm CLI (Linux / macOS) - a passthrough to this clone's own venv.
# Examples:  ./localm.sh gui     ./localm.sh run <model>     ./localm.sh pull <spec>
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "No .venv found. Run:  bash setup.sh" >&2
  exit 1
fi
if [ ! -x .venv/bin/localm ]; then
  echo "The .venv exists but its localm CLI entry point is missing (a known" >&2
  echo "setup.sh install quirk on some filesystems). Reinstall it with:" >&2
  echo "  uv pip install -p .venv -e \".[coder,voice,monitor]\" --reinstall" >&2
  exit 1
fi
exec .venv/bin/localm "$@"
