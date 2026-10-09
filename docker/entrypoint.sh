#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Container entrypoint for the localm image.
#
#   (no arguments), serve [ARGS], -ARGS   start the server on 0.0.0.0, refusing
#                                         to start without an API key
#   sh, bash                              run that shell
#   anything else                         run as `localm <arguments>`
#                                         (key generate, pull, doctor, ...)
set -eu

PORT="${LOCALM_CONTAINER_PORT:-8642}"

mode=serve
case "${1:-serve}" in
  serve) if [ "$#" -gt 0 ]; then shift; fi ;;
  -*) ;;
  sh|bash) mode=exec ;;
  *) mode=cli ;;
esac

if [ "$mode" = exec ]; then
  exec "$@"
fi
if [ "$mode" = cli ]; then
  exec localm "$@"
fi

insecure=0
for arg in "$@"; do
  if [ "$arg" = "--insecure" ]; then
    insecure=1
  fi
done

if [ "$insecure" = 0 ]; then
  errfile="$(mktemp)"
  state="$(python -c 'from localm.cli import _exposed_bind_warning as w; print("open" if w("0.0.0.0") else "ok")' 2>"$errfile" | tail -n 1)"
  case "$state" in
    ok) ;;
    open)
      cat >&2 <<'EOF'
localm: refusing to start. This container listens on every interface, so it needs
an API key of at least 8 characters. None is set, or the one that is set is shorter.

Create one in the data volume (printed once), then start the container again:

  docker run --rm -v localm-data:/data ghcr.io/matlan1/localm key generate

or pass your own for this run:

  docker run -e LOCALM_API_KEY=<8+ characters> ...

Pass --insecure to serve without a key on a trusted, isolated network.
EOF
      rm -f "$errfile"
      exit 2
      ;;
    *)
      echo "localm: could not check the API key configuration:" >&2
      cat "$errfile" >&2
      rm -f "$errfile"
      exit 1
      ;;
  esac
  rm -f "$errfile"
fi

exec localm serve -H 0.0.0.0 -p "$PORT" "$@"
