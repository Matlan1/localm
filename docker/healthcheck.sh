#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Container healthcheck: healthy when /whoami answers 200, which needs neither a
# key nor a loaded model. Tries HTTPS (the default past loopback), then plain HTTP
# (--no-tls). Only a 200 counts; the redirect a TLS server gives plain HTTP does not.
PORT="${LOCALM_CONTAINER_PORT:-8642}"
for url in "https://127.0.0.1:${PORT}/whoami" "http://127.0.0.1:${PORT}/whoami"; do
  code="$(curl -sk --max-time 4 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || true)"
  if [ "$code" = "200" ]; then
    exit 0
  fi
done
exit 1
