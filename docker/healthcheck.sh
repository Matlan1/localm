#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Container healthcheck: /whoami answers without a key and without a loaded
# model. Tries HTTPS (the default past loopback) and then plain HTTP (--no-tls).
PORT="${LOCALM_CONTAINER_PORT:-8642}"
curl -fsk --max-time 4 "https://127.0.0.1:${PORT}/whoami" >/dev/null 2>&1 \
  || curl -fs --max-time 4 "http://127.0.0.1:${PORT}/whoami" >/dev/null
