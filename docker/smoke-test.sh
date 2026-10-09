#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Smoke test for a built localm image:  docker/smoke-test.sh IMAGE
#
# Checks that the container
#   - refuses to start without an API key,
#   - accepts a key created in the data volume with `key generate`,
#   - answers /v1/models with that key and refuses it without (or with a wrong) key,
#   - runs as a non-root user and reports healthy,
#   - accepts a key passed in LOCALM_API_KEY.
set -euo pipefail

IMAGE="${1:?usage: docker/smoke-test.sh IMAGE}"
NAME="localm-smoke-$$"
VOLUME="${NAME}-data"

cleanup() {
  docker rm -f "$NAME" "${NAME}-env" "${NAME}-refuse" >/dev/null 2>&1 || true
  docker volume rm "$VOLUME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  for c in "$NAME" "${NAME}-env" "${NAME}-refuse"; do
    if docker inspect "$c" >/dev/null 2>&1; then
      echo "---- logs of $c ----" >&2
      docker logs "$c" >&2 || true
    fi
  done
  exit 1
}

http_code() {
  curl -sk --max-time 10 -o /dev/null -w '%{http_code}' "$@" || true
}

wait_for_server() {
  local container="$1" port="$2" deadline=$((SECONDS + 150))
  until curl -fsk --max-time 5 "https://127.0.0.1:${port}/whoami" >/dev/null 2>&1; do
    if [ "$(docker inspect --format '{{.State.Running}}' "$container")" != "true" ]; then
      fail "container $container exited before it answered"
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      fail "container $container did not answer /whoami within 150 s"
    fi
    sleep 2
  done
}

published_port() {
  docker port "$1" 8642/tcp | head -n 1 | sed 's/.*://'
}

echo "== no key: the container must refuse to start"
set +e
refusal="$(timeout 90 docker run --name "${NAME}-refuse" "$IMAGE" 2>&1)"
rc=$?
set -e
[ "$rc" -eq 2 ] || fail "expected exit code 2 without a key, got $rc: $refusal"
grep -q "needs" <<<"$refusal" && grep -q "API key" <<<"$refusal" \
  || fail "the refusal did not explain the missing key: $refusal"

echo "== key generate: the key is created in the data volume"
generated="$(docker run --rm -v "${VOLUME}:/data" "$IMAGE" key generate)"
KEY="$(printf '%s\n' "$generated" | sed -n '/New API key/{n;s/^[[:space:]]*//;s/[[:space:]]*$//;p;}')"
[[ "$KEY" =~ ^[A-Za-z0-9_-]{8,}$ ]] || fail "could not read the generated key from: $generated"

echo "== start with the volume"
docker run -d --name "$NAME" -v "${VOLUME}:/data" -p 127.0.0.1::8642 "$IMAGE" >/dev/null
PORT="$(published_port "$NAME")"
wait_for_server "$NAME" "$PORT"

echo "== /v1/models without and with a key"
BASE="https://127.0.0.1:${PORT}"
code="$(http_code "${BASE}/v1/models")"
[ "$code" = "401" ] || fail "no key: expected 401, got $code"
code="$(http_code -H "Authorization: Bearer wrong-key-${RANDOM}" "${BASE}/v1/models")"
[ "$code" = "401" ] || fail "wrong key: expected 401, got $code"
code="$(http_code -H "Authorization: Bearer ${KEY}" "${BASE}/v1/models")"
[ "$code" = "200" ] || fail "right key: expected 200, got $code"
body="$(curl -sk --max-time 10 -H "Authorization: Bearer ${KEY}" "${BASE}/v1/models")"
grep -q '"data"' <<<"$body" || fail "/v1/models did not return a model list: $body"

echo "== plain HTTP is redirected to HTTPS, not served"
code="$(http_code "http://127.0.0.1:${PORT}/v1/models")"
[ "$code" = "308" ] || fail "plain HTTP: expected a 308 redirect, got $code"

echo "== non-root, certificate in the volume, healthy"
uid="$(docker exec "$NAME" id -u)"
[ "$uid" != "0" ] || fail "the server runs as root"
docker exec "$NAME" test -f /data/tls/ca.crt || fail "no CA certificate in /data/tls"
deadline=$((SECONDS + 150))
until [ "$(docker inspect --format '{{.State.Health.Status}}' "$NAME")" = "healthy" ]; do
  [ "$SECONDS" -lt "$deadline" ] || fail "container did not become healthy"
  sleep 3
done

echo "== LOCALM_API_KEY"
ENV_KEY="smoke-env-key-${RANDOM}${RANDOM}"
docker run -d --name "${NAME}-env" -e "LOCALM_API_KEY=${ENV_KEY}" -p 127.0.0.1::8642 "$IMAGE" >/dev/null
ENV_PORT="$(published_port "${NAME}-env")"
wait_for_server "${NAME}-env" "$ENV_PORT"
code="$(http_code "https://127.0.0.1:${ENV_PORT}/v1/models")"
[ "$code" = "401" ] || fail "env key, no header: expected 401, got $code"
code="$(http_code -H "Authorization: Bearer ${ENV_KEY}" "https://127.0.0.1:${ENV_PORT}/v1/models")"
[ "$code" = "200" ] || fail "env key: expected 200, got $code"

echo "smoke test passed for ${IMAGE}"
