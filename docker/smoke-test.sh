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
# For a cuda or cuda13 image, which cannot be given a GPU here, it also checks that
#   - the CUDA build, the CUDA runtime libraries and the staged-runtime record are
#     in the image, for the right CUDA line,
#   - the container refuses to serve without a GPU and says why,
#   - `localm doctor` reports the runtime as fetched without a GPU,
# and runs the checks above with LOCALM_ALLOW_NO_GPU=1.
set -euo pipefail

IMAGE="${1:?usage: docker/smoke-test.sh IMAGE}"
NAME="localm-smoke-$$"
VOLUME="${NAME}-data"

BACKEND_LABEL="$(docker image inspect --format '{{index .Config.Labels "io.localm.backend"}}' "$IMAGE")"
GPU_ARGS=()
case "$BACKEND_LABEL" in
  cuda|cuda13) GPU_ARGS=(-e LOCALM_ALLOW_NO_GPU=1) ;;
esac

cleanup() {
  docker rm -fv "$NAME" "${NAME}-env" "${NAME}-refuse" "${NAME}-nogpu" >/dev/null 2>&1 || true
  docker volume rm "$VOLUME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  for c in "$NAME" "${NAME}-env" "${NAME}-refuse" "${NAME}-nogpu"; do
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
docker run -d --name "$NAME" "${GPU_ARGS[@]}" -v "${VOLUME}:/data" -p 127.0.0.1::8642 "$IMAGE" >/dev/null
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
docker run -d --name "${NAME}-env" "${GPU_ARGS[@]}" -e "LOCALM_API_KEY=${ENV_KEY}" -p 127.0.0.1::8642 "$IMAGE" >/dev/null
ENV_PORT="$(published_port "${NAME}-env")"
wait_for_server "${NAME}-env" "$ENV_PORT"
code="$(http_code "https://127.0.0.1:${ENV_PORT}/v1/models")"
[ "$code" = "401" ] || fail "env key, no header: expected 401, got $code"
code="$(http_code -H "Authorization: Bearer ${ENV_KEY}" "https://127.0.0.1:${ENV_PORT}/v1/models")"
[ "$code" = "200" ] || fail "env key: expected 200, got $code"

case "$BACKEND_LABEL" in
  cuda|cuda13)
    if [ "$BACKEND_LABEL" = cuda ]; then LINE=cuda-12; else LINE=cuda-13; fi

    echo "== ${BACKEND_LABEL}: the CUDA build, runtime libraries and staged record are in the image"
    inventory="$(docker run --rm --entrypoint python "$IMAGE" -c '
import localm_llama_runtime
from pathlib import Path
lib = Path(localm_llama_runtime.LIB_DIR)
print("files:" + " ".join(sorted(p.name for p in lib.iterdir())))
print("backend:" + (lib / ".localm-backend").read_text().split()[0])
print("staged:" + (lib / ".localm-cuda-staged").read_text().strip())
')" || fail "could not read the runtime directory of ${IMAGE}"
    grep -q '^backend:cuda$' <<<"$inventory" || fail "the runtime is not recorded as cuda: $inventory"
    grep -q "^staged:${LINE}\$" <<<"$inventory" || fail "the staged line is not ${LINE}: $inventory"
    for lib in libggml-cuda.so libcudart.so libcublas.so libcublasLt.so; do
      grep -q "^files:.*[ ]${lib}" <<<"$inventory" || fail "${lib} is not in the runtime directory: $inventory"
    done

    echo "== ${BACKEND_LABEL}: serving without a GPU is refused and says why"
    set +e
    nogpu="$(timeout 120 docker run --name "${NAME}-nogpu" -e LOCALM_API_KEY=smoke-nogpu-key-12345 "$IMAGE" 2>&1)"
    rc=$?
    set -e
    [ "$rc" -eq 3 ] || fail "expected exit code 3 without a GPU, got $rc: $nogpu"
    grep -q "no NVIDIA GPU is visible" <<<"$nogpu" || fail "the refusal did not name the missing GPU: $nogpu"
    grep -q "gpus all" <<<"$nogpu" || fail "the refusal did not say how to give the container a GPU: $nogpu"

    echo "== ${BACKEND_LABEL}: localm doctor reports the runtime as not load-tested at install"
    doctor="$(timeout 300 docker run --rm "${GPU_ARGS[@]}" "$IMAGE" doctor 2>&1)" || true
    grep -q "fetched without a GPU" <<<"$doctor" || fail "doctor did not report the staged runtime: $doctor"
    grep -q "${LINE}" <<<"$doctor" || fail "doctor did not name the ${LINE} line: $doctor"
    ;;
esac

echo "smoke test passed for ${IMAGE}"
