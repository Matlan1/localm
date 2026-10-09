# SPDX-License-Identifier: AGPL-3.0-or-later
#
# localm server image. Build from the repository root:
#
#   docker build -f docker/Dockerfile --build-arg BACKEND=cpu    -t localm:cpu .
#   docker build -f docker/Dockerfile --build-arg BACKEND=vulkan -t localm:vulkan .
#
# BACKEND is the llama.cpp runtime `localm setup-llama` provisions into the image:
# cpu or vulkan. The runtime is part of the image; models, settings, API keys and
# TLS certificates live in /data, which is meant to be a volume.

ARG UBUNTU_IMAGE=ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55
FROM ${UBUNTU_IMAGE}

ARG BACKEND=cpu
ARG EXTRAS=coder,monitor,rag
ARG UV_VERSION=0.12.24
ARG UV_SHA256=b4dfaef47d491a7296981f8374a4595f55dbf84e8937c8ecd2983574d8bb3da6
ARG IMAGE_VERSION=unknown
ARG IMAGE_REVISION=unknown

LABEL org.opencontainers.image.title="localm" \
      org.opencontainers.image.description="localm: an offline local-LLM inference server (OpenAI-compatible API)" \
      org.opencontainers.image.source="https://github.com/Matlan1/localm" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      io.localm.backend="${BACKEND}"

ENV DEBIAN_FRONTEND=noninteractive

RUN case "${BACKEND}" in cpu|vulkan) ;; *) echo "BACKEND must be cpu or vulkan, got '${BACKEND}'" >&2; exit 1 ;; esac \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl tini \
 && if [ "${BACKEND}" = "vulkan" ]; then \
      apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers; \
    fi \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/uv.tar.gz \
      "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
 && echo "${UV_SHA256}  /tmp/uv.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/uv.tar.gz -C /usr/local/bin --strip-components=1 uv-x86_64-unknown-linux-gnu/uv \
 && rm /tmp/uv.tar.gz

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin localm \
 && mkdir /opt/localm /opt/venv /opt/python /data \
 && chown localm:localm /opt/localm /opt/venv /opt/python /data

USER localm

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_NO_CACHE=1

COPY --chown=localm:localm pyproject.toml VERSION /opt/localm/
COPY --chown=localm:localm docs/pypi-readme.md /opt/localm/docs/pypi-readme.md
COPY --chown=localm:localm localm /opt/localm/localm
COPY --chown=localm:localm runtime /opt/localm/runtime

RUN uv venv --python 3.12 /opt/venv \
 && uv pip install --python /opt/venv/bin/python -e "/opt/localm[${EXTRAS}]" -e /opt/localm/runtime

RUN LOCALM_HOME=/tmp/localm-build-home /opt/venv/bin/localm setup-llama --backend "${BACKEND}" --yes \
 && rm -rf /tmp/localm-build-home

COPY --chown=root:root --chmod=0755 docker/entrypoint.sh /usr/local/bin/localm-docker-entrypoint
COPY --chown=root:root --chmod=0755 docker/healthcheck.sh /usr/local/bin/localm-docker-healthcheck

ENV PATH=/opt/venv/bin:${PATH} \
    LOCALM_HOME=/data \
    LOCALM_CONTAINER_PORT=8642 \
    PYTHONUNBUFFERED=1

WORKDIR /data
VOLUME ["/data"]
EXPOSE 8642

HEALTHCHECK --interval=30s --timeout=6s --start-period=60s --retries=3 \
  CMD ["/usr/local/bin/localm-docker-healthcheck"]

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/localm-docker-entrypoint"]
CMD ["serve"]
