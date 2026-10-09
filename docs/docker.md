# Running localm in Docker

The image runs `localm serve`: the OpenAI-compatible API server, without the web
GUI. It listens on every interface inside the container, so it **refuses to start
without an API key**, and it serves **HTTPS** with localm's built-in certificate
(see [tls.md](tls.md)).

Images are published to `ghcr.io/matlan1/localm` when a release is published; see
the package page for the tags that exist. You can always build the same image from
a checkout (see [Build it yourself](#build-it-yourself)).

| Tag | Contents |
|---|---|
| `<version>`, `<version>-cpu`, `latest`, `cpu` | llama.cpp CPU runtime |
| `<version>-vulkan`, `vulkan` | llama.cpp Vulkan runtime (Mesa drivers included) |

`latest`, `cpu` and `vulkan` move with each release that is not a pre-release.
Images are `linux/amd64` only (on an arm64 host, build with `--platform linux/amd64`). There is no CUDA image yet.

## First run

Create the API key in a volume. It is printed once:

```bash
docker run --rm -v localm-data:/data ghcr.io/matlan1/localm key generate
```

Download a model into the same volume (any `localm pull` spec works):

```bash
docker run --rm -v localm-data:/data ghcr.io/matlan1/localm pull owner/repo:model.gguf
```

Start the server:

```bash
docker run -d --name localm -v localm-data:/data -p 8642:8642 ghcr.io/matlan1/localm
```

The server loads the first registered chat model at startup. Call it with the key:

```bash
curl -k https://localhost:8642/v1/models -H "Authorization: Bearer <your key>"
```

Without a key, or with a wrong one, the API answers 401. With no key at all the
container exits with code 2 and prints how to create one.

To use a key you already have, pass it instead of creating one in the volume (8 or
more characters; prefer `localm key generate` for a hard-to-guess one):

```bash
docker run -d -v localm-data:/data -p 8642:8642 -e LOCALM_API_KEY=<key> ghcr.io/matlan1/localm
```

Scoped keys for individual clients are created the same way, for example
`docker run --rm -v localm-data:/data ghcr.io/matlan1/localm key create dashboard --scope models:read`.

## Verifying an image

Each published image carries a build provenance attestation that ties its digest to the workflow run and commit that built it:

```bash
gh attestation verify oci://ghcr.io/matlan1/localm:<version> --repo Matlan1/localm
```

## Compose

```yaml
services:
  localm:
    image: ghcr.io/matlan1/localm:latest
    ports:
      - "8642:8642"
    volumes:
      - localm-data:/data
    environment:
      LOCALM_API_KEY: ${LOCALM_API_KEY:?set LOCALM_API_KEY in .env}
volumes:
  localm-data:
```

## What lives where

| Path | Contents |
|---|---|
| `/data` (volume, `LOCALM_HOME`) | models, settings, the API key (`auth.key`), TLS certificates (`tls/`), logs |
| `/opt/localm`, `/opt/venv` | localm and its Python 3.12, installed the same way `setup.sh` installs them; part of the image |

The server runs as user `localm` (uid 10001). A named volume is owned by that user
automatically. For a bind mount, make the directory writable by uid 10001
(`chown 10001 <dir>`) or run with `--user`.

Session persistence defaults to privacy mode (nothing is saved). Change it with
`--mode` or in the settings, as on any other install.

## Certificate

The certificate is signed by a local CA created on first start under
`/data/tls/`. Clients either skip verification (`curl -k`) or trust the CA, which
the server hands out at `https://<host>:8642/localm-ca.crt`. The certificate
covers the names localm can find inside the container, so reaching the container
by another hostname shows a certificate warning even with the CA trusted. Put a
reverse proxy with your own certificate in front for a public name, or pass
`--tls-cert` and `--tls-key` with files mounted into the container.

A plain HTTP request to the port is answered with a redirect to HTTPS. To serve plain HTTP, for example behind a reverse proxy that terminates TLS, add
`--no-tls` after the image name:

```bash
docker run -d -v localm-data:/data -p 8642:8642 ghcr.io/matlan1/localm serve --no-tls
```

The API key is then sent in cleartext between the proxy and the container, so keep
that hop on a private network.

## Options

Arguments after the image name go to `localm serve`, and the server is always
started with `-H 0.0.0.0`. Any other first argument runs as a `localm` command
(`key`, `pull`, `doctor`, `list`, ...). `sh` and `bash` open a shell.

- Another port inside the container: `-e LOCALM_CONTAINER_PORT=9000`, and publish
  that port.
- A specific model: `serve <model-name>`. A context size: `serve -c 8192`.
- `--insecure` serves without a key, to the whole network; use it only on an
  isolated network.

The container reports `healthy` once the server answers.

## GPU

The CPU image runs on any host. The Vulkan image contains the Vulkan loader and
Mesa drivers; to use an AMD or Intel GPU, pass the render device through:

```bash
docker run -d --device /dev/dri -v localm-data:/data -p 8642:8642 ghcr.io/matlan1/localm:vulkan
```

If the render device on your host is restricted to a group, add that group with `--group-add`. The automated tests run the images on CPU-only runners, so GPU use is not covered
by them. `localm doctor` inside the container (`docker exec <container> localm
doctor`) reports which devices the runtime found. NVIDIA GPUs through CUDA are not
supported by these images yet.

## Build it yourself

From the repository root:

```bash
docker build --platform linux/amd64 -f docker/Dockerfile --build-arg BACKEND=cpu -t localm:cpu .
docker build --platform linux/amd64 -f docker/Dockerfile --build-arg BACKEND=vulkan -t localm:vulkan .
```

`bash docker/smoke-test.sh localm:cpu` runs the same checks the project's CI runs
on every image: it refuses to start without a key, `/v1/models` answers 401
without a key and 200 with it, the server runs as a non-root user, and the
container becomes healthy.
