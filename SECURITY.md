# Security Policy

## Reporting a vulnerability

Please report security issues privately using GitHub's **"Report a vulnerability"**
button (the repository's *Security* tab -> *Advisories*), rather than opening a
public issue. We will acknowledge the report and work with you on a fix and a
disclosure timeline.

## Authentication model

localm is a local-first, single-owner application. API access is gated by a
**bearer key** (see `localm/inference/http_server.py`):

- When **no key is configured**, the server is **fail-open by design** - it binds
  to localhost and serves requests without auth, for frictionless local use.
- When a key **is** configured (`LOCALM_API_KEY`), every `/v1` and `/api` route
  requires `Authorization: Bearer <key>`, gated by capability scopes: model-read
  routes (`GET /v1/models`, `GET /v1/models/{id}`) need `models:read`, plugin
  routes their per-plugin scope, key/config/plugin administration their privileged
  scopes (the owner key implies every scope). The sole exception is `GET /health`,
  an unauthenticated liveness probe that returns only the model name and load state.

Because the default is fail-open, **do not bind localm to a non-localhost interface
without setting an API key** (and ideally TLS - see `docs/tls.md`). Exposing the GUI
also exposes the coder agent, which can run shell commands.

State-changing endpoints (`POST`/`PUT`/`PATCH`/`DELETE`) additionally require the
request to be same-origin (or an explicitly configured `cors_origins`), so a web page
on another `localhost` port cannot drive them from your browser. Non-browser clients
(the CLI and SDKs) send no `Origin` and are unaffected.

## Capability scopes grant host access - only issue keys to trusted clients

Some capabilities reach the host filesystem and process by design, bounded by the
localm process's own permissions rather than a sandbox:

- **`coder`** runs shell commands and reads/writes files (the `--scope` glob narrows
  *which* files; `run_shell` is intentionally unscoped).
- **`rag`** indexing reads any file the localm process can read (you point it at your
  documents); a `rag`-scoped key can therefore read server-readable files back through
  a query.
- **`config:write` / `plugins:admin` / `keys:admin`** are privileged and are never
  granted implicitly - only the owner key may mint keys carrying them.

localm is single-owner and local-first: these are deliberate grants to *you*. Treat a
scoped key (or an exposed GUI) as granting the holder that capability on your machine,
and only issue keys to - or expose the GUI to - clients you trust.

## Supported versions

localm is pre-1.0; security fixes land on the latest `master`.
