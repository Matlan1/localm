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
  requires `Authorization: Bearer <key>`, and plugin routes are additionally gated
  by per-plugin capability scopes.

Because the default is fail-open, **do not bind localm to a non-localhost interface
without setting an API key** (and ideally TLS - see `docs/tls.md`). Exposing the GUI
also exposes the coder agent, which can run shell commands.

## Supported versions

localm is pre-1.0; security fixes land on the latest `master`.
