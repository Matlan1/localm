# Security Policy

## Reporting a vulnerability

Please report security issues privately using GitHub's **"Report a vulnerability"**
button (the repository's *Security* tab -> *Advisories*), rather than opening a
public issue. We will acknowledge the report and work with you on a fix and a
disclosure timeline.

## Authentication model

localm is a local-first, single-owner application. API access is gated by a
**bearer key**:

- When **no key is configured**, reads and the inference API are **fail-open by
  design** - the server binds to localhost and serves them without auth, for
  frictionless local use. State-changing management routes still require the
  loopback shell token (see *State-changing endpoints* below), so another local
  program cannot silently drive a keyless install.
- When a key **is** configured (`LOCALM_API_KEY`), every `/v1` and `/api` route
  requires `Authorization: Bearer <key>`, gated by capability scopes: model-read
  routes (`GET /v1/models`, `GET /v1/models/{id}`) need `models:read`, plugin
  routes their per-plugin scope, key/config/plugin administration their privileged
  scopes (the owner key implies every scope). The sole exception is `GET /health`,
  an unauthenticated liveness probe that returns the status, model name, and load state.

Manage the key from the CLI: `localm key show` / `generate` / `set` / `clear`, and
mint named, scope-limited keys with `localm key create --scope <scope>` (privileged
scopes are never minted into a named key).

Because the default is fail-open for reads, a network bind without a key is unsafe,
so **both `localm gui` and `localm serve` refuse to bind past loopback unless an API
key is set** (printing how to set one). `--insecure` overrides this for a trusted,
isolated network - it then serves unauthenticated, the GUI's coder agent included. A
network bind also gets built-in TLS automatically (see `docs/tls.md`), so the key and
all traffic are encrypted. Exposing the GUI exposes the coder agent, which can run
shell commands.

### State-changing endpoints

Every state-changing endpoint (`POST`/`PUT`/`PATCH`/`DELETE`) except the
OpenAI-compatible inference API (`/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings`) is **same-origin only** by default, so a web page on another
`localhost` port (a dev server, an npm postinstall page) cannot drive it from your
browser. The guard is allowlist-by-default: it covers plugin data routes
(`/api/rag`, `/api/coder`, ...) too, not just key/config/plugin administration, and
a new route is protected the moment it is added. The inference API stays
cross-origin callable so a local app can use it; `"cors_origins"` opts specific
origins (or `"*"`) into cross-origin use.

When **no key is configured** (open mode), those same state-changing routes also
require a per-process **shell token** that only the loopback GUI shell carries. So a
local non-browser client (curl, a script) cannot mint a key, change config, install
a plugin, load a model, drive the coder agent, or index files in open mode: manage
through the loopback GUI, or set a key (`localm key generate`). Reads and the
inference API stay open. A configured `cors_origins` is trusted for cross-origin
*reads*, but state changes still need a key or the shell token, so a forged `Origin`
header cannot be used as a management credential.

## Capability scopes grant host access - only issue keys to trusted clients

Some capabilities reach the host filesystem and process by design, bounded by the
localm process's own permissions rather than a sandbox:

- **`coder:full`** runs shell commands and reads/writes files (the `--scope` glob
  narrows *which* files; `run_shell` is intentionally unscoped). The plain **`coder`**
  scope is restricted - read plus confined file edits within the scope, no shell.
- **`rag`** indexing over the HTTP API is confined to your home folder and the
  working directory and refuses the localm data dir and credential folders
  (`~/.ssh`, ...), so an API client cannot index-and-read arbitrary system files.
  The `localm rag` CLI is unconfined (a local user can already read their own files).
  A `rag`-scoped key can still read documents under the allowed roots back through a
  query, so issue one only to clients you trust.
- **`config:write` / `plugins:admin` / `keys:admin`** are privileged and are never
  granted implicitly - only the owner key may mint keys carrying them.

These are deliberate grants to *you*: a scoped key (or an exposed GUI) grants its
holder that capability on your machine, so only issue keys to - or expose the GUI to -
clients you trust.

## Software updates

localm never updates itself. An update runs only when you initiate it (`localm
update`, or the GUI "Update now" button). The client is signature-verifying:

- **Signed builds.** A downloaded build is verified against an **Ed25519** public
  key **pinned in localm's own source** before anything is extracted or executed.
  A missing, invalid, or tampered signature is refused before any file is swapped
  (fail closed). The pinned key is a list, so a key can be rotated in before an
  old one retires.
- **No downgrades.** A validly signed but older, equal, or version-less build is
  refused: a signature proves authenticity, not freshness.
- **HTTPS only.** The download endpoint must be HTTPS and a redirect that would
  downgrade to plain HTTP is blocked.
- **Your data is never in the swap.** The updater never touches the venv, `.git`,
  your data directory (models, config, sessions), or `.localcoder`; provisioned
  native binaries (the llama.cpp runtime) are preserved across the swap. It backs
  up first and rolls back on failure rather than leaving a half-applied tree.

Honest limits: signature verification enforces only while a key is pinned. Shipped
builds do pin one (a test keeps the pin non-empty), but if the pin were ever empty
the updater would fall back to transport trust (HTTPS plus the private channel)
rather than brick itself. And there is no post-restart health check: a build that
applies cleanly but misbehaves after restart is recovered with `localm update
--rollback` (or by restoring the backup directory the update left behind), not
automatically.

## Supported versions

localm is pre-1.0; security fixes land on the latest `master`.
