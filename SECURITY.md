# Security Policy

## Reporting a vulnerability

Please report security issues privately using GitHub's **"Report a vulnerability"**
button (the repository's *Security* tab -> *Advisories*), rather than opening a
public issue. We will acknowledge the report and work with you on a fix and a
disclosure timeline.

## Authentication model

localm is a local-first, single-owner application. API access is gated by a
**bearer key**:

- When **no key is configured**, the inference API and ordinary reads (model
  listing, health) are **fail-open by design** - the server binds to localhost
  and serves them without auth, for frictionless local use. State-changing
  management routes, and reads of management/metadata endpoints specifically
  (named keys, server config, host stats, the filesystem browser), still
  require the loopback shell token (see *State-changing endpoints* below) -
  that stops a malicious **web page**, not another **local program**, which
  can obtain the token the same way the GUI shell does (see below).
- When a key **is** configured (`LOCALM_API_KEY`), every `/v1` and `/api` route
  requires `Authorization: Bearer <key>`, gated by capability scopes: model-read
  routes (`GET /v1/models`, `GET /v1/models/{id}`) need `models:read`, plugin
  routes their per-plugin scope, key/config/plugin administration their privileged
  scopes (the owner key implies every scope). The sole exception is `GET /health`,
  an unauthenticated liveness probe that returns the status, model name, and load state.

Manage the key from the CLI: `localm key show` / `generate` / `set` / `clear`, and
mint named, scope-limited keys with `localm key create --scope <scope>` - by
default `key create` refuses to mint a privileged scope into a named key (pass
`--allow-privileged` to override, since a terminal on this machine is already
owner-equivalent trust); an owner-authenticated API call (`POST /v1/keys`) can
mint one too (see *Capability scopes* below).

A key you chose yourself (`localm key set`, `LOCALM_API_KEY`, or writing
`auth.key` by hand) can be short or memorable, so its fingerprint - recorded in
`sessions.json` and `jobs.json` to check ownership - uses a slow, salted
derivation (scrypt) rather than a fast hash, so it cannot be brute-forced
offline from those files. A key localm generates for you is random and long
enough that this does not matter, and keeps using the cheap path.

Because the default is fail-open for reads, a network bind without a key is unsafe,
so **both `localm gui` and `localm serve` refuse to bind past loopback unless an API
key is set** (printing how to set one). `--insecure` overrides this for a trusted,
isolated network - it then serves unauthenticated, the GUI's coder agent included. A
network bind also gets built-in TLS automatically (see `docs/tls.md`), so the key and
all traffic are encrypted. Exposing the GUI exposes the coder agent, which can run
shell commands.

### State-changing endpoints

Every state-changing endpoint (`POST`/`PUT`/`PATCH`/`DELETE`) is **same-origin
only** by default, so a web page on another `localhost` port (a dev server, an
npm postinstall page) cannot drive it from your browser. The guard is
allowlist-by-default: it covers plugin data routes (`/api/rag`, `/api/coder`,
...) too, not just key/config/plugin administration, and a new route is
protected the moment it is added. Three groups are exempt from the check itself,
each because it carries its own credential instead: the OpenAI-compatible
inference API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`),
left cross-origin callable so a local app can use it; `/v1/surfaces/*`
(on-demand GUI mount), gated on its own attach token or the owner key; and
`/v1/instances/*` (multi-instance GPU coordination), gated on a
per-coordination token. A configured `"cors_origins"` (or `"*"`) opts specific
origins into cross-origin use for everything else.

When **no key is configured** (open mode), those same state-changing routes also
require a per-process **shell token**, served by `GET /` to whatever passes the
same-origin-or-loopback-`Host` check above. The loopback GUI shell is the intended
recipient, but that check cannot tell it apart from any other local program making
the identical request - so a local non-browser client (curl, a script) needs one
extra, unauthenticated `GET /` to obtain the token, and from there has the same
reach as the shell: mint a key, change config, install a plugin, load a model,
drive the coder agent, index files. This matches open mode's own
frictionless-local-use design (see above); it is not a credential boundary between
the owner and other software already running on the machine. Set a key (`localm
key generate`) if that reach should not be available to other local programs. Reads
and the inference API stay open. A configured `cors_origins` (including `"*"`) is
trusted for cross-origin *reads* of ordinary API responses, but state changes still
need a key or the shell token, so a forged `Origin` header cannot be used as a
management credential. **The one exception is `GET /` itself**: the page that
carries the shell token is served to same-origin requests only, regardless of
`cors_origins` - CORS governs whether another origin may read a response, not
whether the token was safe to put in one, so the token's own delivery is never
covered by that trust.

### Management/metadata reads (open mode)

The same-origin requirement above also applies to `GET` reads of management and
metadata endpoints (named keys, server config, host stats, uploads/conversations/
plugins listings, and the filesystem browser used by the file picker) when no key
is configured: the open-mode shell token those routes accept is bound to the SAME
same-origin-or-allowlist check as a state change, not just to token possession. This
closes a specific gap CORS's own permissive `localhost`/`127.0.0.1` origin policy
would otherwise create: without it, any other program on the machine that can read
the loopback GUI shell's HTML (CORS trusts every `localhost:PORT` origin to read a
matching response) could lift the embedded token and replay it cross-origin against
these routes. `GET /v1/models` and `GET /health` are exempt (unauthenticated by
design, matching the inference API's own cross-origin posture).

## Capability scopes grant host access - only issue keys to trusted clients

Some capabilities reach the host filesystem and process by design, bounded by the
localm process's own permissions rather than a sandbox:

- **`coder:full`** runs shell commands and reads/writes files (the `--scope` glob
  narrows *which* files; `run_shell` is intentionally unscoped). The plain **`coder`**
  scope is restricted - read plus confined file edits within the scope, no shell.
- **`rag`** indexing over the HTTP API is confined to your home folder, the
  working directory, and any folders you explicitly allow, and always refuses
  credential folders (`~/.ssh`, ...) wherever they appear, so an API client
  cannot index-and-read arbitrary system files. The localm data directory is
  NOT specially excluded from that confinement - if it falls within an allowed
  folder, its contents are indexable like any other file, on the reasoning that
  the owner already has direct filesystem access to their own data; this is a
  deliberate choice for a local, single-user tool, not an oversight.
  The `localm rag` CLI is unconfined (a local user can already read their own files).
  A `rag`-scoped key can still read documents under the allowed roots back through a
  query, so issue one only to clients you trust. A key can also be confined to a
  specific folder allowlist at mint time (`rag_roots`) instead of that default
  reach - which REPLACES it rather than narrowing it, so it can point at a folder
  outside the default reach too, and only the owner key may grant it.
- **`image` / `video` / `music`** read a source image for img2img only from the
  uploads folder and the generated-media galleries - never the rest of the data
  directory (which holds your owner key and sessions) and never the localm
  install directory - and a file with no image signature is refused before it is
  uploaded to ComfyUI (which may be another machine, over plain http). Moving a
  generated file OUT of the data directory needs host filesystem access, the
  same dial as the folder picker that chooses the destination. The same rule
  covers the `generate_image` MCP tool, which reaches the same upload.
  Known limit, if you issue several media keys: those source folders are
  SHARED, so one media key can use another's generated image as an img2img
  source. The uploads inbox has no per-key ownership at all, so this is a
  property of the folders rather than of one route. It is narrower than it
  looks - the folders hold generated media and files you uploaded, not your
  keys or sessions - but if that matters to you, issue one media key.
- **Host filesystem access** is a separate dial, and it gates the model and
  media routes that can name a path on the server: pulling a model by naming a
  path that already exists there, scanning for ComfyUI models, downloading a
  curated ComfyUI model, and moving a generated media file out of the data
  directory (see above). Without it, a key holding those scopes is confined to
  HuggingFace-by-name pulls and the folders localm already manages. It is a
  per-route gate on those routes, not a blanket property of every scope.
- **Installing a plugin** (a store name, or a third-party directory via
  `localm plugin install <path>` / `POST /api/plugins/install-external`)
  refuses a source tree containing any symlink or Windows junction, so a
  malicious plugin source cannot smuggle a file's contents out through a link
  or drive a large copy through a self-referencing cycle.
- **`config:write` / `plugins:admin` / `keys:admin` / `coder:full` / `admin`**
  are privileged and are never granted implicitly: an owner-authenticated
  `POST /v1/keys` call, or `localm key create --allow-privileged` from this
  machine's terminal, must ask for one deliberately.

These are deliberate grants to *you*: a scoped key (or an exposed GUI) grants its
holder that capability on your machine, so only issue keys to - or expose the GUI to -
clients you trust.

## Model trust boundaries

- **A model name from the API, an MCP client, or a scheduled job must be one
  you have already registered** - it is never treated as a filesystem path, so
  a request cannot point the server at an arbitrary folder (which, for a
  HuggingFace-format model, would otherwise run that folder's own bundled
  Python unconditionally). Naming a model straight from a path on the command
  line (`localm run D:\models\foo.gguf`, `localm gui <path>`, `localm mcp
  --model <path>`) is unchanged and still allowed - you typed it yourself.
- **A model's own bundled code does not run just because you loaded it.**
  Custom model code (`trust_remote_code`) is off by default; a model that
  needs it is refused with an explanation, and the owner-only "Allow
  model-bundled custom code" setting (Settings -> Security) turns it back on
  for a model you trust.
- **A downloaded model or vision-projector filename** cannot resolve to
  something other than the plain file it appears to be: a repo-supplied name
  is rejected if it contains a colon (which can open a hidden Windows
  alternate-data-stream), matches an 8.3 short-name alias for an unrelated
  file you already have, names a reserved Windows device, or ends in a dot or
  space (which Windows silently strips). This applies to `localm pull`, the
  same-repo vision-projector auto-attach, and `--mmproj`.

## Outbound network policy

localm is offline-first, and the paths that CAN reach the network run through
one policy choke point (`netpolicy.check_url`). The full model, including the
domain lists and the mode semantics, is in [docs/network.md](docs/network.md);
this section states the security properties and, more importantly, their edges.

- **What it covers.** The coder's and chat's web access, HuggingFace model
  discovery, and model pulls all go through the policy. It is not only a
  model-facing guard.
- **What it does not cover.** Several outbound paths deliberately do not use
  it, including bug-report upload and requests to your ComfyUI instance.
  ComfyUI has its own, narrower guards instead: a configured `comfy_api_url`
  that targets a link-local / cloud-metadata address is refused
  (CHK-COMFY-APIURL), and the connection itself refuses any HTTP redirect
  outright (CHK-COMFY-REDIRECT), so a hostile or compromised ComfyUI cannot use
  a 3xx response to steer the request elsewhere. Loopback and LAN are both
  normal, unchecked ComfyUI deployments - treat the policy as governing the
  paths named above, not as a blanket statement about every socket localm
  opens.
- **No redirect off HTTPS, on any of them.** Separate from the policy above and
  narrower: every outbound client that uses localm's shared verified opener
  (setup-llama's runtime download and its GitHub and PyPI lookups, the update
  check and download, the issues list, the bug-report upload) refuses a redirect
  that leaves HTTPS for a weaker scheme. Verifying the first hop's certificate
  says nothing about the hops after it, and a redirect target is chosen by the
  server, after any check on the URL you configured has already run. This is a
  transport guarantee only: it says the bytes stay encrypted in transit, not
  that the host they came from was policy-checked.
- **`off` is the meaningful setting.** At the policy layer `ask` and `allow`
  are the same thing: the only mode branch that refuses is `off`. The prompt
  you see for `ask` comes from the coder's own confirmation step, one layer up,
  and an auto-approve session does not show it. Do not read `ask` as a
  guarantee that something will stop and ask.
- **`off` has one documented exception.** An admin-only setting
  (`update_ignore_net_policy`, off by default) lets the update check run
  regardless. Nothing else opts out.
- **Private-address guard.** Requests to loopback, link-local, CGNAT and
  private ranges are refused, and the check is re-applied to the resolved
  address rather than the name. It classifies by ADDRESS TYPE, so a service
  reachable on a globally-routable address is not "internal" to this guard.
  Setting `net_allow_private` true removes both the pre-flight check and the
  pin-time re-check, not just the former.
- **DNS-rebinding pin.** A permitted request is pinned to the address that was
  validated, so a name cannot resolve to something else between the check and
  the connection. The pinned session also disables environment trust, so a
  proxy environment variable cannot route the connection somewhere the pin
  never saw, and `.netrc` credentials are not auto-attached to a request the
  caller never asked to authenticate. The pin applies to sessions built for
  this purpose, not to every HTTP client in the process.
- **Redirects.** Page fetches and model pulls re-validate each hop, so a
  permitted URL cannot redirect its way to a refused one. That re-validation is
  not present on every network path.
- **Domain allow and deny lists.** These are read from config per call. If that
  read fails, they are dropped for that call with a warning rather than failing
  closed, so a denied host would pass.
- **Response size.** Fetches are capped, but the cap is a default that callers
  may raise; it is not a fixed ceiling.

**What this is not.** The policy decides whether a request may be made. It says
nothing about whether the content that comes back is trustworthy. Fetched pages
and search snippets are untrusted input to the model, and
[docs/network.md](docs/network.md) is explicit about that.

## Transport security on a network bind

- **Automatic past loopback, not before it.** A bind beyond loopback generates a
  local certificate authority and a leaf certificate and serves HTTPS. A default
  loopback bind is plain HTTP and generates no certificate at all, which is why
  a normal local install has none.
- **What the certificate covers.** The SANs are built from this host's own
  addresses and its primary LAN address. Reaching localm over a VPN or overlay
  network may therefore land on an address the certificate does not name.
  Address enumeration also depends on the optional `[monitor]` extra; without
  it, that set is empty.
- **Regeneration.** The leaf is regenerated when it no longer covers a required
  name, not on every address change.
- **Key file permissions.** The private key is written with owner-only
  permissions on POSIX. On Windows that call does no filtering, so the key
  inherits the directory's own permissions; treat the data directory's access
  control as the real boundary there.
- **Trusting the CA.** Clients need the generated CA to validate the connection.
  `GET /localm-ca.crt` serves it, and that route is deliberately public.
- **A self-call caveat.** When the CA file is missing, localm's own internal
  calls fall back to not verifying rather than failing.

Setup, distribution to phones and browsers, and reverse-proxy alternatives are
in [docs/tls.md](docs/tls.md).

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
rather than brick itself. The GUI "Update now" button's restart (the only
transition that happens with nobody watching) is followed by a detached watchdog
that polls the relaunched build's `/whoami` for the expected version and
auto-rolls-back if it does not come up healthy within 90 seconds. `localm update`
from the CLI applies the same way but does not restart for you - you relaunch by
hand, so a build that misbehaves after that manual restart has no automatic
watchdog and is recovered with `localm update --rollback` (or by restoring the
backup directory the update left behind).

## Supported versions

localm is pre-1.0; security fixes land on the latest `master`.
