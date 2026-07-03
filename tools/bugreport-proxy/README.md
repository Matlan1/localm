# localm proxy (bug reports + issues + updates)

A tiny [Cloudflare Worker](https://workers.cloudflare.com/) that gives 1-2
private-repo testers three account-less surfaces, each backed by a **server-side**
GitHub token - **without** a GitHub account and **without** shipping any token in
the app:

- **Report** a bug -> files a GitHub issue (`POST /`).
- **Track** issues -> lists open/closed issues (`GET /issues`).
- **Update** -> latest release + streams the build zip (`GET /update`,
  `GET /update/download`).

The **issue token** (Issues:write) cannot read code; the **separate update token**
(Contents:read) cannot write. The shared secret gates the routes and is **required**
for the update routes (they serve private builds).

## Why a proxy instead of a token in the app

A GitHub token baked into the distributed app leaks the moment someone opens the
build. Even a fine-grained `Issues: write` token (which cannot read your code) can
still read **every** issue and spam the repo. So the token lives here, server-side:

- The app POSTs the user-reviewed report to this Worker.
- The Worker holds the token as an encrypted secret and creates the issue.
- The token can only file issues through this rate-limitable endpoint, and you
  rotate it here without re-shipping the app.

This keeps the repo private and the testers account-less, which a baked-in token
cannot do safely. (Free Worker tier is far more than enough for 1-2 testers.)

## Deploy (one time)

1. **Install wrangler** and log in:
   ```sh
   npm install -g wrangler
   wrangler login
   ```

2. **Create a fine-grained GitHub PAT** at
   <https://github.com/settings/personal-access-tokens/new>:
   - Resource owner: your account; Repository access: **Only select repositories ->
     your private repo** (e.g. `Matlan1/localm`).
   - Permissions: **Issues -> Read and write** (that is all; it cannot read code).
   - Copy the token (starts with `github_pat_...`).

3. **Set the repo** in `wrangler.toml` (`TARGET_REPO`), then **deploy**:
   ```sh
   cd tools/bugreport-proxy
   wrangler deploy
   ```
   Note the deployed URL, e.g. `https://localm-bugreport-proxy.<you>.workers.dev`.

4. **Set the secrets**:
   ```sh
   wrangler secret put GITHUB_TOKEN     # the Issues PAT (report + issues list)
   wrangler secret put SHARED_SECRET    # required for updates; recommended always
   ```
   Without `SHARED_SECRET` the issue routes accept a POST from anyone who finds the
   URL (they can only file issues, never read the repo, but could spam). The
   **update routes are disabled** until `SHARED_SECRET` is set.

   To enable updates, also mint a **second** fine-grained PAT - **Contents: Read**
   only, same repo - and set it separately:
   ```sh
   wrangler secret put UPDATE_GITHUB_TOKEN   # Contents:read PAT (releases / builds)
   ```
   Keeping the two tokens distinct means a leak of one does not grant the other, and
   neither can push or admin.

5. **That is it - localm already points at the Worker.** The Worker URL and the
   public client token ship as DEFAULTS in `localm/config.py` (`bugreport_upload_url`
   / `bugreport_upload_token`), so a fresh download works with **zero config**: the
   GUI shows **Report a bug**, an **Issues** view, and the **Update** check, and the
   CLI has `localm issues` / `localm update`, out of the box. One Worker hosts all
   three surfaces, so those two defaults also drive the issues list and the updater
   (`update_url`/`update_token` fall back to them; set those only to host updates on a
   different Worker).

   The token is a PUBLIC, low-value client token (like a Sentry DSN), not a secret: it
   only gates the endpoint against drive-by spam and can never read the repo. If you
   redeploy to a different Worker or rotate `SHARED_SECRET`, update the two defaults in
   `localm/config.py` to match. To opt a build OUT of the hosted channel, set either
   key to `""` in `config.json` (a report then just saves to a file / opens email).

## Spam / abuse control

- Set `SHARED_SECRET` so random scanners that find the URL cannot file issues.
- Add a Cloudflare **Rate limiting rule** (dashboard -> your Worker -> Security ->
  Rate limiting) capping requests per IP if you want a hard ceiling.
- Issues are created by the token's identity; you can label/triage/close as usual.

## Protocol

All routes accept `X-Localm-Token: <SHARED_SECRET>` (required where the secret is
set; mandatory for `/update*`). Failures return a 4xx/5xx `{ "error": "..." }` that
the app surfaces honestly (a failed action is never reported as success).

```
POST /                      { "title", "body" }   -> 201 { ok, url, number }
GET  /issues[?state=all|open|closed][&per_page=N] -> 200 { ok, issues: [...] }
GET  /issues?number=N                             -> 200 { ok, issue }
GET  /update                -> 200 { ok, version, notes, published_at, asset:{id,name,size} }
                               (version null when there are no releases yet)
GET  /update/download?id=N  -> 200 application/zip  (streams the release asset)
```

`/issues` items are trimmed to `{number, title, state, created_at, closed_at,
html_url, labels}` and PR entries are filtered out. `/update*` use the separate
Contents:read token and require `SHARED_SECRET`.

## Releases (for the updater)

The updater offers a build when the latest **GitHub Release** tag is newer than the
running `VERSION`. To publish one:

1. Bump `VERSION` (repo root) and `pyproject.toml` version; commit.
2. Package a build zip (the repo tree minus `.venv/`, the data dir, and `.git/`) with
   `tools/bugreport-proxy/make-release.ps1` (or `.sh`).
3. `gh release create vX.Y.Z dist/localm-vX.Y.Z.zip --notes "..."` (private repo).

Testers' apps see it on their next check (a quiet startup check + a manual button);
applying is always their explicit action.

## Local smoke test

```sh
wrangler dev          # serves the Worker locally
curl -X POST http://localhost:8787 -H 'content-type: application/json' \
  -H 'x-localm-token: <SHARED_SECRET>' \
  -d '{"title":"proxy smoke test","body":"hello from curl"}'
```

## Unit test (no Cloudflare account needed)

The Worker's request handling (method gate, secret gate, validation, and the
GitHub call shape with `fetch` stubbed) is covered by
`tools/bugreport-proxy/worker.test.mjs`:

```sh
node --test tools/bugreport-proxy/worker.test.mjs
```
