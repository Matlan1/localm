# localm bug-report proxy

A tiny [Cloudflare Worker](https://workers.cloudflare.com/) that turns an in-app
"Send to maintainer" click into a GitHub issue - **without** giving testers a
GitHub account and **without** shipping a GitHub token inside the app.

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
   wrangler secret put GITHUB_TOKEN     # paste the fine-grained PAT
   wrangler secret put SHARED_SECRET    # optional: any random string (recommended)
   ```

5. **Point localm at it** - in the build you hand to testers, set in
   `~/.localm/config.json` (or via Settings):
   ```json
   {
     "bugreport_upload_url": "https://localm-bugreport-proxy.<you>.workers.dev",
     "bugreport_upload_token": "<the SHARED_SECRET, if you set one>"
   }
   ```
   With this set, the GUI shows a **Send to maintainer** button and the CLI report
   menu offers **Send now**. Without it, reports are saved-to-file + emailed as
   before (nothing breaks; the upload channel is simply hidden).

## Spam / abuse control

- Set `SHARED_SECRET` so random scanners that find the URL cannot file issues.
- Add a Cloudflare **Rate limiting rule** (dashboard -> your Worker -> Security ->
  Rate limiting) capping requests per IP if you want a hard ceiling.
- Issues are created by the token's identity; you can label/triage/close as usual.

## Protocol

```
POST <worker-url>
Content-Type: application/json
X-Localm-Token: <SHARED_SECRET>        # only if SHARED_SECRET is set

{ "title": "...", "body": "...(markdown report)..." }
```

Success: `201 { "ok": true, "url": "<issue url>", "number": 123 }`.
Failure: a 4xx/5xx with `{ "error": "..." }` - the app surfaces this and keeps the
saved report file (a failed send is never reported as success).

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
