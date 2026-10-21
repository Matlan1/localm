// SPDX-License-Identifier: AGPL-3.0-or-later
//
// localm proxy - a tiny Cloudflare Worker giving 1-2 private-repo testers three
// account-less surfaces, each backed by a SERVER-SIDE GitHub token:
//
//   POST /                       file a bug report      [GitHub App or GITHUB_TOKEN, Issues:write]
//   GET  /issues                 list open/closed issues[GitHub App or GITHUB_TOKEN, Issues:read]
//   GET  /issues?number=N        one issue's state      [GitHub App or GITHUB_TOKEN]
//   GET  /update                 latest release meta    [UPDATE_GITHUB_TOKEN, Contents:read]
//   GET  /update/download?id=N   stream a build asset   [UPDATE_GITHUB_TOKEN]
//
// No token ships in the app. The issue token cannot read code; the SEPARATE update
// token can read releases/contents but cannot write. The shared secret gates the
// routes and is MANDATORY for the update routes (they serve private builds).
//
// Deploy + secrets: see README.md.

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return cors(new Response(null, { status: 204 }));
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";
    try {
      if (request.method === "POST" && (path === "/" || path === "/report")) {
        return await fileReport(request, env);
      }
      if (request.method === "GET" && path === "/issues") {
        return await listIssues(request, env, url);
      }
      if (request.method === "GET" && path === "/update") {
        return await latestRelease(request, env, url);
      }
      if (request.method === "GET" && path === "/update/download") {
        return await downloadBuild(request, env, url);
      }
      return json({ error: "not found" }, 404);
    } catch (e) {
      // This catch sits OUTSIDE secretOk(), and json() serves the body with
      // Access-Control-Allow-Origin: * (see cors()), so whatever it carries is
      // readable by any web origin. It used to return String(e).slice(0, 200) -
      // raw runtime error text describing internal state. Reachable without the
      // shared secret whenever SHARED_SECRET is UNSET, which is the documented
      // open configuration for the issue routes (secretOk returns true), so an
      // upstream failure inside fileReport/listIssues surfaced its message to an
      // anonymous caller.
      //
      // Return an opaque id and log the REAL error, with its stack, server-side.
      // That is strictly BETTER for debugging than the old 200-character prefix
      // (a full stack in `wrangler tail` instead of a truncated message) while
      // the caller gets only an id they can quote in a report.
      const rid = crypto.randomUUID();
      console.error("proxy error", rid, (e && e.stack) || String(e));
      return json({ error: "proxy error", request_id: rid }, 500);
    }
  },
};

// ----------------------------- gates ------------------------------------

function secretOk(request, env) {
  if (!env.SHARED_SECRET) return true; // unset => open (issue routes only)
  const got = request.headers.get("X-Localm-Token") || "";
  return timingSafeEqual(got, env.SHARED_SECRET);
}

// Update routes REQUIRE a configured + matching secret (they can serve private
// builds): no secret -> the channel is disabled (fail closed), not wide open.
function updateGateOrError(request, env) {
  if (!env.SHARED_SECRET) return json({ error: "update channel disabled: set SHARED_SECRET" }, 503);
  if (!secretOk(request, env)) return json({ error: "unauthorized" }, 401);
  if (!env.UPDATE_GITHUB_TOKEN || !env.TARGET_REPO) {
    return json({ error: "update not configured: set UPDATE_GITHUB_TOKEN and TARGET_REPO" }, 500);
  }
  return null;
}

// ------------------------ issue token ----------------------------------

// The token the issue routes use. With GITHUB_APP_ID, GITHUB_APP_INSTALLATION_ID
// and GITHUB_APP_PRIVATE_KEY all set, it is a GitHub App installation token, so
// issues are filed as the app's bot account; a failed app token request is an
// error, never a fall back to GITHUB_TOKEN. With none of them set, GITHUB_TOKEN
// is used. Returns { token } or { error: Response }.
async function issueToken(env) {
  if (!env.TARGET_REPO) {
    return { error: json({ error: "proxy misconfigured: set TARGET_REPO" }, 500) };
  }
  const app = [env.GITHUB_APP_ID, env.GITHUB_APP_INSTALLATION_ID, env.GITHUB_APP_PRIVATE_KEY];
  const set = app.filter(Boolean).length;
  if (set === 0) {
    if (!env.GITHUB_TOKEN) {
      return { error: json({ error: "proxy misconfigured: set GITHUB_TOKEN or the GitHub App secrets" }, 500) };
    }
    return { token: env.GITHUB_TOKEN };
  }
  if (set !== app.length) {
    return { error: json({ error: "proxy misconfigured: set all of GITHUB_APP_ID, GITHUB_APP_INSTALLATION_ID and GITHUB_APP_PRIVATE_KEY" }, 500) };
  }
  try {
    return { token: await appInstallationToken(env) };
  } catch (e) {
    console.error("github app token failed", (e && e.stack) || String(e));
    return { error: json({ error: "GitHub App token request failed" }, 502) };
  }
}

// Cached installation token: { key, token, expiresAt } (ms since epoch).
let appTokenCache = null;

// An installation token for the configured GitHub App, reused until five minutes
// before it expires. Throws on any failure.
async function appInstallationToken(env) {
  const key = `${env.GITHUB_APP_ID}/${env.GITHUB_APP_INSTALLATION_ID}`;
  const now = Date.now();
  if (appTokenCache && appTokenCache.key === key && appTokenCache.expiresAt - 300000 > now) {
    return appTokenCache.token;
  }
  const jwt = await appJwt(env.GITHUB_APP_ID, env.GITHUB_APP_PRIVATE_KEY, Math.floor(now / 1000));
  const res = await fetch(
    `https://api.github.com/app/installations/${encodeURIComponent(env.GITHUB_APP_INSTALLATION_ID)}/access_tokens`,
    { method: "POST", headers: ghHeaders(jwt) });
  if (!res.ok) throw new Error(`installation token request returned ${res.status}`);
  const data = await res.json();
  if (!data || typeof data.token !== "string" || !data.token) {
    throw new Error("installation token response has no token");
  }
  const expiresAt = Date.parse(data.expires_at || "") || now + 3600000;
  appTokenCache = { key, token: data.token, expiresAt };
  return data.token;
}

// An RS256 JWT for GitHub App authentication: issued 60 s in the past (clock
// skew), valid for nine minutes, issuer = app id. *pem* is the app's private key,
// PKCS#1 ("BEGIN RSA PRIVATE KEY", as GitHub issues it) or PKCS#8.
async function appJwt(appId, pem, nowSeconds) {
  const enc = (obj) => base64url(new TextEncoder().encode(JSON.stringify(obj)));
  const head = enc({ alg: "RS256", typ: "JWT" });
  const body = enc({ iat: nowSeconds - 60, exp: nowSeconds + 540, iss: String(appId) });
  const key = await crypto.subtle.importKey(
    "pkcs8", pemToPkcs8(pem), { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("RSASSA-PKCS1-v1_5", key, new TextEncoder().encode(`${head}.${body}`));
  return `${head}.${body}.${base64url(new Uint8Array(sig))}`;
}

// DER bytes of a PKCS#8 private key from a PEM string; a PKCS#1 RSA key is wrapped
// in the PKCS#8 PrivateKeyInfo structure. Throws on anything else.
function pemToPkcs8(pem) {
  const text = String(pem).replace(/\\n/g, "\n");
  const m = text.match(/-----BEGIN (RSA )?PRIVATE KEY-----([\s\S]*?)-----END (RSA )?PRIVATE KEY-----/);
  if (!m) throw new Error("GITHUB_APP_PRIVATE_KEY is not a PEM private key");
  const der = Uint8Array.from(atob(m[2].replace(/\s+/g, "")), (c) => c.charCodeAt(0));
  if (!m[1]) return der;
  const rsaAlgorithm = [0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01, 0x05, 0x00];
  const inner = concatBytes([0x02, 0x01, 0x00], rsaAlgorithm, derHeader(0x04, der.length), der);
  return concatBytes(derHeader(0x30, inner.length), inner);
}

function derHeader(tag, length) {
  if (length < 0x80) return [tag, length];
  const bytes = [];
  for (let n = length; n > 0; n >>= 8) bytes.unshift(n & 0xff);
  return [tag, 0x80 | bytes.length, ...bytes];
}

function concatBytes(...parts) {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

function base64url(bytes) {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

// --------------------------- bug report ---------------------------------

async function fileReport(request, env) {
  if (!secretOk(request, env)) return json({ error: "unauthorized" }, 401);
  // Spam control: the report endpoint ships ENABLED in every localm install and its
  // client token is public, so throttle issue creation PER CLIENT IP. The binding is
  // optional: absent (local `wrangler dev`, or before it is provisioned) means no
  // limiting and the Worker still works. Only enforced when deployed. See the
  // [[ratelimits]] block in wrangler.toml.
  if (env.RATE_LIMIT) {
    const ip = request.headers.get("cf-connecting-ip") || "unknown";
    const { success } = await env.RATE_LIMIT.limit({ key: `report:${ip}` });
    if (!success) {
      // Sliding window: ~30s frees enough tokens for a legitimate retry. The client
      // reads retry_after (body) / Retry-After (header) to count down and retry once.
      const retryAfter = 30;
      const resp = json({ error: "rate limited: too many bug reports from your network, please wait", retry_after: retryAfter }, 429);
      resp.headers.set("Retry-After", String(retryAfter));
      return resp;
    }
  }
  const auth = await issueToken(env);
  if (auth.error) return auth.error;
  let payload;
  try { payload = await request.json(); } catch { return json({ error: "invalid JSON" }, 400); }
  const title = (String(payload.title || "").slice(0, 200).trim()) || "localm bug report";
  const report = String(payload.body || "").slice(0, 60000);
  if (!report.trim()) return json({ error: "empty report body" }, 400);
  const note = looksLikeLocalmReport(report) ? REPORTER_NOTE : REPORTER_NOTE_UNSHAPED;
  const issue = { title, body: note + report };
  if (env.ISSUE_LABELS) {
    const labels = env.ISSUE_LABELS.split(",").map((s) => s.trim()).filter(Boolean);
    if (labels.length) issue.labels = labels;
  }
  const gh = await ghFetch(auth.token, `/repos/${env.TARGET_REPO}/issues`, {
    method: "POST", body: JSON.stringify(issue),
  });
  if (!gh.ok) {
    const detail = (await gh.text().catch(() => "")).slice(0, 500);
    return json({ error: "GitHub rejected the issue", status: gh.status, detail }, 502);
  }
  const data = await gh.json();
  return json({ ok: true, url: data.html_url, number: data.number }, 201);
}

// Prepended to the body of every issue this proxy files.
const REPORTER_NOTE =
  "> Submitted by a localm user through the built-in bug reporter.\n\n";

// Prepended instead of REPORTER_NOTE when the body lacks the REPORT_SECTIONS headings.
const REPORTER_NOTE_UNSHAPED =
  "> Submitted by a localm user through the built-in bug reporter.\n" +
  "> It does not have the sections a report built by localm contains.\n\n";

// Section headings every report built by localm's reporter contains.
const REPORT_SECTIONS = ["## What happened", "## Environment"];

// True when *text* has every heading in REPORT_SECTIONS on a line of its own.
function looksLikeLocalmReport(text) {
  const lines = text.split(/\r?\n/).map((line) => line.trim());
  return REPORT_SECTIONS.every((h) => lines.includes(h));
}

// ---------------------------- issues ------------------------------------

async function listIssues(request, env, url) {
  if (!secretOk(request, env)) return json({ error: "unauthorized" }, 401);
  const auth = await issueToken(env);
  if (auth.error) return auth.error;
  const one = url.searchParams.get("number");
  if (one) {
    if (!/^\d+$/.test(one)) return json({ error: "bad issue number" }, 400);
    const gh = await ghFetch(auth.token, `/repos/${env.TARGET_REPO}/issues/${one}`);
    if (!gh.ok) return json({ error: "issue lookup failed", status: gh.status }, 502);
    return json({ ok: true, issue: trimIssue(await gh.json()) });
  }
  const want = url.searchParams.get("state");
  const state = ["open", "closed", "all"].includes(want) ? want : "all";
  const per = Math.min(Math.max(parseInt(url.searchParams.get("per_page") || "30", 10) || 30, 1), 100);
  const gh = await ghFetch(auth.token,
    `/repos/${env.TARGET_REPO}/issues?state=${state}&per_page=${per}&sort=updated`);
  if (!gh.ok) return json({ error: "issues list failed", status: gh.status }, 502);
  const arr = await gh.json();
  const issues = (Array.isArray(arr) ? arr : [])
    .filter((it) => !it.pull_request) // the issues API returns PRs too - drop them
    .map(trimIssue);
  return json({ ok: true, issues });
}

function trimIssue(it) {
  return {
    number: it.number, title: it.title, state: it.state,
    created_at: it.created_at, closed_at: it.closed_at, html_url: it.html_url,
    labels: (it.labels || []).map((l) => (typeof l === "string" ? l : l.name)),
  };
}

// ------------------------ update: latest release ------------------------

async function latestRelease(request, env, url) {
  const gate = updateGateOrError(request, env);
  if (gate) return gate;
  // Opt-in prerelease channel (localm's update_allow_prerelease setting, admin_only,
  // default off).
  const wantsPrerelease = url.searchParams.get("channel") === "prerelease";
  const rel = await latestAppRelease(env, wantsPrerelease);
  if (rel === null) return json({ ok: true, version: null, note: "no releases yet" });
  if (rel === undefined) return json({ error: "release lookup failed", status: 502 }, 502);
  const assets = (rel.assets || []).map((a) => ({ id: a.id, name: a.name, size: a.size }));
  // Prefer a .zip asset (the build); else the first asset; else none.
  const asset = assets.find((a) => /\.zip$/i.test(a.name)) || assets[0] || null;
  // Serve the signature: the text of the first *.sig asset, verbatim (trimmed). From
  // 0.3.1 on that is a JSON signature envelope the client verifies against the key
  // set it trusts (updater.verify_signature); the 0.3.0 .sig is a bare base64
  // signature for 0.2.0 clients. Absent -> null: the client refuses the unsigned build.
  let signature = null;
  const sigAsset = assets.find((a) => /\.sig$/i.test(a.name));
  if (sigAsset) {
    const txt = await fetchAssetText(env, sigAsset.id);
    if (txt != null) signature = txt.trim();
  }
  return json({
    ok: true, version: rel.tag_name, name: rel.name,
    notes: String(rel.body || "").slice(0, 8000), published_at: rel.published_at,
    asset, signature,
  });
}

// Matches an app release tag (v1.2.3, v1.2.3rc1, v1.2.3-beta, ...) and nothing
// else. This repo also hosts non-app releases in the SAME release list - e.g.
// llama-cuda-linux-<tag>, the self-built Linux CUDA runtime - and GitHub's own
// "latest release" concept has no notion of release TYPE, so without this
// filter any such release can silently outrank the real app build the moment
// it is newer. Measured live 2026-08-07: /releases/latest returned
// llama-cuda-linux-b9870 ahead of v0.1.4 for the few minutes between that build
// publishing and its --prerelease flag being set by hand. The CI workflow that
// publishes those builds now always sets --prerelease too (belt), but this
// filter is what makes the mistake structurally impossible regardless of that
// (suspenders) - it does not depend on every future release-creating workflow
// remembering the flag correctly.
const APP_TAG_RE = /^v\d+\.\d+\.\d+/;

// The single most recent NON-DRAFT release matching APP_TAG_RE, optionally
// including prereleases. Returns the release object, null (no qualifying
// release - the /update route reports this the same as a 404), or undefined
// (the GitHub call itself failed - reported as a 502).
//
// Uses /releases (the list endpoint), not /releases/latest, for BOTH the
// default and prerelease-channel paths: /releases/latest cannot be asked "the
// latest release matching this tag pattern", only "the latest release, full
// stop" - so the only way to apply APP_TAG_RE at all is to list and filter by
// hand. Two things NOT trusted from GitHub's docs while doing so:
//   - Sort order: the List Releases docs name NO ordering guarantee, so this
//     picks the max by published_at explicitly rather than trusting array
//     order or index 0.
//   - Draft visibility: GitHub's docs say only a push-access token even
//     RECEIVES draft entries in this listing, so UPDATE_GITHUB_TOKEN (a
//     Contents:read token, not a collaborator) most likely never sees one -
//     but a draft must never be offered to ANYONE, so this filters
//     draft === true explicitly rather than relying on that token-permission
//     behavior holding.
async function latestAppRelease(env, includePrerelease) {
  const gh = await ghFetch(env.UPDATE_GITHUB_TOKEN, `/repos/${env.TARGET_REPO}/releases?per_page=30`);
  if (!gh.ok) return undefined;
  const all = await gh.json();
  const candidates = (Array.isArray(all) ? all : [])
    .filter((r) => r.draft !== true)
    .filter((r) => includePrerelease || r.prerelease !== true)
    .filter((r) => APP_TAG_RE.test(String(r.tag_name || "")));
  if (!candidates.length) return null;
  return candidates.reduce((best, r) => {
    const rt = Date.parse(r.published_at || r.created_at || 0) || 0;
    const bt = Date.parse(best.published_at || best.created_at || 0) || 0;
    return rt > bt ? r : best;
  });
}

// Read a small release asset (the .sig) as text, following GitHub's signed-URL
// redirect by hand and CONSTRAINING the redirect host (same SSRF guard as the build
// download). Returns null on any failure or an implausibly large body - a missing
// signature must degrade to "unsigned", never to an arbitrary fetch.
async function fetchAssetText(env, id) {
  const api = `https://api.github.com/repos/${env.TARGET_REPO}/releases/assets/${id}`;
  const r1 = await fetch(api, {
    headers: ghHeaders(env.UPDATE_GITHUB_TOKEN, "application/octet-stream"),
    redirect: "manual",
  });
  let upstream = r1;
  if (r1.status === 301 || r1.status === 302 || r1.status === 307 || r1.status === 308) {
    const loc = r1.headers.get("location");
    if (!loc || !redirectHostAllowed(loc)) return null;
    upstream = await fetch(loc); // signed URL: no auth header
  }
  if (!upstream.ok) return null;
  const txt = await upstream.text();
  return txt.length <= 4096 ? txt : null;   // the client refuses an envelope over 4096 characters
}

// ----------------------- update: stream a build -------------------------

async function downloadBuild(request, env, url) {
  const gate = updateGateOrError(request, env);
  if (gate) return gate;
  const id = url.searchParams.get("id");
  if (!id || !/^\d+$/.test(id)) return json({ error: "missing or bad asset id" }, 400);
  const api = `https://api.github.com/repos/${env.TARGET_REPO}/releases/assets/${id}`;
  // The asset API 302-redirects to a signed URL that must be fetched WITHOUT the
  // Authorization header, so follow the redirect by hand.
  const r1 = await fetch(api, {
    headers: ghHeaders(env.UPDATE_GITHUB_TOKEN, "application/octet-stream"),
    redirect: "manual",
  });
  let upstream = r1;
  if (r1.status === 301 || r1.status === 302 || r1.status === 307 || r1.status === 308) {
    const loc = r1.headers.get("location");
    if (!loc) return json({ error: "asset redirect had no location" }, 502);
    // Constrain the redirect target: GitHub asset downloads redirect to a signed URL
    // on a GitHub/AWS asset host. Refuse anything else so a surprising Location cannot
    // turn this into an SSRF that fetches an arbitrary URL.
    if (!redirectHostAllowed(loc)) {
      return json({ error: "asset redirect to an unexpected host" }, 502);
    }
    upstream = await fetch(loc); // signed URL: no auth header
  }
  if (!upstream.ok || !upstream.body) {
    return json({ error: "asset download failed", status: upstream.status }, 502);
  }
  const headers = new Headers();
  headers.set("Content-Type", "application/zip");
  headers.set("Content-Disposition", `attachment; filename="localm-build-${id}.zip"`);
  const len = upstream.headers.get("content-length");
  if (len) headers.set("Content-Length", len);
  headers.set("Access-Control-Allow-Origin", "*");
  return new Response(upstream.body, { status: 200, headers });
}

// ----------------------------- helpers ----------------------------------

function ghHeaders(token, accept) {
  return {
    "Authorization": `Bearer ${token}`,
    "Accept": accept || "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "localm-proxy",
  };
}

function ghFetch(token, pathOrUrl, opts = {}) {
  const u = pathOrUrl.startsWith("http") ? pathOrUrl : `https://api.github.com${pathOrUrl}`;
  const headers = ghHeaders(token);
  if (opts.body) headers["Content-Type"] = "application/json";
  return fetch(u, { method: opts.method || "GET", headers, body: opts.body });
}

function json(obj, status = 200) {
  return cors(new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } }));
}

function cors(resp) {
  resp.headers.set("Access-Control-Allow-Origin", "*");
  resp.headers.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  resp.headers.set("Access-Control-Allow-Headers", "Content-Type, X-Localm-Token");
  return resp;
}

// Allow only GitHub / GitHub-AWS asset hosts as a release-asset redirect target, so
// the manual redirect-follow in /update/download cannot be steered into an SSRF.
function redirectHostAllowed(loc) {
  let host;
  try { host = new URL(loc).hostname.toLowerCase(); } catch { return false; }
  return host === "github.com" ||
    host === "api.github.com" ||
    host === "codeload.github.com" ||
    host.endsWith(".githubusercontent.com") ||   // objects/release-assets.githubusercontent.com
    host.endsWith(".s3.amazonaws.com") ||         // github-cloud.s3.amazonaws.com
    host === "github-cloud.s3.amazonaws.com";
}

// Constant-time-ish compare so the secret check does not leak its contents via
// per-character timing. (Length is not hidden; acceptable for a low-value secret.)
function timingSafeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let out = 0;
  for (let i = 0; i < a.length; i++) out |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return out === 0;
}
