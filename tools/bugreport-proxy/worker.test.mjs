// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for the localm proxy Worker - NO Cloudflare account needed. We import
// the handler and stub global fetch to assert routing, the gate logic, the token
// used per route, and the GitHub call shape.
// Run: node --test tools/bugreport-proxy/worker.test.mjs

import { test } from "node:test";
import assert from "node:assert/strict";
import worker from "./worker.js";

const ENV = { GITHUB_TOKEN: "ght_issue", TARGET_REPO: "Matlan1/localm" };
const ENV_UP = { ...ENV, SHARED_SECRET: "s3cret", UPDATE_GITHUB_TOKEN: "ght_update" };

function req(body, { method = "POST", headers = {}, path = "/" } = {}) {
  return new Request(`https://proxy.example${path}`, {
    method,
    headers: { "content-type": "application/json", ...headers },
    body: body === undefined ? undefined
      : (typeof body === "string" ? body : JSON.stringify(body)),
  });
}

function stubFetch(impl) {
  const orig = globalThis.fetch;
  globalThis.fetch = impl;
  return () => { globalThis.fetch = orig; };
}

// ------------------------------ routing ---------------------------------

test("unknown route (GET /) -> 404", async () => {
  const r = await worker.fetch(req(undefined, { method: "GET" }), ENV);
  assert.equal(r.status, 404);
});

test("CORS preflight (OPTIONS) is allowed", async () => {
  const r = await worker.fetch(req(undefined, { method: "OPTIONS" }), ENV);
  assert.equal(r.status, 204);
  assert.equal(r.headers.get("Access-Control-Allow-Origin"), "*");
});

// --------------------------- bug report ---------------------------------

test("report: misconfigured proxy (no token) fails closed", async () => {
  const r = await worker.fetch(req({ title: "t", body: "b" }), { TARGET_REPO: "x/y" });
  assert.equal(r.status, 500);
});

test("report: invalid JSON is rejected", async () => {
  const r = await worker.fetch(req("{not valid", {}), ENV);
  assert.equal(r.status, 400);
});

test("report: empty body is rejected", async () => {
  const r = await worker.fetch(req({ title: "t", body: "   " }), ENV);
  assert.equal(r.status, 400);
});

test("report: wrong shared secret -> 401 and NO GitHub call", async () => {
  let called = false;
  const restore = stubFetch(async () => { called = true; return new Response("{}", { status: 201 }); });
  try {
    const r = await worker.fetch(
      req({ title: "t", body: "b" }, { headers: { "X-Localm-Token": "wrong" } }),
      { ...ENV, SHARED_SECRET: "right" });
    assert.equal(r.status, 401);
    assert.equal(called, false);
  } finally { restore(); }
});

test("report: rate limited -> 429 and NO GitHub call", async () => {
  let called = false;
  const restore = stubFetch(async () => { called = true; return new Response("{}", { status: 201 }); });
  const limiter = { limit: async ({ key }) => { limiter.key = key; return { success: false }; } };
  try {
    const r = await worker.fetch(
      req({ title: "t", body: "b" }, { headers: { "cf-connecting-ip": "1.2.3.4" } }),
      { ...ENV, RATE_LIMIT: limiter });
    assert.equal(r.status, 429);
    assert.equal(called, false);                  // throttled BEFORE the GitHub call
    assert.ok(limiter.key.includes("1.2.3.4"));   // keyed per client IP, not global
    const body = await r.json();
    assert.ok(body.retry_after > 0);              // client reads this to count down
    assert.ok(Number(r.headers.get("Retry-After")) > 0);
  } finally { restore(); }
});

test("report: under the rate limit proceeds to file the issue", async () => {
  const restore = stubFetch(async () => new Response(
    JSON.stringify({ html_url: "https://github.com/Matlan1/localm/issues/1", number: 1 }),
    { status: 201 }));
  const limiter = { limit: async () => ({ success: true }) };
  try {
    const r = await worker.fetch(req({ title: "t", body: "b" }), { ...ENV, RATE_LIMIT: limiter });
    assert.equal(r.status, 201);
  } finally { restore(); }
});

test("report: happy path creates the issue with the ISSUE token", async () => {
  const seen = {};
  const restore = stubFetch(async (url, opts) => {
    seen.url = url; seen.opts = opts;
    return new Response(
      JSON.stringify({ html_url: "https://github.com/Matlan1/localm/issues/7", number: 7 }),
      { status: 201 });
  });
  try {
    const r = await worker.fetch(req({ title: "froze", body: "## report" }), ENV);
    assert.equal(r.status, 201);
    const out = await r.json();
    assert.equal(out.url, "https://github.com/Matlan1/localm/issues/7");
    assert.equal(seen.url, "https://api.github.com/repos/Matlan1/localm/issues");
    assert.equal(seen.opts.headers.Authorization, "Bearer ght_issue"); // ISSUE token
  } finally { restore(); }
});

test("report: a GitHub error becomes a 502", async () => {
  const restore = stubFetch(async () => new Response("forbidden", { status: 403 }));
  try {
    const r = await worker.fetch(req({ title: "t", body: "b" }), ENV);
    assert.equal(r.status, 502);
  } finally { restore(); }
});

// ----------------------------- issues -----------------------------------

test("issues: lists issues with the ISSUE token, filtering out PRs", async () => {
  const seen = {};
  const restore = stubFetch(async (url, opts) => {
    seen.url = url; seen.opts = opts;
    return new Response(JSON.stringify([
      { number: 5, title: "real bug", state: "open", html_url: "u5", labels: [{ name: "bug" }] },
      { number: 6, title: "a PR", state: "open", pull_request: { url: "x" }, labels: [] },
    ]), { status: 200 });
  });
  try {
    const r = await worker.fetch(req(undefined, { method: "GET", path: "/issues" }), ENV);
    assert.equal(r.status, 200);
    const out = await r.json();
    assert.equal(out.issues.length, 1, "PR entry filtered out");
    assert.equal(out.issues[0].number, 5);
    assert.deepEqual(out.issues[0].labels, ["bug"]);
    assert.match(seen.url, /\/repos\/Matlan1\/localm\/issues\?state=all/);
    assert.equal(seen.opts.headers.Authorization, "Bearer ght_issue");
  } finally { restore(); }
});

test("issues: ?number=N returns one issue's state", async () => {
  const restore = stubFetch(async () =>
    new Response(JSON.stringify({ number: 12, title: "t", state: "closed", html_url: "u" }), { status: 200 }));
  try {
    const r = await worker.fetch(req(undefined, { method: "GET", path: "/issues?number=12" }), ENV);
    const out = await r.json();
    assert.equal(out.issue.number, 12);
    assert.equal(out.issue.state, "closed");
  } finally { restore(); }
});

// ----------------------------- update -----------------------------------

test("update: disabled (503) when SHARED_SECRET is not set", async () => {
  const r = await worker.fetch(req(undefined, { method: "GET", path: "/update" }),
    { ...ENV, UPDATE_GITHUB_TOKEN: "ght_update" });
  assert.equal(r.status, 503);
});

test("update: wrong secret -> 401, no GitHub call", async () => {
  let called = false;
  const restore = stubFetch(async () => { called = true; return new Response("{}"); });
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update", headers: { "X-Localm-Token": "nope" } }), ENV_UP);
    assert.equal(r.status, 401);
    assert.equal(called, false);
  } finally { restore(); }
});

test("update: latest release uses the UPDATE token and prefers the .zip asset", async () => {
  const seen = {};
  const restore = stubFetch(async (url, opts) => {
    seen.url = url; seen.opts = opts;
    return new Response(JSON.stringify([{
      tag_name: "v0.2.0", name: "0.2.0", body: "notes", published_at: "2026-07-01T00:00:00Z",
      draft: false, prerelease: false,
      assets: [
        { id: 1, name: "notes.txt", size: 10 },
        { id: 2, name: "localm-0.2.0.zip", size: 1234 },
      ],
    }]), { status: 200 });
  });
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update", headers: { "X-Localm-Token": "s3cret" } }), ENV_UP);
    assert.equal(r.status, 200);
    const out = await r.json();
    assert.equal(out.version, "v0.2.0");
    assert.equal(out.asset.id, 2, "the .zip asset is preferred");
    assert.match(seen.url, /\/releases\?per_page=30$/);
    assert.equal(seen.opts.headers.Authorization, "Bearer ght_update"); // UPDATE token, not issue
  } finally { restore(); }
});

test("update: serves the detached .sig signature alongside the .zip asset", async () => {
  const restore = stubFetch(async (url) => {
    if (url.includes("/releases?")) {
      return new Response(JSON.stringify([{
        tag_name: "v0.2.0", name: "0.2.0", body: "n", published_at: "2026-07-01T00:00:00Z",
        draft: false, prerelease: false,
        assets: [
          { id: 2, name: "localm-0.2.0.zip", size: 1234 },
          { id: 3, name: "localm-0.2.0.zip.sig", size: 89 },
        ],
      }]), { status: 200 });
    }
    if (url.includes("/releases/assets/3")) {   // the .sig asset API -> signed redirect
      return { status: 302, ok: false, headers: { get: (k) =>
        k.toLowerCase() === "location" ? "https://objects.githubusercontent.com/sig" : null } };
    }
    if (url === "https://objects.githubusercontent.com/sig") {
      return new Response("Base64SigValue==\n", { status: 200 });
    }
    return new Response("{}", { status: 404 });
  });
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update", headers: { "X-Localm-Token": "s3cret" } }), ENV_UP);
    const out = await r.json();
    assert.equal(out.asset.id, 2, "the .zip is the build asset");
    assert.equal(out.signature, "Base64SigValue==", "the .sig content is served, trimmed");
  } finally { restore(); }
});

test("update: no .sig asset -> signature is null (the client refuses an unsigned build)", async () => {
  const restore = stubFetch(async (url) => {
    if (url.includes("/releases?")) {
      return new Response(JSON.stringify([{
        tag_name: "v0.2.0", name: "0.2.0", body: "n", published_at: "2026-07-01T00:00:00Z",
        draft: false, prerelease: false,
        assets: [{ id: 2, name: "localm-0.2.0.zip", size: 1234 }],
      }]), { status: 200 });
    }
    return new Response("{}", { status: 404 });
  });
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update", headers: { "X-Localm-Token": "s3cret" } }), ENV_UP);
    const out = await r.json();
    assert.equal(out.signature, null);
  } finally { restore(); }
});

test("update: no releases yet -> ok with version null", async () => {
  // The list endpoint (used for both channels now) never 404s for "no releases" -
  // it returns 200 with an empty array. Unlike the old /releases/latest special
  // case, an actual 404/error here is a real failure (see the 502 test below).
  const restore = stubFetch(async () => new Response("[]", { status: 200 }));
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update", headers: { "X-Localm-Token": "s3cret" } }), ENV_UP);
    const out = await r.json();
    assert.equal(out.ok, true);
    assert.equal(out.version, null);
  } finally { restore(); }
});

// ------------------------ update: prerelease channel ----------------------

test("update: no channel param -> uses the releases list, filtered by app tag pattern", async () => {
  // Regression test for the 2026-08-07 incident: this repo also hosts non-app
  // releases in the SAME release list (llama-cuda-linux-<tag>, the self-built
  // Linux CUDA runtime). A newer non-app release must never be offered as the
  // "latest" app version. This reproduces the exact shape that happened live:
  // a CUDA-linux release published AFTER the real app release, non-draft,
  // non-prerelease - exactly what the old, unfiltered /releases/latest call
  // returned for several minutes before both the CI workflow (--prerelease)
  // and this filter were fixed.
  const seen = {};
  const restore = stubFetch(async (url, opts) => {
    seen.url = url; seen.opts = opts;
    return new Response(JSON.stringify([
      { tag_name: "llama-cuda-linux-b9870", name: "llama.cpp CUDA (Linux) - b9870",
        draft: false, prerelease: false, published_at: "2026-08-07T16:19:04Z",
        assets: [{ id: 9, name: "llama-cuda-linux-b9870.tar.gz", size: 999 }] },
      { tag_name: "v0.1.4", name: "0.1.4",
        draft: false, prerelease: false, published_at: "2026-08-06T20:35:53Z",
        assets: [{ id: 2, name: "localm-0.1.4.zip", size: 1234 }] },
    ]), { status: 200 });
  });
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update", headers: { "X-Localm-Token": "s3cret" } }), ENV_UP);
    assert.equal(r.status, 200);
    const out = await r.json();
    assert.equal(out.version, "v0.1.4", "the non-app release must never win, even though it is newer");
    assert.equal(out.asset.id, 2);
    assert.match(seen.url, /\/releases\?per_page=30$/, "uses the list endpoint, not /releases/latest");
  } finally { restore(); }
});

test("update: channel=prerelease also excludes a non-app-tagged release, even though it is newer", async () => {
  // Same regression as the default-channel test above, for the prerelease
  // channel's own code path. The pre-fix code (latestIncludingPrerelease) had
  // NO tag filtering at all here - a second, independent copy of the same bug,
  // just never triggered live because update_allow_prerelease defaults off.
  const restore = stubFetch(async (url) => {
    if (url.includes("/releases?")) {
      return new Response(JSON.stringify([
        { tag_name: "llama-cuda-linux-b9870", draft: false, prerelease: false,
          published_at: "2026-08-07T16:19:04Z", assets: [] },
        { tag_name: "v0.1.5-rc1", draft: false, prerelease: true,
          published_at: "2026-08-07T10:00:00Z",
          assets: [{ id: 5, name: "localm-0.1.5-rc1.zip", size: 1 }] },
      ]), { status: 200 });
    }
    return new Response("{}", { status: 404 });
  });
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update?channel=prerelease",
                       headers: { "X-Localm-Token": "s3cret" } }), ENV_UP);
    const out = await r.json();
    assert.equal(out.version, "v0.1.5-rc1", "the non-app release must never win here either");
  } finally { restore(); }
});

test("update: an unrecognized channel value falls through to the default (stable) path", async () => {
  // Anything other than the literal string "prerelease" must be treated as absent -
  // never partially matched, never a second code path with its own bugs to find.
  const seen = {};
  const restore = stubFetch(async (url) => {
    seen.url = url;
    return new Response(JSON.stringify([
      { tag_name: "v0.2.0", draft: false, prerelease: false, assets: [], published_at: "2026-07-01T00:00:00Z" },
    ]), { status: 200 });
  });
  try {
    await worker.fetch(
      req(undefined, { method: "GET", path: "/update?channel=stable", headers: { "X-Localm-Token": "s3cret" } }),
      ENV_UP);
    assert.match(seen.url, /\/releases\?per_page=30$/);
  } finally { restore(); }
});

test("update: channel=prerelease picks the single most recent NON-DRAFT release by published_at, not array order", async () => {
  // Deliberately out of chronological order and with a DRAFT that is the newest
  // entry by date - proves this does not trust index 0 (GitHub's List Releases
  // docs name no sort-order guarantee) and does not offer a draft to anyone.
  const restore = stubFetch(async (url) => {
    if (url.includes("/releases?")) {
      return new Response(JSON.stringify([
        { tag_name: "v0.1.4-rc1", draft: false, prerelease: true,
          published_at: "2026-08-01T00:00:00Z", assets: [] },
        { tag_name: "v0.1.5-draft", draft: true, prerelease: true,
          published_at: "2026-08-10T00:00:00Z", assets: [] },   // newest by date, but a DRAFT
        { tag_name: "v0.1.4-rc2", draft: false, prerelease: true,
          published_at: "2026-08-05T00:00:00Z",                  // newest NON-draft
          assets: [{ id: 9, name: "localm-0.1.4-rc2.zip", size: 1 }] },
        { tag_name: "v0.1.3", draft: false, prerelease: false,
          published_at: "2026-07-01T00:00:00Z", assets: [] },
      ]), { status: 200 });
    }
    return new Response("{}", { status: 404 });
  });
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update?channel=prerelease",
                       headers: { "X-Localm-Token": "s3cret" } }), ENV_UP);
    assert.equal(r.status, 200);
    const out = await r.json();
    assert.equal(out.version, "v0.1.4-rc2", "newest NON-draft by date, not by array position");
    assert.equal(out.asset.id, 9);
  } finally { restore(); }
});

test("update: channel=prerelease with only drafts available -> no releases yet, never offers a draft", async () => {
  const restore = stubFetch(async (url) => {
    if (url.includes("/releases?")) {
      return new Response(JSON.stringify([
        { tag_name: "v0.2.0-draft", draft: true, published_at: "2026-08-01T00:00:00Z", assets: [] },
      ]), { status: 200 });
    }
    return new Response("{}", { status: 404 });
  });
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update?channel=prerelease",
                       headers: { "X-Localm-Token": "s3cret" } }), ENV_UP);
    const out = await r.json();
    assert.equal(out.ok, true);
    assert.equal(out.version, null);
  } finally { restore(); }
});

test("update: channel=prerelease surfaces a GitHub list-releases failure as 502", async () => {
  const restore = stubFetch(async (url) => {
    if (url.includes("/releases?")) return new Response("forbidden", { status: 403 });
    return new Response("{}", { status: 404 });
  });
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update?channel=prerelease",
                       headers: { "X-Localm-Token": "s3cret" } }), ENV_UP);
    assert.equal(r.status, 502);
  } finally { restore(); }
});

test("update: channel=prerelease can pick a stable release when it IS the newest overall", async () => {
  // Opted-in does not mean "always show an rc" - it means "consider the whole
  // list", and a plain stable release winning that comparison is correct.
  const restore = stubFetch(async (url) => {
    if (url.includes("/releases?")) {
      return new Response(JSON.stringify([
        { tag_name: "v0.1.4-rc1", draft: false, prerelease: true,
          published_at: "2026-07-20T00:00:00Z", assets: [] },
        { tag_name: "v0.1.4", draft: false, prerelease: false,
          published_at: "2026-08-01T00:00:00Z", assets: [] },
      ]), { status: 200 });
    }
    return new Response("{}", { status: 404 });
  });
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update?channel=prerelease",
                       headers: { "X-Localm-Token": "s3cret" } }), ENV_UP);
    const out = await r.json();
    assert.equal(out.version, "v0.1.4");
  } finally { restore(); }
});

test("update/download: streams the asset via the signed redirect", async () => {
  const restore = stubFetch(async (url) => {
    if (url.includes("/releases/assets/")) {
      // asset API: a 302 to a signed URL (no auth on the follow-up)
      return { status: 302, ok: false, headers: { get: (k) =>
        k.toLowerCase() === "location" ? "https://objects.githubusercontent.com/blob" : null } };
    }
    if (url === "https://objects.githubusercontent.com/blob") {
      return new Response("PKbuildzip", { status: 200, headers: { "content-length": "10" } });
    }
    return new Response("{}", { status: 404 });
  });
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update/download?id=2", headers: { "X-Localm-Token": "s3cret" } }), ENV_UP);
    assert.equal(r.status, 200);
    assert.equal(r.headers.get("Content-Type"), "application/zip");
    const text = await r.text();
    assert.match(text, /buildzip/);
  } finally { restore(); }
});

test("update/download: rejects a redirect to a non-GitHub host (no SSRF)", async () => {
  let followed = false;
  const restore = stubFetch(async (url) => {
    if (url.includes("/releases/assets/")) {
      return { status: 302, ok: false, headers: { get: (k) =>
        k.toLowerCase() === "location" ? "https://evil.example/secret" : null } };
    }
    followed = true; // must NOT be reached
    return new Response("leaked", { status: 200 });
  });
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update/download?id=2", headers: { "X-Localm-Token": "s3cret" } }), ENV_UP);
    assert.equal(r.status, 502);
    assert.equal(followed, false, "must not fetch an unexpected redirect host");
  } finally { restore(); }
});

test("update/download: missing asset id -> 400", async () => {
  const r = await worker.fetch(
    req(undefined, { method: "GET", path: "/update/download", headers: { "X-Localm-Token": "s3cret" } }), ENV_UP);
  assert.equal(r.status, 400);
});

// ------------------- update: choose a release by version -----------------

const RELEASES = [
  { tag_name: "v0.3.0", name: "0.3.0", body: "old notes", published_at: "2026-10-15T06:29:13Z",
    draft: false, prerelease: false,
    assets: [
      { id: 30, name: "localm-0.3.0.zip", size: 100 },
      { id: 31, name: "localm-0.3.0.zip.sig", size: 88 },
      { id: 32, name: "localm-0.3.0.zip.envelope", size: 221 },
    ] },
  { tag_name: "llama-cuda-linux-b9870", name: "cuda", draft: false, prerelease: false,
    published_at: "2026-10-30T00:00:00Z", assets: [{ id: 90, name: "x.tar.gz", size: 1 }] },
  { tag_name: "v0.4.0rc1", name: "0.4.0rc1", body: "rc", published_at: "2026-10-28T00:00:00Z",
    draft: false, prerelease: true,
    assets: [{ id: 50, name: "localm-0.4.0rc1.zip", size: 1 }, { id: 51, name: "localm-0.4.0rc1.zip.sig", size: 221 }] },
  { tag_name: "v0.5.0", name: "draft", draft: true, prerelease: false,
    published_at: "2026-10-29T00:00:00Z", assets: [{ id: 60, name: "localm-0.5.0.zip", size: 1 }] },
  { tag_name: "v0.3.1", name: "0.3.1", body: "new notes", published_at: "2026-10-22T22:06:00Z",
    draft: false, prerelease: false,
    assets: [{ id: 40, name: "localm-0.3.1.zip", size: 100 }, { id: 41, name: "localm-0.3.1.zip.sig", size: 221 }] },
  { tag_name: "v0.1.0", name: "v0.1.0", body: "", published_at: "2026-07-04T17:03:04Z",
    draft: false, prerelease: false, assets: [{ id: 10, name: "localm-0.1.0.zip", size: 1 }] },
];

function releasesFetch(seen = {}) {
  return async (url, opts) => {
    (seen.urls = seen.urls || []).push(url);
    if (url.includes("/releases?")) {
      seen.auth = opts.headers.Authorization;
      return new Response(JSON.stringify(RELEASES), { status: 200 });
    }
    const byTag = url.match(/\/releases\/tags\/([^/]+)$/);
    if (byTag) {
      seen.auth = opts.headers.Authorization;
      const rel = RELEASES.find((r) => r.tag_name === decodeURIComponent(byTag[1]));
      return rel ? new Response(JSON.stringify(rel), { status: 200 })
        : new Response("{}", { status: 404 });
    }
    const m = url.match(/\/releases\/assets\/(\d+)$/);
    if (m) {
      return { status: 302, ok: false, headers: { get: (k) =>
        k.toLowerCase() === "location" ? `https://objects.githubusercontent.com/a${m[1]}` : null } };
    }
    const a = url.match(/^https:\/\/objects\.githubusercontent\.com\/a(\d+)$/);
    if (a) return new Response(`sig-of-${a[1]}\n`, { status: 200 });
    return new Response("{}", { status: 404 });
  };
}

const UP_HDR = { "X-Localm-Token": "s3cret" };

test("update/releases: app releases only, newest first, no drafts, no prereleases by default", async () => {
  const seen = {};
  const restore = stubFetch(releasesFetch(seen));
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update/releases", headers: UP_HDR }), ENV_UP);
    assert.equal(r.status, 200);
    const out = await r.json();
    assert.deepEqual(out.releases.map((x) => x.version), ["v0.3.1", "v0.3.0", "v0.1.0"]);
    assert.equal(seen.auth, "Bearer ght_update");
    assert.equal(seen.urls.length, 1, "the list fetches no signature assets");
    const [r031, r030, r010] = out.releases;
    assert.equal(r031.asset.id, 40);
    assert.equal(r031.signed, true);
    assert.equal(r030.signed, true);
    assert.equal(r010.signed, false, "a release with no signature asset is reported unsigned");
    assert.equal(r031.notes, "new notes");
    assert.equal(r031.prerelease, false);
  } finally { restore(); }
});

test("update/releases: channel=prerelease includes prereleases, still never a draft", async () => {
  const restore = stubFetch(releasesFetch());
  try {
    const r = await worker.fetch(req(undefined,
      { method: "GET", path: "/update/releases?channel=prerelease", headers: UP_HDR }), ENV_UP);
    const out = await r.json();
    assert.deepEqual(out.releases.map((x) => x.version), ["v0.4.0rc1", "v0.3.1", "v0.3.0", "v0.1.0"]);
    assert.equal(out.releases[0].prerelease, true);
  } finally { restore(); }
});

test("update/releases: wrong secret -> 401 and no GitHub call; no secret -> 503", async () => {
  let called = false;
  const restore = stubFetch(async () => { called = true; return new Response("[]"); });
  try {
    const r = await worker.fetch(req(undefined,
      { method: "GET", path: "/update/releases", headers: { "X-Localm-Token": "nope" } }), ENV_UP);
    assert.equal(r.status, 401);
    const r2 = await worker.fetch(req(undefined, { method: "GET", path: "/update/releases" }),
      { ...ENV, UPDATE_GITHUB_TOKEN: "ght_update" });
    assert.equal(r2.status, 503);
    assert.equal(called, false);
  } finally { restore(); }
});

test("update/releases: a GitHub list failure is a 502", async () => {
  const restore = stubFetch(async () => new Response("boom", { status: 500 }));
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update/releases", headers: UP_HDR }), ENV_UP);
    assert.equal(r.status, 502);
  } finally { restore(); }
});

test("update/release: serves the chosen release, preferring the .envelope over a bare .sig", async () => {
  const seen = {};
  const restore = stubFetch(releasesFetch(seen));
  try {
    const r = await worker.fetch(req(undefined,
      { method: "GET", path: "/update/release?version=0.3.0", headers: UP_HDR }), ENV_UP);
    assert.equal(r.status, 200);
    const out = await r.json();
    assert.equal(out.found, true);
    assert.equal(out.version, "v0.3.0");
    assert.equal(out.asset.id, 30);
    assert.equal(out.signature, "sig-of-32", "the envelope asset, not the 0.2.0-format .sig");
    assert.ok(!seen.urls.some((u) => u.endsWith("/assets/31")), "the bare .sig is never fetched");
  } finally { restore(); }
});

test("update/release: a release with only a .sig serves that .sig", async () => {
  const restore = stubFetch(releasesFetch());
  try {
    const r = await worker.fetch(req(undefined,
      { method: "GET", path: "/update/release?version=v0.3.1", headers: UP_HDR }), ENV_UP);
    const out = await r.json();
    assert.equal(out.version, "v0.3.1");
    assert.equal(out.signature, "sig-of-41");
  } finally { restore(); }
});

test("update/release: unknown, draft, non-app or (without the channel) prerelease tags are not found", async () => {
  const restore = stubFetch(releasesFetch());
  try {
    for (const v of ["0.9.9", "0.5.0", "0.4.0rc1"]) {
      const r = await worker.fetch(req(undefined,
        { method: "GET", path: `/update/release?version=${v}`, headers: UP_HDR }), ENV_UP);
      assert.equal(r.status, 200, v);
      const out = await r.json();
      assert.equal(out.found, false, v);
      assert.equal(out.version, null, v);
    }
    const r = await worker.fetch(req(undefined, { method: "GET",
      path: "/update/release?version=0.4.0rc1&channel=prerelease", headers: UP_HDR }), ENV_UP);
    assert.equal((await r.json()).version, "v0.4.0rc1");
  } finally { restore(); }
});

test("update/release: a missing or malformed version is a 400 with no GitHub call", async () => {
  let called = false;
  const restore = stubFetch(async () => { called = true; return new Response("[]"); });
  try {
    for (const q of ["", "?version=", "?version=llama-cuda-linux-b9870", "?version=../x",
      `?version=1.2.3${"a".repeat(40)}`]) {
      const r = await worker.fetch(req(undefined,
        { method: "GET", path: `/update/release${q}`, headers: UP_HDR }), ENV_UP);
      assert.equal(r.status, 400, q);
    }
    assert.equal(called, false);
  } finally { restore(); }
});

test("update/release: wrong secret -> 401 and no GitHub call", async () => {
  let called = false;
  const restore = stubFetch(async () => { called = true; return new Response("[]"); });
  try {
    const r = await worker.fetch(req(undefined, { method: "GET",
      path: "/update/release?version=0.3.0", headers: { "X-Localm-Token": "nope" } }), ENV_UP);
    assert.equal(r.status, 401);
    assert.equal(called, false);
  } finally { restore(); }
});

test("update/release: looks the tag up directly, so a release far down the list is found", async () => {
  const seen = [];
  const restore = stubFetch(async (url) => {
    seen.push(url);
    if (url.includes("/releases?")) return new Response("[]", { status: 200 });
    if (url.endsWith("/releases/tags/v0.0.1")) {
      return new Response(JSON.stringify({ tag_name: "v0.0.1", draft: false, prerelease: false,
        published_at: "2020-01-01T00:00:00Z", assets: [{ id: 5, name: "localm-0.0.1.zip", size: 1 }] }),
      { status: 200 });
    }
    return new Response("{}", { status: 404 });
  });
  try {
    const r = await worker.fetch(req(undefined,
      { method: "GET", path: "/update/release?version=0.0.1", headers: UP_HDR }), ENV_UP);
    const out = await r.json();
    assert.equal(out.found, true);
    assert.equal(out.asset.id, 5);
    assert.ok(!seen.some((u) => u.includes("/releases?")), "no list read for one tag");
  } finally { restore(); }
});

test("update/release: a GitHub error other than 404 on the tag lookup is a 502", async () => {
  const restore = stubFetch(async () => new Response("boom", { status: 500 }));
  try {
    const r = await worker.fetch(req(undefined,
      { method: "GET", path: "/update/release?version=0.3.0", headers: UP_HDR }), ENV_UP);
    assert.equal(r.status, 502);
  } finally { restore(); }
});

test("update/releases: reads further pages while a page is full, and stops at a short page", async () => {
  const mk = (n) => ({ id: n, tag_name: `v0.0.${n}`, draft: false, prerelease: false,
    published_at: new Date(Date.UTC(2026, 0, 1) + n * 1000).toISOString(),
    assets: [{ id: 1000 + n, name: `localm-0.0.${n}.zip`, size: 1 }] });
  const pages = { 1: Array.from({ length: 30 }, (_, i) => mk(i + 1)),
    2: Array.from({ length: 5 }, (_, i) => mk(i + 31)) };
  const urls = [];
  const restore = stubFetch(async (url) => {
    urls.push(url);
    const page = Number((url.match(/[?&]page=(\d+)/) || [0, 1])[1]);
    return new Response(JSON.stringify(pages[page] || []), { status: 200 });
  });
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update/releases", headers: UP_HDR }), ENV_UP);
    const out = await r.json();
    assert.equal(out.releases.length, 35);
    assert.equal(out.releases[0].version, "v0.0.35");
    assert.equal(urls.length, 2, "page 2 was short, so no page 3");
  } finally { restore(); }
});

test("update: /update is unchanged by the new routes (latest stable, .sig only)", async () => {
  const restore = stubFetch(releasesFetch());
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update", headers: UP_HDR }), ENV_UP);
    const out = await r.json();
    assert.equal(out.version, "v0.3.1");
    assert.equal(out.signature, "sig-of-41");
  } finally { restore(); }
});

test("update: a legacy client is never served a tag lookup answer that is not the ceiling release", async () => {
  const restore = stubFetch(async (url) => {
    if (url.includes("/releases/tags/")) {
      return new Response(JSON.stringify({ tag_name: "v9.9.9", draft: false, prerelease: false,
        published_at: "2026-12-01T00:00:00Z", assets: [{ id: 99, name: "localm-9.9.9.zip", size: 1 }] }),
      { status: 200 });
    }
    return new Response(JSON.stringify([{ tag_name: "v0.3.1", draft: false, prerelease: false,
      published_at: "2026-10-22T00:00:00Z", assets: [{ id: 31, name: "localm-0.3.1.zip", size: 1 }] }]),
    { status: 200 });
  });
  try {
    const r = await worker.fetch(
      req(undefined, { method: "GET", path: "/update", headers: UP_HDR }), ENV_UP);
    const out = await r.json();
    assert.equal(out.version, "v0.3.1", "falls back to the list, filtered to the ceiling");
    assert.equal(out.legacy_ceiling, "v0.3.2");
  } finally { restore(); }
});

test("update: a legacy client never gets a tag that only looks like the ceiling (v0.3.2.1, -hotfix, +b, a)", async () => {
  for (const tag of ["v0.3.2.1", "v0.3.2-hotfix", "v0.3.2+b", "v0.3.2a"]) {
    const restore = stubFetch(async (url) => {
      if (url.includes("/releases/tags/")) return new Response("{}", { status: 404 });
      return new Response(JSON.stringify([
        { tag_name: tag, draft: false, prerelease: false, published_at: "2026-11-01T00:00:00Z",
          assets: [{ id: 9, name: "localm-x.zip", size: 1 }] },
        { tag_name: "v0.3.1", draft: false, prerelease: false, published_at: "2026-10-22T00:00:00Z",
          assets: [{ id: 31, name: "localm-0.3.1.zip", size: 1 }] },
      ]), { status: 200 });
    });
    try {
      const r = await worker.fetch(
        req(undefined, { method: "GET", path: "/update", headers: UP_HDR }), ENV_UP);
      const out = await r.json();
      assert.equal(out.version, "v0.3.1", tag);
    } finally { restore(); }
  }
});

test("update: a legacy client still gets a numbered prerelease of the ceiling on the prerelease channel", async () => {
  const restore = stubFetch(async (url) => {
    if (url.includes("/releases/tags/")) return new Response("{}", { status: 404 });
    return new Response(JSON.stringify([
      { tag_name: "v0.3.2rc1", draft: false, prerelease: true, published_at: "2026-11-01T00:00:00Z",
        assets: [{ id: 9, name: "localm-0.3.2rc1.zip", size: 1 }] },
      { tag_name: "v0.3.1", draft: false, prerelease: false, published_at: "2026-10-22T00:00:00Z",
        assets: [{ id: 31, name: "localm-0.3.1.zip", size: 1 }] },
    ]), { status: 200 });
  });
  try {
    const r = await worker.fetch(req(undefined,
      { method: "GET", path: "/update?channel=prerelease", headers: UP_HDR }), ENV_UP);
    assert.equal((await r.json()).version, "v0.3.2rc1");
  } finally { restore(); }
});
