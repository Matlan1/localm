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

test("update: no .sig asset -> signature is null (client decides enforce vs open)", async () => {
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
