// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Unit tests for the localm bug-report proxy Worker - NO Cloudflare account
// needed. We import the handler and stub global fetch to assert the gate logic
// and the GitHub call shape. Run: node --test tools/bugreport-proxy/worker.test.mjs

import { test } from "node:test";
import assert from "node:assert/strict";
import worker from "./worker.js";

const ENV = { GITHUB_TOKEN: "ght_secret", TARGET_REPO: "Matlan1/localm" };

function req(body, { method = "POST", headers = {} } = {}) {
  return new Request("https://proxy.example/", {
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

test("rejects a non-POST method", async () => {
  const r = await worker.fetch(req(undefined, { method: "GET" }), ENV);
  assert.equal(r.status, 405);
});

test("CORS preflight (OPTIONS) is allowed", async () => {
  const r = await worker.fetch(req(undefined, { method: "OPTIONS" }), ENV);
  assert.equal(r.status, 204);
  assert.equal(r.headers.get("Access-Control-Allow-Origin"), "*");
});

test("misconfigured proxy (no token) fails closed", async () => {
  const r = await worker.fetch(req({ title: "t", body: "b" }), { TARGET_REPO: "x/y" });
  assert.equal(r.status, 500);
});

test("invalid JSON is rejected", async () => {
  const r = await worker.fetch(req("{not valid", {}), ENV);
  assert.equal(r.status, 400);
});

test("empty report body is rejected", async () => {
  const r = await worker.fetch(req({ title: "t", body: "   " }), ENV);
  assert.equal(r.status, 400);
});

test("wrong shared secret -> 401 and NO GitHub call", async () => {
  let called = false;
  const restore = stubFetch(async () => { called = true; return new Response("{}", { status: 201 }); });
  try {
    const r = await worker.fetch(
      req({ title: "t", body: "b" }, { headers: { "X-Localm-Token": "wrong" } }),
      { ...ENV, SHARED_SECRET: "right" });
    assert.equal(r.status, 401);
    assert.equal(called, false, "must not reach GitHub when the secret is wrong");
  } finally { restore(); }
});

test("happy path: creates the issue and returns its url + number", async () => {
  const seen = {};
  const restore = stubFetch(async (url, opts) => {
    seen.url = url;
    seen.opts = opts;
    return new Response(
      JSON.stringify({ html_url: "https://github.com/Matlan1/localm/issues/7", number: 7 }),
      { status: 201 });
  });
  try {
    const r = await worker.fetch(req({ title: "image gen froze", body: "## report\nbody" }), ENV);
    assert.equal(r.status, 201);
    const out = await r.json();
    assert.equal(out.ok, true);
    assert.equal(out.url, "https://github.com/Matlan1/localm/issues/7");
    assert.equal(out.number, 7);
    // The outbound GitHub call is shaped correctly.
    assert.equal(seen.url, "https://api.github.com/repos/Matlan1/localm/issues");
    assert.equal(seen.opts.headers.Authorization, "Bearer ght_secret");
    const payload = JSON.parse(seen.opts.body);
    assert.equal(payload.title, "image gen froze");
    assert.equal(payload.body, "## report\nbody");
  } finally { restore(); }
});

test("correct shared secret is accepted", async () => {
  const restore = stubFetch(async () =>
    new Response(JSON.stringify({ html_url: "https://x/issues/1", number: 1 }), { status: 201 }));
  try {
    const r = await worker.fetch(
      req({ title: "t", body: "b" }, { headers: { "X-Localm-Token": "right" } }),
      { ...ENV, SHARED_SECRET: "right" });
    assert.equal(r.status, 201);
  } finally { restore(); }
});

test("a GitHub error becomes a 502 (failure is surfaced, not faked)", async () => {
  const restore = stubFetch(async () => new Response("forbidden", { status: 403 }));
  try {
    const r = await worker.fetch(req({ title: "t", body: "b" }), ENV);
    assert.equal(r.status, 502);
    const out = await r.json();
    assert.equal(out.status, 403);
  } finally { restore(); }
});
