// SPDX-License-Identifier: AGPL-3.0-or-later
// Covers the Cloudflare bug-report proxy Worker: its top-level catch returns an
// opaque request_id instead of raw error text, and logs the real error.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const WORKER = join(dirname(fileURLToPath(import.meta.url)),
  "..", "tools", "bugreport-proxy", "worker.js");

// tools/bugreport-proxy has no package.json, so the source is loaded through a
// data: URL to have it evaluated as ESM rather than CommonJS.
async function loadWorker() {
  const src = readFileSync(WORKER, "utf8");
  const url = "data:text/javascript;base64," + Buffer.from(src).toString("base64");
  return (await import(url)).default;
}

const CANARY = "internal-detail-do-not-disclose-9f3a";

// Drive the handler into its top-level catch by making the upstream GitHub call
// throw, then capture what the caller sees and what was logged.
async function throwingRequest(worker, { url, headers, env }) {
  const realFetch = globalThis.fetch;
  const realError = console.error;
  const logged = [];
  globalThis.fetch = async () => { throw new TypeError(CANARY); };
  console.error = (...args) => logged.push(args.map(String).join(" "));
  try {
    const res = await worker.fetch(new Request(url, { method: "GET", headers }), env);
    return { res, body: await res.json(), logged };
  } finally {
    globalThis.fetch = realFetch;
    console.error = realError;
  }
}

const SECRET_ENV = {
  SHARED_SECRET: "s3cret", UPDATE_GITHUB_TOKEN: "t", TARGET_REPO: "a/b",
  GITHUB_TOKEN: "g",
};

test("a thrown error is not disclosed to the caller", async () => {
  const worker = await loadWorker();
  const { res, body } = await throwingRequest(worker, {
    url: "https://proxy.example/update",
    headers: { "X-Localm-Token": "s3cret" },
    env: SECRET_ENV,
  });
  assert.equal(res.status, 500);
  const blob = JSON.stringify(body);
  assert.ok(!blob.includes(CANARY), `error text leaked to the caller: ${blob}`);
  assert.ok(!blob.includes("TypeError"), `error type leaked: ${blob}`);
  assert.equal(body.detail, undefined, "the old `detail` field is gone");
  assert.equal(body.error, "proxy error");
});

test("the caller gets an opaque id it can quote", async () => {
  const worker = await loadWorker();
  const { body } = await throwingRequest(worker, {
    url: "https://proxy.example/update",
    headers: { "X-Localm-Token": "s3cret" },
    env: SECRET_ENV,
  });
  assert.match(body.request_id,
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/);
});

test("the real error IS logged server-side, with its stack", async () => {
  const worker = await loadWorker();
  const { body, logged } = await throwingRequest(worker, {
    url: "https://proxy.example/update",
    headers: { "X-Localm-Token": "s3cret" },
    env: SECRET_ENV,
  });
  const line = logged.join("\n");
  assert.ok(line.includes(CANARY), `the real cause was not logged: ${line}`);
  assert.ok(line.includes(body.request_id),
    "the log line must carry the same id the caller was given, or the id is useless");
  assert.ok(line.includes("at "), `no stack frames in the log: ${line}`);
});

test("an ANONYMOUS caller gets no detail either (SHARED_SECRET unset)", async () => {
  // With SHARED_SECRET unset, secretOk() returns true and the issue routes are open.
  const worker = await loadWorker();
  const { res, body } = await throwingRequest(worker, {
    url: "https://proxy.example/issues",
    headers: {},
    env: { GITHUB_TOKEN: "g", TARGET_REPO: "a/b" },
  });
  assert.equal(res.status, 500);
  const blob = JSON.stringify(body);
  assert.ok(!blob.includes(CANARY), `anonymous caller saw internals: ${blob}`);
  assert.ok(body.request_id, "an anonymous caller still gets a quotable id");
});

test("the wide-open CORS header is still on the error response", async () => {
  const worker = await loadWorker();
  const { res } = await throwingRequest(worker, {
    url: "https://proxy.example/update",
    headers: { "X-Localm-Token": "s3cret" },
    env: SECRET_ENV,
  });
  assert.equal(res.headers.get("access-control-allow-origin"), "*");
});

test("non-error routing is untouched", async () => {
  const worker = await loadWorker();
  const worker404 = await worker.fetch(
    new Request("https://proxy.example/nope", { method: "GET" }), SECRET_ENV);
  assert.equal(worker404.status, 404);
  assert.equal((await worker404.json()).error, "not found");

  const unauth = await worker.fetch(
    new Request("https://proxy.example/update", { method: "GET" }), SECRET_ENV);
  assert.equal(unauth.status, 401, "a wrong/absent secret still 401s first");
});

const FIXTURE = join(dirname(fileURLToPath(import.meta.url)),
  "..", "tests", "fixtures", "bugreport_contract", "automatic_report.md");
const REPORT_ENV = { GITHUB_TOKEN: "g", TARGET_REPO: "a/b" };

// POST a report to the Worker with GitHub stubbed; returns the response, its JSON
// body, and every request the Worker sent upstream.
async function postReport(worker, payload) {
  const realFetch = globalThis.fetch;
  const upstream = [];
  globalThis.fetch = async (u, init) => {
    upstream.push({ url: String(u), body: init && init.body ? JSON.parse(init.body) : null });
    return new Response(JSON.stringify({ html_url: "https://github.com/a/b/issues/1", number: 1 }),
      { status: 201, headers: { "Content-Type": "application/json" } });
  };
  try {
    const res = await worker.fetch(new Request("https://proxy.example/", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload) }), REPORT_ENV);
    return { res, body: await res.json(), upstream };
  } finally {
    globalThis.fetch = realFetch;
  }
}

test("a body without the report's sections is still filed, flagged in the note", async () => {
  const worker = await loadWorker();
  for (const payload of [{ title: "%n", body: "%s%s%s" }, { title: "A".repeat(128), body: "name" }]) {
    const { res, upstream } = await postReport(worker, payload);
    assert.equal(res.status, 201, "a report is never refused for its shape");
    assert.equal(upstream.length, 1);
    const filed = upstream[0].body.body;
    assert.ok(filed.startsWith("> Submitted by a localm user through the built-in bug reporter."), filed.slice(0, 120));
    assert.match(filed, /does not have the sections a report built by localm contains/);
    assert.ok(filed.endsWith(payload.body));
  }
});

test("a report built by localm is filed, prefixed with the reporter note", async () => {
  const worker = await loadWorker();
  const report = readFileSync(FIXTURE, "utf8");
  const { res, body, upstream } = await postReport(worker, { title: "localm bug report", body: report });
  assert.equal(res.status, 201);
  assert.equal(body.number, 1);
  assert.equal(upstream.length, 1);
  const filed = upstream[0].body.body;
  assert.ok(filed.startsWith("> Submitted by a localm user through the built-in bug reporter."), filed.slice(0, 120));
  assert.ok(filed.endsWith(report), "the report follows the note unchanged");
  assert.doesNotMatch(filed, /does not have the sections/, "a real report is not flagged");
});