// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// The Knowledge page's re-embed button calls the server's re-embed endpoint,
// which recomputes vectors from the chunk text already stored in the
// collection. It sends no paths, fetches no document list, and does no
// batching.

const tick = () => new Promise((r) => setTimeout(r, 0));

function setup(fetchImpl) {
  const { window } = loadAppWithPages({ fetchImpl });
  runScript(window, `streamJob = () => Promise.resolve({ status: "done" });`);
  return window;
}

function recorder(extra = {}) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    calls.push({ url: String(url), opts });
    const u = String(url);
    if (u.includes("/reembed"))
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ job_id: "j1" }), ...extra };
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ collections: [] }) };
  };
  return { calls, fetchImpl };
}

const reembedCalls = (calls) => calls.filter((c) => c.url.includes("/reembed"));

test("re-embed POSTs to the collection's reembed endpoint", async () => {
  const { calls, fetchImpl } = recorder();
  const window = setup(fetchImpl);
  runScript(window, `kbConfirmReembed = () => Promise.resolve(true);`);
  runScript(window, "kbReembedCollection('mycoll');");
  for (let i = 0; i < 8; i++) await tick();

  const hits = reembedCalls(calls);
  assert.equal(hits.length, 1, "exactly one /reembed POST");
  assert.equal(hits[0].url, "/api/rag/collections/mycoll/reembed");
  assert.equal(hits[0].opts.method, "POST");
});

test("re-embed sends no paths and never calls /add", async () => {
  const { calls, fetchImpl } = recorder();
  const window = setup(fetchImpl);
  runScript(window, `kbConfirmReembed = () => Promise.resolve(true);`);
  runScript(window, "kbReembedCollection('mycoll');");
  for (let i = 0; i < 8; i++) await tick();

  assert.equal(calls.filter((c) => c.url.includes("/add")).length, 0,
    "re-embed must not re-add documents");
  const body = reembedCalls(calls)[0].opts.body;
  assert.ok(body === undefined || body === null,
    "the endpoint takes no body - nothing to filter, batch or exclude");
});

test("re-embed does not fetch the collection's document list first", async () => {
  const { calls, fetchImpl } = recorder();
  const window = setup(fetchImpl);
  runScript(window, `kbConfirmReembed = () => Promise.resolve(true);`);
  runScript(window, "kbReembedCollection('mycoll');");
  for (let i = 0; i < 8; i++) await tick();

  const docListGets = calls.filter((c) =>
    c.url === "/api/rag/collections/mycoll" && (c.opts.method || "GET") === "GET");
  assert.equal(docListGets.length, 0, "no doc-list GET is needed any more");
});

test("an uploaded-only collection IS re-embedded, not refused", async () => {
  // Uploaded documents have no path on the server disk; the endpoint works
  // from stored chunk text, so they re-embed like any other document.
  const { calls, fetchImpl } = recorder();
  const window = setup(fetchImpl);
  runScript(window, `kbConfirmReembed = () => Promise.resolve(true);`);
  runScript(window, "kbReembedCollection('mycoll');");
  for (let i = 0; i < 8; i++) await tick();

  assert.equal(reembedCalls(calls).length, 1,
    "an uploaded-only collection must still reach the server");
});

test("declining the confirm sends nothing at all", async () => {
  const { calls, fetchImpl } = recorder();
  const window = setup(fetchImpl);
  runScript(window, `kbConfirmReembed = () => Promise.resolve(false);`);
  runScript(window, "kbReembedCollection('mycoll');");
  for (let i = 0; i < 8; i++) await tick();
  assert.equal(reembedCalls(calls).length, 0, "no request when the user cancels");
});

test("a server error is surfaced, not swallowed", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    calls.push({ url: String(url), opts });
    if (String(url).includes("/reembed"))
      return { ok: false, status: 400, text: async () => "",
               json: async () => ({ detail: "No embedding model is available" }) };
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ collections: [] }) };
  };
  const window = setup(fetchImpl);
  runScript(window, `kbConfirmReembed = () => Promise.resolve(true);`);
  runScript(window, "kbReembedCollection('mycoll');");
  for (let i = 0; i < 8; i++) await tick();

  const log = window.document.getElementById("kb-log");
  assert.match(log.textContent, /failed: No embedding model is available/,
    "the server's own reason must reach the user, not a generic failure");
});

test("plain 'add docs' (kbRunAdd default) still does NOT force reindex", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    calls.push({ url: String(url), opts });
    // kbRunAdd calls refreshKnowledgePage() when the job finishes, which needs
    // a collections list, so unmatched URLs fall through to one below.
    if (String(url).includes("/add"))
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ job_id: "j1" }) };
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ collections: [] }) };
  };
  const window = setup(fetchImpl);
  runScript(window, `kbRunAdd('mycoll', ['/a/one.md'], true, $("kb-log"));`);
  await tick();
  await tick();
  const add = calls.find((c) => c.url.includes("/add"));
  assert.ok(add, "an /add POST was sent");
  assert.equal(JSON.parse(add.opts.body).reindex, false,
    "a normal add-docs click must not force-reindex unchanged files");
});
