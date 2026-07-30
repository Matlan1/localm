// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// The Knowledge page's re-embed button must call the server's own re-embed
// endpoint, which recomputes vectors from the chunk text already stored in the
// collection.
//
// It used to re-ADD the collection's documents with force instead, and this
// file used to pin that behaviour in seven tests - including one asserting that
// uploaded documents being SKIPPED was correct, and four testing the 50-path
// batching needed to stay under the server's /add cap. All of that machinery
// existed only because the implementation was re-reading source files, and none
// of it could fix the case the button exists for: after an embedding-model
// switch, re-adding trips the very dimension guard the user is trying to get
// past, and it excludes uploads entirely. The old tests were not missing the
// bug, they specified it - which is why it survived to ship.
//
// What the endpoint gives instead: no source file has to still exist, uploads
// are included like any other document, nothing is deleted, and the previous
// index stays in place until the new one is complete. So there are no paths to
// send, no doc list to fetch, and no batching.

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
  // The regression this guards: re-adding source files is what could not fix a
  // dimension mismatch. If a /add POST ever reappears here, the button has been
  // wired back to the mechanism that cannot recover the user.
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
  // Fetching it was how the old implementation obtained paths to re-add. Needing
  // it again would mean the path-based approach had returned.
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
  // THE REGRESSION THAT MATTERS, and the exact inverse of what this file used to
  // assert. Uploaded documents have no path on the server disk, so the old
  // implementation filtered them out with `!d.uploaded` and told the user to
  // re-upload everything. The endpoint works from stored chunk text, so an
  // uploaded-only collection re-embeds like any other - and the client cannot
  // even express the old exclusion, because it sends no paths at all.
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
  // Unchanged from before and still valid: this is about the add flow, which
  // keeps its reindex flag. Kept so the re-embed rewrite cannot quietly change
  // what a normal add-docs click does.
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    calls.push({ url: String(url), opts });
    // kbRunAdd calls refreshKnowledgePage() when the job finishes, which needs a
    // collections list - without this fallthrough the refresh throws rather than
    // the assertion below failing, which would look like a product bug.
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
