// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// A collection that was indexed before an embedding model was set up (or
// before switching models) stays BM25/lexical forever unless its existing
// documents are force-reindexed - and clicking "add docs" again on an
// unchanged folder is a no-op (the server skips unchanged files). The
// "reindex" button reuses the collection's own known document list and
// forces the server to re-embed regardless.

const tick = () => new Promise((r) => setTimeout(r, 0));
const addBody = (calls) => {
  const c = calls.find((x) => x.url.includes("/add"));
  return c ? JSON.parse(c.opts.body) : null;
};

function setup(fetchImpl) {
  const { window } = loadAppWithPages({ fetchImpl });
  runScript(window, `streamJob = () => Promise.resolve({ status: "done" });`);
  return window;
}

function fetchFor(docs) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.includes("/add"))
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ job_id: "j1" }) };
    if (u.includes("/api/rag/collections/mycoll"))
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ name: "mycoll", n_docs: docs.length,
                                    n_chunks: docs.length, has_vectors: false,
                                    docs }) };
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ collections: [] }) };
  };
}

test("reindex fetches the collection's own doc list and force-reindexes them", async () => {
  const calls = [];
  const docs = [{ path: "/a/one.md", chunks: 2 }, { path: "/a/two.md", chunks: 1 }];
  const fetchImpl = async (url, opts) => {
    calls.push({ url: String(url), opts });
    return fetchFor(docs)(url, opts);
  };
  const window = setup(fetchImpl);
  runScript(window, `kbConfirmReindex = () => Promise.resolve(true);`);
  runScript(window, "kbReindexCollection('mycoll');");
  for (let i = 0; i < 8; i++) await tick();

  const body = addBody(calls);
  assert.ok(body, "an /add POST was sent");
  assert.deepEqual(body.paths.slice().sort(), ["/a/one.md", "/a/two.md"]);
  assert.equal(body.reindex, true, "reindex must be forced, or unchanged files are skipped");
});

test("plain 'add docs' (kbRunAdd default) does NOT force reindex", async () => {
  const calls = [];
  const fetchImpl = async (url, opts) => {
    calls.push({ url: String(url), opts });
    return fetchFor([])(url, opts);
  };
  const window = setup(fetchImpl);
  runScript(window, `kbRunAdd('mycoll', ['/a/one.md'], true, $("kb-log"));`);
  await tick();
  await tick();
  const body = addBody(calls);
  assert.ok(body, "an /add POST was sent");
  assert.equal(body.reindex, false, "a normal add-docs click must not force-reindex unchanged files");
});

test("declining the reindex confirm sends no /add request", async () => {
  const calls = [];
  const docs = [{ path: "/a/one.md" }];
  const fetchImpl = async (url, opts) => {
    calls.push({ url: String(url), opts });
    return fetchFor(docs)(url, opts);
  };
  const window = setup(fetchImpl);
  runScript(window, `kbConfirmReindex = () => Promise.resolve(false);`);
  runScript(window, "kbReindexCollection('mycoll');");
  for (let i = 0; i < 8; i++) await tick();
  assert.equal(calls.filter((c) => c.url.includes("/add")).length, 0,
    "no /add POST when the user cancels");
});

test("a collection with only uploaded docs has nothing to reindex", async () => {
  const calls = [];
  const docs = [{ path: "upload:report.pdf", uploaded: true }];
  const fetchImpl = async (url, opts) => {
    calls.push({ url: String(url), opts });
    return fetchFor(docs)(url, opts);
  };
  const window = setup(fetchImpl);
  runScript(window, `kbConfirmReindex = () => Promise.resolve(true);`);
  runScript(window, "kbReindexCollection('mycoll');");
  for (let i = 0; i < 8; i++) await tick();
  assert.equal(calls.filter((c) => c.url.includes("/add")).length, 0,
    "uploaded-only docs can't be re-read from server disk, so no /add is sent");
});

test("uploaded docs are skipped but server-path docs still reindex", async () => {
  const calls = [];
  const docs = [{ path: "/a/one.md" }, { path: "upload:report.pdf", uploaded: true }];
  const fetchImpl = async (url, opts) => {
    calls.push({ url: String(url), opts });
    return fetchFor(docs)(url, opts);
  };
  const window = setup(fetchImpl);
  runScript(window, `kbConfirmReindex = () => Promise.resolve(true);`);
  runScript(window, "kbReindexCollection('mycoll');");
  for (let i = 0; i < 8; i++) await tick();
  const body = addBody(calls);
  assert.ok(body, "an /add POST was sent");
  assert.deepEqual(body.paths, ["/a/one.md"], "only the re-readable doc is sent");
});

// The server caps a single /add request at 50 paths ("Too many paths in one
// request (max 50)", rag/plug.py). A collection with more documents than that
// must be split into multiple <=50-path batches client-side, or reindexing a
// large collection fails outright.
test("reindexing more than 50 docs is split into batches of <=50 paths", async () => {
  const calls = [];
  const docs = Array.from({ length: 152 }, (_, i) => ({ path: `/a/doc${i}.md` }));
  const fetchImpl = async (url, opts) => {
    calls.push({ url: String(url), opts });
    return fetchFor(docs)(url, opts);
  };
  const window = setup(fetchImpl);
  runScript(window, `kbConfirmReindex = () => Promise.resolve(true);`);
  runScript(window, "kbReindexCollection('mycoll');");
  for (let i = 0; i < 20; i++) await tick();

  const addCalls = calls.filter((c) => c.url.includes("/add"));
  assert.equal(addCalls.length, 4, "152 docs at 50/batch must be 4 requests (50+50+50+2)");
  const batchSizes = addCalls.map((c) => JSON.parse(c.opts.body).paths.length);
  assert.deepEqual(batchSizes, [50, 50, 50, 2]);
  for (const c of addCalls)
    assert.ok(JSON.parse(c.opts.body).paths.length <= 50, "no batch exceeds the server's 50-path cap");

  const allPaths = addCalls.flatMap((c) => JSON.parse(c.opts.body).paths);
  assert.deepEqual(allPaths.slice().sort(), docs.map((d) => d.path).sort(),
    "every document is covered exactly once across batches");
});

test("reindexing 50 or fewer docs sends a single batch (no unnecessary split)", async () => {
  const calls = [];
  const docs = Array.from({ length: 50 }, (_, i) => ({ path: `/a/doc${i}.md` }));
  const fetchImpl = async (url, opts) => {
    calls.push({ url: String(url), opts });
    return fetchFor(docs)(url, opts);
  };
  const window = setup(fetchImpl);
  runScript(window, `kbConfirmReindex = () => Promise.resolve(true);`);
  runScript(window, "kbReindexCollection('mycoll');");
  for (let i = 0; i < 8; i++) await tick();

  const addCalls = calls.filter((c) => c.url.includes("/add"));
  assert.equal(addCalls.length, 1, "exactly 50 docs must not be split into a 2nd empty batch");
  assert.equal(JSON.parse(addCalls[0].opts.body).paths.length, 50);
});
