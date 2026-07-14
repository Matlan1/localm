// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// A collection can show "BM25" for two very different reasons: embeddings were
// never turned on (nothing actionable), or the embedding model IS ready but
// this collection predates it / its vector index went stale (actionable -
// "reindex" fixes it). Before this, the only place that distinction surfaced
// was a small warning inside the per-collection info modal; the collections
// table itself just said "BM25" with no hint that anything could be done. Now
// a ready-but-BM25 row gets a visible inline badge and the reindex button is
// highlighted as the fix.

const tick = () => new Promise((r) => setTimeout(r, 0));

function setup({ collections, embedding }) {
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (/\/api\/rag\/collections$/.test(u))
      return { ok: true, status: 200, text: async () => "", json: async () => ({ collections }) };
    if (u.includes("/api/rag/embedding"))
      return { ok: true, status: 200, text: async () => "", json: async () => embedding };
    if (u.includes("/api/models"))
      return { ok: true, status: 200, text: async () => "", json: async () => ({ models: [] }) };
    return { ok: true, status: 200, text: async () => "", json: async () => ({}) };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  return window;
}

const READY = { status: "ready", model: "bge-small-en-v1.5", dim: 384, internal: [], error: null };
const NOT_INSTALLED = { status: "not_installed", model: "bge-small-en-v1.5", dim: null, internal: [], error: null };

test("BM25 row gets a 'reindex needed' badge + warn button when the embedding model is ready", async () => {
  const window = setup({
    collections: [{ name: "kb", n_docs: 1, n_chunks: 2, has_vectors: false,
                    vector_degrade_reason: null }],
    embedding: READY,
  });
  runScript(window, "refreshKnowledgePage();");
  await tick(); await tick(); await tick();

  const table = window.document.getElementById("kb-table");
  const badge = table.querySelector(".retrieval-badge");
  assert.ok(badge, "a .retrieval-badge is rendered inline in the row");
  assert.equal(badge.textContent, "reindex needed");

  const reindexBtn = [...table.querySelectorAll("button")]
    .find((b) => b.textContent.includes("reindex"));
  assert.ok(reindexBtn, "the reindex button is present");
  assert.equal(reindexBtn.className, "warn", "it is styled as the fix for the badge");
  assert.equal(reindexBtn.textContent, "reindex needed");
});

test("BM25 row shows the vector_degrade_reason in the badge tooltip when present", async () => {
  const window = setup({
    collections: [{ name: "kb", n_docs: 152, n_chunks: 1166, has_vectors: false,
                    vector_degrade_reason:
                      "vectors.json has 1192 vectors for 1166 chunks "
                      + "(orphaned entries from a prior, larger index); "
                      + "using BM25 lexical retrieval only" }],
    embedding: READY,
  });
  runScript(window, "refreshKnowledgePage();");
  await tick(); await tick(); await tick();

  const badge = window.document.getElementById("kb-table").querySelector(".retrieval-badge");
  assert.ok(badge, "badge is rendered");
  assert.match(badge.title, /orphaned entries/);
});

test("BM25 row has no badge and a plain reindex button when the embedding model is not installed", async () => {
  const window = setup({
    collections: [{ name: "kb", n_docs: 1, n_chunks: 2, has_vectors: false,
                    vector_degrade_reason: null }],
    embedding: NOT_INSTALLED,
  });
  runScript(window, "refreshKnowledgePage();");
  await tick(); await tick(); await tick();

  const table = window.document.getElementById("kb-table");
  assert.equal(table.querySelector(".retrieval-badge"), null,
    "no badge: nothing to reindex INTO since there is no ready embedding model");
  const reindexBtn = [...table.querySelectorAll("button")]
    .find((b) => b.textContent.includes("reindex"));
  assert.ok(reindexBtn);
  assert.equal(reindexBtn.className, "", "not highlighted - reindexing wouldn't help right now");
  assert.equal(reindexBtn.textContent, "reindex");
});

test("a hybrid row (has_vectors true) never gets the badge, even when the embedding model is ready", async () => {
  const window = setup({
    collections: [{ name: "kb", n_docs: 1, n_chunks: 2, has_vectors: true,
                    vector_degrade_reason: null }],
    embedding: READY,
  });
  runScript(window, "refreshKnowledgePage();");
  await tick(); await tick(); await tick();

  const table = window.document.getElementById("kb-table");
  assert.equal(table.querySelector(".retrieval-badge"), null);
  assert.match(table.textContent, /hybrid/);
  const reindexBtn = [...table.querySelectorAll("button")]
    .find((b) => b.textContent.includes("reindex"));
  assert.equal(reindexBtn.className, "");
});

test("the embedding-ready panel copy qualifies that existing collections may need reindex", async () => {
  const window = setup({
    collections: [{ name: "kb", n_docs: 1, n_chunks: 2, has_vectors: true,
                    vector_degrade_reason: null }],
    embedding: READY,
  });
  runScript(window, "refreshKnowledgePage();");
  await tick(); await tick(); await tick();

  const status = window.document.getElementById("kb-embed-status");
  assert.match(status.textContent, /semantic search is on for new indexing/);
  assert.match(status.textContent, /Existing collections may need "reindex"/);
});

test("clicking the highlighted reindex button still force-reindexes the collection", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    if (/\/api\/rag\/collections$/.test(u))
      return { ok: true, status: 200, text: async () => "", json: async () => ({
        collections: [{ name: "kb", n_docs: 1, n_chunks: 1, has_vectors: false,
                        vector_degrade_reason: null }] }) };
    if (u.includes("/api/rag/embedding"))
      return { ok: true, status: 200, text: async () => "", json: async () => READY };
    if (u.includes("/api/models"))
      return { ok: true, status: 200, text: async () => "", json: async () => ({ models: [] }) };
    if (/\/api\/rag\/collections\/kb$/.test(u))
      return { ok: true, status: 200, text: async () => "", json: async () => ({
        name: "kb", n_docs: 1, n_chunks: 1, has_vectors: false,
        docs: [{ path: "/a/one.md" }] }) };
    if (u.includes("/add"))
      return { ok: true, status: 200, text: async () => "", json: async () => ({ job_id: "j1" }) };
    return { ok: true, status: 200, text: async () => "", json: async () => ({}) };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  runScript(window, `streamJob = () => Promise.resolve({ status: "done" });`);
  runScript(window, `kbConfirmReindex = () => Promise.resolve(true);`);
  runScript(window, "refreshKnowledgePage();");
  await tick(); await tick(); await tick();

  const reindexBtn = [...window.document.getElementById("kb-table").querySelectorAll("button")]
    .find((b) => b.textContent.includes("reindex"));
  reindexBtn.onclick();
  for (let i = 0; i < 8; i++) await tick();

  const addCall = calls.find((c) => c.url.includes("/add"));
  assert.ok(addCall, "an /add POST was sent");
  const body = JSON.parse(addCall.opts.body);
  assert.equal(body.reindex, true);
  assert.deepEqual(body.paths, ["/a/one.md"]);
});
