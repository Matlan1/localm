// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// A BM25 row in the collections table carries an inline "re-embed needed"
// badge and a highlighted re-embed button when the embedding model is ready,
// and neither when it is not installed.

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

test("BM25 row gets a 're-embed needed' badge + warn button when the embedding model is ready", async () => {
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
  assert.equal(badge.textContent, "re-embed needed");

  const reembedBtn = [...table.querySelectorAll("button")]
    .find((b) => b.textContent.includes("re-embed"));
  assert.ok(reembedBtn, "the re-embed button is present");
  assert.equal(reembedBtn.className, "warn", "it is styled as the fix for the badge");
  assert.equal(reembedBtn.textContent, "re-embed needed");
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

test("BM25 row has no badge and a plain re-embed button when the embedding model is not installed", async () => {
  const window = setup({
    collections: [{ name: "kb", n_docs: 1, n_chunks: 2, has_vectors: false,
                    vector_degrade_reason: null }],
    embedding: NOT_INSTALLED,
  });
  runScript(window, "refreshKnowledgePage();");
  await tick(); await tick(); await tick();

  const table = window.document.getElementById("kb-table");
  assert.equal(table.querySelector(".retrieval-badge"), null,
    "no badge: nothing to re-embed INTO since there is no ready embedding model");
  const reembedBtn = [...table.querySelectorAll("button")]
    .find((b) => b.textContent.includes("re-embed"));
  assert.ok(reembedBtn);
  assert.equal(reembedBtn.className, "secondary", "not highlighted - re-embedding wouldn't help right now");
  assert.equal(reembedBtn.textContent, "re-embed");
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
  const reembedBtn = [...table.querySelectorAll("button")]
    .find((b) => b.textContent.includes("re-embed"));
  assert.equal(reembedBtn.className, "secondary");
});

test("the embedding-ready panel copy qualifies that existing collections may need re-embed", async () => {
  const window = setup({
    collections: [{ name: "kb", n_docs: 1, n_chunks: 2, has_vectors: true,
                    vector_degrade_reason: null }],
    embedding: READY,
  });
  runScript(window, "refreshKnowledgePage();");
  await tick(); await tick(); await tick();

  const status = window.document.getElementById("kb-embed-status");
  assert.match(status.textContent, /semantic search is on for new indexing/);
  assert.match(status.textContent, /Existing collections may need "re-embed"/);
});

test("clicking the highlighted re-embed button calls the reembed endpoint", async () => {
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
    if (u.includes("/reembed"))
      return { ok: true, status: 200, text: async () => "", json: async () => ({ job_id: "j1" }) };
    if (u.includes("/add"))
      return { ok: true, status: 200, text: async () => "", json: async () => ({ job_id: "j1" }) };
    return { ok: true, status: 200, text: async () => "", json: async () => ({}) };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  runScript(window, `streamJob = () => Promise.resolve({ status: "done" });`);
  runScript(window, `kbConfirmReembed = () => Promise.resolve(true);`);
  runScript(window, "refreshKnowledgePage();");
  await tick(); await tick(); await tick();

  const reembedBtn = [...window.document.getElementById("kb-table").querySelectorAll("button")]
    .find((b) => b.textContent.includes("re-embed"));
  reembedBtn.onclick();
  for (let i = 0; i < 8; i++) await tick();

  const reembedCall = calls.find((c) => c.url.includes("/reembed"));
  assert.ok(reembedCall, "a /reembed POST was sent");
  assert.equal(reembedCall.url, "/api/rag/collections/kb/reembed");
  assert.equal(reembedCall.opts.method, "POST");
  assert.equal(calls.filter((c) => c.url.includes("/add")).length, 0,
    "re-embedding must not fall back to re-adding source files");
});
