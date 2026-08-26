// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// A degraded vectors index is shown in the collection info modal. The server
// reports it as vector_degrade_reason on GET /api/rag/collections/<name>.

function setup(collData) {
  const fetchImpl = async (url) => {
    const u = String(url);
    if (/\/api\/rag\/collections\/[^/]+$/.test(u))
      return { ok: true, status: 200, text: async () => "", json: async () => collData };
    return { ok: true, status: 200, text: async () => "", json: async () => ({ collections: [] }) };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  return window;
}
const tick = () => new Promise((r) => setTimeout(r, 0));

test("info modal shows the vector degrade reason when the index is degraded", async () => {
  const win = setup({
    name: "kb", n_docs: 1, n_chunks: 2, has_vectors: false, docs: [],
    vector_degrade_reason:
      "vectors.json is unreadable (JSONDecodeError); using BM25 lexical retrieval only",
  });
  runScript(win, 'kbInfoModal("kb");');
  await tick();
  await tick();
  const modal = win.document.getElementById("modal");
  assert.match(modal.textContent, /fell back to BM25/);
  assert.match(modal.textContent, /unreadable/);
});

test("info modal shows no degrade warning when the index is healthy", async () => {
  const win = setup({
    name: "kb", n_docs: 1, n_chunks: 2, has_vectors: true,
    vector_degrade_reason: null, docs: [],
  });
  runScript(win, 'kbInfoModal("kb");');
  await tick();
  await tick();
  const modal = win.document.getElementById("modal");
  assert.equal(/fell back to BM25/.test(modal.textContent), false);
});
