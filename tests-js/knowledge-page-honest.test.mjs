// SPDX-License-Identifier: AGPL-3.0-or-later
// The Knowledge page reports every failed request, creates on Enter, shows the
// reranker's score for reranked hits, and does not call a model whose last use
// failed "ready".
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));
const settle = async () => { for (let i = 0; i < 10; i++) await tick(); };
const ok = (body, status = 200) => ({ ok: status < 400, status, statusText: "", text: async () => "",
                                      json: async () => body });

function setup(routes = {}) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    for (const [part, fn] of Object.entries(routes)) if (u.includes(part)) return fn(u, opts);
    if (/\/api\/rag\/collections$/.test(u)) return ok({ collections: [] });
    return ok({});
  };
  const { window } = loadAppWithPages({ fetchImpl });
  return { window, calls };
}

const errorToasts = (win) => win.document.getElementById("toast-errors").textContent;

test("Enter in the new collection name creates it", async () => {
  const { window: win, calls } = setup({});
  const input = win.document.getElementById("kb-name");
  input.value = "notes";
  input.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  await settle();
  const posts = calls.filter((c) => /\/api\/rag\/collections$/.test(c.url) && c.opts.method === "POST");
  assert.equal(posts.length, 1);
  assert.deepEqual(JSON.parse(posts[0].opts.body), { name: "notes" });
});

test("creating a collection with the server unreachable says so", async () => {
  const { window: win } = setup({
    "/api/rag/collections": (u, opts) => {
      if (opts.method === "POST") throw new TypeError("Failed to fetch");
      return ok({ collections: [] });
    },
  });
  win.document.getElementById("kb-name").value = "notes";
  win.document.getElementById("kb-create").click();
  await settle();
  assert.match(errorToasts(win), /Create failed: Failed to fetch/);
});

test("a collection's info with a reply that is not JSON reports the HTTP failure", async () => {
  const { window: win } = setup({
    "/api/rag/collections/x": () => ({ ok: false, status: 502, statusText: "Bad Gateway",
      text: async () => "<html>", json: async () => { throw new SyntaxError("Unexpected token <"); } }),
  });
  runScript(win, 'kbInfoModal("x");');
  await settle();
  assert.match(errorToasts(win), /Could not load the collection: Bad Gateway/);
});

test("a reranked search hit shows the rerank score first", async () => {
  const { window: win } = setup({
    "/query": () => ok({ hits: [
      { score: 0.21, rerank_score: 0.97, source: "/d/a.md", pos: 3, text: "alpha" },
      { score: 0.55, rerank_score: 0.12, source: "/d/b.md", pos: 1, text: "beta" }] }),
  });
  runScript(win, 'kbSearchModal("x");');
  const input = win.document.querySelector(".kb-search-input");
  input.value = "q";
  input.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Enter" }));
  await settle();
  const lines = [...win.document.querySelectorAll("#modal-body .log-entry .t")].map((s) => s.textContent);
  assert.match(lines[0], /^rerank 0\.97 \(search 0\.21\)/);
  assert.match(lines[1], /^rerank 0\.12 \(search 0\.55\)/);
});

test("an installed embedding model whose last use failed is not called ready", async () => {
  const { window: win } = setup({
    "/api/rag/embedding": () => ok({ status: "ready", model: "chatmodel", dim: 0, internal: [],
                                     error: "this model cannot embed" }),
    "/api/models": () => ok({ models: [] }),
  });
  runScript(win, "refreshEmbeddingPanel();");
  await settle();
  const text = win.document.getElementById("kb-embed-status").textContent;
  assert.doesNotMatch(text, /semantic search is on/);
  assert.match(text, /last use failed/);
  assert.match(text, /this model cannot embed/);
});
