// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// The Knowledge page's Reranking card: the rag_rerank toggle and one line saying
// what retrieval will do, from GET /api/rag/rerank.

const tick = () => new Promise((r) => setTimeout(r, 0));
const json = (body, ok = true) =>
  ({ ok, status: ok ? 200 : 500, text: async () => "", json: async () => body });

function setup(state, { retrievalNote = null } = {}) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    if (u.endsWith("/api/rag/rerank")) return json(state.current);
    if (u.endsWith("/v1/config")) {
      state.current = { ...state.current, enabled: JSON.parse(opts.body).rag_rerank };
      return json({});
    }
    if (u.includes("/query"))
      return json({ hits: [], rerank_note: retrievalNote, reranked: false });
    if (/\/api\/rag\/collections$/.test(u)) return json({ collections: [] });
    if (u.includes("/api/rag/embedding")) return json({ status: "ready", model: "m", internal: [] });
    if (u.includes("/api/models")) return json({ models: [] });
    return json({});
  };
  const { window } = loadAppWithPages({ fetchImpl });
  return { window, calls };
}

const panel = (window) => ({
  box: window.document.getElementById("kb-rerank"),
  status: window.document.getElementById("kb-rerank-status"),
});

test("a usable reranker: the box is checked and the line names the model and count", async () => {
  const state = { current: { enabled: true, installed: ["qwen"], model: "qwen",
                             candidates: 20, note: null } };
  const { window } = setup(state);
  runScript(window, "refreshRerankPanel();");
  await tick(); await tick();
  const { box, status } = panel(window);
  assert.equal(box.checked, true);
  assert.equal(box.disabled, false);
  assert.match(status.textContent, /Reranking the best 20 matches with qwen/);
});

test("enabled with no reranker installed: the line says so and gives the pull command", async () => {
  const state = { current: { enabled: true, installed: [], model: null,
                             candidates: 20, note: null } };
  const { window } = setup(state);
  runScript(window, "refreshRerankPanel();");
  await tick(); await tick();
  const { box, status } = panel(window);
  assert.equal(box.checked, true);
  assert.match(status.textContent, /No reranker model is installed/);
  assert.match(status.textContent, /localm pull ggml-org\/Qwen3-Reranker/);
});

test("switched off: the line says results are returned as searched", async () => {
  const state = { current: { enabled: false, installed: ["qwen"], model: "qwen",
                             candidates: 20, note: null } };
  const { window } = setup(state);
  runScript(window, "refreshRerankPanel();");
  await tick(); await tick();
  const { box, status } = panel(window);
  assert.equal(box.checked, false);
  assert.match(status.textContent, /Reranking is off/);
});

test("a configured reranker that cannot be used shows the server's reason", async () => {
  const state = { current: { enabled: true, installed: ["a", "b"], model: null,
                             candidates: 20,
                             note: "Several rerankers are registered (a, b); name one." } };
  const { window } = setup(state);
  runScript(window, "refreshRerankPanel();");
  await tick(); await tick();
  assert.match(panel(window).status.textContent,
    /Reranking cannot run: Several rerankers are registered/);
});

test("a failed status read disables the box and says why", async () => {
  const window = loadAppWithPages({
    fetchImpl: async (url) => String(url).endsWith("/api/rag/rerank")
      ? json({ detail: "boom" }, false) : json({ collections: [], models: [] }),
  }).window;
  runScript(window, "refreshRerankPanel();");
  await tick(); await tick();
  const { box, status } = panel(window);
  assert.equal(box.disabled, true);
  assert.match(status.textContent, /Could not read the reranking state: boom/);
});

test("ticking the box PATCHes rag_rerank and repaints from the server", async () => {
  const state = { current: { enabled: true, installed: ["qwen"], model: "qwen",
                             candidates: 20, note: null } };
  const { window, calls } = setup(state);
  runScript(window, "refreshRerankPanel();");
  await tick(); await tick();
  const { box, status } = panel(window);
  box.checked = false;
  box.dispatchEvent(new window.Event("change"));
  await tick(); await tick(); await tick();
  const patch = calls.find((c) => c.url.endsWith("/v1/config") && c.opts.method === "PATCH");
  assert.ok(patch, "a config PATCH was sent");
  assert.deepEqual(JSON.parse(patch.opts.body), { rag_rerank: false });
  assert.match(status.textContent, /Reranking is off/);
  assert.equal(box.disabled, false, "the box is usable again afterwards");
});

test("a chat retrieval whose rerank was skipped toasts the reason once", async () => {
  const state = { current: { enabled: true, installed: [], model: null,
                             candidates: 20, note: null } };
  const { window } = setup(state, { retrievalNote: "reranking failed (RuntimeError); using the unreranked order" });
  runScript(window, `
    document.getElementById("p-kb").innerHTML = '<option value="kb" selected>kb</option>';
    document.getElementById("p-kb").value = "kb";
    globalThis.__conv = { messages: [] };
    retrieveKnowledge(globalThis.__conv, "how do I do x?");
  `);
  await tick(); await tick(); await tick();
  const toast = window.document.getElementById("toast");
  assert.match(toast.textContent, /Knowledge results were not reranked: reranking failed/);
  toast.textContent = "";
  runScript(window, `retrieveKnowledge(globalThis.__conv, "and y?");`);
  await tick(); await tick(); await tick();
  assert.equal(toast.textContent, "", "the same reason is not toasted twice in a row");
});
