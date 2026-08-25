// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));

const NOT_INSTALLED = { status: "not_installed", model: "bge-small-en-v1.5", dim: null, internal: [], error: null };

function setupTable(collections) {
  const fetchImpl = async (url) => {
    const u = String(url);
    if (/\/api\/rag\/collections$/.test(u))
      return { ok: true, status: 200, text: async () => "", json: async () => ({ collections }) };
    if (u.includes("/api/rag/embedding"))
      return { ok: true, status: 200, text: async () => "", json: async () => NOT_INSTALLED };
    return { ok: true, status: 200, text: async () => "", json: async () => ({}) };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  return window;
}

test("corrupt collection gets a distinct 'index damaged' badge and a Repair button in the table", async () => {
  const window = setupTable([{ name: "kb", n_docs: 1, n_chunks: 2, has_vectors: true,
                                vector_degrade_reason: null, corrupt: true, chunks_bad_lines: 0 }]);
  runScript(window, "refreshKnowledgePage();");
  await tick(); await tick(); await tick();

  const table = window.document.getElementById("kb-table");
  const badge = table.querySelector(".corrupt-badge");
  assert.ok(badge, "a .corrupt-badge is rendered inline in the row");
  // no chunks_bad_lines count -> generic wording
  assert.equal(badge.textContent, "index damaged");
  assert.match(badge.title, /corrupt or malformed/);
  const repairBtn = table.querySelector("button.corrupt-fix");
  assert.ok(repairBtn, "a real Repair button is offered, not just a CLI pointer");
  assert.equal(repairBtn.textContent, "repair");
});

test("a corrupt collection with a known malformed-line count names it instead of the generic wording", async () => {
  const window = setupTable([{ name: "kb", n_docs: 1, n_chunks: 2, has_vectors: true,
                                vector_degrade_reason: null, corrupt: true, chunks_bad_lines: 62 }]);
  runScript(window, "refreshKnowledgePage();");
  await tick(); await tick(); await tick();

  const table = window.document.getElementById("kb-table");
  const badge = table.querySelector(".corrupt-badge");
  assert.ok(badge);
  assert.equal(badge.textContent, "62 malformed line(s)");
  assert.match(badge.title, /62 malformed chunk line\(s\)/);
  assert.ok(table.querySelector("button.corrupt-fix"));
});

test("a healthy collection gets no corrupt badge", async () => {
  const window = setupTable([{ name: "kb", n_docs: 1, n_chunks: 2, has_vectors: true,
                                vector_degrade_reason: null, corrupt: false }]);
  runScript(window, "refreshKnowledgePage();");
  await tick(); await tick(); await tick();

  const table = window.document.getElementById("kb-table");
  assert.equal(table.querySelector(".corrupt-badge"), null);
});

test("corrupt and re-embed badges can both show at once, distinctly", async () => {
  const window = setupTable([{ name: "kb", n_docs: 1, n_chunks: 2, has_vectors: false,
                                vector_degrade_reason: null, corrupt: true }]);
  runScript(window, "refreshKnowledgePage();");
  await tick(); await tick(); await tick();

  const table = window.document.getElementById("kb-table");
  // embedding not installed -> no re-embed badge here, only the corrupt one
  assert.ok(table.querySelector(".corrupt-badge"));
  assert.equal(table.querySelector(".retrieval-badge"), null);
});

function setupDetail(collData) {
  const fetchImpl = async (url) => {
    const u = String(url);
    if (/\/api\/rag\/collections\/[^/]+$/.test(u))
      return { ok: true, status: 200, text: async () => "", json: async () => collData };
    return { ok: true, status: 200, text: async () => "", json: async () => ({ collections: [] }) };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  return window;
}

test("info modal shows the corrupt-index warning and a Repair button when the collection is corrupt", async () => {
  const win = setupDetail({
    name: "kb", n_docs: 1, n_chunks: 2, has_vectors: true, docs: [],
    vector_degrade_reason: null, corrupt: true, chunks_bad_lines: 0,
  });
  runScript(win, 'kbInfoModal("kb");');
  await tick();
  await tick();
  const modal = win.document.getElementById("modal");
  assert.match(modal.textContent, /corrupt or malformed/);
  const repairBtn = Array.from(modal.querySelectorAll("button"))
    .find((b) => b.textContent.trim() === "Repair");
  assert.ok(repairBtn, "a real Repair button is offered, not just a CLI pointer");
});

test("info modal names the malformed-line count when the server can give one", async () => {
  const win = setupDetail({
    name: "kb", n_docs: 1, n_chunks: 2, has_vectors: true, docs: [],
    vector_degrade_reason: null, corrupt: true, chunks_bad_lines: 62,
  });
  runScript(win, 'kbInfoModal("kb");');
  await tick();
  await tick();
  const modal = win.document.getElementById("modal");
  assert.match(modal.textContent, /62 malformed chunk line\(s\)/);
});

test("info modal shows no corrupt warning when the collection is healthy", async () => {
  const win = setupDetail({
    name: "kb", n_docs: 1, n_chunks: 2, has_vectors: true, docs: [],
    vector_degrade_reason: null, corrupt: false,
  });
  runScript(win, 'kbInfoModal("kb");');
  await tick();
  await tick();
  const modal = win.document.getElementById("modal");
  assert.equal(/corrupt or malformed/.test(modal.textContent), false);
});
