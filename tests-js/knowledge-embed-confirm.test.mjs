// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// Switching the embedding model dry-runs first (POST without confirm); if that
// names affected collections the user is asked, then a second POST with
// confirm:true makes the change.

const tick = () => new Promise((r) => setTimeout(r, 0));

function setup({ dryRun, jobOk = true }) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    if (u.includes("/api/rag/embedding") && opts.method === "POST") {
      const body = JSON.parse(opts.body);
      if (!body.confirm) {
        return { ok: true, status: 200, text: async () => "", json: async () => dryRun };
      }
      return { ok: true, status: 200, text: async () => "", json: async () => ({ job_id: "j1" }) };
    }
    if (u.includes("/api/rag/embedding"))
      return { ok: true, status: 200, text: async () => "", json: async () => ({
        status: "ready", model: "bge-small-en-v1.5", dim: 384, internal: [], error: null }) };
    if (u.includes("/api/models"))
      return { ok: true, status: 200, text: async () => "", json: async () => ({ models: [] }) };
    return { ok: true, status: 200, text: async () => "", json: async () => ({ collections: [] }) };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  runScript(window, `streamJob = () => Promise.resolve({ status: "${jobOk ? "done" : "failed"}" });`);
  return { window, calls };
}

function confirmBody(calls) {
  const c = calls.find((x) => x.url.includes("/api/rag/embedding") && x.opts.method === "POST"
    && JSON.parse(x.opts.body).confirm);
  return c ? JSON.parse(c.opts.body) : null;
}

test("nothing affected: no modal, dry-run then confirm both fire, job runs", async () => {
  const { window, calls } = setup({
    dryRun: { needs_confirm: true, model: "new-model", collections: [],
              note: "No existing collection currently has embeddings, so switching "
                    + "to 'new-model' has nothing to invalidate." },
  });
  runScript(window, `applyEmbeddingModel("new-model");`);
  await tick(); await tick(); await tick(); await tick(); await tick();

  const modal = window.document.getElementById("modal");
  assert.equal(modal.style.display, "none", "no confirm modal when nothing is affected");
  const dryCalls = calls.filter((c) => c.url.includes("/api/rag/embedding")
    && c.opts.method === "POST" && !JSON.parse(c.opts.body).confirm);
  assert.equal(dryCalls.length, 1, "exactly one dry-run POST");
  const confirmed = confirmBody(calls);
  assert.ok(confirmed, "the confirmed POST was sent");
  assert.equal(confirmed.model, "new-model");
});

test("affected collections: the confirm modal shows them before any confirmed POST", async () => {
  const { window, calls } = setup({
    dryRun: {
      needs_confirm: true, model: "new-model",
      collections: [{ name: "docs", built_with: "old-model", n_chunks: 42 }],
      note: "Switching to 'new-model' may invalidate the semantic search of "
            + "1 existing collection(s) until they are re-embedded.",
    },
  });
  runScript(window, `applyEmbeddingModel("new-model");`);
  await tick(); await tick(); await tick();

  const modal = window.document.getElementById("modal");
  assert.notEqual(modal.style.display, "none", "confirm modal is shown");
  assert.match(modal.textContent, /may invalidate/);
  assert.match(modal.textContent, /docs/);
  assert.match(modal.textContent, /old-model/);
  assert.match(modal.textContent, /42 chunks/);
  assert.equal(confirmBody(calls), null, "no confirmed POST before the user answers");
});

test("confirming the modal sends the confirm:true POST and runs the job", async () => {
  const { window, calls } = setup({
    dryRun: {
      needs_confirm: true, model: "new-model",
      collections: [{ name: "docs", built_with: "old-model", n_chunks: 42 }],
      note: "may invalidate",
    },
  });
  runScript(window, `applyEmbeddingModel("new-model");`);
  await tick(); await tick();

  const buttons = [...window.document.getElementById("modal").querySelectorAll("button")];
  const ok = buttons.find((b) => b.textContent.includes("Switch anyway"));
  assert.ok(ok, "the confirm button is present");
  ok.onclick();
  await tick(); await tick(); await tick(); await tick();

  const confirmed = confirmBody(calls);
  assert.ok(confirmed, "the confirmed POST was sent after clicking through");
  assert.equal(confirmed.model, "new-model");
  assert.equal(confirmed.confirm, true);
});

test("cancelling the modal sends no confirmed POST and no job", async () => {
  const { window, calls } = setup({
    dryRun: {
      needs_confirm: true, model: "new-model",
      collections: [{ name: "docs", built_with: "old-model", n_chunks: 42 }],
      note: "may invalidate",
    },
  });
  runScript(window, `applyEmbeddingModel("new-model");`);
  await tick(); await tick();

  const buttons = [...window.document.getElementById("modal").querySelectorAll("button")];
  const cancel = buttons.find((b) => b.textContent.includes("Cancel"));
  assert.ok(cancel, "the cancel button is present");
  cancel.onclick();
  await tick(); await tick(); await tick();

  assert.equal(confirmBody(calls), null, "cancelling never sends the confirmed POST");
  const log = window.document.getElementById("kb-embed-log");
  assert.match(log.textContent, /Cancelled/);
});
