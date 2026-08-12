// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// NEW-RAG-DIM-NO-REEMBED (2026-08-12 follow-up): PATCH /v1/config is the
// OTHER GUI writer of embedding_model (the Settings page's editable field),
// alongside the RAG picker's POST /api/rag/embedding (see
// knowledge-embed-confirm.test.mjs). saveSettingsSection() now handles a
// needs_confirm response from PATCH /v1/config the same way applyEmbeddingModel
// handles one from the RAG-picker route: show the in-page confirm modal, and
// only on "Switch anyway" re-PATCH the same body plus confirm:true. Unlike
// that route, PATCH /v1/config only gates when there is something to lose, so
// the no-risk case is a single request with no modal at all.

const SCHEMA = { fields: [
  { key: "embedding_model", widget: "text", label: "Embedding model", help: "",
    group: "Models", owner: "core", default: "bge-small-en-v1.5",
    shipped_default: "bge-small-en-v1.5" },
] };

function makeFetch(patchResponses) {
  const calls = [];
  let patchCall = 0;
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    if (u === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    }
    if (u === "/v1/config" && (opts.method || "GET") === "PATCH") {
      const data = patchResponses[Math.min(patchCall, patchResponses.length - 1)];
      patchCall += 1;
      return { ok: true, status: 200, json: async () => data, text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
  return { fetchImpl, calls };
}

function confirmedBody(calls) {
  const c = calls.find((x) => x.url === "/v1/config" && x.opts.method === "PATCH"
    && JSON.parse(x.opts.body).confirm);
  return c ? JSON.parse(c.opts.body) : null;
}

function patchCalls(calls) {
  return calls.filter((x) => x.url === "/v1/config" && x.opts.method === "PATCH");
}

async function renderAndEditEmbeddingModel(win, value) {
  runScript(win, "refreshSettingsPage();");
  await new Promise((r) => setTimeout(r, 0));
  const input = win.document.querySelector('input[data-key="embedding_model"]');
  input.value = value;
  input.dispatchEvent(new win.Event("input"));
  const secId = input.closest(".settings-section").dataset.sec;
  runScript(win, `saveSettingsSection(${JSON.stringify(secId)});`);
  await new Promise((r) => setTimeout(r, 0));
}

test("no affected collections: a single PATCH writes directly, no modal", async () => {
  const { fetchImpl, calls } = makeFetch([
    { embedding_model: "new-model" },   // the ordinary merged-config response
  ]);
  const { window: win } = loadAppWithPages({ fetchImpl });

  await renderAndEditEmbeddingModel(win, "new-model");
  await new Promise((r) => setTimeout(r, 0));

  const modal = win.document.getElementById("modal");
  assert.equal(modal.style.display, "none", "no confirm modal when nothing is at risk");
  assert.equal(patchCalls(calls).length, 1, "exactly one PATCH, no round trip");
  assert.equal(JSON.parse(patchCalls(calls)[0].opts.body).embedding_model, "new-model");
});

test("affected collections: the confirm modal shows them before any confirmed PATCH", async () => {
  const { fetchImpl, calls } = makeFetch([
    {
      needs_confirm: true, model: "new-model",
      collections: [{ name: "docs", built_with: "old-model", n_chunks: 42 }],
      note: "Switching to 'new-model' may invalidate the semantic search of "
            + "1 existing collection(s) until they are re-embedded.",
    },
  ]);
  const { window: win } = loadAppWithPages({ fetchImpl });

  await renderAndEditEmbeddingModel(win, "new-model");
  await new Promise((r) => setTimeout(r, 0));

  const modal = win.document.getElementById("modal");
  assert.notEqual(modal.style.display, "none", "confirm modal is shown");
  assert.match(modal.textContent, /may invalidate/);
  assert.match(modal.textContent, /docs/);
  assert.match(modal.textContent, /old-model/);
  assert.match(modal.textContent, /42 chunks/);
  assert.equal(confirmedBody(calls), null, "no confirmed PATCH before the user answers");
});

test("confirming the modal re-PATCHes the same body plus confirm:true", async () => {
  const { fetchImpl, calls } = makeFetch([
    {
      needs_confirm: true, model: "new-model",
      collections: [{ name: "docs", built_with: "old-model", n_chunks: 42 }],
      note: "may invalidate",
    },
    { embedding_model: "new-model" },
  ]);
  const { window: win } = loadAppWithPages({ fetchImpl });

  await renderAndEditEmbeddingModel(win, "new-model");

  const buttons = [...win.document.getElementById("modal").querySelectorAll("button")];
  const ok = buttons.find((b) => b.textContent.includes("Switch anyway"));
  assert.ok(ok, "the confirm button is present");
  ok.onclick();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  const confirmed = confirmedBody(calls);
  assert.ok(confirmed, "the confirmed PATCH was sent after clicking through");
  assert.equal(confirmed.embedding_model, "new-model");
  assert.equal(confirmed.confirm, true);
  assert.equal(patchCalls(calls).length, 2, "dry-run then confirmed - no extra requests");
});

test("cancelling the modal sends no confirmed PATCH and reports Cancelled", async () => {
  const { fetchImpl, calls } = makeFetch([
    {
      needs_confirm: true, model: "new-model",
      collections: [{ name: "docs", built_with: "old-model", n_chunks: 42 }],
      note: "may invalidate",
    },
  ]);
  const { window: win } = loadAppWithPages({ fetchImpl });

  await renderAndEditEmbeddingModel(win, "new-model");

  const buttons = [...win.document.getElementById("modal").querySelectorAll("button")];
  const cancel = buttons.find((b) => b.textContent.includes("Cancel"));
  assert.ok(cancel, "the cancel button is present");
  cancel.onclick();
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(confirmedBody(calls), null, "cancelling never sends the confirmed PATCH");
  assert.equal(patchCalls(calls).length, 1, "only the dry-run PATCH was ever sent");
  assert.match(win.document.getElementById("toast").textContent, /Cancelled/);
});
