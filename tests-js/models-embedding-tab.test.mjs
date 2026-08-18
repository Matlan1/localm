// SPDX-License-Identifier: AGPL-3.0-or-later
// The Models page "Embedding" tab: a dedicated tab-nav button that filters the
// table to embedding models (data-type="embedding"), plus the per-row set-type
// <select> offering "embedding" as one of the selectable model types. Mirrors
// tests-js/models-set-type.test.mjs (same page, same harness/fetch-mock style).

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(models, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    calls.push(u);
    if (u.startsWith("/api/models/type")) {
      const body = opts.body ? JSON.parse(opts.body) : {};
      return {
        ok: true, status: 200,
        json: async () => ({ status: "typed", model: body.model, model_type: body.model_type }),
        text: async () => "",
      };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return {
        ok: true, status: 200,
        json: async () => ({ models, active: models.find((m) => m.active)?.name || null }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

test("models-embedding-tab: the Embedding tab button exists with the right data-type and label", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([], []) });

  const nav = window.document.querySelector("#models-tab-nav");
  assert.ok(nav, "the tab-nav container exists");
  const btn = nav.querySelector('.tab-btn[data-type="embedding"]');
  assert.ok(btn, "a tab-btn with data-type=\"embedding\" exists in the nav");
  assert.equal(btn.textContent.trim(), "Embedding", "the tab is labelled 'Embedding'");
});

test("models-embedding-tab: clicking the Embedding tab activates it and re-fetches type=embedding", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([], calls) });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));

  const nav = window.document.querySelector("#models-tab-nav");
  const allBtn = nav.querySelector('.tab-btn[data-type="all"]');
  const embeddingBtn = nav.querySelector('.tab-btn[data-type="embedding"]');
  assert.ok(allBtn.classList.contains("active"), "the All tab starts active");
  assert.ok(!embeddingBtn.classList.contains("active"), "the Embedding tab starts inactive");

  calls.length = 0;   // only care about what the click itself triggers
  embeddingBtn.click();
  await new Promise((r) => setTimeout(r, 0));

  assert.ok(embeddingBtn.classList.contains("active"), "clicking the Embedding tab marks it active");
  assert.ok(!allBtn.classList.contains("active"), "clicking the Embedding tab deactivates its sibling (All)");
  for (const b of nav.querySelectorAll(".tab-btn")) {
    if (b !== embeddingBtn) {
      assert.ok(!b.classList.contains("active"), `sibling tab '${b.dataset.type}' is not active`);
    }
  }

  assert.ok(calls.includes("/api/models?type=embedding"),
    "the click re-fetched the models list filtered to type=embedding");
});

test("models-embedding-tab: an embedding-type row renders and its set-type control offers 'embedding'", async () => {
  const models = [
    { name: "bge-small", active: false, loaded: false, model_type: "embedding", size_bytes: 1000 },
  ];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models, []) });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));

  const row = window.document.querySelector("#models-table tbody tr");
  assert.ok(row, "the embedding model renders a row");
  assert.ok(row.textContent.includes("bge-small"), "the row shows the model name");

  const sel = row.querySelector("select");
  assert.ok(sel, "the row exposes the set-type <select>");
  assert.equal(sel.value, "embedding", "the control reflects the model's current type");
  const opts = [...sel.querySelectorAll("option")].map((o) => o.value);
  assert.ok(opts.includes("embedding"), "the control offers 'embedding' as a selectable type");
  assert.ok(opts.includes("llm"), "the control still offers the other real model types");
});
