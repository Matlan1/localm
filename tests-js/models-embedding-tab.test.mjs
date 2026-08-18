// SPDX-License-Identifier: AGPL-3.0-or-later
// The Models page "Embedding" type chip: a dedicated .reg-type chip that filters
// the table to embedding models (data-type="embedding"), plus the per-row
// set-type <select> offering "embedding" as one of the selectable model types.
// Mirrors tests-js/models-set-type.test.mjs (same page, harness, fetch-mock).

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

test("models-embedding-tab: the Embedding type chip exists with the right data-type and label", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([], []) });

  const row = window.document.querySelector("#models-type-filter");
  assert.ok(row, "the Registered-models type filter row exists");
  const chip = row.querySelector('.filter-chip[data-type="embedding"]');
  assert.ok(chip, "a chip with data-type=\"embedding\" exists in the row");
  assert.ok(chip.querySelector('input.reg-type[value="embedding"]'),
    "the chip wraps a real checkbox, so it is keyboard-focusable and multi-select");
  // The count <i> rides inside the pill, so read the label text without it.
  assert.equal(chip.querySelector("span").firstChild.textContent.trim(), "Embedding",
    "the chip is labelled 'Embedding'");
});

test("models-embedding-tab: narrowing to the Embedding chip leaves only embedding rows in the table", async () => {
  // This used to assert the request URL carried ?type=embedding. The filter is
  // multi-select now and narrows client-side, so the URL no longer says
  // anything - and the RENDERED TABLE is the property that actually mattered.
  // Asserting on what the user sees also survives the next transport change.
  const models = [
    { name: "bge-small", active: false, loaded: false, model_type: "embedding", size_bytes: 1000 },
    { name: "qwen-7b", active: false, loaded: false, model_type: "llm", size_bytes: 2000 },
  ];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models, []) });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));

  const names = () => [...window.document.querySelectorAll("#models-table tbody tr .name")]
    .map((n) => n.textContent);
  assert.deepEqual(names().sort(), ["bge-small", "qwen-7b"], "every type ticked shows both rows");

  // .reg-chip, not a bare .filter-chip[data-type=]: the search row and this row
  // share the component AND the data-type values, so an unscoped selector picks
  // up whichever comes first in the DOM (the search chip).
  const chip = window.document.querySelector('.reg-chip[data-type="embedding"]');
  const box = chip.querySelector("input.reg-type");
  assert.ok(box.checked, "the Embedding chip starts ticked (the default is every type)");

  for (const b of window.document.querySelectorAll(".reg-type")) {
    b.checked = (b.value === "embedding");
    b.dispatchEvent(new window.Event("change"));
  }
  await new Promise((r) => setTimeout(r, 0));

  assert.ok(chip.classList.contains("on"),
    "the ticked chip carries the .on class the CSS colours on");
  for (const other of window.document.querySelectorAll(".reg-chip")) {
    if (other === chip) continue;
    assert.ok(!other.classList.contains("on"),
      `unticked chip '${other.dataset.type}' is not shown as active`);
  }
  assert.deepEqual(names(), ["bge-small"],
    "the table now shows the embedding row and nothing else");
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
