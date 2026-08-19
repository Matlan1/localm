// SPDX-License-Identifier: AGPL-3.0-or-later
// Branch A: the Models page one-click set-type control. Every row (including a
// type='unknown' model, which otherwise has no actionable controls) gets a type
// <select>; changing it POSTs the chosen type to /api/models/type. This is how a
// bulk-imported / mis-detected 'unknown' model is corrected without the CLI.
//
// Both fixtures here hold ONE model, and it is an 'unknown' - which is a type
// with no tab of its own, so it now lives on the Other tab rather than in All
// (All lists it too once the merge toggle asks for it, and says so meanwhile).
// So each test navigates to Other first. That is also the honest repair path a
// user walks: the Other tab is where a mis-detected model is found and fixed.
// The assertions about the control itself are unchanged.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(models, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models/type")) {
      const body = opts.body ? JSON.parse(opts.body) : {};
      calls.push({ url: u, body });
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

test("models-set-type: an unknown-type row shows a type control preset to 'unknown'", async () => {
  const models = [
    { name: "mystery", active: false, loaded: false, model_type: "unknown", size_bytes: 100 },
  ];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models, []) });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));
  window.document.querySelector('#models-tab-nav .tab-btn[data-type="other"]').click();
  await new Promise((r) => setTimeout(r, 0));

  const row = window.document.querySelector("#models-table tbody tr");
  assert.ok(row, "the unknown model renders a row");
  const sel = row.querySelector("select");
  assert.ok(sel, "the unknown row exposes a set-type <select> (it has no other controls)");
  assert.equal(sel.value, "unknown", "the control reflects the model's current type");
  const opts = [...sel.querySelectorAll("option")].map((o) => o.value);
  assert.ok(opts.includes("llm") && opts.includes("unknown"),
    "the control offers the real model types");
});

test("models-set-type: changing the control POSTs the chosen type once", async () => {
  const calls = [];
  const models = [
    { name: "mystery", active: false, loaded: false, model_type: "unknown", size_bytes: 100 },
  ];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models, calls) });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));
  window.document.querySelector('#models-tab-nav .tab-btn[data-type="other"]').click();
  await new Promise((r) => setTimeout(r, 0));

  const sel = window.document.querySelector("#models-table tbody tr select");
  sel.value = "llm";
  sel.dispatchEvent(new window.Event("change"));
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(calls.length, 1, "changing the type posted exactly once");
  assert.equal(calls[0].url.replace(/\?.*/, ""), "/api/models/type");
  assert.deepEqual(calls[0].body, { model: "mystery", model_type: "llm" },
    "the control posts the model name and the chosen type");
});
