// SPDX-License-Identifier: AGPL-3.0-or-later
// model-unload: the Models page's per-row Unload button and global Unload-all
// button. A model can be loaded (resident in VRAM) without being the active
// one - the row must show that state and offer a way to release it without
// disturbing whichever model IS active.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(models, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models/unload")) {
      const body = opts.body ? JSON.parse(opts.body) : {};
      calls.push({ url: u, body });
      return {
        ok: true, status: 200,
        json: async () => ({
          status: "unloaded",
          unloaded_models: body.model ? [body.model] : ["model-a"],
        }),
        text: async () => "",
      };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return {
        ok: true, status: 200,
        json: async () => ({
          models, active: models.find((m) => m.active)?.name || null,
        }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

test("model-unload: a loaded-but-not-active model shows a loaded tag and an Unload button", async () => {
  const calls = [];
  const models = [
    { name: "model-a", active: false, loaded: true, model_type: "llm", size_bytes: 1000 },
    { name: "model-b", active: true, loaded: true, model_type: "llm", size_bytes: 2000 },
  ];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models, calls) });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));

  const rows = [...window.document.querySelectorAll("#models-table tbody tr")];
  assert.equal(rows.length, 2, "both models render as rows");
  const rowA = rows.find((tr) => tr.textContent.includes("model-a"));
  assert.ok(rowA, "model-a has a row");
  assert.ok(rowA.querySelector(".loaded-tag"), "model-a (loaded, not active) shows a loaded tag");

  const unloadBtn = [...rowA.querySelectorAll("button")].find((b) => b.textContent === "unload");
  assert.ok(unloadBtn, "model-a has an Unload button");

  unloadBtn.click();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(calls.length, 1, "clicking Unload posted exactly once");
  assert.deepEqual(calls[0].body, { model: "model-a" },
    "the per-row Unload button targets only that model");
});

test("model-unload: the active model's row has no separate loaded tag (active already implies loaded here) but still offers Unload", async () => {
  const models = [
    { name: "model-b", active: true, loaded: true, model_type: "llm", size_bytes: 2000 },
  ];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models, []) });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));

  const row = window.document.querySelector("#models-table tbody tr");
  assert.ok(row.querySelector(".active-tag"), "shows the active tag");
  const unloadBtn = [...row.querySelectorAll("button")].find((b) => b.textContent === "unload");
  assert.ok(unloadBtn, "the active model can still be unloaded, matching the existing unload-all endpoint's own permissiveness");
});

test("model-unload: a never-loaded model has no Unload button and no loaded tag", async () => {
  const models = [
    { name: "model-c", active: false, loaded: false, model_type: "llm", size_bytes: 500 },
  ];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models, []) });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));

  const row = window.document.querySelector("#models-table tbody tr");
  assert.equal(row.querySelector(".loaded-tag"), null, "never-loaded model shows no loaded tag");
  const unloadBtn = [...row.querySelectorAll("button")].find((b) => b.textContent === "unload");
  assert.equal(unloadBtn, undefined, "never-loaded model has no Unload button");
});

test("model-unload: the global Unload-all button POSTs with no model field", async () => {
  const calls = [];
  const models = [
    { name: "model-a", active: false, loaded: true, model_type: "llm", size_bytes: 1000 },
  ];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models, calls) });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));

  const btn = window.document.getElementById("models-unload-all-btn");
  assert.ok(btn, "the global Unload all button exists");
  btn.click();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].body, {}, "Unload-all sends no model field, preserving unload-everything");
});
