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

// unload_one_model() (http_server.py) answers HTTP 200 with status:"in_use"
// when the target engine is mid-generation right now - a real, distinct,
// non-error outcome, not a completed unload. r.ok alone cannot tell the two
// apart, and the per-row handler used to check only r.ok (same class of bug
// already fixed once for the sidebar's own quick-unload button, PR #1298).
//
// The GET /api/models mock below deliberately answers "model-a is still
// loaded and active" NO MATTER what the unload call's outcome was - that is
// what actually happened server-side (nothing was released), so it is the
// honest mock, not a shortcut. It also means refreshModelsPage()'s own
// reconciling fetch would make the row look "correct" (still loaded) under
// EITHER the buggy or the fixed handler, so row/button state cannot be the
// discriminator here (this is exactly what let the sidebar's first version of
// this test pass against unfixed code - see dev-notes/NEW-MODEL-DROPDOWN-2026-08-13.md).
// The one signal only the correct branch produces before any reconciling
// refresh runs is what the handler itself said happened: the toast text.
test("model-unload: an in-use engine (HTTP 200, status 'in_use') is not reported as unloaded", async () => {
  const models = [
    { name: "model-a", active: true, loaded: true, model_type: "llm", size_bytes: 1000 },
  ];
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models/unload")) {
      calls.push({ url: u, body: opts.body ? JSON.parse(opts.body) : {} });
      return {
        ok: true, status: 200,
        json: async () => ({ status: "in_use", model: "model-a", vram_freed: 0 }),
        text: async () => "",
      };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return {
        ok: true, status: 200,
        json: async () => ({ models, active: "model-a" }),
        text: async () => "",
      };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));

  const row = window.document.querySelector("#models-table tbody tr");
  const unloadBtn = [...row.querySelectorAll("button")].find((b) => b.textContent === "unload");
  assert.ok(unloadBtn, "model-a has an Unload button");

  unloadBtn.click();
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(calls.length, 1, "exactly one unload POST was attempted");
  const toastText = window.document.getElementById("toast").textContent;
  assert.match(toastText, /still generating/,
    `the toast must say the model is still in use, not claim it was unloaded (got: ${toastText})`);
  assert.doesNotMatch(toastText, /^Unloaded/,
    `must never claim success for an unload that did not happen (got: ${toastText})`);
});

// Same gap, "Unload all" side: unload_all_models() reports any pinned
// (mid-generation) engine in skipped_in_use rather than unloading it - it WAS
// loaded, it just was not released. The button's toast used to look only at
// unloaded_models/embedder_unloaded, so an all-pinned response (n === 0,
// skipped_in_use non-empty) fell through to "Nothing was loaded" - false,
// since something genuinely was loaded and resident.
test("model-unload: Unload-all does not claim 'Nothing was loaded' when everything loaded is pinned in-use", async () => {
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u.startsWith("/api/models/unload")) {
      return {
        ok: true, status: 200,
        json: async () => ({
          status: "in_use", model: "none", unloaded_models: [],
          embedder_unloaded: false, skipped_in_use: ["model-a"],
        }),
        text: async () => "",
      };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return {
        ok: true, status: 200,
        json: async () => ({
          models: [{ name: "model-a", active: true, loaded: true, model_type: "llm", size_bytes: 1000 }],
          active: "model-a",
        }),
        text: async () => "",
      };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));

  const btn = window.document.getElementById("models-unload-all-btn");
  btn.click();
  await new Promise((r) => setTimeout(r, 0));

  const toastText = window.document.getElementById("toast").textContent;
  assert.notEqual(toastText, "Nothing was loaded",
    `must not claim nothing was loaded when a model WAS loaded and only skipped for being in use (got: ${toastText})`);
  assert.match(toastText, /still generating/,
    `should name the pinned model(s) as still generating (got: ${toastText})`);
});

// The partial case: some models unload cleanly, one is pinned. The old code
// silently dropped the skip and reported only the successful count.
test("model-unload: Unload-all reports a partial result honestly (some unloaded, one still in use)", async () => {
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u.startsWith("/api/models/unload")) {
      return {
        ok: true, status: 200,
        json: async () => ({
          status: "unloaded", model: "model-b", unloaded_models: ["model-b"],
          embedder_unloaded: false, skipped_in_use: ["model-a"],
        }),
        text: async () => "",
      };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return {
        ok: true, status: 200,
        json: async () => ({
          models: [{ name: "model-a", active: true, loaded: true, model_type: "llm", size_bytes: 1000 }],
          active: "model-a",
        }),
        text: async () => "",
      };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));

  const btn = window.document.getElementById("models-unload-all-btn");
  btn.click();
  await new Promise((r) => setTimeout(r, 0));

  const toastText = window.document.getElementById("toast").textContent;
  assert.match(toastText, /Unloaded 1 model/,
    `should still report the genuine success count (got: ${toastText})`);
  assert.match(toastText, /still generating/,
    `must not silently drop the skipped-as-in-use model (got: ${toastText})`);
});
