// SPDX-License-Identifier: AGPL-3.0-or-later
// The Models page's per-row Unload button and global Unload-all button. A model
// can be loaded (resident in VRAM) without being the active one.

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

// unload_one_model() (http_server.py) answers HTTP 200 with
// status:"confirm_required" when the target engine is still in use, which is
// not a completed unload.
//
// The GET /api/models mock below answers "model-a is still loaded and active"
// whatever the unload call returned, so row and button state are the same either
// way and the toast text is the discriminating signal.
function makeConfirmRequiredFetch(models, calls, { onForcedUnload } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models/unload")) {
      const body = opts.body ? JSON.parse(opts.body) : {};
      calls.push({ url: u, body });
      if (!body.force) {
        return {
          ok: true, status: 200,
          json: async () => ({
            status: "confirm_required", model: "model-a",
            detail: "'model-a' is still generating",
          }),
          text: async () => "",
        };
      }
      if (onForcedUnload) onForcedUnload();
      return {
        ok: true, status: 200,
        json: async () => ({ status: "unloaded", unloaded_models: ["model-a"] }),
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
}

test("model-unload: a busy engine (confirm_required) is not reported as unloaded, and declining stops after one request", async () => {
  const models = [
    { name: "model-a", active: true, loaded: true, model_type: "llm", size_bytes: 1000 },
  ];
  const calls = [];
  const fetchImpl = makeConfirmRequiredFetch(models, calls, {
    onForcedUnload: () => assert.fail("must not force-unload without confirmation"),
  });
  const { window } = loadAppWithPages({ fetchImpl });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));
  window.confirmDangerAsync = async () => false;   // decline

  const row = window.document.querySelector("#models-table tbody tr");
  const unloadBtn = [...row.querySelectorAll("button")].find((b) => b.textContent === "unload");
  assert.ok(unloadBtn, "model-a has an Unload button");

  unloadBtn.click();
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(calls.length, 1, "declining must not post a forced retry");
  const toastText = window.document.getElementById("toast").textContent;
  assert.doesNotMatch(toastText, /^Unloaded/,
    `must never claim success for an unload that did not happen (got: ${toastText})`);
});

test("model-unload: a busy engine (confirm_required) unloads once confirmed, via a forced retry", async () => {
  const models = [
    { name: "model-a", active: true, loaded: true, model_type: "llm", size_bytes: 1000 },
  ];
  const calls = [];
  let forced = false;
  const fetchImpl = makeConfirmRequiredFetch(models, calls, { onForcedUnload: () => { forced = true; } });
  const { window } = loadAppWithPages({ fetchImpl });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));
  window.confirmDangerAsync = async () => true;   // confirm

  const row = window.document.querySelector("#models-table tbody tr");
  const unloadBtn = [...row.querySelectorAll("button")].find((b) => b.textContent === "unload");
  unloadBtn.click();
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(calls.length, 2, "confirming retries exactly once, with force");
  assert.equal(calls[1].body.force, true, "the retry must carry force: true");
  assert.ok(forced, "the forced retry actually reached the server's force path");
  const toastText = window.document.getElementById("toast").textContent;
  assert.match(toastText, /^Unloaded/, "confirming and forcing through reports the real success");
});

// unload_all_models() reports a pinned (mid-generation) engine in skipped_in_use
// rather than unloading it.
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

// The partial case: some models unload cleanly, one is pinned.
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

// Confirming the force-retry offer once something is skipped_in_use must
// actually re-post with force: true and report the follow-up success -
// otherwise "Unload all" would have no way to push through a genuinely
// stuck model at all.
test("model-unload: Unload-all's force-retry offer, once confirmed, unloads the rest with force", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models/unload")) {
      const body = opts.body ? JSON.parse(opts.body) : {};
      calls.push(body);
      if (!body.force) {
        return {
          ok: true, status: 200,
          json: async () => ({
            status: "in_use", unloaded_models: [],
            embedder_unloaded: false, skipped_in_use: ["model-a"],
          }),
          text: async () => "",
        };
      }
      return {
        ok: true, status: 200,
        json: async () => ({ status: "unloaded", unloaded_models: ["model-a"] }),
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
  window.confirmDangerAsync = async () => true;   // confirm the force-retry offer

  const btn = window.document.getElementById("models-unload-all-btn");
  btn.click();
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0));

  assert.equal(calls.length, 2, "the confirm must trigger exactly one forced retry");
  assert.equal(calls[1].force, true, "the retry must carry force: true");
  assert.equal(calls[1].model, undefined, "the retry is still an unload-everything call");
  const toastText = window.document.getElementById("toast").textContent;
  assert.match(toastText, /Unloaded 1 model/,
    `confirming and forcing through should report the real success (got: ${toastText})`);
});
