// SPDX-License-Identifier: AGPL-3.0-or-later
// Drives the Models-page "use" button click through to switchModel().

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const MODELS_PAYLOAD = {
  models: [{ name: "other-model", model_type: "llm", active: false, size_bytes: 1e9 }],
  active: "current-model",
};

/** A controllable load response: the /api/models/load call resolves only when
 *  the test calls `resolveLoad()`. */
function makeFetch({ loadBody = { status: "loaded", model: "other-model" } } = {}) {
  let resolveLoad;
  const loadPromise = new Promise((r) => { resolveLoad = r; });
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => MODELS_PAYLOAD, text: async () => "" };
    }
    if (u === "/api/models/load" && (opts.method || "GET") === "POST") {
      await loadPromise;
      return { ok: true, status: 200, json: async () => loadBody, text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  return { fetchImpl, resolveLoad: () => resolveLoad() };
}

async function tick(n = 1) { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); }

function findUseButton(win) {
  const row = [...win.document.querySelectorAll("#models-table tbody tr")]
    .find((r) => r.querySelector(".name")?.textContent === "other-model");
  return [...row.querySelectorAll("button")].find((b) => b.textContent === "use" || b.textContent === "loading…");
}

test("use button: shows 'loading…' and stays disabled WHILE the switch is in flight", async () => {
  const { fetchImpl, resolveLoad } = makeFetch();
  const { window: win } = loadAppWithPages({ fetchImpl });
  await win.refreshModelsPage();
  await tick();
  const btn = findUseButton(win);
  assert.equal(btn.textContent, "use", "starts with the normal label");

  const clickPromise = btn.onclick();
  await tick(2);   // let the click handler run up to the awaited fetch
  assert.equal(btn.textContent, "loading…", "shows the inline loading cue while switchModel() is in flight");
  assert.equal(btn.disabled, true);

  resolveLoad();
  await clickPromise;
  await tick(3);
  // a successful switch re-renders the whole table, so the original button node
  // is gone and is not asserted on further here
});

test("use button: label restores on a FAILED load, not stuck at 'loading…'", async () => {
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => MODELS_PAYLOAD, text: async () => "" };
    }
    if (u === "/api/models/load") {
      return { ok: false, status: 500, json: async () => ({ detail: "boom" }), text: async () => "boom" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await win.refreshModelsPage();
  await tick();
  const btn = findUseButton(win);
  await btn.onclick();
  await tick(3);
  assert.equal(btn.textContent, "use", "restored to the normal label after a failed load");
  assert.equal(btn.disabled, false, "re-enabled after a failed load");
});

test("use button: label restores on 'superseded' (a newer selection took over), button usable again", async () => {
  const { fetchImpl, resolveLoad } = makeFetch({ loadBody: { status: "superseded", model: "other-model" } });
  const { window: win } = loadAppWithPages({ fetchImpl });
  await win.refreshModelsPage();
  await tick();
  const btn = findUseButton(win);
  const clickPromise = btn.onclick();
  await tick(2);
  assert.equal(btn.textContent, "loading…");
  resolveLoad();
  await clickPromise;
  await tick(2);
  assert.equal(btn.textContent, "use", "restored even on the quiet 'superseded' path");
  assert.equal(btn.disabled, false);
});

test("use button: a successful switch still refreshes the table and toasts (unchanged behavior)", async () => {
  const { fetchImpl, resolveLoad } = makeFetch();
  const { window: win } = loadAppWithPages({ fetchImpl });
  await win.refreshModelsPage();
  await tick();
  const btn = findUseButton(win);
  const clickPromise = btn.onclick();
  await tick();
  resolveLoad();
  await clickPromise;
  await tick(3);
  const toastEl = win.document.getElementById("toast");
  assert.match(toastEl.textContent, /Model switched to other-model/,
    "the loading-label addition must not have broken the existing success toast");
});
