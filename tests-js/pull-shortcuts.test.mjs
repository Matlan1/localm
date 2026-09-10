// SPDX-License-Identifier: AGPL-3.0-or-later
// A failed /api/models/shortcuts fetch must not permanently empty the picker:
// _pullShortcutsLoaded is cleared on failure so the next refreshModelsPage()
// retries. See models.js's _pullShortcutsLoaded flag and _loadPullShortcuts().

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const SHORTCUTS_PAYLOAD = {
  shortcuts: [
    { alias: "llama3.2-1b",
      spec: "bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf",
      size: "~0.7 GB" },
    { alias: "qwen2.5-7b",
      spec: "bartowski/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf",
      size: "~4.7 GB" },
  ],
};

const ok = (payload) => (
  { ok: true, status: 200, json: async () => payload, text: async () => "" });
const modelsOk = ok({ models: [], active: null });
const tick = () => new Promise((r) => setTimeout(r, 0));

// Every call to /api/models/shortcuts before the Nth fails via `firstFail`;
// the Nth and every later call succeed. Every other URL (the page's own
// /api/models fetch) resolves with an empty model list.
function makeFlakyFetch(firstFail) {
  let shortcutsCalls = 0;
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u.includes("/api/models/shortcuts")) {
      shortcutsCalls++;
      if (shortcutsCalls === 1) return firstFail();
      return ok(SHORTCUTS_PAYLOAD);
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) return modelsOk;
    return ok({});
  };
  return { fetchImpl, callCount: () => shortcutsCalls };
}

test("pull shortcuts: a 500 on the first fetch is retried on the next refresh", async () => {
  const { fetchImpl, callCount } = makeFlakyFetch(
    () => ({ ok: false, status: 500, json: async () => ({}), text: async () => "" }));
  const { window } = loadAppWithPages({ fetchImpl });

  await window.refreshModelsPage();
  await tick();
  const sel = window.document.getElementById("pull-shortcut");
  assert.equal(sel.options.length, 1,
    "only the placeholder remains after the first (failed) fetch");

  await window.refreshModelsPage();
  await tick();
  // +1 for the "Curated shortcuts…" placeholder option.
  assert.equal(sel.options.length, SHORTCUTS_PAYLOAD.shortcuts.length + 1,
    "the retry populates the picker; the failed attempt left no partial options");
  assert.equal(sel.options[1].textContent, "llama3.2-1b (~0.7 GB)");
  assert.equal(sel.options[2].textContent, "qwen2.5-7b (~4.7 GB)");
  assert.equal(callCount(), 2, "fetched again after the failure, exactly once");
});

test("pull shortcuts: a thrown fetch on the first attempt is retried on the next refresh", async () => {
  const { fetchImpl, callCount } = makeFlakyFetch(() => { throw new Error("network down"); });
  const { window } = loadAppWithPages({ fetchImpl });

  await window.refreshModelsPage();
  await tick();
  const sel = window.document.getElementById("pull-shortcut");
  assert.equal(sel.options.length, 1,
    "only the placeholder remains after the first (thrown) fetch");

  await window.refreshModelsPage();
  await tick();
  assert.equal(sel.options.length, SHORTCUTS_PAYLOAD.shortcuts.length + 1,
    "the retry populates the picker after a thrown fetch");
  assert.equal(callCount(), 2, "fetched again after the throw, exactly once");
});

test("pull shortcuts: a refresh after success does not re-fetch or duplicate options", async () => {
  const { fetchImpl, callCount } = makeFlakyFetch(
    () => ({ ok: false, status: 500, json: async () => ({}), text: async () => "" }));
  const { window } = loadAppWithPages({ fetchImpl });

  await window.refreshModelsPage();   // fails
  await tick();
  await window.refreshModelsPage();   // succeeds
  await tick();
  await window.refreshModelsPage();   // must not re-fetch or re-append
  await tick();

  const sel = window.document.getElementById("pull-shortcut");
  assert.equal(sel.options.length, SHORTCUTS_PAYLOAD.shortcuts.length + 1,
    "options are not duplicated by a refresh that follows a success");
  assert.equal(callCount(), 2, "a successful load is not retried on a later refresh");
});
