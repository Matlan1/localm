// SPDX-License-Identifier: AGPL-3.0-or-later
// The shortcuts fetch is not eager at module load, so these tests drive it
// through refreshModelsPage().

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

function makeFetch(shortcutsResponse) {
  return async (url) => {
    const u = String(url);
    if (u.includes("/api/models/shortcuts")) return shortcutsResponse;
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models: [], active: null }),
               text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}
const ok = (payload) => (
  { ok: true, status: 200, json: async () => payload, text: async () => "" });

const tick = () => new Promise((r) => setTimeout(r, 0));

test("model shortcuts: the picker is populated with alias + size on load", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(ok(SHORTCUTS_PAYLOAD)) });
  await window.refreshModelsPage();
  await tick();
  const sel = window.document.getElementById("pull-shortcut");
  assert.ok(sel, "the shortcut picker exists on the Models page");
  // +1 for the "Curated shortcuts…" placeholder option.
  assert.equal(sel.options.length, SHORTCUTS_PAYLOAD.shortcuts.length + 1);
  assert.equal(sel.options[1].textContent, "llama3.2-1b (~0.7 GB)");
  assert.equal(sel.options[2].textContent, "qwen2.5-7b (~4.7 GB)");
});

test("model shortcuts: it is fetched only once across repeated page refreshes", async () => {
  const calls = [];
  const counting = makeFetch(ok(SHORTCUTS_PAYLOAD));
  const { window } = loadAppWithPages({
    fetchImpl: async (url) => { calls.push(String(url)); return counting(url); },
  });
  await window.refreshModelsPage();
  await tick();
  await window.refreshModelsPage();   // e.g. re-sorting, switching tabs
  await tick();
  assert.equal(calls.filter((u) => u.includes("/api/models/shortcuts")).length, 1,
    "a fixed local list needs fetching once per page lifetime, not on every refresh");
});

test("model shortcuts: picking one fills the RESOLVED spec and the alias as the name", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(ok(SHORTCUTS_PAYLOAD)) });
  await window.refreshModelsPage();
  await tick();
  const sel = window.document.getElementById("pull-shortcut");
  sel.value = SHORTCUTS_PAYLOAD.shortcuts[1].spec;
  sel.dispatchEvent(new window.Event("change"));

  assert.equal(
    window.document.getElementById("pull-spec").value,
    "bartowski/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "pull-spec gets the full resolved repo:file, not the bare alias");
  assert.equal(window.document.getElementById("pull-name").value, "qwen2.5-7b",
    "pull-name gets the short alias, not a repo-derived filename");
  assert.equal(sel.selectedIndex, 0, "the picker resets to the placeholder after use");
});

test("model shortcuts: picking the placeholder itself is a no-op", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(ok(SHORTCUTS_PAYLOAD)) });
  await window.refreshModelsPage();
  await tick();
  const spec = window.document.getElementById("pull-spec");
  spec.value = "untouched";
  const sel = window.document.getElementById("pull-shortcut");
  sel.value = "";
  sel.dispatchEvent(new window.Event("change"));
  assert.equal(spec.value, "untouched");
});

test("model shortcuts: a fetch failure leaves only the placeholder, nothing throws", async () => {
  const failed = { ok: false, status: 500, json: async () => ({}), text: async () => "" };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(failed) });
  await window.refreshModelsPage();
  await tick();
  const sel = window.document.getElementById("pull-shortcut");
  assert.equal(sel.options.length, 1, "only the placeholder remains when the fetch fails");
});
