// SPDX-License-Identifier: AGPL-3.0-or-later
// A plugin turned off while the page is open (Plugins page, another tab, the
// CLI): the chat knowledge list empties and the mic follows voice, no reload.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const drain = async (n = 12) => { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); };
const res = (body, ok = true, status = 200) =>
  ({ ok, status, json: async () => body, text: async () => "" });

function makeFetch(state) {
  return async (url) => {
    const u = String(url);
    if (u === "/api/capabilities") {
      return res({ plugins: ["knowledge", "rag", "voice"]
        .filter((n) => state.on.has(n))
        .map((n) => ({ name: n, active: true, tab: n === "knowledge" ? "knowledge" : "" })) });
    }
    if (u === "/api/rag/collections") {
      return state.on.has("rag")
        ? res({ collections: [{ name: "kb1", n_docs: 0, n_chunks: 0 }] })
        : res({}, false, 404);
    }
    if (u === "/api/voice/status") {
      return state.on.has("voice") ? res({ available: true, reason: "" }) : res({}, false, 404);
    }
    return res({ models: [], active: "", conversations: [], plugins: [] });
  };
}

function fireStorage(window) {
  const ev = new window.Event("storage");
  Object.defineProperty(ev, "key", { value: "localm.pluginsRev", configurable: true });
  window.dispatchEvent(ev);
}

test("disabling plugins live: kb list empty, mic greyed", async () => {
  const state = { on: new Set(["knowledge", "rag", "voice"]) };
  const { window } = loadApp({ fetchImpl: makeFetch(state) });
  await drain();
  const doc = window.document;
  const kb = doc.getElementById("p-kb");
  const mic = doc.getElementById("chat-mic");
  assert.ok([...kb.options].some((o) => o.value === "kb1"), "precondition: kb1 listed");
  assert.ok(!mic.classList.contains("unavailable"), "precondition: mic available");

  state.on = new Set();
  fireStorage(window);
  await drain();

  assert.ok(![...kb.options].some((o) => o.value === "kb1"), "kb1 no longer offered");
  assert.ok(mic.classList.contains("unavailable"), "mic greyed");
  assert.match(mic.title, /not installed/);
});

test("enabling voice live un-greys the mic", async () => {
  const state = { on: new Set() };
  const { window } = loadApp({ fetchImpl: makeFetch(state) });
  await drain();
  const mic = window.document.getElementById("chat-mic");
  assert.ok(mic.classList.contains("unavailable"), "precondition: greyed");
  state.on = new Set(["voice"]);
  fireStorage(window);
  await drain();
  assert.ok(!mic.classList.contains("unavailable"));
});
