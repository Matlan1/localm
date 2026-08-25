// SPDX-License-Identifier: AGPL-3.0-or-later
// Toggling a plugin bumps a shared localStorage rev. The storage event fires in
// the other tab, which re-syncs the plugin set and, via reconcileActiveView,
// leaves a view whose plugin is now inactive for chat.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function fireStorage(window, key, value) {
  let ev;
  try {
    ev = new window.StorageEvent("storage", { key, newValue: value });
  } catch (e) {
    ev = new window.Event("storage");
    Object.defineProperty(ev, "key", { value: key, configurable: true });
  }
  window.dispatchEvent(ev);
}

async function drain(times = 8) {
  for (let i = 0; i < times; i++) await new Promise((r) => setTimeout(r, 0));
}

/** /api/plugins reports the knowledge plugin as inactive (disabled elsewhere). */
function fetchWithKnowledgeDisabled(count) {
  return async (url) => {
    if (String(url) === "/api/plugins") {
      count.n += 1;
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ plugins: [{ name: "knowledge", active: false,
          tab: "knowledge", group: "", icon: "book", label: "Knowledge", commands: [] }],
          suggest_plugins: true }) };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

test("R50: a cross-tab plugin-disable signal redirects a parked tab to chat", async () => {
  const count = { n: 0 };
  const { window } = loadApp({ fetchImpl: fetchWithKnowledgeDisabled(count) });
  await drain();   // let the boot settle

  // This tab is parked on the (static) knowledge view.
  runScript(window, 'document.querySelectorAll(".view").forEach((v) =>' +
    ' v.classList.toggle("active", v.id === "view-knowledge"));');
  assert.equal(window.document.querySelector(".view.active").id, "view-knowledge",
    "starts parked on the knowledge view");

  // The other tab disabled the plugin -> a storage event fires here.
  fireStorage(window, "localm.pluginsRev", "1");
  await drain();

  assert.equal(window.document.querySelector(".view.active").id, "view-chat",
    "the parked tab was redirected to chat, not left erroring on dead routes");
});

test("R50: an unrelated storage key does not trigger a re-sync", async () => {
  const count = { n: 0 };
  const { window } = loadApp({ fetchImpl: fetchWithKnowledgeDisabled(count) });
  await drain();
  const before = count.n;
  fireStorage(window, "localm.someOtherKey", "x");
  await drain();
  assert.equal(count.n, before, "only the localm.pluginsRev key re-syncs plugins");
});
