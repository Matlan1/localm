// SPDX-License-Identifier: AGPL-3.0-or-later
// U2 (GUI half): the plugin catalog now shows a load-status indicator and
// populates its rows one after another (token-guarded) instead of flashing the
// whole table at once.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function payload() {
  return {
    plugins: [
      { name: "chat", label: "Chat", builtin: true, installed: true,
        enabled: true, active: true, protected: true, description: "core",
        requires: [], missing_requires: [] },
      { name: "coder", label: "Coder", builtin: true, installed: true,
        enabled: true, active: true, protected: false, description: "agent",
        requires: [], missing_requires: [] },
      { name: "broken", label: "Broken", builtin: true, installed: true,
        enabled: true, active: false, protected: false, description: "oops",
        requires: [], missing_requires: [], error: "load failed: boom" },
      { name: "image", label: "Image", builtin: true, installed: false,
        enabled: false, active: false, protected: false, description: "media",
        requires: [], missing_requires: [] },
    ],
  };
}

function makeFetch() {
  return async (url) => {
    if (url === "/api/plugins") {
      return { ok: true, status: 200, json: async () => payload(), text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [], commands: [] }),
    };
  };
}

async function settle(win) {
  for (let i = 0; i < 30; i++) {
    await new Promise((r) => setTimeout(r, 0));
    const s = win.document.querySelector(".catalog-status");
    if (s && !/Loading/.test(s.textContent)) return s;
  }
  return win.document.querySelector(".catalog-status");
}

test("catalog shows a status indicator and fills in all rows", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  runScript(win, "_catalogStaggerMs = 0; renderCatalogPlugins();");
  const status = await settle(win);

  assert.ok(status, "a status indicator is rendered");
  // 2 of the 4 builtin plugins are active; 1 errored.
  assert.match(status.textContent, /2\/4 plugins active/);
  assert.match(status.textContent, /1 failed/);

  const rows = win.document.querySelectorAll("#catalog-table tbody tr");
  assert.equal(rows.length, 4, "every plugin row is populated");
});

test("a newer render cancels the stale staggered populate (no duplicate rows)", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  // Use a non-zero stagger so the first render is still mid-populate when the
  // second starts; the token guard must abandon the first.
  runScript(win, "_catalogStaggerMs = 5; renderCatalogPlugins(); renderCatalogPlugins();");
  await settle(win);
  // Extra drain in case the higher stagger needs more time.
  for (let i = 0; i < 30; i++) await new Promise((r) => setTimeout(r, 5));

  const rows = win.document.querySelectorAll("#catalog-table tbody tr");
  assert.equal(rows.length, 4, "exactly one set of rows, not doubled");
  const statuses = win.document.querySelectorAll(".catalog-status");
  assert.equal(statuses.length, 1, "one status indicator");
});
