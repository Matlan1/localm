// SPDX-License-Identifier: AGPL-3.0-or-later
// The plugin catalog offers a "Refresh" button on every installed first-party
// plugin. Clicking it POSTs /api/plugins/<name>/refresh.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function pluginsPayload() {
  return {
    plugins: [
      { name: "chat", label: "Chat", builtin: true, installed: true,
        enabled: true, active: true, protected: true, description: "core",
        requires: [], missing_requires: [] },
      { name: "image", label: "Images", builtin: true, installed: true,
        enabled: true, active: true, protected: false, description: "image gen",
        requires: [], missing_requires: [] },
      { name: "voice", label: "Voice", builtin: true, installed: false,
        enabled: false, active: false, protected: false, description: "stt",
        requires: [], missing_requires: [] },
    ],
  };
}

function makeFetch(calls) {
  return async (url, opts = {}) => {
    if (url === "/api/plugins") {
      return { ok: true, status: 200, json: async () => pluginsPayload(), text: async () => "" };
    }
    if (typeof url === "string" && url.includes("/refresh")) {
      calls.push(url);
      return { ok: true, status: 200, json: async () => ({ status: "refreshed", name: "image" }), text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [], commands: [] }),
    };
  };
}

async function renderCatalog(win) {
  runScript(win, "_catalogStaggerMs = 0; renderCatalogPlugins();");
  for (let i = 0; i < 20; i++) {
    await new Promise((r) => setTimeout(r, 0));
    const s = win.document.querySelector(".catalog-status");
    if (s && !/Loading/.test(s.textContent)) break;
  }
}

function refreshBtn(box, name) {
  return [...box.querySelectorAll("button")].find(
    (b) => b.textContent === "Refresh" &&
      b.closest("tr") &&
      b.closest("tr").textContent.includes(name));
}

test("installed first-party plugins get a Refresh button; available + protected do not", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await renderCatalog(win);
  const box = win.document.getElementById("catalog-table");

  assert.ok(refreshBtn(box, "Images"), "installed plugin has a Refresh button");
  assert.equal(refreshBtn(box, "Voice"), undefined,
    "an available (not installed) plugin has no Refresh button");
  assert.equal(refreshBtn(box, "Chat"), undefined,
    "the protected chat plugin has no Refresh button");
});

test("an installed third-party (non-builtin) plugin gets no Refresh button", async () => {
  // The catalog renders only builtins, and the Refresh button is further
  // guarded on p.builtin.
  const payload = pluginsPayload();
  payload.plugins.push({
    name: "custom", label: "Custom", builtin: false, installed: true,
    enabled: true, active: true, protected: false, description: "third-party",
    requires: [], missing_requires: [],
  });
  const fetchImpl = async (url) => {
    if (url === "/api/plugins") {
      return { ok: true, status: 200, json: async () => payload, text: async () => "" };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [], commands: [] }) };
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await renderCatalog(win);
  const box = win.document.getElementById("catalog-table");
  assert.equal(refreshBtn(box, "Custom"), undefined,
    "a third-party plugin must not get a store-Refresh button");
});

test("clicking Refresh POSTs /api/plugins/<name>/refresh", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  await renderCatalog(win);
  const box = win.document.getElementById("catalog-table");

  refreshBtn(box, "Images").click();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  assert.ok(calls.some((u) => u.includes("/api/plugins/image/refresh")),
    "refresh hits the per-plugin refresh route");
});
