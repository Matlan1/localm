// SPDX-License-Identifier: AGPL-3.0-or-later
// A settings card's Save with the server unreachable says so instead of doing
// nothing, and reports the save as failed.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = {
  fields: [
    { key: "coder_unattended_network", widget: "select", label: "server label",
      help: "server help", group: "Coder", owner: "coder",
      options: ["deny", "allow"], default: "deny", shipped_default: "deny" },
  ],
};

function makeFetch(state) {
  return async (url, opts = {}) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    }
    if (url === "/v1/config" && (opts.method || "GET") === "PATCH") {
      state.patches += 1;
      throw new TypeError("Failed to fetch");
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

test("Save with the server unreachable toasts a failure and returns false", async () => {
  const state = { patches: 0 };
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(state) });
  const rejections = [];
  win.addEventListener("unhandledrejection", (e) => rejections.push(e.reason));
  runScript(win, "refreshSettingsPage();");
  await new Promise((r) => setTimeout(r, 0));
  const sel = win.document.querySelector('select[data-key="coder_unattended_network"]');
  sel.value = "allow";
  sel.dispatchEvent(new win.Event("change"));
  runScript(win, 'window.__save = saveSettingsSection([...document.querySelectorAll(".settings-section")]' +
    '.find((s) => s.querySelector(\'[data-key="coder_unattended_network"]\')).dataset.sec);');
  const result = await win.__save;
  assert.equal(state.patches, 1);
  assert.equal(result, false);
  const toast = win.document.getElementById("toast");
  assert.match(toast.textContent, /Save failed \(Failed to fetch\)/);
  assert.equal(toast.className.includes("error"), true);
  assert.deepEqual(rejections, []);
});
