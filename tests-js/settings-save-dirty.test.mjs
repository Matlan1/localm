// SPDX-License-Identifier: AGPL-3.0-or-later
// R09: Ctrl+S on the Settings page saves the active section (instead of the
// browser's "Save page as"). R10: editing a setting marks the page dirty and
// leaving Settings without saving warns first.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = {
  fields: [
    { key: "n_ctx", widget: "number", label: "Context window", help: "",
      group: "Engine", owner: "core", min: 512, step: 512, default: 4096 },
    { key: "require_auth", widget: "toggle", label: "Require an API key",
      help: "", group: "Security", owner: "core", default: false },
  ],
};

function makeFetch(patches) {
  return async (url, opts = {}) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    }
    if (url === "/v1/config" && (opts.method || "GET") === "PATCH") {
      patches.push(JSON.parse(opts.body));
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

async function render(win) {
  runScript(win, "refreshSettingsPage();");
  await new Promise((r) => setTimeout(r, 0));
}
async function drain(times = 6) {
  for (let i = 0; i < times; i++) await new Promise((r) => setTimeout(r, 0));
}
function makeSettingsActive(win) {
  // Mark the Settings VIEW active without going through showView (which would
  // re-render and reset the dirty flag we want to test).
  runScript(win, 'document.querySelectorAll(".view").forEach((v) =>' +
    ' v.classList.toggle("active", v.id === "view-settings"));');
}
function ctrlS(win) {
  win.document.dispatchEvent(new win.KeyboardEvent("keydown",
    { key: "s", ctrlKey: true, bubbles: true, cancelable: true }));
}

test("R09: Ctrl+S on the Settings page saves the active section", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(patches) });
  await render(win);
  makeSettingsActive(win);
  assert.ok(win.document.querySelector("#settings-content .settings-section.active"),
    "a settings section is active");
  ctrlS(win);
  await drain();
  assert.ok(patches.length >= 1, "Ctrl+S saved the active section (a PATCH was sent)");
});

test("R09: Ctrl+S off the Settings page does not save", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(patches) });
  await render(win);
  runScript(win, 'document.querySelectorAll(".view").forEach((v) =>' +
    ' v.classList.toggle("active", v.id === "view-chat"));');
  ctrlS(win);
  await drain();
  assert.equal(patches.length, 0, "Ctrl+S on a non-Settings view saves nothing");
});

test("R10: editing a setting marks the page dirty; a fresh render is clean", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  assert.equal(win.settingsDirty(), false, "clean right after render");
  const nctx = win.document.querySelector('input[data-key="n_ctx"]');
  nctx.value = "8192";
  nctx.dispatchEvent(new win.Event("input", { bubbles: true }));
  assert.equal(win.settingsDirty(), true, "editing a control marks it dirty");
  await render(win);
  assert.equal(win.settingsDirty(), false, "a re-render resets the baseline");
});

test("R10: leaving Settings with unsaved edits warns; declining stays put", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  makeSettingsActive(win);
  const nctx = win.document.querySelector('input[data-key="n_ctx"]');
  nctx.value = "8192";
  nctx.dispatchEvent(new win.Event("input", { bubbles: true }));

  let asked = 0;
  win.confirm = () => { asked += 1; return false; };   // user declines
  runScript(win, 'showView("chat");');
  assert.equal(asked, 1, "the unsaved-changes prompt fired");
  assert.equal(win.isSettingsView(), true, "declining keeps you on Settings");

  win.confirm = () => true;                              // user confirms
  runScript(win, 'showView("chat");');
  assert.equal(win.isSettingsView(), false, "confirming leaves Settings");
});

test("R10: leaving a clean Settings page does not prompt", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  makeSettingsActive(win);
  let asked = 0;
  win.confirm = () => { asked += 1; return false; };
  runScript(win, 'showView("chat");');
  assert.equal(asked, 0, "no prompt when there are no unsaved edits");
  assert.equal(win.isSettingsView(), false, "navigation proceeds");
});
