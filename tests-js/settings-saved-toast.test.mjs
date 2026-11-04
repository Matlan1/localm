// SPDX-License-Identifier: AGPL-3.0-or-later
// The toast after a section save in Settings (savedToastKey in pages/settings.js):
// the saved values that changed decide what it says.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function toastKey(fields, updates) {
  const { window: win } = loadAppWithPages({
    fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
  });
  win.__fields = fields;
  win.__updates = updates;
  runScript(win, `
    _settingsControls = window.__fields.map((field) => ({ field }));
    window.__key = savedToastKey(window.__updates);
    window.__text = t(window.__key);
  `);
  return { key: win.__key, text: win.__text };
}

const DIAGNOSTICS = { key: "keep_diagnostics", widget: "toggle", applies: "restart", default: false };
const BIND = { key: "bind_host", widget: "text", applies: "restart", default: "127.0.0.1" };
const LIVE = { key: "show_tips", widget: "toggle", applies: "live", default: true };

test("Keep diagnostics alone gets its own toast, which does not ask for a restart", () => {
  const { key, text } = toastKey([DIAGNOSTICS, LIVE], { keep_diagnostics: true, show_tips: true });
  assert.equal(key, "settings.toast.savedDiagnostics");
  assert.match(text, /next server start, the rest at once/);
});

test("Keep diagnostics with another restart setting keeps the plain restart toast", () => {
  const { key } = toastKey([DIAGNOSTICS, BIND], { keep_diagnostics: true, bind_host: "0.0.0.0" });
  assert.equal(key, "settings.toast.savedRestart");
});

test("an unchanged Keep diagnostics does not take over the toast", () => {
  const { key } = toastKey([DIAGNOSTICS, BIND], { keep_diagnostics: false, bind_host: "0.0.0.0" });
  assert.equal(key, "settings.toast.savedRestart");
  assert.equal(toastKey([DIAGNOSTICS, LIVE], { keep_diagnostics: false, show_tips: false }).key,
               "settings.toast.saved");
});
