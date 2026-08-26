// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages } from "./harness.mjs";

const safe = async () => ({
  ok: true, status: 200,
  json: async () => ({ models: [], active: "", plugins: [], keys: [],
                       is_owner: true, presets: [],
                       core: { chat: true, models: true, plugins: true, settings: true } }),
  text: async () => "",
});

test("palette offers 'Manage keys & devices' when the keys panel is available", () => {
  const { window } = loadApp({ fetchImpl: safe });
  window.document.getElementById("keys-card").classList.remove("sec-hidden");
  const labels = window.cmdkCommands().map((c) => c.label);
  assert.ok(labels.includes("Manage keys & devices"),
    "owner/keys-manager sees the direct jump");
});

test("palette hides the keys jump when the panel is gated-hidden (non-manager)", () => {
  const { window } = loadApp({ fetchImpl: safe });
  window.document.getElementById("keys-card").classList.add("sec-hidden");
  const labels = window.cmdkCommands().map((c) => c.label);
  assert.ok(!labels.includes("Manage keys & devices"),
    "a key that cannot manage keys gets no dead-end command");
});

test("the keys jump opens Settings without throwing (app-only: helper guarded)", () => {
  const { window } = loadApp({ fetchImpl: safe });
  window.document.getElementById("keys-card").classList.remove("sec-hidden");
  const cmd = window.cmdkCommands().find((c) => c.label === "Manage keys & devices");
  assert.ok(cmd, "command present");
  cmd.run();   // gotoSettingsSection lives in pages.js, not loaded here
  assert.ok(window.document.getElementById("view-settings").classList.contains("active"),
    "Settings view is shown");
});

test("gotoSettingsSection activates and remembers the target section", () => {
  const { window } = loadAppWithPages({ fetchImpl: safe });
  window.gotoSettingsSection("keys-card");
  assert.ok(window.document.getElementById("keys-card").classList.contains("active"),
    "the keys section is the active settings section");
});
