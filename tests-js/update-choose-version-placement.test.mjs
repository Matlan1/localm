// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings > Updates: "Choose a version" sits in the App update action row.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const fetchImpl = async (url) => {
  const u = String(url);
  const body = u.includes("/api/update/releases")
    ? { available: true, current: "0.3.0", releases: [
        { version: "v0.4.0", direction: "newer", installable: true, prerelease: false, warnings: [] }] }
    : { models: [], active: "", conversations: [], plugins: [] };
  return { ok: true, status: 200, text: async () => "", json: async () => body };
};

test("Choose a version is in the same row as Check for updates, outside the settings fields", () => {
  const { window } = loadAppWithPages({ fetchImpl });
  const doc = window.document;
  const btn = doc.getElementById("update-versions");
  const check = doc.getElementById("update-check");
  assert.ok(btn);
  assert.equal(btn.parentElement, check.parentElement, "same action row");
  assert.equal(btn.className, check.className, "same button style");
  assert.equal(btn.closest("#update-toggles-block"), null, "not inside the settings form");
  assert.equal(doc.getElementById("update-toggles-block").hidden, true, "settings fields not loaded");
  assert.equal(btn.closest("[hidden]")?.id, "app-update-block", "gated only by the app-update block");
  assert.ok(doc.querySelector('[data-i18n="settings.update.versionsHelp"]'), "description line present");
  const toggles = doc.getElementById("update-toggles-block");
  for (const id of ["update-version-picker", "update-version-status"]) {
    const el = doc.getElementById(id);
    assert.ok(btn.compareDocumentPosition(el) & 4, id + " follows the button");
    assert.ok(el.compareDocumentPosition(toggles) & 4, id + " sits above the settings fields");
  }
});

test("clicking Choose a version opens the picker without the settings fields", async () => {
  const { window } = loadAppWithPages({ fetchImpl });
  const doc = window.document;
  doc.getElementById("update-versions").click();
  for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
  assert.equal(doc.getElementById("update-version-picker").hidden, false);
  assert.equal(doc.getElementById("update-toggles-block").hidden, true);
});
