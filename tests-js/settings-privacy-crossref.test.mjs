// SPDX-License-Identifier: AGPL-3.0-or-later
// NEW-SETTINGS-IA-REVIEW, resolved 2026-08-13.
//
// This file used to pin the MITIGATION: settingsSectionOf routed by `owner`, so a
// plugin-owned field declaring group="Privacy" rendered on its plugin's tab, and
// the core Privacy panel carried a "Related settings also live on plugin tabs: ..."
// line so browsing Privacy did not look like the whole story.
//
// The root cause is now fixed instead. `group` is the single source of truth for
// placement, so those fields render IN Privacy & data and there is nothing left to
// cross-reference - a section pointing at itself would be worse than saying nothing.
// These tests therefore pin the REUNION, and that the note is gone.
//
// Privacy was the case that justified the whole change: measured on master, its nine
// fields had FOUR different owners (core 2, memory 5, chat 1, coder 1), so seven of
// them left the Privacy panel. It is the only genuinely cross-cutting group in the
// schema; every other displaced group was simply a plugin's name spelled twice.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = {
  fields: [
    { key: "mode", widget: "select", label: "Session persistence",
      help: "how much is saved", group: "Privacy", owner: "core",
      options: ["privacy", "log", "full"], default: "log" },
    // Plugin-owned but group="Privacy" - the case the whole overhaul is about.
    { key: "chat_mode", widget: "select", label: "Chat persistence override",
      help: "Overrides the global persistence for chat only.", group: "Privacy",
      owner: "chat", options: ["", "privacy", "log", "full"], default: "" },
    { key: "coder_mode", widget: "select", label: "Coder persistence override",
      help: "Overrides the global persistence for the coder only.", group: "Privacy",
      owner: "coder", options: ["", "privacy", "log", "full"], default: "" },
    // memory-owned, and now grouped as Memory: still under the Privacy & data NAV
    // group, but its own panel.
    { key: "memory_recall_in_privacy", widget: "toggle",
      label: "Allow memory recall in privacy mode", help: "", group: "Memory",
      owner: "memory", default: false },
    // A generation default: owner chat, group Chat -> Model, not Plugins.
    { key: "temperature", widget: "number", label: "Temperature", help: "",
      group: "Chat", owner: "chat", min: 0, max: 2, default: 0.8 },
  ],
};

function makeFetch() {
  return async (url) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

async function render(win) {
  runScript(win, "refreshSettingsPage();");
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
}

test("the persistence parent and BOTH its overrides render in one Privacy section", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  const doc = win.document;

  const privacySec = doc.getElementById("settings-sec-core-Privacy");
  assert.ok(privacySec, "the Privacy section renders");

  for (const key of ["mode", "chat_mode", "coder_mode"]) {
    const node = doc.querySelector(`[data-field-key="${key}"]`);
    assert.ok(node, `${key} renders`);
    assert.equal(node.closest(".settings-section"), privacySec,
      `${key} must render INSIDE the Privacy section - an override is unreadable `
      + "away from the setting it overrides, and its help says so");
  }
  assert.equal(privacySec.dataset.group, "privacy",
    "the Privacy section sits in the Privacy & data nav group");
});

test("memory keys sit in their own panel but the SAME nav group as privacy", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  const doc = win.document;

  const mem = doc.querySelector('[data-field-key="memory_recall_in_privacy"]');
  assert.ok(mem, "the memory toggle renders");
  const memSec = mem.closest(".settings-section");
  assert.equal(memSec.dataset.group, "privacy",
    "a user opening Privacy & data to check memory recall now finds it there");
  assert.notEqual(memSec, doc.getElementById("settings-sec-core-Privacy"),
    "it is still its own panel, not dumped into the persistence block");
});

test("the cross-reference note is gone - nothing is filed away from its group", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  const doc = win.document;

  // Scope to the RENDERED sections, not document.body: the harness inlines the
  // module source into the page, so body.textContent also contains settings.js's
  // own comments - including the one explaining why this note was removed. That
  // false positive is what the first version of this assertion tripped on.
  const rendered = [...doc.querySelectorAll("section.settings-section")]
    .map((s) => s.textContent).join("\n");
  assert.ok(rendered.length > 0, "sections rendered (guard: the check below is "
    + "vacuous against an empty page)");
  assert.doesNotMatch(rendered, /Related settings also live/,
    "with placement following `group`, no section is missing part of itself, so "
    + "the mitigation note must not appear in any rendered section");
});

test("a plugin-owned generation default files under Model, not Plugins", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  const doc = win.document;

  const sec = doc.querySelector('[data-field-key="temperature"]').closest(".settings-section");
  assert.equal(sec.dataset.group, "model",
    "owner=chat must not drag a sampling knob into the optional-plugins drawer");
});
