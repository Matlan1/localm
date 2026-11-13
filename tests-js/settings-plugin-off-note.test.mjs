// SPDX-License-Identifier: AGPL-3.0-or-later
// A Settings card whose plugin is not installed or is turned off says so, so a
// person does not save settings that nothing will use. A card of an active
// plugin, and a card the capability list does not name, carry no note.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const FIELDS = [
  { key: "coder_max_turns", widget: "number", label: "Max turns", help: "",
    group: "Coder", owner: "coder", default: 30 },
  { key: "coder_remember_projects", widget: "toggle", label: "Remember projects", help: "",
    group: "Coder", owner: "core", default: true },
  { key: "browser_inline_live_view", widget: "toggle", label: "Live view", help: "",
    group: "Coder", owner: "browser", default: true },
  { key: "bind_host", widget: "text", label: "Bind address", help: "",
    group: "Network", owner: "core", default: "" },
  { key: "net_mode", widget: "select", label: "Network access", help: "",
    group: "Network", owner: "browser", options: ["off", "ask", "allow"], default: "ask" },
  { key: "rag_top_k", widget: "number", label: "Chunks", help: "",
    group: "Knowledge", owner: "rag", default: 4 },
  { key: "memory_enabled", widget: "toggle", label: "Memory recall", help: "",
    group: "Memory", owner: "memory", default: true },
  { key: "web_backend", widget: "text", label: "Search backend", help: "",
    group: "Web", owner: "web", default: "" },
];

const PLUGINS = [
  { name: "coder", installed: false, enabled: false, active: false },
  { name: "rag", installed: true, enabled: false, active: false },
  { name: "memory", installed: true, enabled: true, active: true },
  { name: "browser", installed: false, enabled: false, active: false },
];

function makeFetch() {
  return async (url) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => ({ fields: FIELDS }), text: async () => "" };
    }
    if (url === "/api/capabilities") {
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ plugins: PLUGINS, core: {}, fs_access: "none" }) };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

async function settle() {
  for (let i = 0; i < 12; i++) await new Promise((r) => setTimeout(r, 0));
}

function note(win, group) {
  const sec = win.document.getElementById("settings-sec-core-" + group);
  assert.ok(sec, `the ${group} card is drawn`);
  const n = sec.querySelector(".plugin-off-note");
  return n ? n.textContent : null;
}

test("cards of a missing or turned-off plugin say so; others do not", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  runScript(win, "refreshPluginCommands().then(() => refreshSettingsPage());");
  await settle();
  assert.match(note(win, "Coder") || "", /plugin is not installed, so these settings do nothing yet/);
  assert.match(note(win, "Knowledge") || "", /plugin is turned off/);
  assert.equal(note(win, "Memory"), null, "an active plugin's card has no note");
  assert.equal(note(win, "Web"), null, "a plugin the list does not name gets no note");
  assert.equal(note(win, "Network"), null, "a card of core fields and a plugin's field gets no note");
});
