// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings: when the server's environment sets the session mode (LOCALM_MODE /
// --mode) or the network access mode (LOCALM_NET_MODE) over the saved value, a
// note under that field says what is in force.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const FIELDS = [
  { key: "mode", widget: "select", label: "Session persistence", help: "h",
    group: "Privacy", owner: "core", options: ["privacy", "log", "full"], default: "privacy" },
  { key: "net_mode", widget: "select", label: "Network access", help: "h",
    group: "Network", owner: "web", options: ["off", "ask", "allow"], default: "ask" },
];

function makeFetch(liveConfig) {
  return async (url, opts = {}) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => ({ fields: structuredClone(FIELDS) }),
               text: async () => "" };
    }
    if (url === "/v1/config" && (opts.method || "GET") === "GET") {
      return { ok: true, status: 200, json: async () => structuredClone(liveConfig),
               text: async () => "" };
    }
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function settle() {
  for (let i = 0; i < 15; i++) await new Promise((r) => setTimeout(r, 0));
}

async function render(liveConfig) {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(liveConfig) });
  await settle();
  runScript(win, "refreshSettingsPage();");
  await settle();
  return win;
}

test("an environment-set session and network mode are noted under their fields", async () => {
  const win = await render({ mode: "privacy", effective_mode: "full", mode_from_env: true,
                             net_mode: "ask", effective_net_mode: "off", net_mode_from_env: true });
  const mode = win.document.querySelector('[data-field-key="mode"] #mode-env-note');
  const net = win.document.querySelector('[data-field-key="net_mode"] #net-mode-env-note');
  assert.ok(mode, "a note under Session persistence");
  assert.match(mode.textContent, /^In force now: full, set by LOCALM_MODE/);
  assert.ok(net, "a note under Network access");
  assert.match(net.textContent, /^In force now: off, set by LOCALM_NET_MODE/);
});

test("no note when the environment sets neither", async () => {
  const win = await render({ mode: "privacy", effective_mode: "privacy", mode_from_env: false,
                             net_mode: "ask", effective_net_mode: "ask", net_mode_from_env: false });
  assert.ok(win.document.querySelector('[data-field-key="mode"]'), "the field is drawn");
  assert.equal(win.document.getElementById("mode-env-note"), null);
  assert.equal(win.document.getElementById("net-mode-env-note"), null);
});

test("a later read without the override removes the note", async () => {
  const win = await render({ effective_mode: "log", mode_from_env: true });
  assert.ok(win.document.getElementById("mode-env-note"));
  win.showEnvOverrideNotes(null);
  assert.equal(win.document.getElementById("mode-env-note"), null);
});
