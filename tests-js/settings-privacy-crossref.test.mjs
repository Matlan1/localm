// SPDX-License-Identifier: AGPL-3.0-or-later
// NEW-SETTINGS-IA-REVIEW: a plugin-owned field can declare group="Privacy" (or
// any other core group) while still being filed under its OWNER's plugin tab
// (settingsSectionOf routes by owner, not group, for owner != "core" - deliberate,
// see settings.js's own comment). Browsing the core Privacy section must not look
// like "that is everything" when related toggles live elsewhere.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = {
  fields: [
    { key: "mode", widget: "select", label: "Session persistence",
      help: "how much is saved", group: "Privacy", owner: "core",
      options: ["privacy", "log", "full"], default: "log" },
    // Plugin-owned but group="Privacy" - the case under test.
    { key: "chat_mode", widget: "select", label: "Chat persistence override",
      help: "Overrides the global persistence for chat only.", group: "Privacy",
      owner: "chat", options: ["", "privacy", "log", "full"], default: "" },
    { key: "coder_mode", widget: "select", label: "Coder persistence override",
      help: "Overrides the global persistence for the coder only.", group: "Privacy",
      owner: "coder", options: ["", "privacy", "log", "full"], default: "" },
    // A plain plugin field whose group MATCHES its own plugin label - must NOT
    // trigger a cross-reference note (no core "Engine"-owned-by-chat mismatch
    // to point out; this exercises that the note is not just "any plugin field").
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
  await new Promise((r) => setTimeout(r, 0));
}

test("core Privacy section points to plugin tabs holding related Privacy-group fields", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  const doc = win.document;

  const privacySec = doc.getElementById("settings-sec-core-Privacy");
  assert.ok(privacySec, "core Privacy section renders");
  // chat_mode/coder_mode themselves must NOT be inside the Privacy section -
  // confirms the routing-by-owner behavior this note exists to compensate for.
  assert.equal(privacySec.querySelector('[data-key="chat_mode"]'), null);
  assert.equal(privacySec.querySelector('[data-key="coder_mode"]'), null);

  const note = privacySec.textContent;
  assert.match(note, /Related settings also live on plugin tabs/);
  assert.match(note, /Chat/);
  assert.match(note, /Coder/);
});

test("chat_mode and coder_mode actually render on their own plugin tabs", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  const doc = win.document;

  const chatSec = doc.getElementById("settings-sec-plugin-chat");
  const coderSec = doc.getElementById("settings-sec-plugin-coder");
  assert.ok(chatSec.querySelector('[data-key="chat_mode"]'), "chat_mode lives on the Chat plugin tab");
  assert.ok(coderSec.querySelector('[data-key="coder_mode"]'), "coder_mode lives on the Coder plugin tab");
});

test("a plugin section whose fields all match its own group gets no cross-reference note", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  const doc = win.document;
  const chatSec = doc.getElementById("settings-sec-plugin-chat");
  // Plugin sections never get the note (only CORE sections do) - this is a
  // Privacy-browsing aid, not a general "this field is elsewhere" annotator.
  assert.doesNotMatch(chatSec.textContent, /Related settings also live/);
});

test("a core section with no cross-filed plugin fields gets no note (e.g. Security)", async () => {
  const schema = {
    fields: [
      { key: "require_auth", widget: "toggle", label: "Require an API key",
        help: "", group: "Security", owner: "core", default: false },
    ],
  };
  const fetchImpl = async (url) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => schema, text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await render(win);
  const sec = win.document.getElementById("settings-sec-core-Security");
  assert.doesNotMatch(sec.textContent, /Related settings also live/);
});
