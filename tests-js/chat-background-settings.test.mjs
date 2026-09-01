// SPDX-License-Identifier: AGPL-3.0-or-later
// The Settings > System > Appearance "Chat background" picker: chat_background
// is a Widget.HIDDEN core field (like logo_style), rendered by the static
// #sec-appearance card in index.html and wired by settings.js's
// setupChatBackgroundPicker(). That wiring must only ever run from
// refreshSettingsPage() (an auth-gated point), never at module load - see
// keygate.test.mjs's "must not reach ... until bootAuthProbe has confirmed".
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = { fields: [] };

function makeFetch(config, patches) {
  return async (url, opts = {}) => {
    const method = opts.method || "GET";
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    }
    if (url === "/v1/config" && method === "GET") {
      return { ok: true, status: 200, json: async () => config, text: async () => "" };
    }
    if (url === "/v1/config" && method === "PATCH") {
      const body = JSON.parse(opts.body);
      patches.push(body);
      Object.assign(config, body);
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

async function render(win) {
  await new Promise((r) => setTimeout(r, 0));
  runScript(win, "refreshSettingsPage();");
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
}

// keygate.test.mjs's "must not reach ... until bootAuthProbe has confirmed"
// cases are the regression guard for setupChatBackgroundPicker() only ever
// being wired from refreshSettingsPage() (this file's own auth-gated boot
// scenario would need to duplicate that harness to test the same property
// correctly, which keygate.test.mjs already owns).

test("the preview seeds from the current chat_background on render", async () => {
  const uri = "data:image/jpeg;base64,iVBORw0KGgo=";
  const config = { chat_background: uri };
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(config, []) });
  await render(win);
  const preview = win.document.getElementById("chat-bg-preview");
  assert.equal(preview.classList.contains("empty"), false);
  assert.match(preview.style.backgroundImage, /iVBORw0KGgo=/);
});

test("with chat_background unset, the preview stays on the empty placeholder", async () => {
  const config = { chat_background: "" };
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(config, []) });
  await render(win);
  const preview = win.document.getElementById("chat-bg-preview");
  assert.equal(preview.classList.contains("empty"), true);
});

test("Clear PATCHes chat_background to empty and resets the preview", async () => {
  const uri = "data:image/jpeg;base64,iVBORw0KGgo=";
  const config = { chat_background: uri };
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(config, patches) });
  await render(win);

  win.document.getElementById("chat-bg-clear").click();
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(patches.length, 1);
  assert.deepEqual(patches[0], { chat_background: "" });
  const preview = win.document.getElementById("chat-bg-preview");
  assert.equal(preview.classList.contains("empty"), true);
});

test("a near-miss chat_background value never reaches a CSS url() - applyChatBackground", async () => {
  const { window: win } = loadAppWithPages();
  runScript(win, `applyChatBackground("data:text/html,<script>1</script>");`);
  const prop = win.document.documentElement.style.getPropertyValue("--chat-bg-image");
  assert.equal(prop.trim(), "none");
});

test("applyChatBackground sets the CSS variable for a genuine raster data URI", async () => {
  const { window: win } = loadAppWithPages();
  const uri = "data:image/jpeg;base64,iVBORw0KGgo=";
  runScript(win, `applyChatBackground(${JSON.stringify(uri)});`);
  const prop = win.document.documentElement.style.getPropertyValue("--chat-bg-image");
  assert.equal(prop.trim(), `url("${uri}")`);
});

test("applyChatBackground clears the CSS variable for an empty value", async () => {
  const { window: win } = loadAppWithPages();
  const uri = "data:image/jpeg;base64,iVBORw0KGgo=";
  runScript(win, `applyChatBackground(${JSON.stringify(uri)});`);
  runScript(win, `applyChatBackground("");`);
  const prop = win.document.documentElement.style.getPropertyValue("--chat-bg-image");
  assert.equal(prop.trim(), "none");
});
