// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings details the matrix found wrong: the folder picker title of each list,
// the secret Clear/Undo tooltip, a cleared text field going back to its default,
// and a save refusal naming the field's label instead of its config key.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const FIELDS = [
  { key: "rag_allowed_roots", widget: "pathlist", label: "Allowed folders", help: "h",
    group: "Knowledge", owner: "rag", default: [] },
  { key: "rag_denied_roots", widget: "pathlist", label: "Denied folders", help: "h",
    group: "Knowledge", owner: "rag", default: [] },
  { key: "hf_token", widget: "secret", label: "Hugging Face token", help: "",
    group: "Models", owner: "core", is_set: true },
  { key: "mdns_name", widget: "text", label: "Network name (mDNS)", help: "",
    group: "Server", owner: "core", applies: "restart", default: "studio",
    shipped_default: "localm" },
  { key: "chat_web_max_lookups", widget: "number", label: "Web lookups per message",
    help: "", group: "Chat", owner: "core", applies: "live", default: 3 },
];

function makeFetch(patches, refusal) {
  return async (url, opts = {}) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => ({ fields: structuredClone(FIELDS) }),
               text: async () => "" };
    }
    if (url === "/v1/config" && (opts.method || "GET") === "PATCH") {
      patches.push(JSON.parse(opts.body));
      if (refusal) return { ok: false, status: 400, json: async () => ({ detail: refusal }) };
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
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

async function open(patches = [], refusal) {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(patches, refusal) });
  runScript(win, 'caps.fsAccess = "host"; refreshSettingsPage();');
  await settle();
  return win;
}

test("each folder list's picker says whether the folder is allowed or denied", async () => {
  const win = await open();
  runScript(win, "globalThis.__titles = []; pickDirectory = (title) => { globalThis.__titles.push(title); return Promise.resolve(null); };");
  for (const key of ["rag_allowed_roots", "rag_denied_roots"]) {
    win.document.querySelector(`[data-field-key="${key}"] .pathlist-add`).click();
  }
  await settle();
  assert.deepEqual(win.__titles, ["Pick a folder to allow", "Pick a folder to deny"]);
});

test("the secret Clear button's tooltip follows its Clear / Undo state", async () => {
  const win = await open();
  const btn = win.document.querySelector('[data-field-key="hf_token"] .secret-clear-btn');
  assert.equal(btn.title, "Clear stored secret on save");
  btn.click();
  assert.equal(btn.textContent, "Undo");
  assert.equal(btn.title, "Keep the stored secret");
  btn.click();
  assert.equal(btn.title, "Clear stored secret on save");
});

test("clearing a changed text field sends its shipped default, not an empty value", async () => {
  const patches = [];
  const win = await open(patches);
  const input = win.document.querySelector('input[data-key="mdns_name"]');
  assert.equal(input.value, "studio");
  input.value = "";
  input.dispatchEvent(new win.Event("input", { bubbles: true }));
  runScript(win, 'saveSettingsSection("core-Server");');
  await settle();
  assert.equal(patches.length, 1);
  assert.equal(patches[0].mdns_name, "localm");
});

test("a save refusal names the field's label, not its config key", async () => {
  const win = await open([], "chat_web_max_lookups: -1 is below the minimum 0");
  const input = win.document.querySelector('input[data-key="chat_web_max_lookups"]');
  input.value = "-1";
  input.dispatchEvent(new win.Event("input", { bubbles: true }));
  runScript(win, 'saveSettingsSection("core-Chat");');
  await settle();
  assert.equal(win.document.getElementById("toast").textContent,
    "Web lookups per message: -1 is below the minimum 0");
});

test("a refusal without a known key prefix is shown unchanged", async () => {
  const win = await open();
  runScript(win, 'globalThis.__out = [labelledRefusal("plain text", _settingsControls),'
    + ' labelledRefusal("no_such_key: x", _settingsControls), labelledRefusal(undefined, [])];');
  assert.deepEqual([...win.__out], ["plain text", "no_such_key: x", undefined]);
});
