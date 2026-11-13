// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings tracks edits by value: a card's Save sends only the controls whose
// value differs from the saved one, a value typed back is not an unsaved edit,
// and a save (or any other redraw) keeps the unsaved edits of the other cards.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const FIELDS = [
  { key: "n_ctx", widget: "number", label: "Context window", help: "", group: "Engine",
    owner: "core", applies: "next_load", default: 8192, shipped_default: 4096 },
  { key: "flash_attn", widget: "toggle", label: "Flash attention", help: "", group: "Engine",
    owner: "core", applies: "next_load", default: true },
  { key: "net_mode", widget: "select", label: "Network access", help: "", group: "Network",
    owner: "core", applies: "live", default: "ask", options: ["off", "ask", "allow"] },
  { key: "net_allow_private", widget: "toggle", label: "Allow private addresses", help: "",
    group: "Network", owner: "core", applies: "live", default: false },
  { key: "comfy_launch_timeout", widget: "number", label: "Launch timeout", help: "",
    group: "Media", owner: "core", applies: "live", default: 300, shipped_default: 300 },
  { key: "comfy_fast_dequant", widget: "toggle", label: "Fast dequant", help: "",
    group: "Media", owner: "core", applies: "live", default: false },
  { key: "comfy_gpu_placement", widget: "toggle", label: "GPU placement", help: "",
    group: "Media", owner: "core", applies: "live", default: false },
];

function makeFetch(patches, fields = FIELDS) {
  return async (url, opts = {}) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => ({ fields }), text: async () => "" };
    }
    if (url === "/v1/config" && (opts.method || "GET") === "PATCH") {
      const body = JSON.parse(opts.body);
      patches.push(body);
      for (const f of fields) if (f.key in body) f.default = body[f.key];
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    if (url === "/v1/media/config") {
      return { ok: true, status: 200, json: async () => ({ plugins: [] }), text: async () => "" };
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

async function open(patches, fields) {
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(patches, structuredClone(fields || FIELDS)),
  });
  runScript(win, "refreshSettingsPage();");
  await settle();
  return win;
}

function control(win, key) {
  return win.document.querySelector(`[data-key="${key}"]`);
}

function type(win, key, value) {
  const input = control(win, key);
  if (input.type === "checkbox") input.checked = value;
  else input.value = String(value);
  input.dispatchEvent(new win.Event("input", { bubbles: true }));
  input.dispatchEvent(new win.Event("change", { bubbles: true }));
}

async function save(win, secId) {
  runScript(win, `window.__saved = saveSettingsSection(${JSON.stringify(secId)});`);
  await settle();
  return win.__saved;
}

test("a card's Save sends only the control that was changed", async () => {
  const patches = [];
  const win = await open(patches);
  type(win, "n_ctx", 16384);
  assert.equal(await save(win, "core-Engine"), true);
  assert.deepEqual(patches, [{ n_ctx: 16384 }],
    "the untouched toggle of the same card is not sent");
});

test("a Save with nothing changed sends nothing and says so", async () => {
  const patches = [];
  const win = await open(patches);
  assert.equal(await save(win, "core-Network"), true,
    "nothing to save still leaves the saved values in place");
  assert.equal(patches.length, 0, "no PATCH for a card nobody edited");
  assert.equal(win.document.getElementById("toast").textContent, "Nothing changed");
});

test("a value typed and then typed back is not an unsaved edit", async () => {
  const win = await open([]);
  type(win, "n_ctx", 16384);
  assert.equal(win.settingsDirty(), true, "a changed value is unsaved");
  type(win, "n_ctx", 8192);
  assert.equal(win.settingsDirty(), false, "the saved value typed back is not");
  type(win, "flash_attn", false);
  type(win, "flash_attn", true);
  assert.equal(win.settingsDirty(), false, "a toggle flipped twice is not");
});

test("saving one card keeps the unsaved edit of another card across the redraw", async () => {
  const patches = [];
  const win = await open(patches);
  type(win, "net_mode", "off");
  type(win, "n_ctx", 16384);
  assert.equal(await save(win, "core-Engine"), true);
  assert.deepEqual(patches, [{ n_ctx: 16384 }]);
  const netMode = control(win, "net_mode");
  assert.equal(netMode.value, "off", "the Network card still shows the unsaved choice");
  assert.equal(win.settingsDirty(), true, "and it still counts as unsaved");
  assert.equal(control(win, "n_ctx").value, "16384", "the saved card shows the saved value");
  assert.equal(await save(win, "core-Network"), true);
  assert.deepEqual(patches[1], { net_mode: "off" }, "the carried edit saves like any edit");
  assert.equal(win.settingsDirty(), false);
});

test("a redraw not caused by a save also keeps unsaved edits", async () => {
  const win = await open([]);
  type(win, "net_allow_private", true);
  runScript(win, "refreshSettingsPage();");
  await settle();
  assert.equal(control(win, "net_allow_private").checked, true);
  assert.equal(win.settingsDirty(), true);
});

test("the Media Shared and Experimental boxes each save only their own controls", async () => {
  const patches = [];
  const win = await open(patches);
  type(win, "comfy_fast_dequant", true);
  type(win, "comfy_gpu_placement", true);
  const shared = control(win, "comfy_fast_dequant").closest(".media-comfy-box");
  const experimental = control(win, "comfy_gpu_placement").closest(".media-comfy-box");
  assert.notEqual(shared, experimental);
  const sharedSave = [...shared.querySelectorAll("button.btn-primary")].pop();
  const experimentalSave = [...experimental.querySelectorAll("button.btn-primary")].pop();
  assert.equal(sharedSave.textContent, "Save Shared");
  assert.equal(experimentalSave.textContent, "Save Experimental");
  sharedSave.click();
  await settle();
  assert.deepEqual(patches, [{ comfy_fast_dequant: true }],
    "the Experimental box's unsaved edit is not saved by the Shared box's button");
  assert.equal(control(win, "comfy_gpu_placement").checked, true,
    "and it survives the redraw");
});
