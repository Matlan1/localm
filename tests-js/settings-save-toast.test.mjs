// SPDX-License-Identifier: AGPL-3.0-or-later
// The toast after a Settings section save follows the `applies` of the values
// that changed: a restart field says to restart, a next-load field says so, a
// live field just says Saved.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const FIELDS = [
  { key: "port", widget: "number", label: "Server port", help: "", group: "Server",
    owner: "core", applies: "restart", default: 8642 },
  { key: "mdns_name", widget: "text", label: "Network name", help: "", group: "Server",
    owner: "core", applies: "restart", default: "localm" },
  { key: "bind_host", widget: "text", label: "Bind address", help: "", group: "Server",
    owner: "core", applies: "restart", default: "" },
  { key: "idle_unload_seconds", widget: "number", label: "Idle unload", help: "", group: "Engine",
    owner: "core", applies: "live", default: 0 },
  { key: "spec_draft_model", widget: "text", label: "Draft model", help: "", group: "Engine",
    owner: "core", applies: "next_load", default: "" },
  { key: "n_ctx", widget: "number", label: "Context window", help: "", group: "Engine",
    owner: "core", applies: "next_load", default: 4096 },
  { key: "temperature", widget: "number", label: "Temperature", help: "", group: "Chat",
    owner: "core", applies: "live", default: 0.8 },
  { key: "top_k", widget: "number", label: "Top-k", help: "", group: "Chat",
    owner: "core", applies: "live", default: 40 },
];

function makeFetch(patchReply = {}) {
  return async (url, opts = {}) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => ({ fields: FIELDS }), text: async () => "" };
    }
    if (url === "/v1/config" && opts.method === "PATCH") {
      return { ok: true, status: 200, json: async () => patchReply, text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

async function settle() {
  for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
}

async function saveWith(key, value, section, patchReply) {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(patchReply) });
  runScript(win, "refreshSettingsPage();");
  await settle();
  const doc = win.document;
  const input = doc.querySelector(`[data-field-key="${key}"] input`);
  assert.ok(input, `${key} renders an input`);
  input.value = String(value);
  input.dispatchEvent(new win.Event("input", { bubbles: true }));
  runScript(win, `window.__saved = saveSettingsSection(${JSON.stringify(section)});`);
  await settle();
  assert.equal(await win.__saved, true, "the save succeeds");
  const toastEl = doc.getElementById("toast");
  saveWith.lastToastClass = toastEl.className;
  return toastEl.textContent;
}

test("a changed restart field says a restart is needed", async () => {
  assert.match(await saveWith("mdns_name", "studio", "core-Server"), /restart localm/);
});

test("a changed port says it needs a full stop and start, since Restart keeps the current port", async () => {
  const text = await saveWith("port", 8700, "core-Server");
  assert.match(text, /full stop and start/);
  assert.match(text, /keeps the current port/);
});

test("a changed next-load field says it applies on the next model load", async () => {
  assert.match(await saveWith("n_ctx", 8192, "core-Engine"), /next model load/);
});

test("a changed live field just says Saved", async () => {
  const text = await saveWith("temperature", 0.5, "core-Chat");
  assert.equal(text.trim(), "Saved");
});

test("a blank text field left blank does not count as changed", async () => {
  const text = await saveWith("idle_unload_seconds", 30, "core-Engine");
  assert.equal(text.trim(), "Saved",
    "the untouched blank Draft model box in the same card must not make it a next-load save");
});

test("an unchanged restart field does not ask for a restart", async () => {
  const text = await saveWith("mdns_name", "localm", "core-Server");
  assert.equal(text.trim(), "Nothing changed", "a value typed back to the saved one is not sent");
});

test("a save the server took with a warning shows the warning in an error toast", async () => {
  const text = await saveWith("temperature", 0.5, "core-Chat",
    { warnings: ["'nomic' names no model yet; pull or add it before indexing."] });
  assert.match(text, /^Saved\. 'nomic' names no model yet; pull or add it before indexing\.$/);
  assert.equal(saveWith.lastToastClass, "error");
});

test("a save with an empty warnings list is a plain Saved", async () => {
  const text = await saveWith("temperature", 0.5, "core-Chat", { warnings: [] });
  assert.equal(text.trim(), "Saved");
  assert.equal(saveWith.lastToastClass, "show");
});
