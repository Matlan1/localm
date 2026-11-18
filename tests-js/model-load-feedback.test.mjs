// SPDX-License-Identifier: AGPL-3.0-or-later
// While a chat model loads the page says so in the chat area and in the model
// dropdown; the empty chat page follows the active model; a reply after an
// unload is attributed to the model the server reloads.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const drain = async () => { for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0)); };

function modelsFetch(payload, calls) {
  return async (url, init) => {
    if (calls) calls.push({ url: String(url), init });
    if (String(url).startsWith("/api/models")) {
      return { ok: true, status: 200, json: async () => payload, text: async () => "" };
    }
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function boot(payload, calls) {
  const { window: win } = loadApp({ fetchImpl: modelsFetch(payload, calls) });
  await drain();
  return win;
}

const sel = (win) => win.document.getElementById("model-select");
const banner = (win) => win.document.getElementById("chat-load-banner");
const picked = (win) => sel(win).selectedOptions[0];

test("a first load from no model: dropdown and chat area both say loading", async () => {
  const win = await boot({ models: [{ name: "big", active: false }], active: "", loading: "big" });
  await win.refreshModels();
  assert.equal(picked(win).value, "", "the placeholder is shown, not a model");
  assert.match(picked(win).textContent, /loading big/);
  assert.doesNotMatch(picked(win).textContent, /No model loaded/);
  assert.equal(banner(win).hidden, false);
  assert.match(banner(win).textContent, /Loading big/);
});

test("a switch: the outgoing model is not shown as the answer, loading is", async () => {
  const win = await boot({ models: [{ name: "old", active: true }, { name: "new", active: false }],
                           active: "old", loading: "new" });
  await win.refreshModels();
  assert.match(picked(win).textContent, /loading new/);
  assert.equal(banner(win).hidden, false);
});

test("the dropdown and banner go back to normal when the load has ended", async () => {
  const win = await boot({ models: [{ name: "old", active: true }], active: "old", loading: "new" });
  await win.refreshModels();
  runScript(win, "modelCache.loading = '';");
  win.fetch = modelsFetch({ models: [{ name: "new", active: true }], active: "new" });
  await win.refreshModels();
  assert.equal(picked(win).value, "new");
  assert.equal(banner(win).hidden, true);
});

test("a load that leaves the selection alone is not announced", async () => {
  const win = await boot({ models: [{ name: "old", active: true }], active: "old" });
  runScript(win, "_switchingTo = 'other'; _switchActivates = false; modelCache.loading = 'other';");
  win.fetch = modelsFetch({ models: [{ name: "old", active: true }], active: "old" });
  await win.refreshModels();
  assert.equal(picked(win).value, "old");
  assert.equal(banner(win).hidden, true);
});

test("the empty chat page names the model once it is active, and again after a switch", async () => {
  const win = await boot({ models: [], active: "" });
  runScript(win, "renderChat();");
  const box = win.document.getElementById("chat-messages");
  assert.match(box.textContent, /Chat with your local model/);
  win.fetch = modelsFetch({ models: [{ name: "m1", active: true }], active: "m1" });
  await win.refreshModels();
  assert.match(box.textContent, /Chat with m1\./);
  win.fetch = modelsFetch({ models: [{ name: "m2", active: true }], active: "m2" });
  await win.refreshModels();
  assert.match(box.textContent, /Chat with m2\./);
  assert.equal(box.querySelectorAll(".empty-hint").length, 1);
});

test("a chat with messages is not redrawn by an active model change", async () => {
  const win = await boot({ models: [], active: "" });
  const box = win.document.getElementById("chat-messages");
  const row = win.document.createElement("div");
  row.className = "msg-row user";
  box.replaceChildren(row);
  win.fetch = modelsFetch({ models: [{ name: "m1", active: true }], active: "m1" });
  await win.refreshModels();
  assert.equal(box.firstElementChild, row);
});

test("after an unload the reply is sent to, and headed with, the resumable model", async () => {
  const calls = [];
  const win = await boot({ models: [{ name: "m1", active: false }], active: "", resumable: "m1" }, calls);
  win.document.getElementById("chat-input").value = "hello";
  win.fetch = async (url, init) => {
    calls.push({ url: String(url), init });
    return { ok: false, status: 503, body: null, json: async () => ({}), text: async () => "" };
  };
  await win.sendChat().catch(() => {});
  const req = calls.find((c) => c.url.includes("/chat/completions"));
  assert.ok(req, "a chat request fires");
  assert.equal(JSON.parse(req.init.body).model, "m1");
  const role = win.document.querySelector("#chat-messages .msg-row:not(.user) .msg-role");
  assert.ok(role, "an assistant row was drawn");
  assert.equal(role.textContent, "m1");
});
