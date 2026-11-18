// SPDX-License-Identifier: AGPL-3.0-or-later
// A message sent while a model loads (a first load, a switch) is held under a
// "waiting for <model>" line and sent to the model that load makes active.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const drain = async () => { for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0)); };

function makeFetch(state, chatBodies) {
  return async (url, init) => {
    const u = String(url);
    if (u.includes("/chat/completions")) {
      chatBodies.push(JSON.parse(init.body));
      return { ok: false, status: 503, body: null, json: async () => ({ detail: "stub" }), text: async () => "stub" };
    }
    if (u.startsWith("/api/models")) {
      return { ok: true, status: 200, json: async () => state.payload, text: async () => "" };
    }
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function boot(payload) {
  const state = { payload };
  const chatBodies = [];
  const { window: win } = loadApp({ fetchImpl: makeFetch(state, chatBodies) });
  await drain();
  await win.refreshModels();
  win.document.getElementById("chat-input").value = "hello";
  return { win, state, chatBodies };
}

test("first load: the send is held, not refused, then goes to the loaded model", async () => {
  const { win, state, chatBodies } = await boot({ models: [{ name: "big", active: false }], active: "", loading: "big" });
  const sent = win.sendChat();
  await drain();
  assert.equal(chatBodies.length, 0, "nothing is sent while the model loads");
  assert.match(win.document.getElementById("chat-messages").textContent, /Waiting for big to load/);
  state.payload = { models: [{ name: "big", active: true }], active: "big" };
  await win.refreshModels();
  await sent;
  assert.equal(chatBodies.length, 1);
  assert.equal(chatBodies[0].model, "big");
});

test("a switch: the outgoing model never answers; the new one does", async () => {
  const { win, state, chatBodies } = await boot({
    models: [{ name: "old", active: true }, { name: "new", active: false }], active: "old", loading: "new" });
  const sent = win.sendChat();
  await drain();
  assert.equal(chatBodies.length, 0, "the old model is not sent the message");
  state.payload = { models: [{ name: "old", active: false }, { name: "new", active: true }], active: "new" };
  await win.refreshModels();
  await sent;
  assert.equal(chatBodies.length, 1);
  assert.equal(chatBodies[0].model, "new");
});

test("Stop cancels the held send and nothing is ever sent", async () => {
  const { win, state, chatBodies } = await boot({ models: [{ name: "big", active: false }], active: "", loading: "big" });
  const sent = win.sendChat();
  await drain();
  runScript(win, "chat.abort.abort();");
  await sent;
  await drain();
  assert.equal(win.document.getElementById("chat-messages").textContent.includes("Waiting for big"), false);
  state.payload = { models: [{ name: "big", active: true }], active: "big" };
  await win.refreshModels();
  await drain();
  assert.equal(chatBodies.length, 0);
});

test("a real switchModel load: the held send goes to the model the load made active", async () => {
  const state = { payload: { models: [{ name: "old", active: true }, { name: "new", active: false }], active: "old" } };
  const chatBodies = [];
  let finishLoad;
  const base = makeFetch(state, chatBodies);
  const fetchImpl = async (url, init) => {
    if (String(url).startsWith("/api/models/load")) {
      await new Promise((r) => { finishLoad = r; });
      state.payload = { models: [{ name: "old", active: false }, { name: "new", active: true }], active: "new" };
      return { ok: true, status: 200, json: async () => ({ status: "ok", model: "new" }), text: async () => "" };
    }
    return base(url, init);
  };
  const { window: win } = loadApp({ fetchImpl });
  await drain();
  await win.refreshModels();
  win.document.getElementById("chat-input").value = "hello";
  const load = runScript(win, "switchModel('new', { local: true })");
  await drain();
  const sent = win.sendChat();
  await drain();
  assert.equal(chatBodies.length, 0);
  finishLoad();
  await load;
  await sent;
  assert.equal(chatBodies.length, 1);
  assert.equal(chatBodies[0].model, "new");
});

test("a load that fails turns the held message into a visible error", async () => {
  const { win, state, chatBodies } = await boot({ models: [{ name: "big", active: false }], active: "", loading: "big" });
  const sent = win.sendChat();
  await drain();
  state.payload = { models: [{ name: "big", active: false }], active: "" };
  await win.refreshModels();
  await sent;
  assert.equal(chatBodies.length, 0);
  assert.match(win.document.getElementById("chat-messages").textContent, /big did not load/);
});
