// SPDX-License-Identifier: AGPL-3.0-or-later
// A model switch made anywhere (Models page "Use", sidebar, another tab) leaves
// the sidebar dropdown on the model that will answer, so an unpinned chat is
// answered by the model the user just switched to. A pinned chat keeps its model.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const MODELS = [{ name: "gemma", size_bytes: 8000 }, { name: "coder", size_bytes: 25000 }];

function makeFetch(state, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    const body = opts.body ? JSON.parse(opts.body) : {};
    calls.push({ url: u, body });
    const headers = { get: () => null };
    if (u.startsWith("/api/models/unload")) {
      state.active = "";
      return { ok: true, status: 200, headers, text: async () => "",
               json: async () => ({ status: "unloaded" }) };
    }
    if (u === "/v1/chat/completions" && state.loadOnChat) state.active = body.model;
    if (u.startsWith("/api/models/load")) {
      state.active = body.model;
      return { ok: true, status: 200, headers, text: async () => "",
               json: async () => ({ status: "loaded", model: body.model }) };
    }
    if (u.startsWith("/api/models?type=llm")) {
      return { ok: true, status: 200, headers, text: async () => "",
               json: async () => ({ models: MODELS.map((m) => ({ ...m, active: m.name === state.active })),
                                    active: state.active }) };
    }
    return { ok: true, status: 200, headers, json: async () => ({}), text: async () => "" };
  };
}

async function settle() {
  for (let i = 0; i < 12; i++) await new Promise((r) => setTimeout(r, 0));
}

async function boot() {
  const state = { active: "gemma" };
  const calls = [];
  const { window: win } = loadApp({ fetchImpl: makeFetch(state, calls) });
  win.__state = state;
  win.readSSE = async (_r, onData) => {
    onData(JSON.stringify({ choices: [{ delta: { content: "an answer" } }] }));
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }));
  };
  const doc = win.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;
  runScript(win, "refreshModels();");
  await settle();
  const select = doc.getElementById("model-select");
  assert.equal(select.value, "gemma", "precondition: gemma is shown loaded");
  return { win, select, calls };
}

async function switchTo(win, name) {
  win.__target = name;
  runScript(win, "window.__done = switchModel(window.__target);");
  await win.__done;
  await settle();
}

async function sendIn(win, conv) {
  win.__testConv = conv;
  runScript(win, "chat.conversations = [window.__testConv]; chat.activeId = window.__testConv.id;");
  await win.runCompletion(conv);
  return conv;
}

const chatPosts = (calls) => calls.filter((c) => c.url === "/v1/chat/completions");
const userTurn = () => ({ id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] });

test("a switch that did not come from the dropdown moves the dropdown to the new model", async () => {
  const { win, select } = await boot();

  await switchTo(win, "coder");

  assert.equal(select.value, "coder");
});

test("an unpinned chat after such a switch is sent to the switched-to model", async () => {
  const { win, calls } = await boot();

  await switchTo(win, "coder");
  await sendIn(win, userTurn());

  const [post] = chatPosts(calls);
  assert.equal(post.body.model, "coder");
  assert.equal(post.body.pin_model, false);
});

test("a chat pinned to gemma keeps asking for gemma after the switch", async () => {
  const { win, calls } = await boot();
  const conv = { ...userTurn(), pinnedModel: "gemma" };

  await switchTo(win, "coder");
  await sendIn(win, conv);

  const [post] = chatPosts(calls);
  assert.equal(post.body.model, "gemma");
  assert.equal(post.body.pin_model, true);
});

test("a pinned chat says so before sending when another model is loaded", async () => {
  const { win } = await boot();
  win.__testConv = { ...userTurn(), pinnedModel: "gemma" };
  runScript(win, "chat.conversations = [window.__testConv]; chat.activeId = window.__testConv.id; renderChat();");
  const note = win.document.getElementById("pin-mismatch");
  assert.equal(note.hidden, true, "gemma is loaded and pinned: nothing to say");

  await switchTo(win, "coder");

  assert.equal(note.hidden, false);
  assert.match(note.textContent, /pinned to gemma/);
  assert.match(note.textContent, /coder is loaded/);
});

test("an unpinned chat shows no pin notice after a switch", async () => {
  const { win } = await boot();
  win.__testConv = userTurn();
  runScript(win, "chat.conversations = [window.__testConv]; chat.activeId = window.__testConv.id; renderChat();");

  await switchTo(win, "coder");

  assert.equal(win.document.getElementById("pin-mismatch").hidden, true);
});

async function pinnedToGemmaWhileCoderLoaded() {
  const ctx = await boot();
  ctx.win.__testConv = { ...userTurn(), pinnedModel: "gemma" };
  runScript(ctx.win, "chat.conversations = [window.__testConv]; chat.activeId = window.__testConv.id; renderChat();");
  await switchTo(ctx.win, "coder");
  const note = ctx.win.document.getElementById("pin-mismatch");
  assert.equal(note.hidden, false, "precondition: the mismatch note shows");
  return { ...ctx, note };
}

test("the pin notice clears once the pinned chat's message has loaded its model", async () => {
  const { win, note } = await pinnedToGemmaWhileCoderLoaded();
  win.__state.loadOnChat = true;

  await win.runCompletion(win.__testConv);
  await settle();

  assert.equal(win.__state.active, "gemma", "precondition: the server loaded gemma to answer");
  assert.equal(note.hidden, true, note.textContent);
});

test("the pin notice clears after the sidebar unload", async () => {
  const { win, note } = await pinnedToGemmaWhileCoderLoaded();

  await win.document.getElementById("sidebar-unload-btn").onclick();
  await settle();

  assert.equal(note.hidden, true, note.textContent);
});

test("the pin notice clears when the poll sees the model was unloaded elsewhere", async () => {
  const { win, note } = await pinnedToGemmaWhileCoderLoaded();
  win.__state.active = "";

  runScript(win, "refreshModels();");
  await settle();

  assert.equal(note.hidden, true, note.textContent);
});
