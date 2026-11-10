// SPDX-License-Identifier: AGPL-3.0-or-later
// After a model is renamed, the page rewrites what it holds under the old name
// (pinned chats, the avatar, the media pickers' remembered choices); a chat
// pinned to a name that no longer exists says so in the chat instead of ending
// in a 404 toast, and keeps the person's message.
import { test } from "node:test";
import assert from "node:assert/strict";

import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

const plain = (v) => JSON.parse(JSON.stringify(v));
const NOTE_A = "Updated 1 scheduled job(s) to the new name";
const NOTE_B = "A per-project .localcoder/config.toml 'model' setting (if any) lives "
  + "outside <data dir> and was NOT updated - fix it by hand in any project that pinned this model.";

const MODELS = [
  { name: "gemma3-12b", active: false, loaded: false, model_type: "llm", size_bytes: 100 },
];

function renameFetch(renameResponse) {
  return async (url) => {
    const u = String(url);
    if (u.startsWith("/api/models/rename")) {
      return { ok: true, status: 200, json: async () => renameResponse, text: async () => "" };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models: MODELS, active: null }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

async function renameOnPage(window, typed) {
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));
  runScript(window, `promptText = async () => ${JSON.stringify(typed)};`);
  const btn = [...window.document.querySelectorAll("#models-table tbody tr button")]
    .find((b) => b.textContent === "rename");
  assert.ok(btn, "the row exposes a rename control");
  await btn.onclick();
  await new Promise((r) => setTimeout(r, 0));
}

test("the rename toast ends every note with a period and joins them", async () => {
  const toasts = [];
  const { window } = loadAppWithPages({
    fetchImpl: renameFetch({ status: "renamed", model: "gemma3-12b", new_name: "daily-driver",
                             notes: [NOTE_A, NOTE_B] }),
  });
  window.toast = (msg) => toasts.push(String(msg));

  await renameOnPage(window, "daily-driver");

  assert.deepEqual(toasts, [`Renamed to 'daily-driver'. ${NOTE_A}. ${NOTE_B}`]);
});

test("a rename made on the page rewrites the chats pinned to it, here, in the browser copy and on the server", async () => {
  const puts = [];
  const { window } = loadAppWithPages({
    fetchImpl: async (url, opts = {}) => {
      const u = String(url);
      if (u.startsWith("/api/conversations/") && opts.method === "PUT") {
        puts.push({ u, body: JSON.parse(opts.body) });
        return { ok: true, status: 200, json: async () => ({}) };
      }
      return renameFetch({ status: "renamed", model: "gemma3-12b", new_name: "daily-driver",
                           notes: [] })(url, opts);
    },
  });
  window.toast = () => {};
  runScript(window, `
    chat.privacy = false; chat.modeKnown = true; chat.persist = true;
    chat.conversations = [
      { id: "mine", title: "t", messages: [{ role: "user", content: "gemma3-12b stays" }],
        pinnedModel: "gemma3-12b" },
      { id: "other", title: "t", messages: [], pinnedModel: "someone-else" },
      { id: "free", title: "t", messages: [] },
    ];
    chat.activeId = "mine";
  `);

  await renameOnPage(window, "daily-driver");

  const saved = window.localStorage.getItem("localm.conversations");
  assert.ok(saved, "the browser copy of the chats was rewritten");
  const snapshot = JSON.parse(saved);
  const byId = Object.fromEntries(snapshot.map((c) => [c.id, c]));
  assert.equal(byId.mine.pinnedModel, "daily-driver");
  assert.equal(byId.mine.messages[0].content, "gemma3-12b stays");
  assert.equal(byId.other.pinnedModel, "someone-else");
  assert.equal(byId.free.pinnedModel, undefined);
  await new Promise((r) => setTimeout(r, 700));
  assert.deepEqual(puts.map((p) => [p.u, p.body.pinned_model]),
    [["/api/conversations/mine", "daily-driver"]]);
});

test("a rename moves the avatar override and the media pickers' remembered choices", () => {
  const { window } = loadAppWithPages({
    seedLocalStorage: {
      "localm.nativeModels.image": JSON.stringify(
        { models: { vae: "gemma3-12b", clip_l: "keep" }, loras: [{ name: "gemma3-12b", strength: 0.5 }] }),
      "localm.nativeModels.music": JSON.stringify({ dit: "gemma3-12b", lm: "keep" }),
    },
  });
  runScript(window, `chat.modelAvatarOverrides = { "gemma3-12b": "OV", "someone-else": "SE" };`);

  window.dispatchEvent(new window.CustomEvent("localm:model-renamed",
    { detail: { from: "gemma3-12b", to: "daily-driver" } }));

  runScript(window, `window.__avatars = chat.modelAvatarOverrides;
    window.__image = imageModelPicker.choice(); window.__music = musicModelPicker.choice();`);
  assert.deepEqual(plain(window.__avatars), { "daily-driver": "OV", "someone-else": "SE" });
  assert.deepEqual(plain(window.__image), {
    native_models: { vae: "daily-driver", clip_l: "keep" },
    loras: [{ name: "daily-driver", strength: 0.5 }] });
  assert.deepEqual(plain(window.__music), { native_models: { dit: "daily-driver", lm: "keep" } });
  assert.deepEqual(JSON.parse(window.localStorage.getItem("localm.nativeModels.image")).models,
    { vae: "daily-driver", clip_l: "keep" });
  assert.deepEqual(JSON.parse(window.localStorage.getItem("localm.nativeModels.music")),
    { dit: "daily-driver", lm: "keep" });
});

// ---------------------------------------------------------------------------
//  A chat pinned to a name that no longer exists
// ---------------------------------------------------------------------------

function chatSetup({ registry = ["live"], stored = null, chat404 = false } = {}) {
  const calls = [];
  const toasts = [];
  const impl = async (url, opts = {}) => {
    const u = String(url);
    let body;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = opts.body; }
    calls.push({ u, body });
    const headers = { get: () => null };
    if (u === "/api/models") {
      return { ok: true, status: 200, headers,
               json: async () => ({ models: registry.map((m) => (typeof m === "string" ? { name: m } : m)),
                                  active: "live" }) };
    }
    if (u.startsWith("/api/conversations/") && !opts.method) {
      return stored
        ? { ok: true, status: 200, headers, json: async () => stored }
        : { ok: false, status: 404, headers, json: async () => ({}) };
    }
    if (u === "/v1/chat/completions" && chat404) {
      return { ok: false, status: 404, headers, statusText: "Not Found",
               json: async () => ({ detail: "Model 'old' is not registered. Registered models in your library: live" }) };
    }
    return { ok: true, status: 200, headers, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl: impl });
  window.toast = (msg, isError) => toasts.push({ msg: String(msg), isError });
  window.maybeCompactConversation = async () => {};
  window.readSSE = async (_r, onData) => {
    onData(JSON.stringify({ choices: [{ delta: { content: "an answer" } }] }));
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }));
  };
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;
  runScript(window, `
    chat.privacy = true;
    modelCache.active = "live"; modelCache.models = [{ name: "live" }];
    chat.conversations = [{ id: "c1", title: "t", messages: [], pinnedModel: "old" }];
    chat.activeId = "c1";
  `);
  return { window, doc, calls, toasts };
}

const chatPosts = (calls) => calls.filter((c) => c.u === "/v1/chat/completions");

test("sending in a chat pinned to a model that is gone asks nothing of the server and keeps the message", async () => {
  const { window, doc, calls, toasts } = chatSetup();
  doc.getElementById("chat-input").value = "hello there";

  await window.sendChat();

  runScript(window, "window.__pin = chat.conversations[0].pinnedModel;");
  assert.equal(window.__pin, undefined, "the pin to a model that does not exist is removed");
  assert.equal(doc.getElementById("chat-input").value, "hello there", "the message is kept");
  const notice = doc.querySelector("#chat-messages .tool-notice");
  assert.ok(notice && notice.textContent.includes("old") && notice.textContent.includes("no longer registered"),
    `the chat says the pinned model is gone, got: ${notice && notice.textContent}`);
  assert.equal(chatPosts(calls).length, 0, "no chat request is made");
  assert.deepEqual(toasts, [], "no error toast");
});

test("a chat pinned to a registered model is sent as before", async () => {
  const { window, doc, calls, toasts } = chatSetup({ registry: ["live", "old"] });
  doc.getElementById("chat-input").value = "hello there";

  await window.sendChat();

  const [post] = chatPosts(calls);
  assert.equal(post.body.model, "old");
  assert.equal(post.body.pin_model, true);
  assert.equal(doc.querySelector("#chat-messages .tool-notice"), null);
  assert.deepEqual(toasts, []);
});

test("a rename made elsewhere is picked up from the server's copy of the chat", async () => {
  const { window, doc, calls, toasts } = chatSetup({
    registry: ["live", "renamed"], stored: { id: "c1", pinned_model: "renamed", messages: [] } });
  runScript(window, "chat.privacy = false; chat.modeKnown = true; chat.persist = true;");
  doc.getElementById("chat-input").value = "hello there";

  await window.sendChat();

  const [post] = chatPosts(calls);
  assert.equal(post.body.model, "renamed", "the chat follows the model to its new name");
  assert.equal(post.body.pin_model, true);
  assert.equal(doc.querySelector("#chat-messages .tool-notice"), null);
  assert.deepEqual(toasts, []);
});

test("a chat pinned to a name the model used to have follows it without the server's copy", async () => {
  const { window, doc, calls, toasts } = chatSetup({
    registry: ["live", { name: "renamed", model_id: "m1", former_names: ["old"] }] });
  doc.getElementById("chat-input").value = "hello there";

  await window.sendChat();

  const [post] = chatPosts(calls);
  assert.equal(post && post.body.model, "renamed", "the chat follows the model to its new name");
  assert.equal(post.body.pin_model, true);
  runScript(window, "window.__id = chat.conversations[0].pinnedModelId;");
  assert.equal(window.__id, "m1");
  assert.equal(doc.querySelector("#chat-messages .tool-notice"), null);
  assert.deepEqual(toasts, []);
});

test("a 404 for the pinned model from a stale model list ends in the same notice, not a toast", async () => {
  const { window, doc, toasts } = chatSetup({ registry: ["live", "old"], chat404: true });
  doc.getElementById("chat-input").value = "hello there";

  await window.sendChat();

  runScript(window, "window.__pin = chat.conversations[0].pinnedModel;");
  assert.equal(window.__pin, undefined);
  const notice = doc.querySelector("#chat-messages .tool-notice");
  assert.ok(notice && notice.textContent.includes("old"),
    `the chat says the pinned model is gone, got: ${notice && notice.textContent}`);
  assert.deepEqual(toasts, [], "no 404 toast");
});

test("the pin banner says the pinned model is gone", () => {
  const { window } = chatSetup();
  runScript(window, "syncPinModelToggle(chat.conversations[0]);");
  const note = window.document.getElementById("pin-mismatch");
  assert.equal(note.hidden, false);
  assert.ok(note.textContent.includes("old") && note.textContent.includes("no longer registered"),
    note.textContent);
});
