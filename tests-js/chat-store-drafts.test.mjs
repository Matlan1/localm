// SPDX-License-Identifier: AGPL-3.0-or-later
// Chat state that must not be lost or leak: two tabs saving one conversation, a
// delete made elsewhere, changes of the session mode, compaction during a send,
// the message box of each chat, refused pictures and think mentions.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { loadApp, runScript } from "./harness.mjs";
import { makeServer } from "./chat-store-server.mjs";
import { makeChannels, makeIndexedDB } from "./fake-indexeddb.mjs";

const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
const drain = async (n = 15) => { for (let i = 0; i < n; i++) await settle(0); };
async function waitFor(fn, timeout = 4000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await settle(15); }
  return false;
}
const plain = (x) => JSON.parse(JSON.stringify(x));
const user = (content) => ({ role: "user", content });
const reply = (content) => ({ role: "assistant", content });

/** A page talking to *server*. *cached* seeds the browser's conversation cache,
 *  *storage* more of it; *idb* and *channels* are the browser's IndexedDB and
 *  BroadcastChannel. */
async function boot(server, { cached = null, storage = {}, idb = null, channels = null, wait = true } = {}) {
  const { window: win } = loadApp({
    fetchImpl: server.fetch,
    seedLocalStorage: { "localm.instanceId": server.instance, ...storage,
      ...(cached ? { "localm.conversations": JSON.stringify(cached) } : {}) },
  });
  if (idb) win.indexedDB = idb.api;
  if (channels) win.BroadcastChannel = channels;
  runScript(win, `window.chat = chat; window.__toasts = [];
    const __toast = toast; toast = (m, e) => { window.__toasts.push(String(m)); __toast(m, e); };
    window.draftStore = draftStore; window.__page = PAGE;
    Object.defineProperty(window, "__restored", { get: () => _draftsRestored });`);
  if (!wait) return win;
  assert.ok(await waitFor(() => win.chat.modeKnown));
  await drain();
  if (idb) {
    assert.ok(await waitFor(() => win.__restored));
    await settle(700);
  }
  return win;
}

const conv = (win, title) => win.chat.conversations.find((c) => c.title === title);
const cache = (win) => JSON.parse(win.localStorage.getItem("localm.conversations") || "[]");
const texts = (c) => c.messages.map((m) => m.content);

async function open(win, title) {
  const row = [...win.document.querySelectorAll("#conv-list .conv-item")]
    .find((r) => r.textContent.includes(title));
  assert.ok(row, "listed: " + title);
  row.click();
  await drain();
  return conv(win, title);
}

function say(win, c, text) {
  c.messages.push(user(text));
  win.saveConversations(c);
}

// ---------------------------------------------------------------- two tabs ---

test("each save names the revision it replaces and takes the new one", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  const first = c.rev;
  say(win, c, "b");
  assert.ok(await waitFor(() => server.puts.length === 1 && !c.unsaved));
  assert.equal(server.puts[0].body.base_rev, first);
  assert.equal(c.rev, server.files.get("c1").rev);
  say(win, c, "c");
  assert.ok(await waitFor(() => server.puts.length === 2 && !c.unsaved));
  assert.notEqual(server.puts[1].body.base_rev, first);
  assert.deepEqual(texts(server.files.get("c1")), ["a", "b", "c"]);
});

test("two tabs that change one chat: both versions are kept and the loser is told", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  server.put("c1", { title: "One", messages: [user("a"), user("from the other tab")] });
  say(win, c, "from this tab");
  assert.ok(await waitFor(() => server.files.size === 2));
  assert.deepEqual(texts(server.files.get("c1")), ["a", "from the other tab"]);
  const copy = [...server.files.values()].find((f) => f.id !== "c1");
  assert.equal(copy.title, "One (copy)");
  assert.deepEqual(texts(copy), ["a", "from this tab"]);
  assert.equal(win.chat.activeId, c.id, "the tab stays on its own version");
  assert.equal(c.id, copy.id);
  assert.ok(win.chat.conversations.some((x) => x.id === "c1"), "the stored one is listed beside it");
  assert.ok(win.__toasts.some((m) => /changed in another tab/.test(m)), win.__toasts.join("|"));
  assert.ok(!server.gui.some((b) => b.includes("One")), "no chat title reaches the server's log");
});

test("a queued message and a draft follow a conversation that became a copy", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  win.document.getElementById("chat-input").value = "half typed";
  win.chat.queue.push({ text: "q", attachments: [], docs: [], clips: [], convId: "c1" });
  server.put("c1", { title: "One", messages: [user("a"), user("other")] });
  say(win, c, "mine");
  assert.ok(await waitFor(() => c.id !== "c1"));
  assert.equal(win.chat.queue[0].convId, c.id);
  await drain();
  win.renderChat();
  assert.equal(win.document.getElementById("chat-input").value, "half typed");
});

test("a rename and a change of the messages elsewhere are merged, no copy is made", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  server.put("c1", { title: "One", messages: [user("a"), reply("b")] });
  c.title = "Renamed here";
  win.saveConversations(c);
  assert.ok(await waitFor(() => server.files.get("c1").title === "Renamed here"));
  assert.deepEqual(texts(server.files.get("c1")), ["a", "b"]);
  assert.deepEqual(texts(c), ["a", "b"]);
  assert.equal(server.files.size, 1);
});

test("messages added here and a folder set elsewhere are merged, no copy is made", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  server.put("c1", { title: "One", folder: "work", messages: [user("a")] });
  say(win, c, "b");
  assert.ok(await waitFor(() => server.files.get("c1").messages.length === 2));
  assert.equal(server.files.get("c1").folder, "work");
  assert.equal(c.folder, "work");
  assert.equal(server.files.size, 1);
});

test("a tab that comes back to the front shows what another tab added", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  server.put("c1", { title: "One", messages: [user("a"), reply("added elsewhere")] });
  await win.syncConversations();
  assert.deepEqual(texts(c), ["a", "added elsewhere"]);
  assert.match(win.document.getElementById("chat-messages").textContent, /added elsewhere/);
  assert.equal(server.puts.length, 0);
});

test("an unsaved turn found at a reload is not dropped for a newer stored copy", async () => {
  const server = makeServer();
  const rev = server.put("c1", { title: "One", messages: [user("a")] });
  server.put("c1", { title: "One", messages: [user("a"), user("other")] });
  const win = await boot(server, { cached: [{ id: "c1", title: "One", updated_at: 5, rev, unsaved: true,
    messages: [user("a"), user("never reached the server")] }] });
  assert.ok(await waitFor(() => server.files.size === 2));
  assert.ok(conv(win, "One (copy)"));
  const copy = [...server.files.values()].find((f) => f.id !== "c1");
  assert.deepEqual(texts(copy), ["a", "never reached the server"]);
});

test("a tab does not drop another tab's unsaved turn from the shared browser cache", async () => {
  const server = makeServer();
  const rev = server.put("c1", { title: "One", messages: [user("a")] });
  server.put("c2", { title: "Two", updated_at: 1, messages: [user("b")] });
  const win = await boot(server);
  await open(win, "One");
  await drain();
  const dirty = { id: "c1", title: "One", rev, unsaved: true, updated_at: Date.now(),
                  messages: [user("a"), user("typed in the other tab")] };
  win.localStorage.setItem("localm.conversations", JSON.stringify([dirty]));
  win.saveConversations();
  await drain();
  assert.match(JSON.stringify(cache(win)), /typed in the other tab/);
});

test("two open pages do not keep rewriting the shared cache at each other", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  await open(win, "One");
  await drain();
  win.localStorage.setItem("localm.conversations", "[]");
  await win.syncConversations();
  await drain();
  assert.equal(win.localStorage.getItem("localm.conversations"), "[]",
    "a sync that finds nothing changed writes nothing");
});

test("text typed for a chat deleted elsewhere stays in the message box", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  server.put("c2", { title: "Two", messages: [user("b")] });
  const win = await boot(server);
  await open(win, "One");
  win.document.getElementById("chat-input").value = "half typed";
  server.remove("c1");
  await win.syncConversations();
  assert.equal(win.document.getElementById("chat-input").value, "half typed");
});

test("pinning a chat that was never opened keeps its messages", async () => {
  const server = makeServer();
  server.put("c0", { title: "Open one", updated_at: Date.now() + 5, messages: [user("x")] });
  server.put("c1", { title: "Closed one", updated_at: 1, messages: [user("a"), reply("b")] });
  const win = await boot(server);
  assert.ok(conv(win, "Closed one")._meta);
  const row = [...win.document.querySelectorAll("#conv-list .conv-item")]
    .find((r) => r.textContent.includes("Closed one"));
  row.querySelector("button.del").click();
  assert.ok(await waitFor(() => server.files.get("c1").pinned));
  assert.deepEqual(texts(server.files.get("c1")), ["a", "b"]);
});

// --------------------------------------------------------- deleted elsewhere ---

test("a chat deleted elsewhere does not come back from a browser that cached it", async () => {
  const server = makeServer();
  server.put("c0", { title: "Other", messages: [user("x")] });
  server.remove("c1");
  const win = await boot(server, { cached: [
    { id: "c1", title: "Gone", updated_at: 5, rev: "r9", messages: [user("a")] }] });
  await settle(800);
  assert.equal(server.puts.length, 0);
  assert.equal(conv(win, "Gone"), undefined);
  assert.ok(!cache(win).some((c) => c.id === "c1"));
});

test("a chat deleted elsewhere does not come back from a cache written before revisions", async () => {
  const server = makeServer();
  server.remove("c1");
  const win = await boot(server, { cached: [
    { id: "c1", title: "Gone", updated_at: 5, messages: [user("a")] }] });
  assert.ok(await waitFor(() => !conv(win, "Gone")));
  assert.equal(server.files.size, 0);
});

test("a chat never stored is uploaded, and one past the listed page is kept", async () => {
  const server = makeServer();
  for (let i = 0; i < 5; i++) server.put("s" + i, { title: "S" + i, messages: [user("x")] });
  server.pageSize = 2;
  const win = await boot(server, { cached: [
    { id: "new1", title: "Never stored", updated_at: 5, messages: [user("a")] },
    { id: "s4", title: "S4", updated_at: 1, rev: server.files.get("s4").rev, messages: [user("x")] }] });
  assert.ok(await waitFor(() => server.files.has("new1")));
  assert.ok(conv(win, "S4"));
  assert.deepEqual(server.puts.map((p) => p.id), ["new1"]);
});

test("a turn added to a chat deleted elsewhere stays in this tab and is not stored", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  server.remove("c1");
  say(win, c, "late words");
  assert.ok(await waitFor(() => c.tabOnly === true));
  await drain();
  assert.deepEqual(texts(c), ["a", "late words"]);
  assert.equal(server.files.size, 0);
  assert.ok(!JSON.stringify(cache(win)).includes("late words"));
  assert.ok(win.__toasts.some((m) => /deleted in another tab/.test(m)));
  say(win, c, "still later");
  await settle(800);
  assert.equal(server.files.size, 0);
  assert.equal(server.puts.length, 1);
});

test("a chat stored while the index was being read is not taken for deleted", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  const stored = server.files.get("c1");
  const real = win.fetch;
  win.fetch = async (url, opts) => {
    if (!String(url).startsWith("/api/conversations?")) return real(url, opts);
    server.files.delete("c1");
    const answer = await real(url, opts);
    server.files.set("c1", stored);
    return answer;
  };
  await win.syncConversations();
  assert.ok(win.chat.conversations.includes(c));
  assert.deepEqual(texts(c), ["a"]);
});

test("an open page drops a chat deleted elsewhere when it comes back to the front", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  server.put("c2", { title: "Two", messages: [user("b")] });
  const win = await boot(server);
  await open(win, "One");
  server.remove("c1");
  await win.syncConversations();
  assert.equal(conv(win, "One"), undefined);
  assert.equal(win.document.querySelector("#conv-list .conv-item.active") !== null, true);
});

// ------------------------------------------------------------- session mode ---

test("a mode changed from a terminal takes effect before anything is written", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  await drain();
  server.mode = "privacy";
  say(win, c, "typed after the switch");
  runScript(win, `lsSetScoped("localm.coderCwd", "a/private/path");`);
  await drain();
  assert.equal(win.chat.privacy, true);
  assert.equal(win.localStorage.getItem("localm.conversations"), null);
  assert.equal(win.localStorage.getItem("localm.coderCwd"), null);
  assert.equal(win.document.getElementById("persist-hint"), null);
  await settle(800);
  assert.equal(server.puts.length, 0, "nothing is sent to the store");
  assert.deepEqual(texts(c), ["a", "typed after the switch"]);
  assert.ok(win.chat.conversations.includes(c), "what was just typed stays on screen");
  assert.equal(c.tabOnly, true);
});

test("a write queued while a mode read is already under way waits for the next one", async () => {
  const server = makeServer();
  const win = await boot(server);
  let release;
  const gate = new Promise((r) => { release = r; });
  const real = win.fetch;
  let first = true;
  win.fetch = async (url, opts) => {
    if (String(url) === "/v1/config" && first) {
      first = false;
      const answer = await real(url, opts);
      await gate;
      return answer;
    }
    return real(url, opts);
  };
  const poll = win.refreshCtxLimit();
  await drain(3);
  server.mode = "privacy";
  runScript(win, `lsSetScoped("localm.webAccess", "1");`);
  release();
  await poll;
  await drain();
  assert.equal(win.localStorage.getItem("localm.webAccess"), null);
  assert.equal(win.chat.privacy, true);
});

test("entering privacy mode says how many saved chats stay on disk", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  server.put("c2", { title: "Two", messages: [user("b")] });
  const win = await boot(server);
  server.mode = "privacy";
  await win.refreshCtxLimit();
  await drain();
  assert.equal(win.document.getElementById("privacy-leftovers").textContent,
    "2 chats saved earlier are still on disk");
  assert.equal(win.chat.conversations.length, 0);
  server.mode = "full";
  await win.refreshCtxLimit();
  await drain();
  assert.equal(win.document.getElementById("privacy-leftovers"), null);
});

test("with nothing on disk privacy mode shows no such note", async () => {
  const server = makeServer();
  server.mode = "privacy";
  const win = await boot(server);
  assert.ok(win.document.getElementById("privacy-hint"));
  assert.equal(win.document.getElementById("privacy-leftovers"), null);
});

test("a chat held in this tab only asks the server to handle it in privacy mode", async () => {
  const server = makeServer();
  const win = await boot(server);
  const headers = (c) => plain(win.chatSurfaceHeaders(c));
  assert.deepEqual(headers({ id: "x" }), { "X-Client-Id": "chat" });
  assert.deepEqual(headers({ id: "x", tabOnly: true }),
    { "X-Client-Id": "chat", "X-Localm-Session-Mode": "privacy" });
  win.chat.privacy = true;
  assert.equal(headers({ id: "x" })["X-Localm-Session-Mode"], "privacy");
});

test("a refused save is never shown as saved, and is shown again once one lands", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  assert.ok(win.document.getElementById("persist-hint"));
  server.failPut = 500;
  say(win, c, "b");
  assert.ok(await waitFor(() => win.document.getElementById("conv-browser-only")));
  assert.equal(win.document.getElementById("persist-hint"), null);
  server.failPut = 0;
  say(win, c, "c");
  assert.ok(await waitFor(() => win.document.getElementById("persist-hint")));
  assert.equal(win.document.getElementById("conv-browser-only"), null);
});

test("a chat too large for the server says the browser keeps it only when it does", async () => {
  for (const [cached, expected] of [[true, /kept only in this browser/], [false, /only while this tab stays open/]]) {
    const server = makeServer();
    server.put("c1", { title: "One", messages: [user("a")] });
    const win = await boot(server);
    const c = await open(win, "One");
    if (!cached) runScript(win, `
      const older = []; for (let i = 0; i < 60; i++) older.push(
        { id: "n" + i, title: "n", updated_at: Date.now() + 1e9 + i, messages: [{ role: "user", content: "x" }] });
      chat.conversations = [...older, ...chat.conversations];`);
    server.failPut = 413;
    say(win, c, "b");
    assert.ok(await waitFor(() => win.__toasts.some((m) => /too large/.test(m))));
    assert.match(win.__toasts.find((m) => /too large/.test(m)), expected);
  }
});

// ------------------------------------------------------------- drawer values ---

test("drawer values are saved with the chat and shown in another browser", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const a = await boot(server);
  await open(a, "One");
  const field = a.document.getElementById("p-system");
  field.value = "You are terse.";
  field.dispatchEvent(new a.Event("input", { bubbles: true }));
  assert.ok(await waitFor(() => server.files.get("c1").params.system === "You are terse."));
  const b = await boot(server);
  await open(b, "One");
  assert.equal(b.document.getElementById("p-system").value, "You are terse.");
});

// ---------------------------------------------------------------- compaction ---

const gatedSummary = () => {
  let release;
  const gate = new Promise((r) => { release = r; });
  return { release: () => release(), reply: async () => {
    await gate;
    return { ok: true, status: 200, headers: { get: () => null }, text: async () => "",
      json: async () => ({ choices: [{ message: { content: "summary" }, finish_reason: "stop" }] }) };
  } };
};

const longChat = () => Array.from({ length: 12 }, (_, i) =>
  (i % 2 ? reply : user)("message " + i + " " + "x".repeat(60)));

test("a message that arrives while a compaction writes its summary is kept", async () => {
  const server = makeServer();
  const sum = gatedSummary();
  server.chatReply = sum.reply;
  const win = await boot(server);
  win.chat.ctxMax = 160;
  const c = { id: "c1", title: "One", messages: longChat() };
  const done = win.compactConversation(c);
  await drain(3);
  c.messages.push(user("asked meanwhile"));
  sum.release();
  assert.equal(await done, true);
  assert.equal(c.messages.at(-1).content, "asked meanwhile");
  assert.match(c.messages[0].content, /summary/);
});

test("a compaction whose older turns changed meanwhile leaves the chat as it is", async () => {
  const server = makeServer();
  const sum = gatedSummary();
  server.chatReply = sum.reply;
  const win = await boot(server);
  win.chat.ctxMax = 160;
  const c = { id: "c1", title: "One", messages: longChat() };
  const done = win.compactConversation(c);
  await drain(3);
  c.messages = [user("edited"), reply("b")];
  sum.release();
  assert.equal(await done, false);
  assert.deepEqual(texts(c), ["edited", "b"]);
});

test("the compact button works as a turn: a message sent meanwhile is queued", async () => {
  const server = makeServer();
  const sum = gatedSummary();
  server.put("c1", { title: "One", messages: longChat() });
  const win = await boot(server);
  const c = await open(win, "One");
  win.chat.ctxMax = 160;
  runScript(win, "modelCache.active = 'm'; runCompletion = async () => ({ stopped: false });");
  server.chatReply = sum.reply;
  const done = win.document.getElementById("compact-conv").onclick();
  await drain(3);
  assert.equal(win.chatBusy(), true);
  win.document.getElementById("chat-input").value = "asked meanwhile";
  await win.sendChat();
  assert.equal(win.chat.queue.length, 1);
  sum.release();
  await done;
  await drain();
  assert.equal(win.chat.queue.length, 0);
  assert.ok(texts(c).includes("asked meanwhile"));
  assert.match(c.messages[0].content, /summary/);
});

// --------------------------------------------------------------- message box ---

async function twoChats(options = {}) {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("first one"), reply("r1")] });
  server.put("c2", { title: "Two", messages: [user("first two"), reply("r2")] });
  const win = await boot(server, options);
  return { server, win, input: win.document.getElementById("chat-input") };
}

test("a draft and its attachments stay with their chat", async () => {
  const { win, input } = await twoChats();
  await open(win, "One");
  input.value = "for one";
  win.chat.attachments.push({ name: "p.png", dataUri: "data:image/png;base64,AA" });
  win.chat.docs.push({ name: "d.txt", text: "t", chars: 1, truncated: false });
  await open(win, "Two");
  assert.equal(input.value, "");
  assert.equal(win.chat.attachments.length + win.chat.docs.length, 0);
  assert.equal(win.document.querySelectorAll("#attach-chips .chip").length, 0);
  input.value = "for two";
  await open(win, "One");
  assert.equal(input.value, "for one");
  assert.equal(win.chat.attachments[0].name, "p.png");
  assert.equal(win.chat.docs[0].name, "d.txt");
  await open(win, "Two");
  assert.equal(input.value, "for two");
});

test("a picture still being read lands in the chat it was added to", async () => {
  const { win } = await twoChats();
  await open(win, "One");
  win.addAttachedFiles([new win.File(["x"], "late.png", { type: "image/png" })]);
  await open(win, "Two");
  await settle(60);
  assert.equal(win.chat.attachments.length, 0);
  await open(win, "One");
  assert.equal(win.chat.attachments[0].name, "late.png");
});

const actionButton = (win, label) => [...win.document.querySelectorAll("#chat-messages .msg-meta button")]
  .find((b) => b.textContent === label);

for (const label of ["edit", "revert"]) {
  test(`${label} asks before it replaces a draft, and keeps it on no`, async () => {
    const { win, input } = await twoChats();
    const c = await open(win, "One");
    input.value = "not sent yet";
    actionButton(win, label).click();
    assert.match(win.document.getElementById("modal-body").textContent, /not sent/);
    assert.equal(input.value, "not sent yet");
    assert.deepEqual(texts(c), ["first one", "r1"]);
    win.document.querySelector("#modal-body button.btn-danger").click();
    await drain();
    assert.equal(input.value, "first one");
    assert.deepEqual(texts(c), []);
  });
}

test("edit brings back the message's pictures, and the old version stays reachable", async () => {
  const server = makeServer();
  const picture = { type: "image_url", image_url: { url: "data:image/png;base64,AA" } };
  server.put("c1", { title: "One", messages: [
    { role: "user", content: [{ type: "text", text: "look" }, picture] }, reply("r1")] });
  const win = await boot(server);
  const c = await open(win, "One");
  actionButton(win, "edit").click();
  await drain();
  assert.equal(win.document.getElementById("chat-input").value, "look");
  assert.equal(win.chat.attachments[0].dataUri, "data:image/png;base64,AA");
  const row = win.document.querySelector(".branch-pending");
  assert.ok(row, "the pending edit shows the way back");
  row.querySelector(".variant button").click();
  assert.equal(c.messages.length, 2);
  assert.match(win.document.getElementById("chat-messages").textContent, /r1/);
});

// What a browser keeps: its IndexedDB, and the pages that hear each other.
const browser = () => ({ idb: makeIndexedDB(), channels: makeChannels() });
const rowsOf = (b) => [...b.idb.rows().values()];
const owned = (extra) => ({ inst: "backend-x", prin: "owner", at: Date.now(), ...extra });

/** Puts *rows* into the browser's store before a page opens. */
function seedRows(b, rows) {
  b.idb.stores.set("rows", new Map(Object.entries(rows)));
  const open = b.idb.api.open;
  let first = true;
  b.idb.api.open = (name, version) => {
    const req = open(name, version);
    if (first) {
      first = false;
      queueMicrotask(() => {
        const set = req.onupgradeneeded;
        req.onupgradeneeded = (e) => { set(e); b.idb.stores.set("rows", new Map(Object.entries(rows))); };
      });
    }
    return req;
  };
}

async function typeDraft(win, text) {
  const input = win.document.getElementById("chat-input");
  input.value = text;
  input.dispatchEvent(new win.Event("input"));
}

/** The page goes away: a reload, or the tab is closed. */
async function leave(win) {
  win.dispatchEvent(new win.Event("pagehide"));
  win.document.getElementById("chat-input").value = "";
  if (win.BroadcastChannel) runScript(win, "if (_pages) _pages.close();");
  await settle(60);
}

test("a draft, its attachments and the queue survive a reload of the tab", async () => {
  const b = browser();
  const { win, server } = await twoChats(b);
  await open(win, "One");
  await typeDraft(win, "unsent words");
  win.chat.attachments.push({ name: "p.png", dataUri: "data:image/png;base64,AA" });
  win.renderAttachChips();
  win.chat.queue.push({ text: "queued words", attachments: [], docs: [], clips: [], convId: "c2" });
  assert.ok(await waitFor(() => rowsOf(b).length === 3), JSON.stringify(rowsOf(b)));
  await leave(win);
  assert.ok(rowsOf(b).every((r) => r.owner === null), "the page gave its rows up");

  const again = await boot(server, b);
  await open(again, "One");
  assert.equal(again.document.getElementById("chat-input").value, "unsent words");
  assert.equal(again.chat.attachments[0].name, "p.png");
  assert.equal(again.chat.queue.map((q) => q.text).join(), "queued words");
  assert.equal(server.puts.length, 0, "a restored queue is not sent on its own");
  assert.ok(await waitFor(() => rowsOf(b).every((r) => r.owner === again.__page)));

  await typeDraft(again, "");
  again.chat.attachments = [];
  again.chat.queue = [];
  again.renderAttachChips();
  assert.ok(await waitFor(() => rowsOf(b).length === 0), "what is gone from the box is gone from the browser");
});

test("a draft typed before the stored one is read is kept beside it", async () => {
  const b = browser();
  seedRows(b, { k1: owned({ conv: "c1", kind: "text", text: "from before", owner: null }) });
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server, { ...b, wait: false });
  win.document.getElementById("chat-input").value = "typed early";
  assert.ok(await waitFor(() => win.__restored));
  await settle(700);
  assert.equal(win.document.getElementById("chat-input").value, "from before\ntyped early");
});

test("the drafts two closed tabs left for one chat are both put back, and neither again later", async () => {
  const b = browser();
  seedRows(b, {
    k1: owned({ conv: "c1", kind: "text", text: "older", owner: null, at: Date.now() - 5000 }),
    k2: owned({ conv: "c1", kind: "text", text: "newer", owner: null }) });
  const { win, server } = await twoChats(b);
  await open(win, "One");
  assert.equal(win.document.getElementById("chat-input").value, "older\nnewer");
  assert.ok(await waitFor(() => rowsOf(b).length === 1));
  assert.equal(rowsOf(b)[0].text, "older\nnewer");
  await typeDraft(win, "");
  assert.ok(await waitFor(() => rowsOf(b).length === 0));
  await leave(win);
  const again = await boot(server, b);
  await open(again, "One");
  assert.equal(again.document.getElementById("chat-input").value, "");
});

test("a row another open page owns is neither taken nor changed", async () => {
  const b = browser();
  const first = await twoChats(b);
  await open(first.win, "One");
  await typeDraft(first.win, "mine");
  first.win.chat.queue.push({ text: "queued once", attachments: [], docs: [], clips: [], convId: "c1" });
  assert.ok(await waitFor(() => rowsOf(b).length === 2));
  const second = await boot(first.server, b);
  await open(second, "One");
  assert.equal(second.document.getElementById("chat-input").value, "");
  assert.equal(second.chat.queue.length, 0);
  await typeDraft(second, "the second tab's");
  assert.ok(await waitFor(() => rowsOf(b).length === 3));
  await typeDraft(second, "");
  assert.ok(await waitFor(() => rowsOf(b).length === 2));
  assert.deepEqual(rowsOf(b).map((r) => r.text || r.item.text).sort(), ["mine", "queued once"]);
});

test("without a way to ask the other pages, only given up or long untouched rows are taken", async () => {
  const b = { idb: makeIndexedDB() };
  seedRows(b, {
    fresh: owned({ conv: "c1", kind: "text", text: "another tab's", owner: "p1" }),
    old: owned({ conv: "c2", kind: "text", text: "abandoned", owner: "p2", at: Date.now() - 3600e3 }),
    queue: owned({ conv: "c1", kind: "queue", state: "queued", owner: "p1",
      item: { id: "q1", convId: "c1", text: "queued", attachments: [], docs: [], clips: [] } }) });
  const { win } = await twoChats(b);
  await open(win, "One");
  assert.equal(win.document.getElementById("chat-input").value, "");
  assert.equal(win.chat.queue.length, 0);
  await open(win, "Two");
  assert.equal(win.document.getElementById("chat-input").value, "abandoned");
});

test("two pages that both show a queued message send it once", async () => {
  const b = browser();
  const { win, server } = await twoChats(b);
  await open(win, "One");
  win.chat.queue.push({ text: "queued once", attachments: [], docs: [], clips: [], convId: "c1" });
  win.renderAttachChips();
  assert.ok(await waitFor(() => rowsOf(b).length === 1));
  // The first tab is frozen: it answers nobody, and its row is taken.
  runScript(win, "_pages.close();");
  const second = await boot(server, b);
  assert.equal(second.chat.queue.length, 1);
  const sent = [];
  for (const w of [win, second]) {
    w.__sent = sent;
    runScript(w, `modelCache.active = 'm'; dispatchChatTurn = async (c, text) => { window.__sent.push(text); };`);
    await open(w, "One");
  }
  await win.processChatQueue();
  await second.processChatQueue();
  await drain();
  assert.deepEqual(sent, ["queued once"]);
  assert.equal(win.chat.queue.length + second.chat.queue.length, 0);
  assert.ok(await waitFor(() => rowsOf(b).length === 0));
});

test("a message another page was sending when it went away waits and asks", async () => {
  const b = browser();
  seedRows(b, { k: owned({ conv: "c1", kind: "queue", state: "sending", owner: "dead",
    item: { id: "q1", convId: "c1", text: "maybe sent", attachments: [], docs: [], clips: [] } }) });
  const { win } = await twoChats(b);
  const sent = [];
  win.__sent = sent;
  runScript(win, `modelCache.active = 'm'; dispatchChatTurn = async (c, text) => { window.__sent.push(text); };`);
  await open(win, "One");
  await win.processChatQueue();
  assert.deepEqual(sent, []);
  const row = win.document.querySelector(".msg-row.queued");
  assert.match(row.textContent, /may have been sent already/);
  row.querySelector("button[title='Send']").click();
  await drain();
  assert.deepEqual(sent, ["maybe sent"]);
});

test("a queued message whose claim fails waits and asks, and is not sent", async () => {
  const b = browser();
  const { win } = await twoChats(b);
  await open(win, "One");
  win.chat.queue.push({ text: "q", attachments: [], docs: [], clips: [], convId: "c1" });
  win.renderAttachChips();
  assert.ok(await waitFor(() => rowsOf(b).length === 1));
  const sent = [];
  win.__sent = sent;
  runScript(win, `modelCache.active = 'm'; dispatchChatTurn = async (c, text) => { window.__sent.push(text); };`);
  b.idb.failWrites = 1;
  await win.processChatQueue();
  assert.deepEqual(sent, []);
  assert.equal(win.chat.queue[0].held, true);
});

test("a claim that is never answered leaves the message in the queue, and the next one is sent", async () => {
  const b = browser();
  const { win } = await twoChats(b);
  await open(win, "One");
  win.chat.queue.push({ text: "q", attachments: [], docs: [], clips: [], convId: "c1" });
  win.renderAttachChips();
  assert.ok(await waitFor(() => rowsOf(b).length === 1));
  const sent = [];
  win.__sent = sent;
  runScript(win, `modelCache.active = 'm'; dispatchChatTurn = async (c, text) => { window.__sent.push(text); };`);
  b.idb.hang = "use";
  const done = win.processChatQueue();
  await settle(300);
  assert.equal(win.chat.queue.length, 1, "in the queue while it is being claimed");
  await done;
  assert.deepEqual(sent, []);
  assert.equal(win.chat.queue[0].held, true);
  win.document.getElementById("chat-input").value = "next";
  await win.sendChat();
  assert.deepEqual(sent, ["next"]);
});

test("a browser database that never answers does not keep the app from starting or sending", async () => {
  const b = browser();
  b.idb.hang = "open";
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server, { ...b, wait: false });
  assert.ok(await waitFor(() => win.document.getElementById("startup-overlay")?.style.display === "none", 8000));
  const sent = [];
  win.__sent = sent;
  runScript(win, `modelCache.active = 'm'; dispatchChatTurn = async (c, text) => { window.__sent.push(text); };`);
  await open(win, "One");
  win.document.getElementById("chat-input").value = "hello";
  await win.sendChat();
  assert.deepEqual(sent, ["hello"]);
  const out = await win.signOutSession();
  assert.equal(out.ok, true);
});

test("a privacy wipe the browser never answers is said and tried again", async () => {
  const b = browser();
  const { win, server } = await twoChats(b);
  await open(win, "One");
  await typeDraft(win, "private words");
  assert.ok(await waitFor(() => rowsOf(b).length === 1));
  b.idb.hang = "use";
  server.mode = "privacy";
  await win.refreshCtxLimit();
  assert.ok(await waitFor(() => win.document.getElementById("privacy-wipe-failed"), 8000));
  b.idb.hang = "";
  assert.ok(await waitFor(() => rowsOf(b).length === 0, 12000));
  assert.ok(await waitFor(() => !win.document.getElementById("privacy-wipe-failed")));
});

test("a page that loses its connection to the browser's storage opens it again", async () => {
  const b = browser();
  const { win } = await twoChats(b);
  await open(win, "One");
  await typeDraft(win, "first");
  assert.ok(await waitFor(() => rowsOf(b).length === 1));
  for (const c of b.idb.connections) c.closed = true;
  await typeDraft(win, "second");
  assert.ok(await waitFor(() => rowsOf(b)[0].text === "second"));
});

test("stored drafts and queued messages of a chat that is deleted are removed", async () => {
  const b = browser();
  seedRows(b, {
    k1: owned({ conv: "c1", kind: "text", text: "x", owner: null }),
    k2: owned({ conv: "gone", kind: "text", text: "y", owner: null }),
    k3: owned({ conv: "c2", kind: "queue", state: "queued", owner: "another", item: { id: "q", convId: "c2", text: "z" } }) });
  const { win } = await twoChats(b);
  assert.ok(await waitFor(() => !rowsOf(b).some((r) => r.conv === "gone")));
  win.retainConversations((c) => c.id !== "c2");
  assert.ok(await waitFor(() => !rowsOf(b).some((r) => r.conv === "c2")));
  assert.ok(rowsOf(b).some((r) => r.conv === "c1"));
});

test("privacy mode keeps no draft in this browser and empties what was there", async () => {
  const b = browser();
  const { win, input, server } = await twoChats(b);
  await open(win, "One");
  await typeDraft(win, "unsent words");
  assert.ok(await waitFor(() => rowsOf(b).length === 1));
  server.mode = "privacy";
  await typeDraft(win, "unsent private words");
  assert.ok(await waitFor(() => win.chat.privacy));
  assert.ok(await waitFor(() => rowsOf(b).length === 0));
  assert.equal(input.value, "unsent private words", "what is typed stays in the box");
  server.mode = "full";
  await win.refreshCtxLimit();
  await typeDraft(win, "unsent private words, more");
  await settle(500);
  assert.equal(rowsOf(b).length, 0, "what was typed under privacy mode is not written after it either");
});

test("a privacy wipe that fails is said and tried again until it succeeds", async () => {
  const b = browser();
  const { win, server } = await twoChats(b);
  await open(win, "One");
  await typeDraft(win, "private words");
  assert.ok(await waitFor(() => rowsOf(b).length === 1));
  b.idb.failWrites = 2;
  server.mode = "privacy";
  await win.refreshCtxLimit();
  assert.ok(await waitFor(() => win.document.getElementById("privacy-wipe-failed")));
  assert.ok(win.__toasts.some((m) => /could not remove/.test(m)));
  assert.equal(rowsOf(b).length, 1);
  assert.ok(await waitFor(() => rowsOf(b).length === 0, 9000));
  assert.ok(await waitFor(() => !win.document.getElementById("privacy-wipe-failed")));
});

test("privacy mode of one install leaves another install's rows alone", async () => {
  const b = browser();
  seedRows(b, { k1: { ...owned({ conv: "z", kind: "text", text: "theirs", owner: null }), inst: "another-install" },
                k2: owned({ conv: "c1", kind: "text", text: "ours", owner: null }) });
  const server = makeServer();
  server.mode = "privacy";
  await boot(server, { ...b, wait: false });
  assert.ok(await waitFor(() => b.idb.writes > 0 && rowsOf(b).length === 1));
  assert.equal(rowsOf(b)[0].text, "theirs");
});

test("another key or install does not see, take or remove a draft or a queued message", async () => {
  const b = browser();
  const item = { id: "q1", convId: "c1", text: "A's queued", attachments: [], docs: [], clips: [] };
  seedRows(b, {
    a: { ...owned({ conv: "c1", kind: "text", text: "A's words", owner: null }), prin: "key-a" },
    b: { ...owned({ conv: "", kind: "text", text: "typed with no chat", owner: null }), prin: "key-a" },
    c: { ...owned({ conv: "c1", kind: "queue", state: "queued", owner: null, item }), prin: "key-a" },
    d: { ...owned({ conv: "", kind: "text", text: "at another install", owner: null }), inst: "elsewhere" } });
  const server = makeServer({ principal: "key-b" });
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server, b);
  await open(win, "One");
  assert.equal(win.document.getElementById("chat-input").value, "");
  assert.equal(win.chat.queue.length, 0);
  win.retainConversations(() => false);
  await settle(300);
  assert.equal(rowsOf(b).length, 4);
});

test("a draft of a sent slash command is not kept", async () => {
  const b = browser();
  const { win } = await twoChats(b);
  await open(win, "One");
  await typeDraft(win, "/pin");
  assert.ok(await waitFor(() => rowsOf(b).length === 1));
  await win.sendChat();
  assert.ok(await waitFor(() => rowsOf(b).length === 0));
});

test("a restored queue waits while there is no model to answer", async () => {
  const { win } = await twoChats();
  const c = await open(win, "One");
  win.chat.queue.push({ text: "q", attachments: [], docs: [], clips: [], convId: c.id });
  runScript(win, "modelCache.active = ''; modelCache.resumable = '';");
  await win.processChatQueue();
  assert.equal(win.chat.queue.length, 1);
});

test("sending takes the box of the chat it was sent from", async () => {
  const { win, input } = await twoChats();
  const c = await open(win, "One");
  input.value = " words ";
  win.chat.docs.push({ name: "d", text: "t", chars: 1 });
  const item = plain(win.takeComposer(c));
  assert.equal(item.text, "words");
  assert.equal(item.docs.length, 1);
  assert.equal(input.value, "");
  assert.equal(win.chat.docs.length, 0);
});

// ---------------------------------------------------------- a server that died ---

test("a reply cut by a dead server keeps its text, and the queue is not sent", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  runScript(win, `modelCache.active = 'm'; maybeCompactConversation = async () => null;
    readSSE = async (r, on) => { on(JSON.stringify({ choices: [{ delta: { content: "half a re" } }] }));
      throw new TypeError("network error"); };`);
  win.chat.queue.push({ text: "next", attachments: [], docs: [], clips: [], convId: "c1" });
  await win.runCompletion(c);
  await drain();
  assert.equal(win.chat.queue.length, 1, "the queued message stays queued");
  const last = c.messages.at(-1);
  assert.equal(last.content, "half a re");
  assert.equal(last.cut, true);
});

// ------------------------------------------------------------ refused pictures ---

async function refusedTurn({ vision, routing }) {
  const server = makeServer();
  const win = await boot(server);
  const pic = (n) => ({ type: "image_url", image_url: { url: "data:image/png;base64," + n } });
  runScript(win, `modelCache.active = 'm'; modelCache.models = [{ name: "m", vision: ${vision} }];
    maybeCompactConversation = async () => null;`);
  win.chat.conversations = [{ id: "c1", title: "One", messages: [
    { role: "user", content: [{ type: "text", text: "good" }, pic("AAAA")] }, reply("fine"),
    { role: "user", content: [{ type: "text", text: "bad" }, pic("BBBB")] }] }];
  win.chat.activeId = "c1";
  const c = win.chat.conversations[0];
  server.chatReply = async () => ({ ok: false, status: 400, statusText: "Bad",
    headers: { get: (k) => (k === "X-Localm-Refusal" ? "image"
      : k === "X-Localm-Model-Routing" ? routing : null) },
    text: async () => "", json: async () => ({ detail: "The attached image could not be read" }) });
  await win.runCompletion(c);
  return { win, first: c.messages[0].content[1], last: c.messages[2].content[1] };
}

test("an unreadable picture does not mark the earlier ones refused", async () => {
  const { win, first, last } = await refusedTurn({ vision: true, routing: null });
  assert.equal(first.refusedBy, undefined);
  assert.deepEqual(plain(last.refusedBy), ["m"]);
  assert.ok(win.__toasts.some((m) => /could not be read/.test(m)), win.__toasts.join("|"));
  assert.ok(!win.__toasts.some((m) => /cannot read images/.test(m)));
});

test("a model that cannot read images still marks every picture", async () => {
  const { first, last } = await refusedTurn({ vision: false, routing: null });
  assert.deepEqual([plain(first.refusedBy), plain(last.refusedBy)], [["m"], ["m"]]);
});

// ------------------------------------------------------------ shared files etc ---

test("files a share left out are named", async () => {
  const server = makeServer();
  const win = await boot(server);
  const real = win.fetch;
  win.fetch = async (url, opts) => (String(url) === "/api/share/pending"
    ? { ok: true, status: 200, json: async () => ({ items: [], skipped: ["a.docx", "b.zip"] }) }
    : real(url, opts));
  await win.ingestSharedFiles();
  assert.ok(win.__toasts.some((m) => m.includes("a.docx, b.zip")), win.__toasts.join("|"));
});

test("a user message keeps a think mention in the digest and in what is summarised", async () => {
  const { window: win } = loadApp();
  const m = user("Why is <think> shown? The rest of the question.");
  assert.match(win.compactionDigest([m]), /The rest of the question/);
  assert.match(win.compactionText(m), /The rest of the question/);
  assert.doesNotMatch(win.compactionText(reply("<think>plan</think>answer")), /plan/);
});

test("a reply that mentions the tag keeps its text; real reasoning is still left out", () => {
  const { window: win } = loadApp();
  for (const text of ["Models write <think> before an answer, then reply.",
    "The <think> and </think> tags wrap reasoning. Here is more.",
    "It opens with (<think>) and goes on."]) {
    assert.equal(win.stripThink(text), text);
    assert.deepEqual(plain(win.splitThink(text)), { think: null, open: false, rest: text });
  }
  assert.equal(win.stripThink("<think>plan"), "");
  assert.equal(win.stripThink("<think>a</think>One <think> mention"), "One <think> mention");
  assert.equal(win.stripThink("Text.\n<think>\nmore plan"), "Text.");
  assert.equal(win.stripThink("Text. <think>\nplan\n</think>\nAnswer."), "Text. \nAnswer.");
});

test("editing the first message keeps a title the user typed", async () => {
  const { window: win } = loadApp();
  const old = [[user("Tell me about seas"), reply("r")], null];
  const c = (title) => ({ title, messages: [], branches: [{ parent: "root", tails: old, current: 1 }] });
  assert.equal(win.titleIsAutomatic(c("Rivers and seas")), false);
  assert.equal(win.titleIsAutomatic(c("Tell me about seas")), true);
  assert.equal(win.titleIsAutomatic({ title: "New chat", messages: [] }), true);
});

test("a renamed model shows no switch divider and its current name", async () => {
  const server = makeServer();
  const win = await boot(server);
  runScript(win, `modelCache.models = [{ name: "new-name", former_names: ["old-name"] }];`);
  win.chat.conversations = [{ id: "c1", title: "One", messages: [
    user("a"), { ...reply("b"), model: "old-name" }, user("c"), { ...reply("d"), model: "new-name" }] }];
  win.chat.activeId = "c1";
  win.renderChat();
  assert.equal(win.document.querySelectorAll("#chat-messages .model-switch").length, 0);
  assert.ok(!win.document.getElementById("chat-messages").textContent.includes("old-name"));
});

test("the first click of a double-click on a closed chat keeps its row", async () => {
  const { win } = await twoChats();
  await open(win, "One");
  const row = [...win.document.querySelectorAll("#conv-list .conv-item")]
    .find((r) => r.textContent.includes("Two"));
  row.click();
  await drain();
  assert.ok(row.isConnected);
  assert.ok(row.classList.contains("active"));
  assert.equal(win.document.querySelectorAll("#conv-list .conv-item.active").length, 1);
  await win.syncConversations();
  assert.ok(row.isConnected, "a sync that finds nothing new leaves the list alone");
  row.querySelector(".title").dispatchEvent(new win.MouseEvent("dblclick", { bubbles: true }));
  const input = row.querySelector("input");
  assert.ok(input);
  win.renderConvList();
  assert.ok(input.isConnected, "the list is not rebuilt under an open rename");
  input.value = "Renamed";
  input.onblur();
  assert.ok(conv(win, "Renamed"));
  assert.equal(win.document.querySelector("#conv-list input"), null);
});

// ----------------------------------------------------- the store's own rules ---

test("the stand-in for the store answers as the contract says", async () => {
  const { steps } = JSON.parse(readFileSync(new URL("./fixtures/chat-store-contract.json", import.meta.url), "utf-8"));
  const server = makeServer();
  const revs = {};
  assert.ok(steps.length > 20);
  for (const [n, step] of steps.entries()) {
    const body = { title: "t" };
    if (step.picture) {
      body.messages = [{ role: "user", content: [{ type: "text", text: "look" },
        { type: "image_url", image_url: { url: "data:image/png;base64," + step.picture } }] }];
    } else if (step.texts) body.messages = step.texts.map((x, i) => (i % 2 ? reply : user)(x));
    if ("base" in step) body.base_rev = step.base ? revs[step.base] : null;
    for (const key of ["instance", "principal"]) if (step[key]) body[key] = step[key];
    const headers = step.store ? { "X-LocaLM-Chat-Store": step.store
      .replace("{instance}", server.instance).replace("{principal}", server.principal) } : {};
    const r = server.store(step.do, step.id, body, headers);
    const data = await r.json();
    const where = `step ${n}: ${JSON.stringify(step)} ${r.status} ${JSON.stringify(data)}`;
    assert.equal(r.status, step.status, where);
    if (step.as) revs[step.as] = data.rev;
    if (step.answer) assert.equal(data.status, step.answer, where);
    if (r.status === 409) {
      assert.equal(data.detail.rev, step.rev ? revs[step.rev] : null, where);
      assert.equal(data.detail.deleted, step.deleted, where);
    }
  }
});

// ----------------------------------------------------- sending while it loads ---

/** Holds the requests *match*(method, url) answers true for until released. */
function gate(server, match) {
  let release;
  const held = new Promise((r) => { release = r; });
  server.hold = async (method, url) => { if (match(method, url)) await held; };
  return () => { server.hold = null; release(); };
}

const answering = (win) => runScript(win, `modelCache.active = 'm'; maybeCompactConversation = async () => null;
  readSSE = async (r, on) => { on(JSON.stringify({ choices: [{ delta: { content: "ANSWER" } }] }));
    on(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })); };`);

async function loadingChat() {
  const server = makeServer();
  server.put("c1", { title: "One", updated_at: Date.now() + 9, messages: [user("a")] });
  server.put("c2", { title: "Two", updated_at: 1, messages: [user("b"), reply("c")] });
  const win = await boot(server);
  answering(win);
  await open(win, "One");
  const release = gate(server, (method, url) => method === "GET" && url.endsWith("/c2"));
  [...win.document.querySelectorAll("#conv-list .conv-item")].find((r) => r.textContent.includes("Two")).click();
  await drain(3);
  return { server, win, release, input: win.document.getElementById("chat-input") };
}

test("a message sent while its chat loads is sent there, after the chat's own messages", async () => {
  const { server, win, release, input } = await loadingChat();
  input.value = "my question";
  const sending = win.sendChat();
  await drain(3);
  assert.equal(server.chatPosts.length, 0, "nothing is sent before the chat is loaded");
  assert.equal(input.value, "");
  release();
  await sending;
  assert.ok(await waitFor(() => server.chatPosts.length === 1));
  const sent = server.chatPosts[0].messages.filter((m) => m.role === "user").map((m) => m.content);
  assert.deepEqual(sent, ["b", "my question"]);
  assert.ok(await waitFor(() => server.files.get("c2").messages.length === 4));
  assert.deepEqual(texts(server.files.get("c2")), ["b", "c", "my question", "ANSWER"]);
  assert.deepEqual(texts(server.files.get("c1")), ["a"]);
});

test("what is typed while a chat loads belongs to it", async () => {
  const { win, release, input } = await loadingChat();
  input.value = "typed for two";
  release();
  await drain();
  assert.equal(input.value, "typed for two");
  await open(win, "One");
  assert.equal(input.value, "");
  await open(win, "Two");
  assert.equal(input.value, "typed for two");
});

test("a chat that cannot be loaded sends nothing and keeps the message", async () => {
  const { server, win, release, input } = await loadingChat();
  input.value = "my question";
  server.files.delete("c2");
  const sending = win.sendChat();
  release();
  await sending;
  assert.equal(server.chatPosts.length, 0);
  assert.equal(win.chat.queue[0].text, "my question");
  assert.ok(win.__toasts.some((m) => /could not be loaded/.test(m)));
});

test("a message sent while the page itself loads keeps the chat's stored messages", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a"), reply("b")] });
  const release = gate(server, (method, url) => method === "GET" && url.endsWith("/c1"));
  server.active = "m";
  const win = await boot(server, { wait: false });
  answering(win);
  assert.ok(await waitFor(() => win.chat.conversations.length === 1));
  await drain();
  win.document.getElementById("chat-input").value = "my question";
  const sending = win.sendChat();
  await drain(3);
  release();
  await sending;
  assert.ok(await waitFor(() => server.files.get("c1").messages.length === 4));
  assert.deepEqual(texts(server.files.get("c1")), ["a", "b", "my question", "ANSWER"]);
});

// --------------------------------------------------------- deletions and losses ---

test("a chat deleted while its first save is on its way is not stored", async () => {
  const server = makeServer();
  const win = await boot(server);
  win.newConversation();
  const c = win.chat.conversations[0];
  c.title = "Secret";
  const release = gate(server, (method) => method === "PUT");
  say(win, c, "secret words");
  assert.ok(await waitFor(() => server.calls.some((x) => x.startsWith("PUT"))));
  const row = [...win.document.querySelectorAll("#conv-list .conv-item")].find((r) => r.textContent.includes("New chat") || r.textContent.includes("Secret"));
  [...row.querySelectorAll("button.del")].at(-1).click();
  win.document.querySelector("#modal-body button.btn-danger").click();
  win.dispatchEvent(new win.Event("pagehide"));
  await drain(3);
  assert.equal(server.calls.filter((x) => x.startsWith("DELETE")).length, 1, "the delete waits for nothing");
  release();
  await settle(50);
  assert.equal(server.files.size, 0);
  assert.deepEqual(plain(win.__toasts), []);
});

test("a delete whose answer never comes is said, but not by a page that is going away", async () => {
  const server = makeServer();
  for (const id of ["c1", "c2"]) server.put(id, { title: id, messages: [user("a")] });
  const win = await boot(server);
  runScript(win, "window.del = (x) => deleteConversationRemote(x)");
  server.hold = (method) => { if (method === "DELETE") throw new TypeError("Failed to fetch"); };
  win.dispatchEvent(new win.Event("pagehide"));
  await win.del(conv(win, "c1"));
  assert.deepEqual(plain(win.__toasts), []);
  win.dispatchEvent(new win.Event("pageshow"));
  await win.del(conv(win, "c2"));
  assert.equal(win.__toasts.length, 1);
});

test("clearing all chats does not wait for a save, and a first save on its way is not stored", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  win.newConversation();
  const c = win.chat.conversations.find((x) => x.title === "New chat");
  const release = gate(server, (method) => method === "PUT");
  say(win, c, "secret words");
  assert.ok(await waitFor(() => server.calls.some((x) => x.startsWith("PUT"))));
  const out = await win.deleteAllServerConversations();
  assert.deepEqual(plain(out.failed), []);
  release();
  await settle(50);
  assert.equal(server.files.size, 0);
});

test("a drawer value changed while a chat loads leaves its other stored values", async () => {
  const server = makeServer();
  server.put("c0", { title: "Open one", updated_at: Date.now() + 5, messages: [user("x")] });
  server.put("c1", { title: "Two", updated_at: 1, messages: [user("a")], params: { system: "BE TERSE", top_k: "40" } });
  const win = await boot(server);
  const release = gate(server, (method, url) => method === "GET" && url.endsWith("/c1"));
  await open(win, "Two");
  for (const [id, value] of [["p-temperature", "0.5"], ["p-top-k", "9"], ["p-top-k", ""]]) {
    const field = win.document.getElementById(id);
    field.value = value;
    field.dispatchEvent(new win.Event("input", { bubbles: true }));
  }
  release();
  assert.ok(await waitFor(() => server.files.get("c1").params.temperature === "0.5"));
  assert.deepEqual(server.files.get("c1").params, { system: "BE TERSE", temperature: "0.5" });
  assert.equal(win.document.getElementById("p-system").value, "BE TERSE");
  assert.deepEqual(texts(server.files.get("c1")), ["a"]);
});

test("a pin set and taken back before the chat loads leaves it unpinned", async () => {
  const server = makeServer();
  server.put("c0", { title: "Open one", updated_at: Date.now() + 5, messages: [user("x")] });
  server.put("c1", { title: "Two", updated_at: 1, messages: [user("a")] });
  const win = await boot(server);
  const release = gate(server, (method, url) => method === "GET" && url.endsWith("/c1"));
  const row = () => [...win.document.querySelectorAll("#conv-list .conv-item")].find((r) => r.textContent.includes("Two"));
  row().querySelector("button.del").click();
  row().querySelector("button.del").click();
  release();
  await settle(1500);
  assert.ok(!server.files.get("c1").pinned);
  assert.ok(!conv(win, "Two").pinned);
});

test("drawer values changed here and elsewhere are merged value by value", async () => {
  const server = makeServer();
  const rev = server.put("c1", { title: "One", messages: [user("a")], params: { system: "old", seed: "1" } });
  assert.ok(rev);
  const win = await boot(server);
  await open(win, "One");
  server.put("c1", { title: "One", messages: [user("a")], params: { system: "theirs", seed: "1" } });
  const field = win.document.getElementById("p-temperature");
  field.value = "0.5";
  field.dispatchEvent(new win.Event("input", { bubbles: true }));
  assert.ok(await waitFor(() => server.files.get("c1").params.temperature === "0.5"));
  assert.deepEqual(server.files.get("c1").params, { system: "theirs", seed: "1", temperature: "0.5" });
});

test("a stored copy that cannot be read after a refused save is asked for again and no copy is made", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  server.put("c1", { title: "One", messages: [user("a"), reply("theirs")] });
  server.failGet = 2;
  c.title = "Renamed";
  win.saveConversations(c);
  assert.ok(await waitFor(() => server.files.get("c1").title === "Renamed", 9000));
  assert.deepEqual(texts(server.files.get("c1")), ["a", "theirs"]);
  assert.equal(server.files.size, 1);
});

test("a save put off again and again waits longer each time", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  server.put("c1", { title: "One", messages: [user("a"), reply("theirs")] });
  win.chat.busy = true;
  c.title = "Renamed";
  win.saveConversations(c);
  await settle(8000);
  assert.ok(server.puts.length >= 2 && server.puts.length <= 4, String(server.puts.length));
  win.chat.busy = false;
});

for (const [what, unsaved] of [["opened", false], ["renamed", true]]) {
  test(`newer turns than a store put back to an older copy holds are kept as a copy (${what})`, async () => {
    const server = makeServer();
    server.put("c1", { title: "One", updated_at: 10, messages: [user("a")] });
    const old = plain(server.files.get("c1"));
    const win = await boot(server);
    const c = await open(win, "One");
    say(win, c, "newer turn");
    assert.ok(await waitFor(() => server.files.get("c1").messages.length === 2 && !c.unsaved));
    const saved = () => cache(win).some((x) => x.rev === c.rev && !x.unsaved);
    win.saveConversations();
    assert.ok(await waitFor(saved));
    server.files.set("c1", { ...old, rev: "restored" });
    const kept = cache(win).map((x) => (unsaved ? { ...x, title: "Renamed", unsaved: true, updated_at: Date.now() } : x));
    const again = await boot(server, { cached: kept });
    assert.ok(await waitFor(() => server.files.size === 2, 6000));
    const copy = [...server.files.values()].find((x) => x.id !== "c1");
    assert.deepEqual(copy.messages.map((m) => m.content), ["a", "newer turn"]);
    assert.match(copy.title, /copy/);
    assert.deepEqual(texts(server.files.get("c1")), ["a"]);
    assert.equal(again.__toasts.filter((m) => /Both versions are kept/.test(m)).length, 1);
  });
}

test("a turn saved here and one added elsewhere afterwards make no copy at the next load", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", updated_at: 10, messages: [user("a")] });
  let win = await boot(server);
  for (const turn of ["first", "second"]) {
    // The second round starts from the cached copy of the first.
    const c = await open(win, "One");
    const before = server.files.get("c1").messages.length;
    say(win, c, turn);
    assert.ok(await waitFor(() => server.files.get("c1").messages.length === before + 1));
    assert.ok(await waitFor(() => cache(win).some((x) => x.rev === c.rev && !x.unsaved)), turn);
    const stored = server.files.get("c1");
    server.put("c1", { ...plain(stored), updated_at: Date.now() + 5, messages: [...plain(stored.messages), reply("elsewhere")] });
    win = await boot(server, { cached: cache(win) });
    await settle(900);
    assert.equal(server.files.size, 1);
  }
  assert.deepEqual(texts(server.files.get("c1")), ["a", "first", "elsewhere", "second", "elsewhere"]);
});

test("messages deleted in another browser do not make a copy", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", updated_at: 10, messages: [user("a"), reply("b")] });
  const win = await boot(server);
  const c = await open(win, "One");
  await drain();
  server.put("c1", { title: "One", updated_at: 20, messages: [user("a")] });
  const again = await boot(server, { cached: cache(win) });
  await settle(900);
  assert.equal(server.files.size, 1);
  assert.ok(conv(again, "One"));
  c.title = "Renamed";
  win.saveConversations(c);
  assert.ok(await waitFor(() => server.files.get("c1").title === "Renamed"));
  assert.equal(server.files.size, 1);
  assert.deepEqual(texts(server.files.get("c1")), ["a"]);
});

test("a queued message of a chat past the listed rows is kept and its chat listed", async () => {
  const b = browser();
  seedRows(b, {
    k: owned({ conv: "c9", kind: "queue", state: "queued", owner: null,
      item: { id: "q9", convId: "c9", text: "for the old chat", attachments: [], docs: [], clips: [] } }),
    d: owned({ conv: "deleted", kind: "text", text: "x", owner: null }) });
  const server = makeServer();
  server.pageSize = 2;
  for (let i = 0; i < 4; i++) server.put("c" + i, { title: "Chat " + i, updated_at: 100 - i, messages: [user("a")] });
  server.put("c9", { title: "Old chat", updated_at: 1, messages: [user("a")] });
  const win = await boot(server, b);
  assert.deepEqual(plain(win.chat.queue.map((q) => q.text)), ["for the old chat"]);
  assert.ok(conv(win, "Old chat"));
  assert.ok(await waitFor(() => rowsOf(b).length === 1), "the row of a chat that is gone is removed");
  await win.syncConversations();
  assert.ok(conv(win, "Old chat"));
});

test("a stored chat whose file is missing is saved again from this browser, and that is said once", async () => {
  const server = makeServer();
  server.put("c0", { title: "Other", messages: [user("x")] });
  const win = await boot(server, { cached: [
    { id: "c1", title: "Missing", updated_at: 5, rev: "r9", messages: [user("a")] },
    { id: "c2", title: "Missing too", updated_at: 4, rev: "r8", messages: [user("b")] }] });
  assert.ok(await waitFor(() => server.files.size === 3));
  assert.deepEqual(texts(server.files.get("c1")), ["a"]);
  assert.equal(win.__toasts.filter((m) => /missing from the data folder/.test(m)).length, 1);
  assert.ok(conv(win, "Missing") && conv(win, "Missing too"));
});

test("a turn added to a chat whose file went missing is stored, not kept in the tab", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  server.files.delete("c1");
  say(win, c, "more");
  assert.ok(await waitFor(() => server.files.has("c1")));
  assert.deepEqual(texts(server.files.get("c1")), ["a", "more"]);
  assert.ok(!c.tabOnly);
});

test("a chat deleted here while a sync reads the list does not come back as a row", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  server.put("c2", { title: "Two", messages: [user("b")] });
  const win = await boot(server);
  await open(win, "One");
  const real = win.fetch;
  win.fetch = async (url, opts) => {
    const answer = await real(url, opts);
    if (String(url).startsWith("/api/conversations?")) {
      const row = [...win.document.querySelectorAll("#conv-list .conv-item")].find((r) => r.textContent.includes("Two"));
      if (row) {
        [...row.querySelectorAll("button.del")].at(-1).click();
        win.document.querySelector("#modal-body button.btn-danger").click();
        await drain();
      }
    }
    return answer;
  };
  await win.syncConversations();
  await drain();
  assert.equal(conv(win, "Two"), undefined);
});

test("a pin set on a chat that was never opened does not undo a rename made elsewhere", async () => {
  const server = makeServer();
  server.put("c0", { title: "Open one", updated_at: Date.now() + 5, messages: [user("x")] });
  server.put("c1", { title: "Old name", updated_at: 1, messages: [user("a")] });
  const win = await boot(server);
  server.put("c1", { title: "New name", folder: "work", updated_at: 2, messages: [user("a")] });
  const row = [...win.document.querySelectorAll("#conv-list .conv-item")].find((r) => r.textContent.includes("Old name"));
  row.querySelector("button.del").click();
  assert.ok(await waitFor(() => server.files.get("c1").pinned));
  assert.equal(server.files.get("c1").title, "New name");
  assert.equal(server.files.get("c1").folder, "work");
});

test("a save whose answer was lost is recognised as this page's own", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  server.loseAnswer = 1;
  say(win, c, "q1");
  assert.ok(await waitFor(() => server.files.get("c1").messages.length === 2));
  await settle(50);
  say(win, c, "q2");
  assert.ok(await waitFor(() => server.files.get("c1").messages.length === 3, 6000));
  assert.deepEqual(texts(server.files.get("c1")), ["a", "q1", "q2"]);
  assert.equal(server.files.size, 1);
  assert.ok(!win.__toasts.some((m) => /changed in another tab/.test(m)));
});

test("a save whose answer was lost is recognised after a reload", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  server.loseAnswer = 1;
  say(win, c, "q1");
  assert.ok(await waitFor(() => server.files.get("c1").messages.length === 2));
  win.saveConversations();
  await drain();
  const again = await boot(server, { cached: cache(win) });
  await settle(800);
  assert.equal(server.files.size, 1);
  assert.deepEqual([...texts(conv(again, "One"))], ["a", "q1"]);
});

// --------------------------------------------------------------- whose chats ---

const cachedFor = (prin, extra) => ({ updated_at: 5, inst: "backend-x", prin, messages: [user(prin + " words")], ...extra });

test("when another server answers at the address, nothing of the old one is shown or saved there", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("of the old server")] });
  const win = await boot(server);
  const c = await open(win, "One");
  say(win, c, "more");
  win.document.getElementById("chat-input").value = "typed for the old server";
  await drain();
  // Another page of this browser noticed first: the cache is the new server's already.
  server.instance = "backend-y";
  server.files.clear();
  server.puts.length = 0;
  win.localStorage.setItem("localm.instanceId", "backend-y");
  win.localStorage.removeItem("localm.conversations");
  win.localStorage.removeItem("localm.chatOwner");
  runScript(win, "window.__read = refreshCtxLimit()");
  await win.__read;
  assert.equal(win.chat.conversations.length, 0);
  assert.equal(win.document.getElementById("chat-input").value, "");
  await settle(1500);
  assert.equal(server.puts.length, 0);
  assert.equal(server.files.size, 0);
  assert.ok(!(win.localStorage.getItem("localm.conversations") || "").includes("old server"));
  assert.deepEqual(plain(win.__toasts), []);
});

test("a cache read before another page noticed the new server is not taken for the new server's", async () => {
  const server = makeServer();
  const cached = [{ id: "old1", title: "Old", updated_at: 5, messages: [user("of the old server")] }];
  const win = await boot(server, { cached, storage: { "localm.instanceId": "backend-old" }, wait: false });
  win.localStorage.setItem("localm.instanceId", server.instance);
  win.localStorage.removeItem("localm.conversations");
  assert.ok(await waitFor(() => win.chat.persist));
  await settle(900);
  assert.equal(win.chat.instanceMatch, true);
  assert.equal(win.chat.conversations.length, 0);
  assert.equal(server.puts.length, 0);
});

test("a delete made just after the key changed does not reach the new key's store", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  server.principal = "key-b";
  runScript(win, "window.del = (x) => deleteConversationRemote(x)");
  await win.del(c);
  assert.ok(server.files.has("c1"));
  assert.ok(!server.deleted.has("c1"));
  assert.equal(win.__toasts.length, 1);
});

test("a cached chat of another key is not shown, uploaded or removed while that key is not the last one", async () => {
  const server = makeServer({ principal: "key-b" });
  const win = await boot(server, {
    storage: { "localm.chatOwner": JSON.stringify({ inst: "backend-x", prin: "key-b" }) },
    cached: [cachedFor("key-a", { id: "ca", title: "A's" }), cachedFor("key-b", { id: "cb", title: "B's" })] });
  assert.ok(await waitFor(() => server.files.has("cb")));
  await settle(700);
  assert.equal(conv(win, "A's"), undefined);
  assert.ok(!win.document.getElementById("conv-list").textContent.includes("A's"));
  assert.deepEqual([...server.files.keys()], ["cb"]);
  assert.ok(cache(win).some((c) => c.id === "ca"), "it is left as it is");
});

test("signing in with another key removes what this browser kept for the old one", async () => {
  const b = browser();
  seedRows(b, { k: { ...owned({ conv: "ca", kind: "text", text: "A's words", owner: null }), prin: "key-a" } });
  const server = makeServer({ principal: "key-b" });
  const win = await boot(server, { ...b,
    storage: { "localm.chatOwner": JSON.stringify({ inst: "backend-x", prin: "key-a" }) },
    cached: [cachedFor("key-a", { id: "ca", title: "A's" }), { id: "old", title: "Unmarked", updated_at: 5, messages: [user("x")] }] });
  await settle(700);
  assert.equal(win.chat.conversations.length, 0);
  assert.equal(server.puts.length, 0, "nothing of key A reaches key B's store");
  assert.ok(!JSON.stringify(cache(win)).includes("key-a words"));
  assert.ok(await waitFor(() => rowsOf(b).length === 0));
  assert.equal(JSON.parse(win.localStorage.getItem("localm.chatOwner")).prin, "key-b");
});

test("a key that changed under an open page keeps the old key's chats out of the new store", async () => {
  const server = makeServer({ principal: "key-a" });
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  server.principal = "key-b";
  server.files.clear();
  say(win, c, "typed under key A");
  assert.ok(await waitFor(() => server.puts.length === 1));
  await settle(200);
  assert.equal(server.files.size, 0);
  assert.equal(win.chat.conversations.length, 0);
});

test("chats cached before rows were marked stay this browser's and are marked", async () => {
  const server = makeServer();
  const win = await boot(server, { cached: [{ id: "c1", title: "Old cache", updated_at: 5, messages: [user("a")] }] });
  assert.ok(await waitFor(() => server.files.has("c1")));
  assert.ok(await waitFor(() => cache(win)[0] && cache(win)[0].prin === "owner"));
});

test("signing out removes what this browser kept, and the page writes nothing more", async () => {
  const b = browser();
  const { win } = await twoChats(b);
  const told = [];
  const set = win.Storage.prototype.setItem;
  win.Storage.prototype.setItem = function (k, v) { told.push(k); return set.call(this, k, v); };
  const c = await open(win, "One");
  await typeDraft(win, "unsent words");
  assert.ok(await waitFor(() => rowsOf(b).length === 1 && cache(win).length > 0));
  const out = await win.signOutSession();
  assert.equal(out.ok, true);
  assert.ok(told.includes("localm.signedOut"), "the other tabs are told");
  assert.equal(win.localStorage.getItem("localm.conversations"), null);
  assert.equal(rowsOf(b).length, 0);
  say(win, c, "late");
  await typeDraft(win, "late words");
  await settle(500);
  assert.equal(win.localStorage.getItem("localm.conversations"), null);
  assert.equal(rowsOf(b).length, 0);
});

test("after the key is refused, a page writes nothing more to this browser and drops what it holds", async () => {
  const b = browser();
  const { win, server } = await twoChats(b);
  const c = await open(win, "One");
  await typeDraft(win, "typed before");
  assert.ok(await waitFor(() => rowsOf(b).length === 1 && cache(win).length > 0));
  const before = [win.localStorage.getItem("localm.conversations"), JSON.stringify(rowsOf(b))];
  server.refuseKey = true;
  c.messages.push(reply("a reply that ended later"));
  win.saveConversations(c);
  await typeDraft(win, "typed after");
  assert.ok(await waitFor(() => win.__localmLocked === true));
  await settle(600);
  assert.equal(win.localStorage.getItem("localm.conversations"), before[0]);
  assert.equal(JSON.stringify(rowsOf(b)), before[1]);
  assert.equal(win.chat.conversations.length, 0);
  assert.equal(win.document.getElementById("chat-input").value, "");
});

test("a tab that sees another tab sign out drops what it holds and what it wrote meanwhile", async () => {
  const b = browser();
  const { win, server } = await twoChats(b);
  const c = await open(win, "One");
  await typeDraft(win, "typed here");
  assert.ok(await waitFor(() => rowsOf(b).length === 1 && cache(win).length > 0));
  server.refuseKey = true;
  win.dispatchEvent(new win.StorageEvent("storage", { key: "localm.signedOut", newValue: "1" }));
  c.messages.push(reply("a reply that ended later"));
  win.saveConversations(c);
  assert.ok(await waitFor(() => rowsOf(b).length === 0));
  await settle(400);
  assert.equal(win.localStorage.getItem("localm.conversations"), null);
  assert.equal(win.chat.conversations.length, 0);
  assert.equal(win.document.getElementById("chat-input").value, "");
});

// ----------------------------------------------------------------- mode reads ---

test("a mode read that never answers holds the browser's copy back for a few seconds only", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  const win = await boot(server);
  const c = await open(win, "One");
  await drain();
  server.hold = (method, url) => (url === "/v1/config" ? new Promise(() => {}) : null);
  say(win, c, "kept here");
  await settle(300);
  assert.ok(!JSON.stringify(cache(win)).includes("kept here"));
  assert.ok(await waitFor(() => JSON.stringify(cache(win)).includes("kept here"), 6000));
});

test("an older answer that lands after a newer one changes nothing", async () => {
  const server = makeServer();
  const win = await boot(server);
  const real = win.fetch;
  let release;
  const held = new Promise((r) => { release = r; });
  let first = true;
  win.fetch = async (url, opts) => {
    if (String(url) === "/v1/config" && first) {
      first = false;
      const answer = await real(url, opts);
      await held;
      return answer;
    }
    return real(url, opts);
  };
  const slow = win.refreshCtxLimit();
  await drain(3);
  server.mode = "privacy";
  await win.refreshCtxLimit();
  assert.equal(win.chat.privacy, true);
  release();
  await slow;
  await drain();
  assert.equal(win.chat.privacy, true);
});

test("a page whose first mode read failed still connects once one answers", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("a")] });
  let down = true;
  server.hold = async (method, url) => {
    if (down && (url === "/v1/config" || url.startsWith("/api/conversations"))) throw new TypeError("Failed to fetch");
  };
  const b = browser();
  seedRows(b, { k: owned({ conv: "c1", kind: "text", text: "kept draft", owner: null }) });
  const win = await boot(server, { ...b, wait: false });
  await settle(100);
  assert.equal(win.chat.modeKnown, false);
  down = false;
  await win.refreshCtxLimit();
  assert.ok(await waitFor(() => win.__restored));
  await settle(700);
  assert.ok(conv(win, "One"));
  await open(win, "One");
  assert.equal(win.document.getElementById("chat-input").value, "kept draft");
});

test("text typed in privacy mode with no chat open does not land in a saved chat", async () => {
  const server = makeServer();
  server.put("c1", { title: "Saved", messages: [user("a")] });
  server.mode = "privacy";
  const win = await boot(server);
  win.document.getElementById("chat-input").value = "private words";
  server.mode = "full";
  await win.refreshCtxLimit();
  assert.ok(await waitFor(() => win.chat.persist));
  await drain();
  assert.equal(win.document.getElementById("chat-input").value, "private words");
  assert.notEqual(win.chat.activeId, "c1");
  await open(win, "Saved");
  assert.equal(win.document.getElementById("chat-input").value, "");
});

test("a message queued in a chat stays with it when privacy mode is entered, and nowhere else", async () => {
  const server = makeServer();
  server.put("c1", { title: "One", updated_at: Date.now() + 9, messages: [user("first one"), reply("r1")] });
  server.put("c2", { title: "Two", updated_at: 1, messages: [user("first two"), reply("r2")] });
  const win = await boot(server);
  const c = await open(win, "One");
  win.chat.queue.push({ text: "q one", attachments: [], docs: [], clips: [], convId: "c1" });
  win.chat.queue.push({ text: "q closed", attachments: [], docs: [], clips: [], convId: "c2" });
  server.mode = "privacy";
  await win.refreshCtxLimit();
  await drain();
  assert.deepEqual([...win.chat.conversations], [c]);
  assert.equal(c.tabOnly, true);
  assert.deepEqual(win.chat.queue.map((q) => [q.text, q.convId]).join(), ["q one", c.id].join());
});
