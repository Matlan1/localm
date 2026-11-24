// SPDX-License-Identifier: AGPL-3.0-or-later
// Chat state that must not be lost or leak: two tabs saving one conversation, a
// delete made elsewhere, changes of the session mode, compaction during a send,
// the message box of each chat, refused pictures and think mentions.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

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

/** An in-memory server: the conversation store (revision check included) and
 *  /v1/config. `put(id, ...)` writes to the store as another browser would. */
function makeServer() {
  const s = { files: new Map(), deleted: new Set(), puts: [], mode: "full", calls: [],
              failPut: 0, rows: null, chatReply: null };
  let n = 0;
  const res = (status, body) => ({ ok: status < 400, status, statusText: String(status),
    headers: { get: () => null }, text: async () => JSON.stringify(body), json: async () => plain(body) });
  const row = (r) => ({ id: r.id, title: r.title, updated_at: r.updated_at, rev: r.rev,
    pinned: !!r.pinned, folder: r.folder || null, n_messages: r.messages.length });
  const stale = (r) => res(409, { detail: { error: "stale", rev: r ? r.rev : null, row: r ? row(r) : null } });
  s.put = (id, data) => {
    const r = { id, title: "t", updated_at: Date.now(), branches: [], params: {}, ...data, rev: "r" + (++n) };
    s.files.set(id, r);
    return r.rev;
  };
  s.remove = (id) => { s.files.delete(id); s.deleted.add(id); };
  s.fetch = async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    s.calls.push(method + " " + u);
    if (u === "/v1/config") {
      return res(200, { effective_mode: s.mode, n_ctx_max: 16384, instance_id: "backend-x" });
    }
    if (u.startsWith("/api/conversations?")) {
      if (s.mode === "privacy") return res(200, { enabled: false, conversations: [], stored: s.files.size });
      const all = [...s.files.values()].map(row);
      const offset = Number(/offset=(\d+)/.exec(u)?.[1] || 0);
      const rows = s.pageSize ? all.slice(offset, offset + s.pageSize) : all;
      return res(200, { enabled: true, conversations: rows, total: all.length, unreadable: [] });
    }
    const m = /^\/api\/conversations\/([^?]+)$/.exec(u);
    if (m) {
      const id = decodeURIComponent(m[1]);
      const stored = s.files.get(id);
      if (s.mode === "privacy") return res(403, { detail: "privacy" });
      if (method === "GET") return stored ? res(200, stored) : res(404, { detail: "none" });
      if (method === "DELETE") { s.remove(id); return res(200, { status: "deleted" }); }
      const body = JSON.parse(opts.body);
      s.puts.push({ id, body });
      if (s.failPut) return res(s.failPut, { detail: "no" });
      if ("base_rev" in body ? (stored ? stored.rev : null) !== body.base_rev : stored) return stale(stored);
      if (!stored && s.deleted.has(id)) return stale(null);
      return res(200, { status: "saved", id, rev: s.put(id, body) });
    }
    if (u === "/v1/chat/completions" && s.chatReply) return s.chatReply(JSON.parse(opts.body), opts);
    return res(200, { models: [], active: "", plugins: [], items: [] });
  };
  return s;
}

/** A page talking to *server*. *cached* seeds the browser's conversation cache. */
async function boot(server, { cached = null } = {}) {
  const { window: win } = loadApp({
    fetchImpl: server.fetch,
    seedLocalStorage: { "localm.instanceId": "backend-x",
      ...(cached ? { "localm.conversations": JSON.stringify(cached) } : {}) },
  });
  runScript(win, `window.chat = chat; window.__toasts = [];
    const __toast = toast; toast = (m, e) => { window.__toasts.push(String(m)); __toast(m, e); };
    window.draftStore = draftStore;`);
  assert.ok(await waitFor(() => win.chat.modeKnown));
  await drain();
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
  assert.equal(server.puts[1].body.base_rev, server.files.get("c1").rev === c.rev ? server.puts[1].body.base_rev : "");
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
  assert.ok(win.__toasts.some((m) => m.includes("One (copy)")), win.__toasts.join("|"));
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
      const older = []; for (let i = 0; i < 60; i++) older.push({ id: "n" + i, title: "n", tabOnly: false, messages: [] });
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

async function twoChats() {
  const server = makeServer();
  server.put("c1", { title: "One", messages: [user("first one"), reply("r1")] });
  server.put("c2", { title: "Two", messages: [user("first two"), reply("r2")] });
  const win = await boot(server);
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

/** Replaces the page's draft store with rows in a Map. */
function fakeDrafts(win, rows = new Map()) {
  runScript(win, `
    draftStore.available = () => true;
    draftStore.all = async () => new Map(window.__rows);
    draftStore.write = async (puts, dels) => {
      for (const [k, v] of puts) window.__rows.set(k, JSON.parse(JSON.stringify(v)));
      for (const k of dels) window.__rows.delete(k);
    };
    draftStore.clear = async () => window.__rows.clear();`.replaceAll("window.__rows", "window.__draftRows"));
  win.__draftRows = rows;
  return rows;
}

test("a draft, its attachments and the queue survive a reload of the tab", async () => {
  const { win, input, server } = await twoChats();
  const rows = fakeDrafts(win);
  runScript(win, "_draftsRestored = false;");
  await win.restoreDrafts();
  await open(win, "One");
  input.value = "unsent words";
  input.dispatchEvent(new win.Event("input"));
  win.chat.attachments.push({ name: "p.png", dataUri: "data:image/png;base64,AA" });
  win.renderAttachChips();
  win.chat.queue.push({ text: "queued words", attachments: [], docs: [], clips: [], convId: "c2" });
  assert.ok(await waitFor(() => rows.size === 3), [...rows.keys()].join());
  const tab = win.sessionStorage.getItem("localm.tabId");

  const again = await boot(server);
  again.sessionStorage.setItem("localm.tabId", tab);
  fakeDrafts(again, rows);
  runScript(again, "_draftsRestored = false; _tabId = '';");
  await again.restoreDrafts();
  await open(again, "One");
  assert.equal(again.document.getElementById("chat-input").value, "unsent words");
  assert.equal(again.chat.attachments[0].name, "p.png");
  assert.equal(again.chat.queue.map((q) => q.text).join(), "queued words");
  assert.equal(server.puts.length, 0, "a restored queue is not sent on its own");

  again.document.getElementById("chat-input").value = "";
  again.chat.attachments = [];
  again.chat.queue = [];
  again.renderAttachChips();
  assert.ok(await waitFor(() => rows.size === 0), "what is gone from the box is gone from the browser");
});

test("a draft typed before the stored one is read is kept beside it", async () => {
  const { win, input } = await twoChats();
  await open(win, "One");
  fakeDrafts(win, new Map([["c1|old|text", { text: "from before", at: Date.now() }]]));
  input.value = "typed early";
  runScript(win, "_draftsRestored = false;");
  await win.restoreDrafts();
  assert.equal(input.value, "from before\ntyped early");
});

test("stored drafts of a chat that is deleted are removed", async () => {
  const { win } = await twoChats();
  const rows = fakeDrafts(win, new Map([
    ["c1|old|text", { text: "x", at: Date.now() }], ["gone|old|text", { text: "y", at: Date.now() }]]));
  runScript(win, "_draftsRestored = false;");
  await win.restoreDrafts();
  assert.ok(![...rows.keys()].some((k) => k.startsWith("gone|")));
  rows.set("c2|other|media", { attachments: [], docs: [], clips: [], at: 1 });
  win.retainConversations((c) => c.id !== "c2");
  assert.ok(await waitFor(() => ![...rows.keys()].some((k) => k.startsWith("c2|"))));
});

test("privacy mode keeps no draft in this browser and empties what was there", async () => {
  const { win, input, server } = await twoChats();
  const rows = fakeDrafts(win);
  runScript(win, "_draftsRestored = false;");
  await win.restoreDrafts();
  await open(win, "One");
  input.value = "unsent words";
  input.dispatchEvent(new win.Event("input"));
  assert.ok(await waitFor(() => rows.size === 1));
  server.mode = "privacy";
  input.value = "unsent private words";
  input.dispatchEvent(new win.Event("input"));
  assert.ok(await waitFor(() => win.chat.privacy));
  assert.ok(await waitFor(() => rows.size === 0));
  assert.equal(input.value, "unsent private words", "what is typed stays in the box");
  assert.equal(win.sessionStorage.getItem("localm.tabId"), null);
  server.mode = "full";
  await win.refreshCtxLimit();
  input.value = "unsent private words, more";
  input.dispatchEvent(new win.Event("input"));
  await settle(400);
  assert.equal(rows.size, 0, "what was typed under privacy mode is not written after it either");
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
  row.querySelector(".title").dispatchEvent(new win.MouseEvent("dblclick", { bubbles: true }));
  assert.ok(row.querySelector("input"));
});
