// SPDX-License-Identifier: AGPL-3.0-or-later
// Chat state that must not be lost or leak: two tabs saving one conversation,
// a delete made elsewhere, the privacy mode switch, per-chat parameters on the
// server, the per-chat message box, edit and revert, the sources line under a
// reply, routing notes and the copy buttons.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { loadApp, runScript } from "./harness.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const CHAT_JS = readFileSync(join(ROOT, "localm/plugins/gui/static/app/chat.js"), "utf-8");
const CAPS_PY = readFileSync(join(ROOT, "localm/model_manager/capabilities.py"), "utf-8");

const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
const drain = async (n = 15) => { for (let i = 0; i < n; i++) await settle(0); };
async function waitFor(fn, timeout = 3000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await settle(15); }
  return false;
}

// An in-memory conversation store answering the routes the way
// localm/plugins/builtin/chat/plug.py does (revision check included).
function makeStore() {
  const files = new Map();
  const puts = [];
  let mode = "full";
  const res = (status, body) => ({
    ok: status < 400, status,
    text: async () => JSON.stringify(body),
    json: async () => JSON.parse(JSON.stringify(body)),
  });
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    if (u === "/v1/config") {
      return res(200, { effective_mode: mode, n_ctx_max: 16384, instance_id: "backend-x" });
    }
    if (u.startsWith("/api/conversations?")) {
      if (mode === "privacy") return res(200, { enabled: false, conversations: [], stored: files.size });
      const rows = [...files.values()].map((r) => ({
        id: r.id, title: r.title, updated_at: r.updated_at, rev: r.rev ?? 0,
        pinned: !!r.pinned, folder: r.folder || null, n_messages: r.messages.length }));
      return res(200, { enabled: true, conversations: rows, total: rows.length, unreadable: [] });
    }
    const m = u.match(/^\/api\/conversations\/([^?]+)$/);
    if (m) {
      const id = decodeURIComponent(m[1]);
      if (mode === "privacy") return res(403, { detail: "privacy" });
      if (method === "GET") {
        return files.has(id) ? res(200, files.get(id)) : res(404, { detail: "No such conversation" });
      }
      if (method === "DELETE") {
        files.delete(id);
        return res(200, { status: "deleted", id });
      }
      if (method === "PUT") {
        const body = JSON.parse(opts.body);
        puts.push({ id, body });
        const cur = files.has(id) ? (files.get(id).rev ?? 0) : null;
        if ("base_rev" in body && cur !== body.base_rev) {
          return res(409, { detail: { error: "stale", rev: cur } });
        }
        const rec = { ...body, id, rev: (cur || 0) + 1 };
        delete rec.base_rev;
        files.set(id, rec);
        return res(200, { status: "saved", id, rev: rec.rev });
      }
    }
    return res(200, { models: [], active: "", conversations: [], plugins: [] });
  };
  return { files, puts, fetchImpl, setMode: (m2) => { mode = m2; } };
}

function stored(id, messages, extra = {}) {
  return { id, title: "Stored " + id, updated_at: 1000, pinned: false, folder: null,
           branches: [], messages, rev: 1, ...extra };
}

async function boot(store, seed = {}) {
  const { window } = loadApp({
    fetchImpl: store.fetchImpl,
    seedLocalStorage: { "localm.instanceId": "backend-x", ...seed },
  });
  runScript(window, "window.chatState = chat;");
  await drain(30);
  await settle(50);
  return window;
}

const toastText = (win) => win.document.getElementById("toast").textContent;
const conv = (win, id) => win.chatState.conversations.find((c) => c.id === id);

/* ---- two tabs, deletes elsewhere ---- */

test("a stale tab's save does not overwrite the other tab's turns: it gets the " +
     "saved copy and a note", async () => {
  const store = makeStore();
  store.files.set("c1", stored("c1", [{ role: "user", content: "q1" },
                                      { role: "assistant", content: "a1" }]));
  const a = await boot(store);
  const b = await boot(store);
  assert.ok(await waitFor(() => conv(a, "c1") && !conv(a, "c1")._meta), "tab A hydrated c1");
  assert.ok(await waitFor(() => conv(b, "c1") && !conv(b, "c1")._meta), "tab B hydrated c1");

  runScript(b, `const c = chat.conversations.find((x) => x.id === "c1");
    c.messages.push({ role: "user", content: "from tab B" }); saveConversations(c);`);
  assert.ok(await waitFor(() => store.files.get("c1").rev === 2), "tab B saved first");

  runScript(a, `const c = chat.conversations.find((x) => x.id === "c1");
    c.messages.push({ role: "user", content: "from tab A" }); saveConversations(c);`);
  assert.ok(await waitFor(() => store.puts.filter((p) => p.id === "c1").length === 2),
    "tab A tried to save");
  assert.ok(await waitFor(() => conv(a, "c1").rev === 2), "tab A took the saved copy");

  const kept = store.files.get("c1").messages.map((x) => x.content);
  assert.deepEqual(kept, ["q1", "a1", "from tab B"], "the store keeps tab B's turn");
  assert.deepEqual(conv(a, "c1").messages.map((x) => x.content), kept,
    "tab A now shows the saved copy");
  assert.match(toastText(a), /changed in another tab or browser/);
  assert.equal(store.puts[1].body.base_rev, 1, "tab A's save named the revision it held");

  // A save queued before the reload carries no change and is not sent.
  const before = store.puts.length;
  runScript(a, `_putConversation(chat.conversations.find((x) => x.id === "c1"));`);
  await settle(50);
  assert.equal(store.puts.length, before);
  assert.equal(store.files.get("c1").rev, 2, "the other tab's next save is not refused");
});

test("pinning a chat whose messages are not loaded yet saves the pin with its " +
     "messages", async () => {
  const store = makeStore();
  store.files.set("open", { ...stored("open", [{ role: "user", content: "o" }]), updated_at: 9000 });
  store.files.set("other", stored("other", [{ role: "user", content: "keep me" }]));
  const win = await boot(store);
  assert.ok(await waitFor(() => conv(win, "other") && conv(win, "other")._meta),
    "precondition: the other chat is an index row");
  runScript(win, `const c = chat.conversations.find((x) => x.id === "other");
    c.pinned = true; saveConversations(c);`);
  assert.ok(await waitFor(() => store.files.get("other").pinned === true), "the pin is stored");
  assert.deepEqual(store.files.get("other").messages.map((x) => x.content), ["keep me"],
    "with the chat's messages, not an empty chat");
});

test("a save of a chat deleted in another tab removes it here instead of " +
     "storing it again", async () => {
  const store = makeStore();
  store.files.set("c1", stored("c1", [{ role: "user", content: "q1" }]));
  const a = await boot(store);
  assert.ok(await waitFor(() => conv(a, "c1") && !conv(a, "c1")._meta));
  store.files.delete("c1");
  runScript(a, `const c = chat.conversations.find((x) => x.id === "c1");
    c.messages.push({ role: "assistant", content: "late" }); saveConversations(c);`);
  assert.ok(await waitFor(() => !conv(a, "c1")), "the deleted chat is removed here");
  assert.equal(store.files.has("c1"), false, "and is not stored again");
  assert.match(toastText(a), /deleted in another tab or browser/);
});

test("a delete-all elsewhere holds: a browser that cached the chats does not " +
     "upload them again on its next load, and a chat never stored is uploaded",
     async () => {
  const store = makeStore();
  const cached = [
    { id: "was-stored", title: "Was stored", updated_at: 5, pinned: false, folder: null,
      branches: [], rev: 3, messages: [{ role: "user", content: "deleted elsewhere" }] },
    { id: "never-stored", title: "Offline", updated_at: 6, pinned: false, folder: null,
      branches: [], messages: [{ role: "user", content: "typed offline" }] },
  ];
  const win = await boot(store, { "localm.conversations": JSON.stringify(cached) });
  assert.ok(await waitFor(() => store.files.has("never-stored")),
    "positive control: the never-stored chat is uploaded");
  assert.equal(store.puts.find((p) => p.id === "never-stored").body.base_rev, null);
  assert.equal(store.puts.some((p) => p.id === "was-stored"), false,
    "the chat deleted elsewhere is not uploaded again");
  assert.equal(conv(win, "was-stored"), undefined, "and is removed from this browser");
  assert.ok(!(win.localStorage.getItem("localm.conversations") || "").includes("deleted elsewhere"));
});

/* ---- privacy mode switch ---- */

test("leaving privacy mode keeps the chat on screen unsaved; a new chat is " +
     "saved", async () => {
  const store = makeStore();
  store.setMode("privacy");
  const win = await boot(store);
  assert.equal(win.chatState.privacy, true);
  runScript(win, `chat.conversations.push({ id: "secret", title: "Secret", updated_at: 1,
      pinned: false, folder: null, branches: [],
      messages: [{ role: "user", content: "private words" }] });
    chat.activeId = "secret"; renderConvList(); renderChat();`);
  store.setMode("full");
  runScript(win, "refreshCtxLimit();");
  assert.ok(await waitFor(() => win.chatState.persist === true), "connected to the store");
  assert.equal(conv(win, "secret").ephemeral, true);
  assert.match(toastText(win), /stays in this tab only/);
  runScript(win, `const s = chat.conversations.find((c) => c.id === "secret");
    s.messages.push({ role: "user", content: "more private words" }); saveConversations(s);
    newConversation(); const n = currentConv();
    n.title = "After"; n.messages.push({ role: "user", content: "public words" });
    saveConversations(n);`);
  const newId = win.chatState.activeId;
  assert.ok(await waitFor(() => store.files.has(newId)), "positive control: the new chat is saved");
  assert.equal(store.puts.some((p) => p.id === "secret"), false,
    "the chat held under privacy mode never reaches the store");
  assert.ok(!(win.localStorage.getItem("localm.conversations") || "").includes("private words"),
    "nor this browser's storage");
});

test("a request made for a chat held from privacy mode asks the server to " +
     "record it under privacy mode", () => {
  const win = bootLocal();
  runScript(win, `window.__h = [convSurfaceHeaders({ ephemeral: true }),
    convSurfaceHeaders({}), convSurfaceHeaders(null)];`);
  assert.equal(win.__h[0]["X-Localm-Session-Mode"], "privacy");
  assert.equal(win.__h[0]["X-Client-Id"], "chat");
  assert.equal(win.__h[1]["X-Localm-Session-Mode"], undefined);
  assert.equal(win.__h[2]["X-Client-Id"], "chat");
  assert.ok(/summaryPass\([^)]*convSurfaceHeaders\(conv\)\)/.test(CHAT_JS),
    "the compaction request of a chat uses its headers");
});

test("privacy mode says how many chats saved earlier are still on disk", async () => {
  const store = makeStore();
  store.files.set("old1", stored("old1", [{ role: "user", content: "x" }]));
  store.files.set("old2", stored("old2", [{ role: "user", content: "y" }]));
  store.setMode("privacy");
  const win = await boot(store);
  assert.ok(await waitFor(() => win.document.getElementById("privacy-leftovers")));
  const note = win.document.getElementById("privacy-leftovers");
  assert.match(note.textContent, /^2 chats saved earlier are still on disk$/);
  assert.match(note.title, /does not delete/);
  store.files.clear();
  runScript(win, "showPrivacyLeftovers();");
  assert.ok(await waitFor(() => !win.document.getElementById("privacy-leftovers")),
    "no note when nothing is left");
});

/* ---- per-chat parameters on the server ---- */

test("a chat's own system prompt is stored with it and shown in another browser",
     async () => {
  const store = makeStore();
  store.files.set("c1", stored("c1", [{ role: "user", content: "q" }]));
  const a = await boot(store);
  assert.ok(await waitFor(() => conv(a, "c1") && !conv(a, "c1")._meta));
  const field = a.document.getElementById("p-system");
  field.value = "Answer in French.";
  field.dispatchEvent(new a.Event("input", { bubbles: true }));
  assert.ok(await waitFor(() => (store.files.get("c1").params || {}).system === "Answer in French."),
    "the parameters reach the store");

  const b = await boot(store);
  assert.ok(await waitFor(() => b.document.getElementById("p-system").value === "Answer in French."),
    "another browser opening the chat shows its system prompt");
});

/* ---- per-chat message box, edit and revert ---- */

function bootLocal() {
  const { window } = loadApp();
  runScript(window, "window.chatState = chat;");
  return window;
}

const IMG = "data:image/png;base64,iVBORw0KGgo=";
const CLIP = { type: "input_audio", input_audio: { data: "UklGRg==", format: "wav" },
               name: "note.wav", seconds: 2 };

test("each chat keeps its own message box and attachments", async () => {
  const win = bootLocal();
  const box = win.document.getElementById("chat-input");
  runScript(win, "newConversation(); currentConv().messages.push({ role: 'user', content: 'hi' });");
  const a = win.chatState.activeId;
  box.value = "draft for A";
  runScript(win, `chat.attachments.push({ name: "a.png", dataUri: "${IMG}" }); renderAttachChips();`);
  await settle(3);
  runScript(win, "newConversation();");
  const b = win.chatState.activeId;
  assert.notEqual(a, b);
  assert.equal(box.value, "", "the new chat starts with an empty box");
  assert.equal(win.chatState.attachments.length, 0, "and without A's picture");
  box.value = "draft for B";
  runScript(win, `chat.activeId = "${a}"; renderChat();`);
  assert.equal(box.value, "draft for A");
  assert.equal(win.chatState.attachments.length, 1);
  assert.equal(win.document.querySelectorAll("#attach-chips .chip").length, 1);
  runScript(win, `chat.activeId = "${b}"; renderChat();`);
  assert.equal(box.value, "draft for B");
  assert.equal(win.chatState.attachments.length, 0);
});

test("a document that finishes loading after a chat switch goes to the chat " +
     "it was attached in", async () => {
  let release;
  const gate = new Promise((r) => { release = r; });
  const { window: win } = loadApp({ fetchImpl: async (url) => {
    if (String(url) === "/api/chat/extract") {
      await gate;
      return { ok: true, status: 200, json: async () => (
        { filename: "doc.txt", text: "body", chars: 4, truncated: false }) };
    }
    return { ok: true, status: 200, text: async () => "", json: async () => ({}) };
  } });
  runScript(win, "window.chatState = chat;");
  runScript(win, "newConversation(); currentConv().messages.push({ role: 'user', content: 'a' });");
  const a = win.chatState.activeId;
  runScript(win, `window.__p = attachDocument(new File(["body"], "doc.txt", { type: "text/plain" }));`);
  await settle(20);
  await settle(3);
  runScript(win, "newConversation();");
  release();
  await win.__p;
  assert.equal(win.chatState.docs.length, 0, "the document is not in the other chat");
  runScript(win, `chat.activeId = "${a}"; renderChat();`);
  assert.deepEqual(win.chatState.docs.map((d) => d.name), ["doc.txt"]);
});

test("edit puts the message's picture and audio clip back and asks before " +
     "replacing an unsent draft", async () => {
  const win = bootLocal();
  runScript(win, `chat.conversations = [{ id: "e1", title: "E", messages: [
      { role: "user", id: "m1", content: [{ type: "text", text: "look at this" },
        { type: "image_url", image_url: { url: "${IMG}" } }, ${JSON.stringify(CLIP)}] },
      { role: "assistant", id: "m2", content: "nice" }] }];
    chat.activeId = "e1"; renderChat();`);
  const box = win.document.getElementById("chat-input");
  box.value = "half-typed";
  runScript(win, "editMessage(currentConv(), 0);");
  assert.equal(win.document.getElementById("modal").style.display, "flex",
    "a draft in the box is not replaced without asking");
  assert.equal(box.value, "half-typed");
  assert.equal(win.chatState.conversations[0].messages.length, 2, "nothing forked yet");

  win.document.querySelector("#modal-body .btn-danger").click();
  assert.equal(box.value, "look at this");
  assert.deepEqual(win.chatState.attachments.map((x) => x.dataUri), [IMG]);
  assert.equal(win.chatState.clips.length, 1);
  assert.equal(win.chatState.clips[0].data, "UklGRg==");
  assert.equal(win.chatState.clips[0].name, "note.wav");
});

test("after edit the earlier messages stay reachable with arrows before the " +
     "edit is sent", async () => {
  const win = bootLocal();
  runScript(win, `chat.conversations = [{ id: "e2", title: "E", messages: [
      { role: "user", id: "m1", content: "first" },
      { role: "assistant", id: "m2", content: "reply" }] }];
    chat.activeId = "e2"; renderChat(); editMessage(currentConv(), 0);`);
  const row = win.document.querySelector("#chat-messages .branch-pending");
  assert.ok(row, "a row stands where the edited message will go");
  assert.match(row.textContent, /1\/2|2\/2/);
  row.querySelector(".variant button").click();
  assert.deepEqual(win.chatState.conversations[0].messages.map((m) => m.content),
    ["first", "reply"], "the arrows bring the earlier version back");
});

test("revert puts the picture back too", async () => {
  const win = bootLocal();
  runScript(win, `chat.conversations = [{ id: "r1", title: "R", messages: [
      { role: "user", id: "m1", content: [{ type: "text", text: "see" },
        { type: "image_url", image_url: { url: "${IMG}" } }] },
      { role: "assistant", id: "m2", content: "ok" }] }];
    chat.activeId = "r1"; renderChat(); revertTo(currentConv(), 0);`);
  assert.equal(win.document.getElementById("chat-input").value, "see");
  assert.deepEqual(win.chatState.attachments.map((x) => x.dataUri), [IMG]);
  assert.equal(win.chatState.conversations[0].messages.length, 0);
});

/* ---- sources under the reply ---- */

test("the sources of a turn are listed under its final reply", async () => {
  const win = bootLocal();
  runScript(win, `chat.conversations = [{ id: "w1", title: "W", messages: [
      { role: "user", content: "what is new" },
      { role: "assistant", content: "" },
      { kind: "tool", tool: "search", status: "done", query: "new",
        sources: [{ id: "S1", url: "https://a.example/x", title: "Example A" },
                  { id: "S2", url: "https://b.example/y", title: "" }] },
      { role: "assistant", content: "Here it is [S1]." },
      { role: "user", content: "thanks" },
      { role: "assistant", content: "welcome" }] }];
    chat.activeId = "w1"; renderChat();`);
  const replies = [...win.document.querySelectorAll("#chat-messages .msg-row.assistant")];
  assert.equal(replies.length, 3);
  assert.equal(replies[0].querySelector(".reply-sources"), null, "not under the tool-call turn");
  const line = replies[1].querySelector(".reply-sources");
  assert.ok(line, "under the final reply of the turn");
  assert.deepEqual([...line.querySelectorAll("a")].map((x) => x.href),
    ["https://a.example/x", "https://b.example/y"]);
  assert.equal(replies[2].querySelector(".reply-sources"), null, "not under a later turn");
});

/* ---- routing note ---- */

test("a routing note with a capability word the catalog lacks reads as words, " +
     "not as a catalog key", () => {
  const win = bootLocal();
  runScript(win, `window.__chip = buildRoutedChip({ from: "m", gaps: ["vision", "long_audio_input"] });
    window.__sugg = buildRoutedChip({ from: "m", suggest: "x", gaps: ["new_thing"] });`);
  assert.match(win.__chip.title, /reading images, long audio input/);
  assert.match(win.__sugg.title, /\(new thing\)/);
  assert.ok(!/chat\.routed\.cap/.test(win.__chip.title + win.__sugg.title));
});

test("every capability the server can name has a catalog entry", () => {
  const win = bootLocal();
  runScript(win, "window.__en = I18N_EN;");
  const names = ["VISION", "AUDIO", "TOOL_USE", "REASONING", "CONTEXT_LENGTH"];
  const words = names.map((n) => {
    const m = CAPS_PY.match(new RegExp("^" + n + ' = "([a-z_]+)"', "m"));
    assert.ok(m, n + " is defined in capabilities.py");
    return m[1];
  });
  const boolTuple = CAPS_PY.match(/^BOOLEAN_CAPABILITIES = \(([^)]*)\)/m);
  assert.ok(boolTuple);
  for (const n of boolTuple[1].split(",").map((x) => x.trim()).filter(Boolean)) {
    assert.ok(names.includes(n), n + " is checked here");
  }
  for (const w of words) assert.ok(win.__en["chat.routed.cap." + w], w);
});

/* ---- copy ---- */

test("chat.js copies text only through copyTextWithFeedback", () => {
  assert.equal((CHAT_JS.match(/clipboard\.writeText/g) || []).length, 0);
  assert.ok((CHAT_JS.match(/copyTextWithFeedback\(/g) || []).length >= 2);
});

test("a blocked clipboard is reported, never shown as copied", async () => {
  const win = bootLocal();
  runScript(win, `chat.conversations = [{ id: "k1", title: "K", messages: [
      { role: "assistant", content: "copy me" }] }];
    chat.activeId = "k1"; renderChat();`);
  const btn = win.document.querySelector("#chat-messages .copy-btn");
  btn.click();
  await drain();
  assert.equal(btn.textContent, "copy");
  assert.match(toastText(win), /Could not copy/);

  const copied = [];
  Object.defineProperty(win.navigator, "clipboard", {
    value: { writeText: async (s) => { copied.push(s); } }, configurable: true });
  btn.click();
  await drain();
  assert.deepEqual(copied, ["copy me"]);
  assert.equal(btn.textContent, "copied");
});
