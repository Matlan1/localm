// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for chat.js behaviour: copying a user message, renaming a chat,
// reusing an empty new chat, the reply stats line, the parameter drawer's
// upper bounds, pinning with no chat open, regenerate during a reply,
// deleting a chat while its reply streams, the too-large save notice and a
// failed page read whose search snippet was used.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const ticks = async (n = 4) => { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); };
const toastText = (win) => win.document.getElementById("toast").textContent;

test("Copy on a user message copies the text as typed, a bare <think> included", async () => {
  const { window: win } = loadApp();
  const copied = [];
  Object.defineProperty(win.navigator, "clipboard",
    { value: { writeText: async (s) => { copied.push(s); } }, configurable: true });
  const box = win.document.getElementById("chat-messages");
  const { row } = win.addMessageRow(box, "user", "before <think> after");
  row.querySelector(".copy-btn").click();
  await ticks();
  assert.deepEqual(copied, ["before <think> after"]);
});

test("Copy on a reply still leaves its reasoning out", async () => {
  const { window: win } = loadApp();
  const copied = [];
  Object.defineProperty(win.navigator, "clipboard",
    { value: { writeText: async (s) => { copied.push(s); } }, configurable: true });
  const box = win.document.getElementById("chat-messages");
  const { row } = win.addMessageRow(box, "assistant", "<think>plan</think>answer");
  row.querySelector(".copy-btn").click();
  await ticks();
  assert.deepEqual(copied, ["answer"]);
});

function startRename(win) {
  const title = win.document.querySelector("#conv-list .conv-item .title");
  title.dispatchEvent(new win.MouseEvent("dblclick", { bubbles: true }));
  return win.document.querySelector("#conv-list .conv-item input");
}

test("Escape while renaming a chat keeps the old name; Enter keeps the new one", () => {
  const { window: win } = loadApp();
  win.newConversation();
  win.currentConv().title = "Old name";
  win.renderConvList();
  let input = startRename(win);
  input.value = "Typed";
  input.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }));
  assert.equal(win.currentConv().title, "Old name");
  input = startRename(win);
  input.value = "New name";
  input.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  assert.equal(win.currentConv().title, "New name");
});

test("a second New chat reuses the empty one instead of adding another", () => {
  const { window: win } = loadApp();
  win.newConversation();
  const first = win.currentConv();
  win.newConversation();
  assert.equal(win.currentConv(), first);
  runScript(win, 'currentConv().messages.push({ role: "user", content: "hi" });');
  win.newConversation();
  assert.notEqual(win.currentConv(), first, "a chat with messages is not reused");
  runScript(win, "globalThis.__n = chat.conversations.length;");
  assert.equal(win.__n, 2);
});

test("the reply stats line gives the first-token time in whole milliseconds", () => {
  const { window: win } = loadApp();
  win.updateUsageDisplay({ total_tokens: 10, ttft_ms: 6326.54 });
  const line = win.document.getElementById("chat-usage").textContent;
  assert.match(line, /TTFT 6327 ms/);
  assert.doesNotMatch(line, /6326\.5/);
});

test("a drawer value above the field's max is sent as the max and shown so", () => {
  const { window: win } = loadApp();
  const temp = win.document.getElementById("p-temperature");
  temp.value = "9";
  const params = win.chatParams();
  assert.equal(params.temperature, 2);
  assert.equal(temp.value, "2", "the field shows the value that is sent");
  temp.value = "0.7";
  assert.equal(win.chatParams().temperature, 0.7);
});

test("ticking Pin with no chat open starts a chat pinned to the model", () => {
  const { window: win } = loadApp();
  runScript(win, "modelCache.active = 'm1';");
  assert.ok(!win.currentConv());
  win.setConversationPin(true);
  const conv = win.currentConv();
  assert.ok(conv, "a chat was started");
  assert.equal(conv.pinnedModel, "m1");
  assert.equal(win.document.getElementById("p-pin-model").checked, true);
});

test("regenerate while a reply streams says to wait", () => {
  const { window: win } = loadApp();
  win.newConversation();
  runScript(win, `currentConv().messages.push({ role: "user", content: "q" }, { role: "assistant", content: "a" });
    chat.busy = true;`);
  win.regenerate(win.currentConv());
  assert.equal(toastText(win), "Wait for the current reply to finish");
  assert.equal(win.currentConv().messages.length, 2);
});

test("a chat deleted while its reply streams is not written back to the server", async () => {
  const puts = [];
  const fetchImpl = async (url, opts = {}) => {
    if ((opts.method || "GET") === "PUT") puts.push(String(url));
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window: win } = loadApp({ fetchImpl });
  runScript(win, "chat.persist = true;");
  win.newConversation();
  const conv = win.currentConv();
  runScript(win, 'currentConv().messages.push({ role: "user", content: "q" });');
  win.renderConvList();
  const del = [...win.document.querySelectorAll("#conv-list .conv-item .del")].find((b) => b.textContent === "×");
  del.click();
  [...win.document.querySelectorAll("#modal-body button")].find((b) => /Delete/.test(b.textContent)).click();
  await ticks();
  puts.length = 0;
  conv.messages.push({ role: "assistant", content: "late reply" });
  win.pushConversation(conv);
  await new Promise((r) => setTimeout(r, 1500));
  assert.deepEqual(puts, [], "the end-of-turn save of the deleted chat sent nothing");
});

test("a failed read whose search snippet is in the evidence says the snippet was used", () => {
  const { window: win } = loadApp();
  const ev = {
    kind: "tool", tool: "search", status: "done", query: "q", grounding: "snippet-only",
    sources: [
      { id: "S1", url: "https://a.example/", title: "A", grounding: "failed", retrieval_status: "failed", snippet: "s" },
      { id: "S2", url: "https://b.example/", title: "B", grounding: "failed", retrieval_status: "failed", snippet: "" },
    ],
    chunks: [{ source_id: "S1", kind: "snippet", text: "s" }],
  };
  const box = win.document.getElementById("chat-messages");
  win.addToolEventRow(box, ev);
  const metas = [...box.querySelectorAll(".tool-source-meta")].map((m) => m.textContent);
  assert.equal(metas[0], "read failed, the search snippet was used");
  assert.match(metas[1], /no evidence/);
});
