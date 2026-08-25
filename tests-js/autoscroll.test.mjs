// SPDX-License-Identifier: AGPL-3.0-or-later
// Chat autoscroll: chat.stick latches from the scroll listener, and the view
// only follows the bottom while it is set.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

test("R31: the scroll listener latches chat.stick to the scroll position", () => {
  const { window } = loadApp();
  runScript(window, "window.chatState = chat;");
  const box = window.document.getElementById("chat-messages");

  // The user scrolled up (nearBottom false) -> autoscroll pauses.
  window.nearBottom = () => false;
  box.dispatchEvent(new window.Event("scroll"));
  assert.equal(window.chatState.stick, false, "scrolling up pauses autoscroll");

  // Back at the bottom -> autoscroll re-arms.
  window.nearBottom = () => true;
  box.dispatchEvent(new window.Event("scroll"));
  assert.equal(window.chatState.stick, true, "returning to the bottom re-arms it");
});

test("R31: scrolling up mid-stream stops the reply from yanking the view down", async () => {
  const { window } = loadApp({
    fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => "" }),
  });
  runScript(window, "window.chatState = chat;");
  window.maybeCompactConversation = async () => {};

  const box = window.document.getElementById("chat-messages");
  let scrollTopVal = 0;
  const writes = [];
  Object.defineProperty(box, "scrollHeight", { configurable: true, get: () => 1000 });
  Object.defineProperty(box, "scrollTop", {
    configurable: true,
    get: () => scrollTopVal,
    set: (v) => { scrollTopVal = v; writes.push(v); },
  });

  window.readSSE = async (_r, onData) => {
    // First token arrives while the user is still at the bottom (armed).
    onData(JSON.stringify({ choices: [{ delta: { content: "line one " } }] }));
    // The user scrolls up to re-read; from here autoscroll must stay paused.
    window.chatState.stick = false;
    writes.length = 0;   // ignore the armed writes before the scroll-up
    onData(JSON.stringify({ choices: [{ delta: { content: "line two " } }] }));
    onData(JSON.stringify({ choices: [{ delta: { content: "line three " } }] }));
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }));
  };

  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;

  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  await window.runCompletion(conv);

  assert.equal(writes.length, 0,
    "no token (nor the finalize renderChat) re-pinned the view after the user scrolled up");
});

test("R31: renderChat does not re-pin to the bottom while the user is scrolled up", () => {
  const { window } = loadApp();
  runScript(window, "window.chatState = chat;");
  runScript(window, `
    chat.conversations = [{ id: "c1", title: "t", messages: [
      { role: "user", content: "hi" }, { role: "assistant", content: "hello there" }] }];
    chat.activeId = "c1";
  `);
  const box = window.document.getElementById("chat-messages");
  let val = 0;
  const writes = [];
  Object.defineProperty(box, "scrollHeight", { configurable: true, get: () => 1000 });
  Object.defineProperty(box, "scrollTop", {
    configurable: true, get: () => val, set: (v) => { val = v; writes.push(v); },
  });

  // Paused: the user is scrolled up.
  window.chatState.stick = false;
  window.renderChat();
  assert.equal(writes.length, 0, "renderChat leaves the scroll alone when paused");

  // Armed: renderChat follows to the bottom.
  window.chatState.stick = true;
  window.renderChat();
  assert.ok(writes.includes(1000), "renderChat re-pins to the bottom when armed");
});

test("R31: a fresh send re-arms autoscroll so the new reply is followed", async () => {
  const { window } = loadApp({
    fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => "" }),
  });
  runScript(window, "window.chatState = chat;");
  window.maybeCompactConversation = async () => {};
  window.chatState.stick = false;   // left paused from a previous reply
  window.readSSE = async (_r, onData) => {
    onData(JSON.stringify({ choices: [{ delta: { content: "hello" } }] }));
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }));
  };
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  await window.runCompletion(conv);
  assert.equal(window.chatState.stick, true, "starting a new reply follows it again");
});
