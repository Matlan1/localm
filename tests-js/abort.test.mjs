// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

// BUG-13: when the user presses Stop mid-stream, runCompletion's read rejects
// with an AbortError. The catch only suppressed the error toast - it then went
// on to persist the partial reply, speak it aloud, and (with web access on)
// fire the web loop / recurse. A Stop must do none of those.
//
// We simulate "the model streamed a (web-call) reply, THEN the user stopped":
// readSSE delivers one content delta (so `full` is non-empty and is a parseable
// web tool call) and then throws AbortError.

const WEB_CALL = '<tool_call>{"name":"web_search","query":"x"}</tool_call>';

async function runAborted({ speak = true, web = true } = {}) {
  const okResponse = { ok: true, status: 200, text: async () => "", json: async () => ({}) };
  const { window } = loadApp({ fetchImpl: () => Promise.resolve(okResponse) });
  window.maybeCompactConversation = async () => {};
  window.runWebCall = async () => {};   // noop: the web path needs no real response
  window.readSSE = async (_r, onData) => {
    onData(JSON.stringify({ choices: [{ delta: { content: WEB_CALL } }] }));
    throw Object.assign(new Error("aborted"), { name: "AbortError" });
  };
  const doc = window.document;
  doc.getElementById("p-speak").checked = speak;
  doc.getElementById("p-web").checked = web;

  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  const fetchCalls = [];
  const realFetch = window.fetch;
  window.fetch = (...a) => { fetchCalls.push(String(a[0])); return realFetch(...a); };

  await window.runCompletion(conv);
  return { window, conv, fetchCalls };
}

test("BUG-13: Stop does not persist the partial assistant reply", async () => {
  const { conv } = await runAborted();
  assert.equal(conv.messages.length, 1, "only the user message remains; no assistant reply saved");
  assert.equal(conv.messages[0].role, "user");
});

test("BUG-13: Stop does not speak the partial reply", async () => {
  const { window } = await runAborted({ speak: true });
  assert.equal(window.__spoken.length, 0, "speechSynthesis.speak must not be called on abort");
});

test("BUG-13: Stop does not fire the web loop / recurse", async () => {
  const { fetchCalls } = await runAborted({ web: true });
  const chatCalls = fetchCalls.filter((u) => u.includes("/v1/chat/completions"));
  assert.equal(chatCalls.length, 1, "exactly one chat call; no recursion after abort");
});

// U-STOP: a stop must be unmistakable - the partial is marked [stopped] on screen
// and any speech already playing is halted (so a stopped reply is never silently
// left looking live or kept being read aloud).
test("U-STOP: Stop marks the partial [stopped] and cancels speech", async () => {
  const okResponse = { ok: true, status: 200, text: async () => "", json: async () => ({}) };
  const { window } = loadApp({ fetchImpl: () => Promise.resolve(okResponse) });
  window.maybeCompactConversation = async () => {};
  let cancelled = 0;
  window.speechSynthesis.cancel = () => { cancelled += 1; };
  window.readSSE = async (_r, onData) => {
    onData(JSON.stringify({ choices: [{ delta: { content: "partial answer" } }] }));
    throw Object.assign(new Error("aborted"), { name: "AbortError" });
  };
  window.document.getElementById("p-speak").checked = false;
  window.document.getElementById("p-web").checked = false;
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  await window.runCompletion(conv);

  assert.match(window.document.getElementById("chat-messages").textContent, /\[stopped\]/,
    "the partial reply is visibly marked as stopped");
  assert.equal(cancelled, 1, "any in-flight speech is cancelled on stop");
  assert.equal(conv.messages.length, 1, "still no assistant reply persisted");
});
