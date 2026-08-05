// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

// runCompletion() only ever mutates the *conv* object passed to it - it never
// touches chat.conversations/activeId itself, so most tests here never seed
// them (production always calls it with the active conversation; the plain
// object these tests build is enough to see runCompletion's OWN behaviour).
// But renderChat() (now called after persisting a stopped or completed reply)
// reads the conversation back through currentConv(), which resolves through
// chat.conversations/activeId - a top-level `const`, not a window property, so
// it can only be reached from inside the shared script realm. Seed it via
// runScript so renderChat() finds the SAME conversation object these tests
// already hold, matching what always holds true in the real app.
function activateConv(window, conv) {
  window.__testConv = conv;
  runScript(window, "chat.conversations = [window.__testConv]; chat.activeId = window.__testConv.id;");
}

// BUG-13: when the user presses Stop mid-stream, runCompletion's read rejects
// with an AbortError. The catch only suppressed the error toast - it then went
// on to persist the partial reply, speak it aloud, and (with web access on)
// fire the web loop / recurse. A Stop must never speak the partial or recurse
// on it - those two guards are what this file protects.
//
// Persistence itself was reconsidered later (see "Stop now persists..." below):
// BUG-13's own bug was never about saving the text, only about a partial being
// mistaken for a finished, continuable reply. The two are separable as long as
// the persisted message is built directly in the abort branch, never by
// falling into the shared post-completion code that leads to TTS/recursion.
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

test("BUG-13: Stop does not speak the partial reply", async () => {
  const { window } = await runAborted({ speak: true });
  assert.equal(window.__spoken.length, 0, "speechSynthesis.speak must not be called on abort");
});

test("BUG-13: Stop does not fire the web loop / recurse", async () => {
  const { fetchCalls } = await runAborted({ web: true });
  const chatCalls = fetchCalls.filter((u) => u.includes("/v1/chat/completions"));
  assert.equal(chatCalls.length, 1, "exactly one chat call; no recursion after abort");
});

test("Stop now persists the partial reply as a terminal, non-continuable message", async () => {
  const { conv } = await runAborted();
  assert.equal(conv.messages.length, 2, "the partial reply is saved alongside the user turn");
  const reply = conv.messages[1];
  assert.equal(reply.role, "assistant");
  assert.equal(reply.stopped, true, "marked stopped, mirroring the existing truncated flag");
  assert.ok(!reply.content.includes("[stopped]"),
    "the [stopped] marker is a render-time annotation, not part of the saved content - " +
    "so a later turn never shows the model its own reply with that literal marker in it");
  assert.ok(reply.content.includes("tool_call"),
    "the raw partial text (here, the mid-flight web tool call) is preserved verbatim");
});

test("Stop with no content at all does not persist an empty assistant turn", async () => {
  const okResponse = { ok: true, status: 200, text: async () => "", json: async () => ({}) };
  const { window } = loadApp({ fetchImpl: () => Promise.resolve(okResponse) });
  window.maybeCompactConversation = async () => {};
  window.readSSE = async () => {
    // Aborted before a single delta arrived.
    throw Object.assign(new Error("aborted"), { name: "AbortError" });
  };
  window.document.getElementById("p-speak").checked = false;
  window.document.getElementById("p-web").checked = false;
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  await window.runCompletion(conv);
  assert.equal(conv.messages.length, 1, "no assistant turn saved when there was nothing to save");
});

// U-STOP: a stop must be unmistakable - the partial is marked [stopped] on screen
// and any speech already playing is halted (so a stopped reply is never silently
// left looking live or kept being read aloud).
test("U-STOP: Stop marks the partial [stopped] on screen and cancels speech", async () => {
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
  activateConv(window, conv);
  await window.runCompletion(conv);

  assert.match(window.document.getElementById("chat-messages").textContent, /\[stopped\]/,
    "the partial reply is visibly marked as stopped");
  assert.equal(cancelled, 1, "any in-flight speech is cancelled on stop");
  assert.equal(conv.messages.length, 2, "the stopped reply is now persisted");
  assert.equal(conv.messages[1].content, "partial answer",
    "the persisted content is the raw text, without the [stopped] marker baked in");
});

// Token-rate persistence: a completed (non-aborted) turn's usage figures used to
// reach only the DOM ($("chat-usage")), never the saved reply object, so they
// were gone the moment the page reloaded. This is unrelated to BUG-13/U-STOP -
// an aborted stream never receives a usage chunk at all (the server only sends
// it on the final SSE frame, which a disconnect prevents from ever arriving) -
// so there was nothing to lose on Stop specifically; the loss was universal,
// for every successful turn.
test("A completed turn's usage is saved on the reply, not just shown in the DOM", async () => {
  const okResponse = { ok: true, status: 200, text: async () => "", json: async () => ({}) };
  const { window } = loadApp({ fetchImpl: () => Promise.resolve(okResponse) });
  window.maybeCompactConversation = async () => {};
  const usage = { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15,
                   ttft_ms: 120, tokens_per_sec: 42 };
  window.readSSE = async (_r, onData) => {
    onData(JSON.stringify({ choices: [{ delta: { content: "answer" } }] }));
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }], usage }));
  };
  window.document.getElementById("p-speak").checked = false;
  window.document.getElementById("p-web").checked = false;
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  activateConv(window, conv);
  await window.runCompletion(conv);

  // Field-by-field, not a whole-object deepEqual: chunk.usage was parsed by
  // JSON.parse INSIDE the jsdom realm, so it carries that realm's Object
  // prototype, not Node's - a whole-object deepEqual fails on that alone even
  // though every field matches.
  const saved = conv.messages[1].usage;
  assert.ok(saved, "usage travels with the reply so it survives a reload");
  assert.equal(saved.total_tokens, 15);
  assert.equal(saved.ttft_ms, 120);
  assert.equal(saved.tokens_per_sec, 42);
  assert.match(window.document.getElementById("chat-usage").textContent, /42 tok\/s/,
    "the live DOM display still works exactly as before");
});

test("updateUsageDisplay renders tok/s and clears it back out for a null usage", async () => {
  const { window } = loadApp();
  window.updateUsageDisplay({ total_tokens: 7, tokens_per_sec: 3 });
  assert.match(window.document.getElementById("chat-usage").textContent, /3 tok\/s/);
  window.updateUsageDisplay(null);
  assert.equal(window.document.getElementById("chat-usage").textContent, "",
    "a conversation (or turn) with no usage must not leave a previous turn's figures on screen");
});
