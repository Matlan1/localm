// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

// Makes `conv` the active conversation. chat.conversations/activeId are
// top-level consts, not window properties, so they must be set from inside the
// script realm for currentConv() to resolve to this same object.
function activateConv(window, conv) {
  window.__testConv = conv;
  runScript(window, "chat.conversations = [window.__testConv]; chat.activeId = window.__testConv.id;");
}

const WEB_CALL = '<tool_call>{"name":"web_search","query":"x"}</tool_call>';

// Streams one content delta carrying a web tool call, then throws AbortError.
async function runAborted({ speak = true, web = true } = {}) {
  const okResponse = { ok: true, status: 200, text: async () => "", json: async () => ({}) };
  const { window } = loadApp({ fetchImpl: () => Promise.resolve(okResponse) });
  window.maybeCompactConversation = async () => {};
  window.runWebCall = async () => {};   // noop stub
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
    // Aborts before any delta arrives.
    throw Object.assign(new Error("aborted"), { name: "AbortError" });
  };
  window.document.getElementById("p-speak").checked = false;
  window.document.getElementById("p-web").checked = false;
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  await window.runCompletion(conv);
  assert.equal(conv.messages.length, 1, "no assistant turn saved when there was nothing to save");
});

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

  // usage is parsed inside the jsdom realm and carries that realm's Object
  // prototype, so it must be compared field by field, not with deepEqual.
  const saved = conv.messages[1].usage;
  assert.ok(saved, "usage travels with the reply so it survives a reload");
  assert.equal(saved.total_tokens, 15);
  assert.equal(saved.ttft_ms, 120);
  assert.equal(saved.tokens_per_sec, 42);
  assert.match(window.document.getElementById("chat-usage").textContent, /42 tok\/s/,
    "the live DOM display still works exactly as before");
});

test("the usage line shows MTP acceptance, a pause, a stop, an off reply or unavailability, and nothing for an idle turn", async () => {
  const { window } = loadApp();
  const usage = $id => window.document.getElementById($id);
  window.updateUsageDisplay({ total_tokens: 7, tokens_per_sec: 3,
                              mtp: { state: "on", drafted: 40, accepted: 30, reason: null } });
  assert.match(usage("chat-usage").textContent, /3 tok\/s · MTP 75% accepted/);
  assert.match(usage("chat-usage").title, /30 of 40/);

  window.updateUsageDisplay({ total_tokens: 7, mtp: { state: "stopped", drafted: 4, accepted: 3,
                                                       reason: "draft-decode-failed:1" } });
  assert.match(usage("chat-usage").textContent, /MTP stopped/);
  assert.match(usage("chat-usage").title, /draft-decode-failed:1/);

  window.updateUsageDisplay({ total_tokens: 7, mtp: { state: "paused", drafted: 6, accepted: 3,
                                                       paused_steps: 40, reason: "slower-than-plain" } });
  assert.match(usage("chat-usage").textContent, /MTP paused/);
  assert.match(usage("chat-usage").title, /40 steps/);

  window.updateUsageDisplay({ total_tokens: 7, mtp: { state: "unavailable", drafted: 0, accepted: 0,
                                                       reason: "no-mtp-graph:llama" } });
  assert.match(usage("chat-usage").textContent, /MTP unavailable/);

  window.updateUsageDisplay({ total_tokens: 7, mtp: { state: "off", drafted: 0, accepted: 0,
                                                       reason: "image" } });
  assert.match(usage("chat-usage").textContent, /MTP off/);
  assert.match(usage("chat-usage").title, /with an image/);

  window.updateUsageDisplay({ total_tokens: 7, mtp: { state: "idle", drafted: 0, accepted: 0 } });
  assert.doesNotMatch(usage("chat-usage").textContent, /MTP/);
  assert.equal(usage("chat-usage").title, "");

  window.updateUsageDisplay({ total_tokens: 7, mtp: { state: "on", drafted: 0, accepted: 0 } });
  assert.doesNotMatch(usage("chat-usage").textContent, /MTP/,
    "a reply that verified no drafts has no acceptance rate to show");
});

test("the usage line shows n-gram drafting from usage.speculation and leaves an mtp source to usage.mtp", async () => {
  const { window } = loadApp();
  const usage = $id => window.document.getElementById($id);
  window.updateUsageDisplay({ total_tokens: 7, tokens_per_sec: 3,
    speculation: { source: "ngram", state: "on", drafted: 20, accepted: 15, reason: null } });
  assert.match(usage("chat-usage").textContent, /3 tok\/s · N-gram 75% accepted/);
  assert.match(usage("chat-usage").title, /15 of 20/);

  window.updateUsageDisplay({ total_tokens: 7, speculation: { source: "ngram", state: "unavailable",
                                                              drafted: 0, accepted: 0, reason: "rewind-unsupported" } });
  assert.match(usage("chat-usage").textContent, /N-gram unavailable/);
  assert.match(usage("chat-usage").title, /rewind-unsupported/);

  window.updateUsageDisplay({ total_tokens: 7, speculation: { source: "ngram", state: "off",
                                                              drafted: 0, accepted: 0, reason: "image" } });
  assert.match(usage("chat-usage").title, /with an image/);

  window.updateUsageDisplay({ total_tokens: 7, speculation: { source: "ngram", state: "idle",
                                                              drafted: 0, accepted: 0 } });
  assert.doesNotMatch(usage("chat-usage").textContent, /N-gram/);

  window.updateUsageDisplay({ total_tokens: 7, tokens_per_sec: 3,
    speculation: { source: "draft", state: "on", drafted: 10, accepted: 6, reason: null } });
  assert.match(usage("chat-usage").textContent, /3 tok\/s · Draft 60% accepted/);
  assert.match(usage("chat-usage").title, /Draft-model drafting: 6 of 10/);

  window.updateUsageDisplay({ total_tokens: 7,
    speculation: { source: "draft", state: "on", drafted: 10, accepted: 6, reason: "draft-on-cpu" } });
  assert.match(usage("chat-usage").textContent, /Draft 60% accepted/);
  assert.match(usage("chat-usage").title, /Draft-model drafting on the CPU: 6 of 10/);

  window.updateUsageDisplay({ total_tokens: 7, speculation: { source: "draft", state: "unavailable",
                                                              drafted: 0, accepted: 0, reason: "draft-model-missing" } });
  assert.match(usage("chat-usage").textContent, /Draft unavailable/);
  assert.match(usage("chat-usage").title, /draft-model-missing/);

  window.updateUsageDisplay({ total_tokens: 7, speculation: { source: "toString", state: "on",
                                                              drafted: 4, accepted: 4 } });
  assert.doesNotMatch(usage("chat-usage").textContent, /accepted/);

  window.updateUsageDisplay({ total_tokens: 7,
    mtp: { state: "on", drafted: 40, accepted: 30 },
    speculation: { source: "mtp", state: "on", drafted: 40, accepted: 30 } });
  assert.match(usage("chat-usage").textContent, /MTP 75% accepted/);
  assert.doesNotMatch(usage("chat-usage").textContent, /N-gram/,
    "an mtp source is shown once, from usage.mtp");
});

test("updateUsageDisplay renders tok/s and clears it back out for a null usage", async () => {
  const { window } = loadApp();
  window.updateUsageDisplay({ total_tokens: 7, tokens_per_sec: 3 });
  assert.match(window.document.getElementById("chat-usage").textContent, /3 tok\/s/);
  window.updateUsageDisplay(null);
  assert.equal(window.document.getElementById("chat-usage").textContent, "",
    "a conversation (or turn) with no usage must not leave a previous turn's figures on screen");
});
