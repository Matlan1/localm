// SPDX-License-Identifier: AGPL-3.0-or-later
// Terminal states of one chat completion. The server reports how a generation
// ended (finish_reason stop / length / error, or the client aborts); the GUI
// must keep those states apart instead of collapsing a failed generation, or a
// reply that only ANNOUNCES an action, into "a completed answer".

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const jsonResp = (obj) => ({
  ok: true, status: 200, json: async () => obj, text: async () => JSON.stringify(obj),
});

function activateConv(window, conv) {
  window.__testConv = conv;
  runScript(window, "chat.conversations = [window.__testConv]; chat.activeId = window.__testConv.id;");
}

const WEB_CALL = '<tool_call>{"name": "web_search", "args": {"query": "x"}}</tool_call>';

/** One streamed round: content deltas then a terminal frame. `rounds` is a
 *  queue, one entry per runCompletion recursion; a `null` entry throws
 *  AbortError after the content (the user pressed Stop). */
function driver({ web = false, speak = false, rounds }) {
  const calls = [];
  const impl = async (url, opts = {}) => {
    let body;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = opts.body; }
    calls.push({ url: String(url), body });
    if (String(url) === "/v1/config") return jsonResp({ net_mode: "allow" });
    if (String(url) === "/api/web/search")
      return jsonResp({ query: "q", results: [{ title: "T", url: "https://example.com/", snippet: "S" }] });
    return jsonResp({});
  };
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  const queue = rounds.slice();
  window.readSSE = async (_r, onData) => {
    const round = queue.shift();
    if (round === undefined) throw new Error("unexpected extra completion round");
    for (const d of round.deltas || []) onData(JSON.stringify(d));
    if (round.abort) throw Object.assign(new Error("aborted"), { name: "AbortError" });
  };
  const doc = window.document;
  doc.getElementById("p-speak").checked = speak;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = web;
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  activateConv(window, conv);
  const completions = () => calls.filter((c) => c.url === "/v1/chat/completions");
  return { window, conv, calls, completions };
}

const content = (s) => ({ choices: [{ delta: { content: s } }] });
const done = (reason) => ({ choices: [{ delta: {}, finish_reason: reason }] });

// ---------------------------------------------------------------------------
//  finish_reason "error" is its own terminal state
// ---------------------------------------------------------------------------

test("error: a failed generation is persisted as a failed turn and marked on screen", async () => {
  const d = driver({ rounds: [{ deltas: [content("partial "), content("[inference error: boom]"), done("error")] }] });
  await d.window.runCompletion(d.conv);
  assert.equal(d.conv.messages.length, 2, "the failed reply is saved next to the user turn");
  const reply = d.conv.messages[1];
  assert.equal(reply.role, "assistant");
  assert.equal(reply.failed, true, "marked failed, mirroring the stopped/truncated flags");
  assert.equal(reply.truncated, undefined);
  assert.equal(reply.stopped, undefined);
  assert.ok(reply.content.includes("[inference error: boom]"),
    "the visible error chunk the server streamed is kept in the content");
  assert.ok(!reply.content.includes("[generation failed]"),
    "the failed marker is a render-time annotation, not baked into the saved content");
  assert.match(d.window.document.getElementById("chat-messages").textContent, /generation failed/,
    "the failed turn is visibly marked");
});

test("error: TTS is never called for a failed generation", async () => {
  const d = driver({ speak: true, rounds: [{ deltas: [content("partial"), done("error")] }] });
  await d.window.runCompletion(d.conv);
  assert.equal(d.window.__spoken.length, 0, "speechSynthesis.speak must not run on a failed turn");
});

test("error: a complete tool call emitted before the failure is never executed or parsed", async () => {
  const d = driver({ web: true, rounds: [{ deltas: [content(WEB_CALL), content("[inference error: boom]"), done("error")] }] });
  let parsed = 0, ran = 0;
  d.window.parseWebCalls = () => { parsed += 1; return []; };
  d.window.runWebCall = async () => { ran += 1; };
  await d.window.runCompletion(d.conv);
  assert.equal(parsed, 0, "parseWebCalls must not be reached for a failed turn");
  assert.equal(ran, 0, "no web tool ran");
  assert.equal(d.completions().length, 1, "no continuation round after a failed turn");
});

test("error: the failed status survives a reload", async () => {
  const d = driver({ rounds: [{ deltas: [content("partial"), done("error")] }] });
  await d.window.runCompletion(d.conv);
  // A reload rebuilds the conversation from its persisted JSON and re-renders.
  const reloaded = JSON.parse(JSON.stringify(d.conv));
  assert.equal(reloaded.messages[1].failed, true, "the flag is in the persisted shape");
  activateConv(d.window, reloaded);
  d.window.renderChat();
  assert.match(d.window.document.getElementById("chat-messages").textContent, /generation failed/,
    "after a reload the turn still renders as failed");
});

test("error with no content at all persists nothing and leaves a failure marker", async () => {
  const d = driver({ rounds: [{ deltas: [done("error")] }] });
  await d.window.runCompletion(d.conv);
  assert.equal(d.conv.messages.length, 1, "no assistant turn saved when nothing was generated");
  assert.match(d.window.document.getElementById("chat-messages").textContent, /generation failed/);
});

// ---------------------------------------------------------------------------
//  stop / length / error / abort / empty-success are five distinct paths
// ---------------------------------------------------------------------------

test("stop: a normal reply is persisted with no terminal flag and is spoken", async () => {
  const d = driver({ speak: true, rounds: [{ deltas: [content("fine"), done("stop")] }] });
  await d.window.runCompletion(d.conv);
  const reply = d.conv.messages[1];
  assert.equal(reply.content, "fine");
  assert.equal(reply.failed, undefined);
  assert.equal(reply.truncated, undefined);
  assert.equal(reply.stopped, undefined);
  assert.equal(d.window.__spoken.length, 1, "a finished reply is read aloud");
});

test("length: a cut-off reply is persisted as truncated, not failed", async () => {
  const d = driver({ rounds: [{ deltas: [content("cut"), done("length")] }] });
  await d.window.runCompletion(d.conv);
  const reply = d.conv.messages[1];
  assert.equal(reply.truncated, true);
  assert.equal(reply.failed, undefined);
  assert.match(d.window.document.getElementById("chat-messages").textContent, /max-tokens limit/);
});

test("abort: a stopped reply is persisted as stopped, not failed", async () => {
  const d = driver({ rounds: [{ deltas: [content("half")], abort: true }] });
  await d.window.runCompletion(d.conv);
  const reply = d.conv.messages[1];
  assert.equal(reply.stopped, true);
  assert.equal(reply.failed, undefined);
  assert.match(d.window.document.getElementById("chat-messages").textContent, /\[stopped\]/);
});

test("empty success: nothing is persisted and nothing is marked failed", async () => {
  const d = driver({ rounds: [{ deltas: [done("stop")] }] });
  await d.window.runCompletion(d.conv);
  assert.equal(d.conv.messages.length, 1);
  assert.doesNotMatch(d.window.document.getElementById("chat-messages").textContent, /generation failed/);
});

// ---------------------------------------------------------------------------
//  a plain-prose action announcement gets exactly one bounded repair round
// ---------------------------------------------------------------------------

test("announcement: 'I will now search' with no call triggers exactly one repair round", async () => {
  const d = driver({ web: true, rounds: [
    { deltas: [content("I will now search the web for that and report back."), done("stop")] },
    { deltas: [content("Here is the answer: 42."), done("stop")] },
  ] });
  await d.window.runCompletion(d.conv);
  const pending = d.conv.messages.filter((m) => /\[pending action\]/.test(String(m.content)));
  assert.equal(pending.length, 1, "exactly one repair note was injected");
  assert.equal(pending[0].web, true, "the note is a dimmed control message, not a user turn");
  assert.equal(d.completions().length, 2, "one original round plus one repair round");
  assert.equal(d.conv.messages[d.conv.messages.length - 1].content, "Here is the answer: 42.");
});

test("announcement: the repair round cannot recurse on its own announcement", async () => {
  const d = driver({ web: true, rounds: [
    { deltas: [content("I will now search for it."), done("stop")] },
    { deltas: [content("Let me look that up and get back to you."), done("stop")] },
  ] });
  await d.window.runCompletion(d.conv);
  const pending = d.conv.messages.filter((m) => /\[pending action\]/.test(String(m.content)));
  assert.equal(pending.length, 1, "a second announcement gets no second repair");
  assert.equal(d.completions().length, 2, "no third round: the repair is bounded to one");
});

test("announcement: a repair reply that emits a real call takes the normal web path", async () => {
  const d = driver({ web: true, rounds: [
    { deltas: [content("I will search for that now."), done("stop")] },
    { deltas: [content(WEB_CALL), done("stop")] },
    { deltas: [content("Answer from results."), done("stop")] },
  ] });
  d.window.confirmWebRequest = async () => true;
  await d.window.runCompletion(d.conv);
  assert.equal(d.calls.filter((c) => c.url === "/api/web/search").length, 1, "the repaired call ran");
  assert.equal(d.completions().length, 3);
  assert.equal(d.conv.messages[d.conv.messages.length - 1].content, "Answer from results.");
});

test("announcement: no repair when web access is off", async () => {
  const d = driver({ web: false, rounds: [
    { deltas: [content("I will now search the web and report back."), done("stop")] },
  ] });
  await d.window.runCompletion(d.conv);
  assert.equal(d.completions().length, 1);
  assert.ok(!d.conv.messages.some((m) => /\[pending action\]/.test(String(m.content))));
});

test("announcement: no repair on a length cut-off (that is the truncation path)", async () => {
  const d = driver({ web: true, rounds: [
    { deltas: [content("I will now search the web and"), done("length")] },
  ] });
  await d.window.runCompletion(d.conv);
  assert.equal(d.completions().length, 1);
  assert.equal(d.conv.messages[1].truncated, true);
});

test("announcement: no repair on a failed generation (that is the failed path)", async () => {
  const d = driver({ web: true, rounds: [
    { deltas: [content("I will now search [inference error: boom]"), done("error")] },
  ] });
  await d.window.runCompletion(d.conv);
  assert.equal(d.completions().length, 1);
  assert.equal(d.conv.messages[1].failed, true);
});

test("looksLikeActionAnnouncement: positives and negatives", () => {
  const { window: w } = loadApp();
  const yes = [
    "I will now search the web and report back.",
    "Let me look that up for you.",
    "I'll check the latest version and get back to you.",
    "Ich werde das jetzt im Internet nachschauen.",
    "I am going to fetch that page now.",
  ];
  const no = [
    "The capital of France is Paris.",
    "I searched and found https://example.com/ which says 42.",
    WEB_CALL,
    "I will send you the summary below:\n\n" + "x".repeat(700),
    "You can search the web yourself at any time.",
    "",
  ];
  for (const s of yes) assert.equal(w.looksLikeActionAnnouncement(s), true, `should match: ${s}`);
  for (const s of no) assert.equal(w.looksLikeActionAnnouncement(s), false, `should not match: ${s.slice(0, 40)}`);
});
