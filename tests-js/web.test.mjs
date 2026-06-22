// SPDX-License-Identifier: AGPL-3.0-or-later
// Chat web access + honesty floor. The model must call the web tools (robustly,
// tolerating the formats local models actually emit) instead of hallucinating,
// and when web access is off it must be told plainly it is offline so it does
// not fabricate current facts or pretend it looked something up.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const jsonResp = (obj) => ({
  ok: true, status: 200, json: async () => obj, text: async () => JSON.stringify(obj),
});

/** A fetch stub that records every call and answers the web endpoints. */
function recordingFetch(webResults) {
  const calls = [];
  const impl = async (url, opts = {}) => {
    let body = null;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch (e) { body = opts.body; }
    calls.push({ url: String(url), body });
    if (String(url) === "/api/web/search") return jsonResp({ query: "q", results: webResults });
    if (String(url) === "/api/web/fetch")
      return jsonResp({ url: "https://example.com/", text: "page text", truncated: false });
    return jsonResp({});   // /v1/chat/completions and everything else
  };
  return { impl, calls };
}

/** One streamed turn: content tokens then a stop chunk. */
const content = (s) => [
  { choices: [{ delta: { content: s } }] },
  { choices: [{ delta: {}, finish_reason: "stop" }] },
];

/** Drive runCompletion with a queue of streamed rounds (one per recursion). */
async function runChat({ web, rounds, webResults = [{ title: "T", url: "https://example.com/", snippet: "S" }] }) {
  const { impl, calls } = recordingFetch(webResults);
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  const queue = rounds.slice();
  window.readSSE = async (_r, onData) => {
    const deltas = queue.shift() || [{ choices: [{ delta: {}, finish_reason: "stop" }] }];
    for (const d of deltas) onData(JSON.stringify(d));
  };
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;       // isolate the system prompt to the floor
  doc.getElementById("p-web").checked = !!web;
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  await window.runCompletion(conv);
  const completions = calls.filter((c) => c.url === "/v1/chat/completions");
  return { window, conv, calls, completions };
}

const systemOf = (completion) =>
  (completion.body.messages.find((m) => m.role === "system") || {}).content || "";

// parseWebCall returns objects created in the jsdom realm (a different
// Object.prototype), so deepStrictEqual rejects them as not reference-equal.
// Compare by value via JSON instead.
const eq = (out, expected) => assert.equal(JSON.stringify(out), JSON.stringify(expected));

// ---------------------------------------------------------------------------
//  parseWebCall: tolerate the formats local models actually emit
// ---------------------------------------------------------------------------

test("parseWebCall: canonical web_search and fetch_url", () => {
  const { window: w } = loadApp();
  eq(
    w.parseWebCall('<tool_call>{"name": "web_search", "args": {"query": "x"}}</tool_call>'),
    { name: "web_search", args: { query: "x" } });
  eq(
    w.parseWebCall('<tool_call>{"name": "fetch_url", "args": {"url": "https://e/"}}</tool_call>'),
    { name: "fetch_url", args: { url: "https://e/" } });
});

test("parseWebCall: mangled <|tool_call|> finetune wrapper", () => {
  const { window: w } = loadApp();
  const out = w.parseWebCall('<|tool_call|>{"name": "web_search", "args": {"query": "x"}}<|tool_call|>');
  eq(out, { name: "web_search", args: { query: "x" } });
});

test("parseWebCall: Gemma native form with the name in a call: prefix", () => {
  const { window: w } = loadApp();
  const out = w.parseWebCall('<|tool_call>call:web_search{"query": "weather"}<tool_call|>');
  eq(out, { name: "web_search", args: { query: "weather" } });
});

test("parseWebCall: ```json and ```tool_call fences", () => {
  const { window: w } = loadApp();
  eq(
    w.parseWebCall('```json\n{"name": "web_search", "args": {"query": "x"}}\n```'),
    { name: "web_search", args: { query: "x" } });
  eq(
    w.parseWebCall('```tool_call\n{"name": "fetch_url", "args": {"url": "https://e/"}}\n```'),
    { name: "fetch_url", args: { url: "https://e/" } });
});

test("parseWebCall: bare top-level JSON with no wrapper", () => {
  const { window: w } = loadApp();
  const out = w.parseWebCall('Sure, let me look that up.\n{"name": "web_search", "args": {"query": "x"}}');
  eq(out, { name: "web_search", args: { query: "x" } });
});

test("parseWebCall: trailing comma, single-quoted keys, and the arguments alias", () => {
  const { window: w } = loadApp();
  eq(
    w.parseWebCall('<tool_call>{"name": "web_search", "args": {"query": "x"},}</tool_call>'),
    { name: "web_search", args: { query: "x" } });
  eq(
    w.parseWebCall(`<tool_call>{'name': "web_search", 'args': {'query': "x"}}</tool_call>`),
    { name: "web_search", args: { query: "x" } });
  eq(
    w.parseWebCall('<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>'),
    { name: "web_search", args: { query: "x" } });
});

test("parseWebCall: a non-web tool name is not treated as a web call", () => {
  const { window: w } = loadApp();
  assert.equal(w.parseWebCall('<tool_call>{"name": "read_file", "args": {"path": "x"}}</tool_call>'), null);
});

test("parseWebCall: plain prose returns null", () => {
  const { window: w } = loadApp();
  assert.equal(w.parseWebCall("I cannot access the internet, so I am not sure."), null);
});

test("parseWebCall: a call only inside <think> is ignored (acts on the answer channel)", () => {
  const { window: w } = loadApp();
  const text = '<think><tool_call>{"name": "web_search", "args": {"query": "x"}}</tool_call></think>Here is my answer.';
  assert.equal(w.parseWebCall(text), null);
});

// ---------------------------------------------------------------------------
//  looksLikeWebToolAttempt: catch a botched call so we can re-prompt
// ---------------------------------------------------------------------------

test("looksLikeWebToolAttempt: true for a broken wrapper, false for clean prose", () => {
  const { window: w } = loadApp();
  assert.equal(w.looksLikeWebToolAttempt("<tool_call>{name: web_search}</tool_call>"), true);
  assert.equal(w.looksLikeWebToolAttempt('{"name": "web_search", broken'), true);
  assert.equal(w.looksLikeWebToolAttempt("Here is a normal answer with no tools."), false);
});

// ---------------------------------------------------------------------------
//  Honesty floor in the system prompt
// ---------------------------------------------------------------------------

test("web OFF: the model is told it is offline and must not fabricate", async () => {
  const { completions } = await runChat({ web: false, rounds: [content("hello")] });
  const sys = systemOf(completions[0]);
  assert.match(sys, /NO internet access/);
  assert.match(sys, /Never claim you looked something up/);
});

test("web ON: the model is taught the tools and the honesty rule", async () => {
  const { completions } = await runChat({ web: true, rounds: [content("hello")] });
  const sys = systemOf(completions[0]);
  assert.match(sys, /access the internet through tools/);
  assert.match(sys, /never invent search results/i);
});

// ---------------------------------------------------------------------------
//  End-to-end: a lenient call actually runs the tool; a botched one re-prompts
// ---------------------------------------------------------------------------

test("web ON: a mangled tool call still runs the real search", async () => {
  const { conv, calls, completions } = await runChat({
    web: true,
    rounds: [
      content('<|tool_call>call:web_search{"query": "weather today"}<tool_call|>'),
      content("It is sunny. Source: https://example.com/"),
    ],
  });
  assert.equal(calls.filter((c) => c.url === "/api/web/search").length, 1,
    "the web search endpoint was actually called");
  assert.ok(conv.messages.some((m) => m.web && /Results of web_search/.test(String(m.content))),
    "search results were injected back into the conversation");
  assert.equal(completions.length, 2, "the model continued after the results arrived");
});

test("web ON: a botched tool call triggers a re-prompt instead of an un-grounded answer", async () => {
  const { conv, completions } = await runChat({
    web: true,
    rounds: [
      content("<tool_call>{name: web_search, args: {query: x}}</tool_call>"),
      content("Sunny, per the search."),
    ],
  });
  assert.ok(conv.messages.some((m) => /\[tool-call format\]/.test(String(m.content))),
    "the model was asked to re-emit the tool call");
  assert.equal(completions.length, 2, "the model got a second chance to call the tool");
});

test("web OFF: a tool-shaped reply is NOT intercepted (no web rounds run)", async () => {
  const { calls, completions } = await runChat({
    web: false,
    rounds: [content('<tool_call>{"name": "web_search", "args": {"query": "x"}}</tool_call>')],
  });
  assert.equal(calls.filter((c) => c.url === "/api/web/search").length, 0);
  assert.equal(completions.length, 1, "no repair / web loop when web access is off");
});

// Regression: the explicit /web command runs a real search even with the toggle
// off (it is direct user consent). The answering turn must then be told to USE
// those results, not handed the offline-denial floor - which would contradict
// the real results sitting in the conversation and make the model deny them.
test("/web with the toggle OFF: real search runs and the answer is grounded, not denied", async () => {
  const { impl, calls } = recordingFetch([{ title: "T", url: "https://example.com/", snippet: "fresh fact" }]);
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  window.readSSE = async (_r, onData) => {
    for (const d of content("answer")) onData(JSON.stringify(d));
  };
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;          // standing toggle OFF

  await window.runWebInChat("price of bitcoin today");

  // A real outbound search fired and its results were injected.
  assert.equal(calls.filter((c) => c.url === "/api/web/search").length, 1);
  assert.ok(window.currentConv().messages.some(
    (m) => m.web && /Results of web_search/.test(String(m.content))),
    "fresh results are in the conversation");

  // The answering turn is grounded, NOT told it is offline.
  const completions = calls.filter((c) => c.url === "/v1/chat/completions");
  const sys = (completions[0].body.messages.find((m) => m.role === "system") || {}).content || "";
  assert.match(sys, /Web search results have been provided/);
  assert.doesNotMatch(sys, /NO internet access/);
});
