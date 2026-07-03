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

// ---------------------------------------------------------------------------
//  R36: the web loop must not spin - dedupe repeats, force a final answer
// ---------------------------------------------------------------------------

const searchCall = (q) =>
  content(`<tool_call>{"name": "web_search", "args": {"query": "${q}"}}</tool_call>`);

test("R36: a repeated identical search is not re-run; the model is told to answer", async () => {
  const { conv, calls, completions } = await runChat({
    web: true,
    rounds: [
      searchCall("weather today"),
      searchCall("weather today"),   // the loop: same query again
      content("It is sunny. Source: https://example.com/"),
    ],
  });
  assert.equal(calls.filter((c) => c.url === "/api/web/search").length, 1,
    "the duplicate search was NOT re-issued");
  assert.ok(conv.messages.some((m) => /\[duplicate web request\]/.test(String(m.content))),
    "the model was told it already searched and to answer from the results");
  assert.ok(conv.messages.some((m) => m.role === "assistant" && /sunny/i.test(String(m.content))),
    "the model produced a final answer");
});

test("R36: when web rounds run out the model is forced to answer, not left mid-search", async () => {
  const { conv, calls } = await runChat({
    web: true,
    rounds: [
      searchCall("q1"), searchCall("q2"), searchCall("q3"),
      searchCall("q4"),                       // a 4th attempt past the cap
      content("Final synthesized answer. Source: https://example.com/"),
    ],
  });
  assert.equal(calls.filter((c) => c.url === "/api/web/search").length, 3,
    "exactly WEB_MAX_ROUNDS searches ran, no more");
  assert.ok(conv.messages.some((m) => /\[web search limit reached\]/.test(String(m.content))),
    "the model was told to stop searching and answer");
  assert.ok(conv.messages.some((m) => m.role === "assistant" &&
    /Final synthesized answer/.test(String(m.content))),
    "the conversation ends on a synthesized answer, not a tool call");
});

// ---------------------------------------------------------------------------
//  WEB-ask: net_mode=ask must APPROVE each model-initiated web request
// ---------------------------------------------------------------------------

function askFetch(netMode, webResults = [{ title: "T", url: "https://example.com/", snippet: "S" }]) {
  const calls = [];
  const impl = async (url, opts = {}) => {
    let body = null;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch (e) { body = opts.body; }
    calls.push({ url: String(url), body });
    if (String(url) === "/v1/config") return jsonResp({ net_mode: netMode });
    if (String(url) === "/api/web/search") return jsonResp({ query: "q", results: webResults });
    if (String(url) === "/api/web/fetch")
      return jsonResp({ url: "https://example.com/", text: "page text", truncated: false });
    return jsonResp({});
  };
  return { impl, calls };
}

async function runAsk({ netMode, approve, rounds }) {
  const { impl, calls } = askFetch(netMode);
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  const prompts = [];
  // Auto-answer the approval dialog instead of opening the real modal.
  window.confirmWebRequest = (call) => { prompts.push(call); return Promise.resolve(approve); };
  const queue = rounds.slice();
  window.readSSE = async (_r, onData) => {
    const deltas = queue.shift() || [{ choices: [{ delta: {}, finish_reason: "stop" }] }];
    for (const d of deltas) onData(JSON.stringify(d));
  };
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = true;
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  await window.runCompletion(conv);
  return { window, conv, calls, prompts };
}

test("WEB-ask: net_mode=ask prompts before a model-initiated search; approve runs it", async () => {
  const { calls, prompts } = await runAsk({
    netMode: "ask", approve: true,
    rounds: [
      content('<tool_call>{"name": "web_search", "args": {"query": "weather"}}</tool_call>'),
      content("It is sunny."),
    ],
  });
  assert.equal(prompts.length, 1, "the user was asked to approve the request");
  assert.equal(calls.filter((c) => c.url === "/api/web/search").length, 1, "approved -> the search ran");
});

test("WEB-ask: net_mode=ask + deny does NOT search and tells the model", async () => {
  const { conv, calls, prompts } = await runAsk({
    netMode: "ask", approve: false,
    rounds: [
      content('<tool_call>{"name": "web_search", "args": {"query": "weather"}}</tool_call>'),
      content("I could not look that up."),
    ],
  });
  assert.equal(prompts.length, 1, "the user was asked");
  assert.equal(calls.filter((c) => c.url === "/api/web/search").length, 0, "denied -> NO search ran");
  assert.ok(conv.messages.some((m) => /\[web access denied\]/.test(String(m.content))),
    "the model was told the request was declined");
});

test("WEB-ask: net_mode=allow does NOT prompt (current behaviour preserved)", async () => {
  const { calls, prompts } = await runAsk({
    netMode: "allow", approve: true,
    rounds: [
      content('<tool_call>{"name": "web_search", "args": {"query": "x"}}</tool_call>'),
      content("done"),
    ],
  });
  assert.equal(prompts.length, 0, "allow mode never prompts");
  assert.equal(calls.filter((c) => c.url === "/api/web/search").length, 1, "the search ran without a prompt");
});

// ---------------------------------------------------------------------------
//  R27: "don't ask again this session" on the web-access popup
// ---------------------------------------------------------------------------

test("R27: ticking 'don't ask again' stops the approval popup re-firing", async () => {
  const { window } = loadApp();
  const doc = window.document;
  const modal = doc.getElementById("modal");

  // First request opens the real modal; tick remember, then Allow.
  const p1 = window.confirmWebRequest({ name: "web_search", args: { query: "x" } });
  assert.notEqual(modal.style.display, "none", "the approval modal opened");
  const cb = modal.querySelector(".web-ask-remember input[type=checkbox]");
  assert.ok(cb, "the remember checkbox is present");
  cb.checked = true;
  const allow = [...modal.querySelectorAll("button")].find((b) => b.textContent === "Allow");
  allow.click();
  assert.equal(await p1, true, "Allow resolves true");
  assert.equal(modal.style.display, "none", "the modal closed");

  // A later request in the same session is auto-approved WITHOUT reopening.
  modal.style.display = "none";
  const p2 = window.confirmWebRequest({ name: "web_search", args: { query: "y" } });
  assert.equal(modal.style.display, "none", "the modal did not reopen");
  assert.equal(await p2, true, "the remembered choice auto-approved");
});

// ---------------------------------------------------------------------------
//  CHAT-TOOL-1: defang EVERY tool-call dialect parseWebCall executes, in the
//  display AND in the context re-sent to the model. A model must never see its
//  own raw <|tool_call> control tokens echoed back - that destabilised some
//  finetunes (a Gemma-4 aeon-abliterated build) into a repetition loop.
// ---------------------------------------------------------------------------

test("formatToolCalls defangs the |-piped / call:-prefixed dialect (not just <tool_call>)", () => {
  const { window: w } = loadApp();
  // The exact shape the reported model emitted (piped wrapper + call: prefix).
  const piped = '<|tool_call>call:{"name": "web_search", "args": {"query": "privacy X"}}<|tool_call|>';
  const out = w.formatToolCalls(piped);
  assert.ok(!/tool_call/.test(out), "no raw tool_call marker survives the defang");
  assert.match(out, /web search: "privacy X"/, "shows a readable note with the query");
  // Canonical form still works, and plain prose is untouched.
  assert.match(w.formatToolCalls('<tool_call>{"name":"fetch_url","args":{"url":"https://x"}}</tool_call>'),
    /read page: https:\/\/x/);
  assert.equal(w.formatToolCalls("just a normal answer"), "just a normal answer");
});

test("CHAT-TOOL-1: the re-sent context defangs the assistant tool-call turn (no raw markers to the model)", async () => {
  const piped = '<|tool_call>call:{"name": "web_search", "args": {"query": "privacy"}}<|tool_call|>';
  const { completions } = await runChat({
    web: true,
    rounds: [content(piped), content("Here is the grounded answer [1].")],
  });
  assert.ok(completions.length >= 2, "the web loop re-completed after running the search");
  // The FINAL (answer) turn's messages must carry the earlier tool-call turn as a
  // clean note, never the raw <|tool_call> tokens the model originally emitted.
  const answerMsgs = completions[completions.length - 1].body.messages;
  const asst = answerMsgs.find((m) => m.role === "assistant");
  assert.ok(asst, "the assistant tool-call turn is present in the re-sent context");
  assert.ok(!/tool_call/.test(String(asst.content)),
    "raw <|tool_call> markers are NOT re-fed to the model");
  assert.match(String(asst.content), /web search/,
    "the tool call is represented as a readable note instead");
});
