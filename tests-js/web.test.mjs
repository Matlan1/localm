// SPDX-License-Identifier: AGPL-3.0-or-later
// Chat web access + honesty floor. The model must call the web tools (robustly,
// tolerating the formats local models actually emit) instead of hallucinating,
// and when web access is off it must be told plainly it is offline so it does
// not fabricate current facts or pretend it looked something up.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";
import { readFile } from "node:fs/promises";

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
async function runChat({ web, rounds, webResults = [{ title: "T", url: "https://example.com/", snippet: "S" }],
                          grammar = "" }) {
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
  doc.getElementById("p-grammar").value = grammar;
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
//  LM-DA-014: requestWebTool fences server-returned content as untrusted, so
//  the model is told (in-band) that it is data, not instructions - the server
//  side (web/plug.py) already defangs literal control tokens; this is the
//  complementary client-side framing layer, matching the coder plugin's own
//  provenance.py treatment of fetch_url/web_search output.
// ---------------------------------------------------------------------------

test("requestWebTool: search results are wrapped in the untrusted_content fence", async () => {
  const { impl } = recordingFetch([{ title: "T", url: "https://example.com/", snippet: "S" }]);
  const { window: w } = loadApp({ fetchImpl: impl });
  const note = await w.requestWebTool({ name: "web_search", args: { query: "x" } });
  assert.match(note, /<untrusted_content>[\s\S]*T[\s\S]*<\/untrusted_content>/);
  assert.match(note, /UNTRUSTED EXTERNAL CONTENT/);
});

test("requestWebTool: fetched page text is wrapped in the untrusted_content fence", async () => {
  const { impl } = recordingFetch([]);
  const { window: w } = loadApp({ fetchImpl: impl });
  const note = await w.requestWebTool({ name: "fetch_url", args: { url: "https://example.com/" } });
  assert.match(note, /<untrusted_content>\npage text\n<\/untrusted_content>/);
  assert.match(note, /UNTRUSTED EXTERNAL CONTENT/);
});

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
  assert.match(sys, /no internet access/i);
  assert.match(sys, /Never claim to have searched/);
});

test("web ON: the model is taught the tools and the honesty rule", async () => {
  const { completions } = await runChat({ web: true, rounds: [content("hello")] });
  const sys = systemOf(completions[0]);
  assert.match(sys, /access the internet through tools/);
  assert.match(sys, /never invent search results/i);
});

// ---------------------------------------------------------------------------
//  Grammar-constrained tool calls: the system prompt above ASKS for the
//  <tool_call>{"name":...,"args":{...}}</tool_call> protocol; these pin that a
//  lazy GBNF grammar also ENFORCES it once web access is on.
// ---------------------------------------------------------------------------

test("web ON: the request is grammar-constrained for tool calls", async () => {
  const { completions, window } = await runChat({ web: true, rounds: [content("hello")] });
  // TOOL_CALLS_ONLY/TOOL_CALL_TRIGGER are top-level const in the injected
  // settings-perf.js classic script - part of the jsdom realm's shared global
  // lexical environment, not a window property and not reachable from this
  // Node module scope directly. Bridge them out the same way the harness's
  // own runScript doc prescribes for reading realm-local state.
  runScript(window, "window.__gbnf = { TOOL_CALLS_ONLY, TOOL_CALL_TRIGGER };");
  const { TOOL_CALLS_ONLY, TOOL_CALL_TRIGGER } = window.__gbnf;
  assert.ok(completions[0].body.grammar, "no grammar was sent");
  assert.equal(completions[0].body.grammar_lazy, true);
  assert.deepEqual(completions[0].body.grammar_triggers, [TOOL_CALL_TRIGGER]);
  assert.equal(completions[0].body.grammar, TOOL_CALLS_ONLY);
});

test("web OFF: no grammar is sent (nothing taught, nothing to enforce)", async () => {
  const { completions } = await runChat({ web: false, rounds: [content("hello")] });
  assert.ok(!("grammar" in completions[0].body));
  assert.ok(!("grammar_lazy" in completions[0].body));
  assert.ok(!("grammar_triggers" in completions[0].body));
});

test("web ON: an explicit persona grammar overrides the web-tool grammar", async () => {
  const { completions } = await runChat({
    web: true, rounds: [content("hello")], grammar: "root ::= \"ok\"",
  });
  assert.equal(completions[0].body.grammar, "root ::= \"ok\"");
  assert.ok(!("grammar_lazy" in completions[0].body),
    "the web-tool lazy/triggers pair must not ride along with a persona's own grammar");
  assert.ok(!("grammar_triggers" in completions[0].body));
});

test("web ON: a backend that refuses the grammar falls back to unconstrained and stops asking", async () => {
  const detail = "This model cannot constrain generation to a grammar, so the "
    + "requested grammar would be ignored and the reply would not match it.";
  let chatCalls = 0;
  const calls = [];
  const impl = async (url, opts = {}) => {
    let body = null;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch (e) { body = opts.body; }
    calls.push({ url: String(url), body });
    if (String(url) === "/v1/chat/completions") {
      chatCalls += 1;
      if (chatCalls === 1) {
        return { ok: false, status: 400, json: async () => ({ detail }),
                 text: async () => JSON.stringify({ detail }) };
      }
    }
    return jsonResp({});
  };
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  const queue = [content("2 + 2 = 4."), content("second turn, plain answer")];
  window.readSSE = async (_r, onData) => {
    const deltas = queue.shift() || [{ choices: [{ delta: {}, finish_reason: "stop" }] }];
    for (const d of deltas) onData(JSON.stringify(d));
  };
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = true;

  await window.runCompletion({ id: "c1", title: "t", messages: [{ role: "user", content: "2+2?" }] });
  await window.runCompletion({
    id: "c1", title: "t",
    messages: [{ role: "user", content: "2+2?" }, { role: "assistant", content: "2 + 2 = 4." },
               { role: "user", content: "and again?" }],
  });

  const completions = calls.filter((c) => c.url === "/v1/chat/completions");
  assert.equal(completions.length, 3,
    "turn 1: refused attempt + unconstrained retry; turn 2: no grammar attempt at all");
  assert.equal(completions[0].body.grammar_lazy, true, "the first attempt asked for the grammar");
  assert.ok(!("grammar" in completions[1].body), "the retry omitted the grammar entirely");
  assert.ok(!("grammar" in completions[2].body),
    "a later turn must not repeat a grammar this backend already refused");
});

test("web ON: the periodic /v1/config poll (refreshCtxLimit) does not resurrect a refused grammar", async () => {
  // chat.toolGrammar mirrors the server's config PREFERENCE and is refreshed
  // by refreshCtxLimit on a 30s poll for the tab's whole lifetime (chat.js).
  // chat.toolGrammarUnsupported is a separate, sticky RUNTIME fact about this
  // backend. A poll landing after a refusal must not undo the second because
  // it refreshed the first - proving that needs an ACTUAL poll call, not just
  // the accidental race the fallback test above happens to exercise.
  const impl = async (url) => {
    if (String(url) === "/v1/config") return jsonResp({ chat_tool_grammar: true });
    return jsonResp({});
  };
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, "chat.toolGrammarUnsupported = true;");

  await window.refreshCtxLimit();

  runScript(window, "window.__unsupported = chat.toolGrammarUnsupported; window.__pref = chat.toolGrammar;");
  assert.equal(window.__pref, true, "the poll DID refresh the preference from config");
  assert.equal(window.__unsupported, true,
    "a config poll must never clear the runtime refusal latch");
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
  assert.match(sys, /Web results were just provided/);
  assert.doesNotMatch(sys, /no internet access/i);
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
//  NEW-WEBSEARCH-UX (1): search returns SNIPPETS, so the model must be told to
//  read a promising result before answering. Without this it answers from the
//  search engine's summary and never opens the page.
// ---------------------------------------------------------------------------

test("web ON: the model is nudged to follow up with fetch_url on a promising result", async () => {
  const { completions } = await runChat({ web: true, rounds: [content("hello")] });
  const sys = systemOf(completions[0]);
  assert.match(sys, /follow up with fetch_url/i,
    "the prompt states the capability but never tells the model to USE it");
  assert.match(sys, /snippets, not page text/i,
    "the model needs the REASON, or it has no way to judge when to follow up");
});

test("web ON: the model is told to emit exactly ONE tool call per reply", async () => {
  // The one-call-per-message design is enforced ONLY by this sentence - there is
  // no grammar constraint on this surface - so the wording is the mechanism, not
  // documentation of it. Added after a fires-control found the JS suite had no
  // test for it at all: only the Python cross-surface drift guard did, which
  // would have let a GUI-side deletion through if that guard were ever removed.
  const { completions } = await runChat({ web: true, rounds: [content("hello")] });
  assert.match(systemOf(completions[0]), /ONLY ONE tool call/);
});

// ---------------------------------------------------------------------------
//  NEW-WEBSEARCH-UX (3): one call per message is the design, but the extras
//  used to vanish in silence. formatToolCalls renders EVERY block, so the user
//  watched two lookups happen when only one did.
// ---------------------------------------------------------------------------

const twoCalls = content(
  '<tool_call>{"name": "web_search", "args": {"query": "weather"}}</tool_call>\n' +
  '<tool_call>{"name": "fetch_url", "args": {"url": "https://example.com/b"}}</tool_call>');

test("parseWebCalls: a single call is ONE call - the bare-JSON layer must not re-count it", () => {
  const { window: w } = loadApp();
  // The JSON inside a wrapper/fence IS also a bare top-level object. If the
  // last-resort layer ran unconditionally, every ordinary reply would look
  // like two calls and the model would be told, every turn, that a second call
  // it never made had been ignored.
  assert.equal(w.parseWebCalls('<tool_call>{"name": "web_search", "args": {"query": "x"}}</tool_call>').length, 1);
  assert.equal(w.parseWebCalls('```json\n{"name": "web_search", "args": {"query": "x"}}\n```').length, 1);
  assert.equal(w.parseWebCalls('{"name": "web_search", "args": {"query": "x"}}').length, 1);
  assert.equal(w.parseWebCalls("plain prose, no call at all").length, 0);
});

test("parseWebCalls: two calls are both reported, and parseWebCall still returns only the first", () => {
  const { window: w } = loadApp();
  const text = '<tool_call>{"name": "web_search", "args": {"query": "a"}}</tool_call>' +
               '<tool_call>{"name": "fetch_url", "args": {"url": "https://e/"}}</tool_call>';
  const all = w.parseWebCalls(text);
  assert.equal(all.length, 2);
  assert.equal(all[0].name, "web_search");
  assert.equal(all[1].name, "fetch_url");
  eq(w.parseWebCall(text), { name: "web_search", args: { query: "a" } });
  // limit stops the scan early without changing which call comes first
  assert.equal(w.parseWebCalls(text, 1).length, 1);
});

test("web ON: a second tool call in one reply is reported as ignored, not silently dropped", async () => {
  const { conv, calls } = await runChat({
    web: true,
    rounds: [twoCalls, content("It is sunny. Source: https://example.com/")],
  });
  assert.equal(calls.filter((c) => c.url === "/api/web/search").length, 1,
    "the first call ran");
  assert.equal(calls.filter((c) => c.url === "/api/web/fetch").length, 0,
    "the second call did NOT run - one call per message is the retained design");
  const note = conv.messages.find(
    (m) => m.web && /only the first tool call ran/.test(String(m.content)));
  assert.ok(note, "the model was never told its second call was ignored");
  assert.match(String(note.content), /fetch_url/,
    "the notice must name what was ignored, not just that something was");
  assert.match(String(note.content), /Results of web_search/,
    "the notice rides on the result message, keeping user/assistant alternation");
  // LM-DA-014: everything inside the fence is DATA the model is told not to obey.
  // A notice that landed in there would be self-defeating - it is our instruction,
  // not fetched content - and "present in the message" cannot tell the two apart.
  const body = String(note.content);
  assert.ok(body.indexOf("only the first tool call ran") > body.lastIndexOf("</untrusted_content>"),
    "the notice must sit OUTSIDE the untrusted-content fence");
});

test("web ON: an ordinary ONE-call reply gets no ignored-call notice", async () => {
  const { conv } = await runChat({
    web: true,
    rounds: [searchCall("weather"), content("It is sunny.")],
  });
  assert.ok(!conv.messages.some((m) => /only the first tool call ran/.test(String(m.content))),
    "a single call must never be reported as though a second was dropped");
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

// ---------------------------------------------------------------------------
//  Chat defaults vs per-chat override: a blank drawer System prompt inherits the
//  Settings "Default system prompt" (chat.systemDefault); a set field overrides.
// ---------------------------------------------------------------------------

async function systemForSend({ drawerSystem, settingsDefault }) {
  const { impl, calls } = recordingFetch([]);
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  window.readSSE = async (_r, onData) =>
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }));
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;   // isolate the system message
  doc.getElementById("p-web").checked = false;
  doc.getElementById("p-system").value = drawerSystem;
  // chat.systemDefault is set from /v1/config; seed it directly (shared realm global).
  runScript(window, `chat.systemDefault = ${JSON.stringify(settingsDefault)};`);
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  await window.runCompletion(conv);
  const completion = calls.find((c) => c.url === "/v1/chat/completions");
  return (completion.body.messages.find((m) => m.role === "system") || {}).content || "";
}

test("a blank System prompt inherits the Settings default system prompt", async () => {
  const sys = await systemForSend({ drawerSystem: "", settingsDefault: "You are a terse pirate." });
  assert.match(sys, /terse pirate/, "the Settings default was used when the drawer is blank");
});

test("a set System prompt overrides the Settings default (not both)", async () => {
  const sys = await systemForSend({
    drawerSystem: "You are a helpful librarian.",
    settingsDefault: "You are a terse pirate.",
  });
  assert.match(sys, /helpful librarian/, "the drawer System prompt is used");
  assert.ok(!/pirate/.test(sys), "the Settings default is NOT also injected");
});

// A SEARCH THE USER DID NOT ASK FOR IS A FAILURE, not a harmless extra step.
//
// Reported live 2026-08-14: "Greet my friend Memo, who is watching right now"
// produced a web_search for "greeting messages", and the reply was a list of
// greeting-card websites instead of a greeting. The prompt told the model when to
// search ("current or uncertain info ... instead of guessing") and never once told
// it when NOT to, so every instruction in it pushed one way.
//
// Asserts the BOUNDARY exists and names the everyday cases, rather than asserting
// the exact wording, so the sentence can be reworded without breaking this.
test("the web tool prompt tells the model when NOT to search", async () => {
  // WEB_TOOL_PROMPT is an ES export and settings-perf.js touches `window` at
  // import time, so neither a bare import nor the classic-script harness reaches
  // it. The subject here is the SHIPPED TEXT, so read the declaration itself.
  const src = await readFile(
    new URL("../localm/plugins/gui/static/app/settings-perf.js", import.meta.url),
    "utf8");
  // \r?\n, not \n: this file is CRLF on disk and Node's readFile does not
  // normalise line endings the way a Python text read does.
  const m = src.match(/export const WEB_TOOL_PROMPT\s*=([\s\S]*?);\r?\n/);
  assert.ok(m, "WEB_TOOL_PROMPT declaration not found - did it move or get renamed?");
  const p = m[1];
  assert.ok(p.length > 200, "the prompt body looks truncated");

  assert.match(p, /do not search/i,
    "the prompt must state a negative boundary, not only when to search");
  assert.match(p, /greet/i,
    "greeting someone is the reported case and must be named as a do-not-search example");
  for (const kind of [/writ/i, /translat/i, /summaris|summariz/i]) {
    assert.match(p, kind,
      `an everyday no-search task is missing from the boundary: ${kind}`);
  }
  // The positive instruction has to survive - this must not turn into "never search".
  assert.match(p, /web_search/,
    "the search tool must still be offered");
  assert.match(p, /current or uncertain/i,
    "the reason TO search must remain");
});
