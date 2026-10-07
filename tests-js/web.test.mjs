// SPDX-License-Identifier: AGPL-3.0-or-later
// Chat web access + honesty floor. The model must call the web tools (robustly,
// tolerating the formats local models actually emit) instead of hallucinating,
// and when web access is off it must be told plainly it is offline so it does
// not fabricate current facts or pretend it looked something up.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";
import { bundleOf } from "./web-fixtures.mjs";
import { readFile } from "node:fs/promises";

const jsonResp = (obj) => ({
  ok: true, status: 200, json: async () => obj, text: async () => JSON.stringify(obj),
});

/** A fetch stub that records every call and answers the web endpoints.
 *  *webResults* feeds /api/web/retrieve (see bundleOf); pass a function to
 *  answer it with an arbitrary bundle. */
function recordingFetch(webResults) {
  const calls = [];
  const impl = async (url, opts = {}) => {
    let body;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = opts.body; }
    calls.push({ url: String(url), body, signal: opts.signal });
    if (String(url) === "/api/web/retrieve") {
      const q = (body && body.query) || "q";
      return jsonResp(typeof webResults === "function" ? webResults(q) : bundleOf(q, webResults));
    }
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
                          grammar = "", setup = null, history = null, fetchImpl = null }) {
  const rec = recordingFetch(webResults);
  const calls = rec.calls;
  const impl = fetchImpl ? fetchImpl(rec) : rec.impl;
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
  const conv = { id: "c1", title: "t",
                 messages: history ? history.slice() : [{ role: "user", content: "hi" }] };
  window.__testConv = conv;
  runScript(window, "chat.conversations = [window.__testConv]; chat.activeId = window.__testConv.id;");
  if (setup) await setup(window);
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

test("requestWebTool: web_search runs the retrieval controller and returns a completed search event", async () => {
  const { impl, calls } = recordingFetch([
    { title: "T", url: "https://example.com/", snippet: "S", page: "PAGE TEXT of T" }]);
  const { window: w } = loadApp({ fetchImpl: impl });
  const ev = await w.requestWebTool({ name: "web_search", args: { query: "x" } });
  const webCalls = calls.filter((c) => c.url.startsWith("/api/web/"));
  assert.deepEqual(webCalls.map((c) => c.url), ["/api/web/retrieve"],
    "web_search goes to the retrieval route, not the raw snippet search");
  assert.equal(webCalls[0].body.query, "x");
  assert.equal(ev.tool, "search");
  assert.equal(ev.status, "done");
  assert.equal(ev.query, "x");
  assert.equal(ev.grounding, "page-backed");
  assert.equal(ev.grounding_summary, "page-backed: 1 of 1 sources read");
  assert.equal(ev.sources.length, 1);
  assert.equal(ev.sources[0].id, "S1");
  assert.equal(ev.chunks.length, 1);
  assert.equal(ev.role, undefined, "a tool event carries no role");
  assert.equal(ev.content, undefined, "a tool event carries no pre-rendered content");
  // The fenced prompt text is rendered from the event, only at assembly time.
  const { content: note } = w.toolEventPrompt(ev);
  assert.match(note, /^\[Results of web_search "x"\] \(page-backed: 1 of 1 sources read\)\n/);
  assert.match(note, /<untrusted_content>[\s\S]*\[S1\] T - https:\/\/example\.com\/ \(page-backed\)[\s\S]*PAGE TEXT of T[\s\S]*<\/untrusted_content>/);
  assert.match(note, /UNTRUSTED EXTERNAL CONTENT/);
});

test("requestWebTool: a snippet-only bundle is labelled as such, never as read pages", async () => {
  const { impl } = recordingFetch([{ title: "T", url: "https://example.com/", snippet: "S" }]);
  const { window: w } = loadApp({ fetchImpl: impl });
  const ev = await w.requestWebTool({ name: "web_search", args: { query: "x" } });
  assert.equal(ev.grounding, "snippet-only");
  const { content: note } = w.toolEventPrompt(ev);
  assert.match(note, /^\[Results of web_search "x"\] \(snippet-only: no page was read, 1 search snippet only\)/);
  assert.doesNotMatch(note.split("<untrusted_content>")[0], /page-backed/);
});

test("requestWebTool: a provider failure inside the bundle takes the failure path", async () => {
  const { impl } = recordingFetch(() => bundleOf("x", [], {
    search_status: "failed", search_error: "RuntimeError: backend rate-limited",
    grounding: "failed", grounding_summary: "failed: no evidence, search failed",
    untrusted_fields: ["prompt_text", "search_error"],
  }));
  const { window: w } = loadApp({ fetchImpl: impl });
  await assert.rejects(
    w.requestWebTool({ name: "web_search", args: { query: "x" } }),
    /Search failed: RuntimeError: backend rate-limited/);
});

test("requestWebTool: fetched page text is stored on the event and fenced at assembly", async () => {
  const { impl } = recordingFetch([]);
  const { window: w } = loadApp({ fetchImpl: impl });
  const ev = await w.requestWebTool({ name: "fetch_url", args: { url: "https://example.com/" } });
  assert.equal(ev.tool, "fetch");
  assert.equal(ev.status, "done");
  assert.deepEqual(JSON.parse(JSON.stringify(ev.page)),
    { url: "https://example.com/", text: "page text", truncated: false });
  const { content: note } = w.toolEventPrompt(ev);
  assert.match(note, /^\[Content of https:\/\/example\.com\/\]\n/);
  assert.match(note, /<untrusted_content>\npage text\n<\/untrusted_content>/);
  assert.match(note, /UNTRUSTED EXTERNAL CONTENT/);
});

// ---------------------------------------------------------------------------
//  AUD-PROVDEFANG: the server declares WHICH response fields are remote-
//  controlled (untrusted_fields, web/plug.py); the client must turn that into
//  untrusted_spans on the outgoing request (Message.untrusted_spans,
//  protocol.py) so the backend tokenises exactly those ranges with special-
//  token parsing off - the same wire contract the coder path already ships
//  (localm/plugins/coder/backends/http.py:_with_untrusted_spans).
// ---------------------------------------------------------------------------

test("requestWebTool: web_search untrusted_spans cover the whole evidence body, never the header", async () => {
  const { impl } = recordingFetch([
    { title: "EVIL_TITLE", url: "https://example.com/page", snippet: "unused",
      page: "EVIL_PAGE_TEXT" },
    { title: "Other", url: "https://other.example/", snippet: "EVIL_SNIPPET" },
  ]);
  const { window: w } = loadApp({ fetchImpl: impl });
  const { content: note, untrusted_spans } = w.toolEventPrompt(
    await w.requestWebTool({ name: "web_search", args: { query: "x" } }));
  assert.equal(untrusted_spans.length, 1, "one span: the server-rendered evidence body");
  const [a, b] = untrusted_spans[0];
  const covered = note.slice(a, b);
  for (const remote of ["EVIL_TITLE", "EVIL_SNIPPET", "EVIL_PAGE_TEXT"]) {
    assert.ok(covered.includes(remote), `${remote} must sit inside the untrusted span`);
  }
  const trusted = note.slice(0, a) + note.slice(b);
  assert.match(trusted, /^\[Results of web_search "x"\] \(page-backed: 1 of 2 sources read\)/,
    "the header (query + grounding summary) is trusted framing");
  assert.doesNotMatch(trusted, /EVIL_/, "no remote text sits in the trusted region");
});

test("requestWebTool: fetch_url untrusted_spans cover exactly the fetched text", async () => {
  const impl = async (url) => {
    if (String(url) === "/api/web/fetch")
      return jsonResp({ url: "https://example.com/", text: "EVIL_PAGE_TEXT", truncated: false,
                         untrusted_fields: ["text"] });
    return jsonResp({});
  };
  const { window: w } = loadApp({ fetchImpl: impl });
  const { content: note, untrusted_spans } = w.toolEventPrompt(
    await w.requestWebTool({ name: "fetch_url", args: { url: "https://example.com/" } }));
  assert.equal(untrusted_spans.length, 1);
  const [a, b] = untrusted_spans[0];
  assert.equal(note.slice(a, b), "EVIL_PAGE_TEXT");
});

test("requestWebTool: a response with no untrusted_fields still fences the evidence as untrusted (never crashes)", async () => {
  const { impl } = recordingFetch(() => bundleOf("x",
    [{ title: "T", url: "https://example.com/", snippet: "S" }], { untrusted_fields: [] }));
  const { window: w } = loadApp({ fetchImpl: impl });
  const { content: note, untrusted_spans } = w.toolEventPrompt(
    await w.requestWebTool({ name: "web_search", args: { query: "x" } }));
  assert.equal(untrusted_spans.length, 1, "the evidence body is always one untrusted span");
  const [a, b] = untrusted_spans[0];
  assert.match(note.slice(a, b), /^Grounding: [\s\S]*\[S1\] T - https:\/\/example\.com\//);
  assert.match(note, /<untrusted_content>[\s\S]*T[\s\S]*<\/untrusted_content>/);
});

test("web ON: the search-result message sent to the model carries untrusted_spans over exactly the remote text", async () => {
  const { impl, calls } = recordingFetch([
    { title: "EVIL_TITLE", url: "https://example.com/page", snippet: "unused",
      page: "EVIL_PAGE" },
    { title: "Other", url: "https://other.example/", snippet: "EVIL_SNIPPET" },
  ]);
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  const queue = [
    content('<tool_call>{"name": "web_search", "args": {"query": "x"}}</tool_call>'),
    content("done"),
  ];
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

  const completions = calls.filter((c) => c.url === "/v1/chat/completions");
  const resultMsg = completions[1].body.messages.find(
    (m) => m.role === "user" && /Results of web_search/.test(m.content));
  assert.ok(resultMsg, "the search-result message was sent to the model");
  assert.ok(Array.isArray(resultMsg.untrusted_spans) && resultMsg.untrusted_spans.length,
    "untrusted_spans travelled over the wire");
  const covered = resultMsg.untrusted_spans.map(([a, b]) => resultMsg.content.slice(a, b)).join("");
  assert.ok(covered.includes("EVIL_TITLE"));
  assert.ok(covered.includes("EVIL_SNIPPET"));
  assert.ok(covered.includes("EVIL_PAGE"));
  assert.ok(!resultMsg.untrusted_spans.some(
    ([a, b]) => resultMsg.content.slice(a, b).includes("Results of web_search")),
    "the header must not be marked untrusted");
});

test("runCompletion: merging a plain user row into an untrusted-spans row shifts the spans correctly", async () => {
  const { impl, calls } = recordingFetch([]);
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  window.readSSE = async (_r, onData) =>
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }));
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;

  const prefix = "look at this doc first";
  const conv = {
    id: "c1", title: "t",
    messages: [
      { role: "user", content: prefix },
      // A persisted web-result tool event (here in the shape a legacy row is
      // migrated to: verbatim text plus its spans) landing right after a plain
      // user row - rendered to user-role text at assembly, the two are merged
      // for strict role alternation.
      { kind: "tool", tool: "search", status: "done", query: "q",
        text: "before EVILTEXT after", untrusted_spans: [[7, 15]] },
    ],
  };
  await window.runCompletion(conv);

  const completion = calls.find((c) => c.url === "/v1/chat/completions");
  const merged = completion.body.messages.find((m) => m.role === "user");
  assert.equal(merged.content, prefix + "\n\n" + "before EVILTEXT after");
  assert.deepEqual(merged.untrusted_spans, [[7 + prefix.length + 2, 15 + prefix.length + 2]]);
  assert.equal(merged.content.slice(...merged.untrusted_spans[0]), "EVILTEXT");
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

test("parseWebCall: the Llama 3 parameters alias carries the query", () => {
  const { window: w } = loadApp();
  eq(
    w.parseWebCall('{"name": "web_search", "parameters": {"query": "seabass price"}}'),
    { name: "web_search", args: { query: "seabass price" } });
  eq(
    w.parseWebCall('<tool_call>{"name": "fetch_url", "parameters": {"url": "https://example.com/"}}</tool_call>'),
    { name: "fetch_url", args: { url: "https://example.com/" } });
});

test("parseWebCall: the XML-tag dialect a non-compliant model emits instead of <tool_call>", () => {
  const { window: w } = loadApp();
  // The exact shape from a live bug report: a Llama3.3 finetune, told the
  // canonical <tool_call>{"name":...} format, emitted the tool name as a
  // literal XML tag with the url as an attribute instead.
  eq(
    w.parseWebCall(
      '<fetch_url url="https://api.open-meteo.com/v1/forecast?latitude=48.3&longitude=14.3"></fetch_url>'),
    { name: "fetch_url", args: {
      url: "https://api.open-meteo.com/v1/forecast?latitude=48.3&longitude=14.3" } });
  eq(
    w.parseWebCall('<web_search query="today weather in Linz" />'),
    { name: "web_search", args: { query: "today weather in Linz" } });
});

test("parseWebCall: an XML tag naming a non-web tool is not a web call", () => {
  const { window: w } = loadApp();
  assert.equal(w.parseWebCall('<read_file path="x.txt"></read_file>'), null);
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
  // TOOL_CALL_SINGLE/TOOL_CALL_TRIGGER are top-level const in the injected
  // settings-perf.js classic script - part of the jsdom realm's shared global
  // lexical environment, not a window property and not reachable from this
  // Node module scope directly. Bridge them out the same way the harness's
  // own runScript doc prescribes for reading realm-local state.
  runScript(window, "window.__gbnf = { TOOL_CALL_SINGLE, TOOL_CALL_TRIGGER };");
  const { TOOL_CALL_SINGLE, TOOL_CALL_TRIGGER } = window.__gbnf;
  assert.ok(completions[0].body.grammar, "no grammar was sent");
  assert.equal(completions[0].body.grammar_lazy, true);
  assert.deepEqual(completions[0].body.grammar_triggers, [TOOL_CALL_TRIGGER]);
  assert.equal(completions[0].body.grammar, TOOL_CALL_SINGLE);
  assert.match(TOOL_CALL_SINGLE, /^root\s+::= opt-ws tool-block opt-ws$/m,
    "one tool-call block per reply, never tool-block+");
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
    let body;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = opts.body; }
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

test("web ON: the XML-tag dialect still runs the real fetch (not silently accepted as an answer)", async () => {
  const { calls, completions } = await runChat({
    web: true,
    rounds: [
      content('<fetch_url url="https://api.open-meteo.com/v1/forecast?latitude=48.3&longitude=14.3"></fetch_url>'),
      content("It is 18C and mostly cloudy. Source: open-meteo."),
    ],
  });
  assert.equal(calls.filter((c) => c.url === "/api/web/fetch").length, 1,
    "the fetch endpoint was actually called instead of the reply being " +
    "accepted as an ordinary, un-grounded answer");
  assert.equal(completions.length, 2, "the model continued after the fetch result arrived");
});

test("web ON: a mangled tool call still runs the real search", async () => {
  const { window, conv, calls, completions } = await runChat({
    web: true,
    rounds: [
      content('<|tool_call>call:web_search{"query": "weather today"}<tool_call|>'),
      content("It is sunny. Source: https://example.com/"),
    ],
  });
  assert.equal(calls.filter((c) => c.url === "/api/web/retrieve").length, 1,
    "the retrieval endpoint was actually called");
  assert.ok(conv.messages.some((m) => m.kind === "tool" && m.tool === "search" &&
    m.status === "done" && /Results of web_search/.test(window.msgText(m))),
    "search results were injected back into the conversation as a tool event");
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
  assert.ok(conv.messages.some((m) => m.kind === "tool" && m.reason === "format" &&
    /\[tool-call format\]/.test(m.note)),
    "the model was asked to re-emit the tool call");
  assert.equal(completions.length, 2, "the model got a second chance to call the tool");
});

test("web OFF: a tool-shaped reply is NOT intercepted (no web rounds run)", async () => {
  const { calls, completions } = await runChat({
    web: false,
    rounds: [content('<tool_call>{"name": "web_search", "args": {"query": "x"}}</tool_call>')],
  });
  assert.equal(calls.filter((c) => c.url === "/api/web/retrieve").length, 0);
  assert.equal(completions.length, 1, "no repair / web loop when web access is off");
});

// Regression: the explicit /web command runs a real search even with the toggle
// off (it is direct user consent). The answering turn must then be told to USE
// those results, not handed the offline-denial floor - which would contradict
// the real results sitting in the conversation and make the model deny them.
async function runSlashWeb(webResults, query = "price of bitcoin today") {
  const { impl, calls } = recordingFetch(webResults);
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  window.readSSE = async (_r, onData) => {
    for (const d of content("answer")) onData(JSON.stringify(d));
  };
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;          // standing toggle OFF
  await window.runWebInChat(query);
  const webMsg = window.currentConv().messages.find((m) => m.kind === "tool");
  const webText = webMsg ? window.msgText(webMsg) : "";
  const completions = calls.filter((c) => c.url === "/v1/chat/completions");
  const sys = (completions[0].body.messages.find((m) => m.role === "system") || {}).content || "";
  return { window, calls, webMsg, webText, sys, completions };
}

test("/web with the toggle OFF: real retrieval runs, page evidence is injected, and the answer is grounded, not denied", async () => {
  const { window, calls, webMsg, webText, sys } = await runSlashWeb([
    { title: "T", url: "https://example.com/", snippet: "fresh fact", page: "PAGE: the price is 42" }]);

  // A real outbound retrieval fired (search + top pages read) and its evidence was injected.
  assert.equal(calls.filter((c) => c.url === "/api/web/retrieve").length, 1);
  assert.equal(calls.filter((c) => c.url === "/api/web/search").length, 0,
    "/web no longer answers from raw snippets (WEB-FUNC-001)");
  assert.ok(webMsg && webMsg.tool === "search" && webMsg.status === "done",
    "fresh results are in the conversation as a completed search event");
  assert.equal(webMsg.query, "price of bitcoin today");
  assert.equal(webMsg.role, undefined, "the event is not a user turn");
  assert.match(webText, /Results of web_search/);
  assert.match(webText, /\(page-backed: 1 of 1 sources read\)/);
  assert.match(webText, /PAGE: the price is 42/, "page text, not only the snippet");
  assert.match(webText, /Cite the source IDs \(S1, S2, \.\.\.\)/,
    "the model is told to cite source ids rather than unread urls");
  assert.doesNotMatch(webText, /No page could be read/);
  assert.equal(window.document.getElementById("toast").textContent, "", "no warning toast");

  // The answering turn is grounded, NOT told it is offline.
  assert.match(sys, /Web results were just provided/);
  assert.match(sys, /cite the source IDs/);
  assert.doesNotMatch(sys, /no internet access/i);
});

test("/web: a snippet-only bundle is labelled for the model and the user, never presented as read pages", async () => {
  const { window, webMsg, webText: note } = await runSlashWeb([
    { title: "T", url: "https://example.com/", snippet: "fresh fact" }]);
  assert.equal(webMsg.grounding, "snippet-only");
  assert.match(note, /\(snippet-only: no page was read, 1 search snippet only\)/);
  assert.match(note, /No page could be read: the evidence above is search snippets only/);
  assert.doesNotMatch(note.split("<untrusted_content>")[0], /page-backed/);
  const toastEl = window.document.getElementById("toast");
  assert.match(toastEl.textContent, /No page could be read/);
  assert.ok(toastEl.className.includes("error"));
});

test("/web: a provider failure is reported as a failed search, not as results", async () => {
  const { window, webMsg, webText, sys } = await runSlashWeb(() => bundleOf("q", [], {
    search_status: "failed", search_error: "RuntimeError: backend rate-limited",
    grounding: "failed", grounding_summary: "failed: no evidence, search failed",
    untrusted_fields: ["prompt_text", "search_error"],
  }));
  assert.equal(webMsg.status, "failed");
  assert.equal(webMsg.error, "Search failed: RuntimeError: backend rate-limited");
  assert.match(webText, /^\[Web request failed: Search failed: RuntimeError: backend rate-limited\] This lookup returned no information\./);
  assert.doesNotMatch(webText, /Answer without the web/,
    "a failed lookup must not invite an answer from the model's own knowledge");
  assert.match(webText, /Do not describe, simulate or guess what it would have found/);
  assert.doesNotMatch(webText, /Results of web_search/);
  assert.match(window.document.getElementById("toast").textContent, /Web search failed/);
  // No results were injected, so the model gets the offline floor, not the grounded one.
  assert.doesNotMatch(sys, /Web results were just provided/);
});

// ---------------------------------------------------------------------------
//  The web loop stops on a repeated call, on lookups that add nothing new and
//  at the Settings ceiling; the completion after that is tool-free, and the
//  turn ends on an answer, never on a bare tool-call marker.
// ---------------------------------------------------------------------------

const searchCall = (q) =>
  content(`<tool_call>{"name": "web_search", "args": {"query": "${q}"}}</tool_call>`);
const fetchCall = (u) =>
  content(`<tool_call>{"name": "fetch_url", "args": {"url": "${u}"}}</tool_call>`);
// A retrieval double whose every query returns its own, never-seen source.
const freshResults = (q) => bundleOf(q, [
  { title: q, url: "https://example.com/" + encodeURIComponent(q), snippet: "about " + q }]);
const retrieves = (calls) => calls.filter((c) => c.url === "/api/web/retrieve").length;
const isFinalRequest = (completion) => {
  const sys = systemOf(completion);
  return /Web lookups for this message are finished/.test(sys) &&
    !/access the internet through tools/.test(sys) && !("grammar" in completion.body);
};

test("R36: a repeated search is not re-run and the next completion is the tool-free last one", async () => {
  const { conv, calls, completions } = await runChat({
    web: true,
    rounds: [
      searchCall("weather today"),
      searchCall("Weather  Today"),   // the same search again, other case and spacing
      content("It is sunny. Source: https://example.com/"),
    ],
  });
  assert.equal(retrieves(calls), 1, "the duplicate search was NOT re-issued");
  assert.ok(conv.messages.some((m) => m.kind === "tool" && m.status === "duplicate" &&
    /\[duplicate web request\]/.test(m.note)),
    "the repeated call got a factual 'not run again' result");
  assert.equal(completions.length, 3);
  assert.ok(!isFinalRequest(completions[1]));
  assert.ok(isFinalRequest(completions[2]), "after the repeat the model gets no tools");
  const last = conv.messages[conv.messages.length - 1];
  assert.equal(last.role, "assistant");
  assert.match(last.content, /sunny/);
});

test("a model that keeps searching stops at the Settings ceiling and the turn still ends on an answer", async () => {
  const { window, conv, calls, completions } = await runChat({
    web: true, webResults: freshResults,
    setup: (w) => runScript(w, "chat.webMaxLookups = 3;"),
    rounds: [
      searchCall("q1"), searchCall("q2"), searchCall("q3"),
      searchCall("q4"),                       // one past the ceiling
      content("Final synthesized answer from S1."),
    ],
  });
  assert.equal(retrieves(calls), 3, "exactly the configured ceiling of lookups ran");
  const skipped = conv.messages.find((m) => m.kind === "tool" && m.status === "skipped");
  assert.ok(skipped, "the call past the ceiling is recorded as not run");
  assert.equal(skipped.query, "q4");
  assert.equal(skipped.limit, 3);
  for (const c of completions.slice(0, 4)) assert.ok(!isFinalRequest(c));
  assert.ok(isFinalRequest(completions[4]), "the completion after the ceiling is tool-free");
  assert.equal(completions.length, 5);
  const last = conv.messages[conv.messages.length - 1];
  assert.equal(last.role, "assistant");
  assert.equal(last.content, "Final synthesized answer from S1.");
  assert.ok(!last.webUnfinished);
  const sent = JSON.stringify(completions.map((c) => c.body.messages));
  assert.doesNotMatch(sent, /web search limit reached|Stop searching/i,
    "no fixed-cap order is sent to the model");
  assert.doesNotMatch(JSON.stringify(conv.messages), /web search limit reached|Stop searching/i);
  const dom = window.document.getElementById("chat-messages").textContent;
  assert.doesNotMatch(dom, /Instruction to the model/);
  assert.match(dom, /limit of 3 web lookups/, "the user is told why the fourth lookup did not run");
});

test("the tool-free last completion that still writes a call ends on text, never on a bare marker", async () => {
  const { window, conv, calls, completions } = await runChat({
    web: true, webResults: freshResults,
    setup: (w) => runScript(w, "chat.webMaxLookups = 1;"),
    rounds: [
      searchCall("q1"),
      searchCall("q2"),                       // past the ceiling of 1
      content("I will attempt to read your repository structure.\n" +
              '<tool_call>{"name": "fetch_url", "args": {"url": "https://github.com/x/y"}}</tool_call>'),
      content("must never be requested"),
    ],
  });
  assert.equal(retrieves(calls), 1);
  assert.equal(calls.filter((c) => c.url === "/api/web/fetch").length, 0,
    "a call written on the last completion does not run");
  assert.equal(completions.length, 3, "nothing is requested after the last completion");
  const last = conv.messages[conv.messages.length - 1];
  assert.equal(last.role, "assistant");
  assert.equal(last.content, "I will attempt to read your repository structure.");
  assert.equal(last.webUnfinished, true);
  const rows = window.document.querySelectorAll("#chat-messages .msg-row");
  const lastRow = rows[rows.length - 1].textContent;
  assert.doesNotMatch(lastRow, /read page|web search|\u{1F310}/u,
    "the last item is never a bare tool-call marker");
  assert.match(lastRow, /none were run/, "the user is told the lookups ended");
});

test("with the default setting, more than three searches run before the answer", async () => {
  const qs = ["a", "b", "c", "d", "e"];
  const { conv, calls } = await runChat({
    web: true, webResults: freshResults,
    rounds: [...qs.map(searchCall), content("Answer.")],
  });
  assert.equal(retrieves(calls), 5);
  assert.ok(!conv.messages.some((m) => m.kind === "tool" &&
    (m.status === "skipped" || m.reason === "limit")));
  assert.equal(conv.messages[conv.messages.length - 1].content, "Answer.");
});

test("the default ceiling is generous, and a ceiling of 0 means none", async () => {
  const qs = Array.from({ length: 22 }, (_, i) => "q" + i);
  const rounds = () => [...qs.map(searchCall), content("Answer.")];
  const byDefault = await runChat({ web: true, webResults: freshResults, rounds: rounds() });
  assert.equal(retrieves(byDefault.calls), 20, "the default stops at 20 lookups");
  const unlimited = await runChat({
    web: true, webResults: freshResults, rounds: rounds(),
    setup: (w) => runScript(w, "chat.webMaxLookups = 0;"),
  });
  assert.equal(retrieves(unlimited.calls), 22, "0 lets every distinct lookup run");
  assert.ok(!unlimited.conv.messages.some((m) => m.kind === "tool" && m.status === "skipped"));
});

test("two lookups in a row that add nothing new end the lookups", async () => {
  // Every query returns the same single source (the default retrieval double).
  const { calls, completions } = await runChat({
    web: true,
    rounds: [searchCall("a"), searchCall("b"), searchCall("c"), content("Answer from S1.")],
  });
  assert.equal(retrieves(calls), 3, "the first lookup found something; the next two found nothing new");
  assert.ok(!isFinalRequest(completions[2]), "one empty lookup does not end the lookups");
  assert.ok(isFinalRequest(completions[3]), "the second one in a row does");
});

test("failed page reads are reported to the model as failures and two in a row end the lookups", async () => {
  const fetchImpl = (rec) => async (url, opts = {}) => {
    if (String(url) === "/api/web/fetch") {
      rec.calls.push({ url: String(url) });
      return { ok: false, status: 502, statusText: "Bad Gateway",
               json: async () => ({ detail: "HTTP 403 from example.org" }) };
    }
    return rec.impl(url, opts);
  };
  const { conv, completions } = await runChat({
    web: true, fetchImpl,
    rounds: [fetchCall("https://example.org/a"), fetchCall("https://example.org/b"),
             content("I could not read either page.")],
  });
  assert.equal(conv.messages.filter((m) => m.kind === "tool" && m.status === "failed").length, 2);
  const sent = completions[2].body.messages.map((m) => m.content).join("\n");
  assert.match(sent, /\[Web request failed: HTTP 403 from example\.org\] This lookup returned no information\. Tell the user plainly that it failed\./);
  assert.doesNotMatch(sent, /Answer without the web/);
  assert.ok(isFinalRequest(completions[2]));
  assert.match(systemOf(completions[2]),
    /never describe, simulate or guess what it would have found/);
});

test("a reply that keeps emitting <tool_call> blocks is cut after the first; that call runs and the request is aborted", async () => {
  const block = (q) => `<tool_call>{"name": "web_search", "args": {"query": "${q}"}}</tool_call>`;
  const runaway = [];
  for (let i = 0; i < 170; i++) {
    runaway.push({ choices: [{ delta: { content: block("Matlan1 LocalM " + i) + "\n" } }] });
  }
  runaway.push({ choices: [{ delta: { content: '<tool_call>{"name": "web_search", "args": {"query": "Ma' } }] });
  const { conv, calls, completions } = await runChat({
    web: true, webResults: freshResults,
    rounds: [runaway, content("Here is what I found [S1].")],
  });
  const first = conv.messages.find((m) => m.role === "assistant");
  assert.equal((first.content.match(/<tool_call>/g) || []).length, 1, "only the first block is kept");
  assert.ok(!first.stopped, "the turn did not need Stop");
  assert.equal(retrieves(calls), 1);
  assert.equal(calls.find((c) => c.url === "/api/web/retrieve").body.query, "Matlan1 LocalM 0");
  assert.equal(completions[0].signal.aborted, true, "the runaway request was aborted at the cut");
  assert.equal(completions[1].signal.aborted, false);
  assert.equal(conv.messages[conv.messages.length - 1].content, "Here is what I found [S1].");
});

// ---------------------------------------------------------------------------
//  The history sent back to the model never carries the UI's tool markers or
//  UI-language text: a call that was answered is re-sent in canonical form,
//  a call nothing answered is dropped.
// ---------------------------------------------------------------------------

const DE = JSON.parse(await readFile(
  new URL("../localm/plugins/gui/static/i18n/de.json", import.meta.url), "utf-8"));

const toolHistory = () => [
  { role: "user", content: "find the repo" },
  { role: "assistant", content: 'Searching.\n<tool_call>{"name": "web_search", "args": {"query": "Matlan1 LocalM"}}</tool_call>' },
  { kind: "tool", tool: "search", status: "done", query: "Matlan1 LocalM", sources: [],
    chunks: [], prompt_text: "evidence", grounding: "snippet-only", grounding_summary: "snippet-only" },
  { role: "assistant", content: '<|tool_call>call:fetch_url{"url": "https://github.com/Matlan1/localm"}<tool_call|>' },
  { kind: "tool", tool: "fetch", status: "failed", url: "https://github.com/Matlan1/localm", error: "timeout" },
  { role: "assistant", content: 'It failed.\n> \u{1F310} *Failed: Web connection is currently unavailable*' },
  { role: "user", content: "try again" },
  { role: "assistant", stopped: true,
    content: Array.from({ length: 172 }, (_, i) => searchCall("q" + i)[0].choices[0].delta.content).join("\n") +
             '\n<tool_call>{"name": "web_search", "args": {"query": "Ma' },
  { role: "user", content: "and now?" },
];

async function historySent(lang) {
  const fetchImpl = (rec) => async (url, opts = {}) => {
    if (String(url).includes("/i18n/de.json")) return jsonResp(DE);
    return rec.impl(url, opts);
  };
  const { window, completions } = await runChat({
    web: true, history: toolHistory(), fetchImpl, rounds: [content("ok")],
    setup: async (w) => {
      if (lang === "de") {
        runScript(w, 'window.__p = applyLanguage("de");');
        await w.__p;
      }
    },
  });
  return { window, messages: completions[0].body.messages };
}

for (const lang of ["en", "de"]) {
  test(`history sent to the model holds no UI tool markers or UI-language text (${lang})`, async () => {
    const { window, messages } = await historySent(lang);
    const display = window.formatToolCalls('<tool_call>{"name": "web_search", "args": {"query": "x"}}</tool_call>');
    assert.match(display, lang === "de" ? /Web-Suche/ : /web search/,
      "control: the DISPLAY transform is localized in this locale");
    const sent = JSON.stringify(messages.filter((m) => m.role !== "system"));
    assert.doesNotMatch(sent, /\u{1F310}/u, "no globe marker");
    assert.doesNotMatch(sent, /web search:|read page:|web request:|Web-Suche:|Seite lesen:|Web-Anfrage:/i,
      "no UI-language tool text");
    const asst = messages.filter((m) => m.role === "assistant").map((m) => m.content);
    assert.deepEqual(asst, [
      'Searching.\n<tool_call>{"name":"web_search","args":{"query":"Matlan1 LocalM"}}</tool_call>',
      '<tool_call>{"name":"fetch_url","args":{"url":"https://github.com/Matlan1/localm"}}</tool_call>',
      "It failed.",
      "",
    ], "answered calls are canonical; the stopped partial nothing answered is sent empty");
    assert.equal((sent.match(/<tool_call>/g) || []).length, 2);
  });
}

test("history is identical whatever the UI language", async () => {
  const en = await historySent("en");
  const de = await historySent("de");
  assert.equal(JSON.stringify(de.messages), JSON.stringify(en.messages));
});

test("a copied display marker is treated as a botched call (re-prompt), not accepted as an answer", async () => {
  const { conv, calls, completions } = await runChat({
    web: true, webResults: freshResults,
    rounds: [
      content('> \u{1F310} *web search: ""*\nThe search query was empty, so the web request failed.'),
      searchCall("Matlan1 LocalM"),
      content("Found it [S1]."),
    ],
  });
  assert.ok(conv.messages.some((m) => m.kind === "tool" && m.reason === "format"),
    "the model was asked to re-emit the call");
  assert.equal(retrieves(calls), 1, "the re-emitted call ran");
  assert.doesNotMatch(JSON.stringify(completions[1].body.messages), /\u{1F310}/u,
    "the copied marker is not re-sent to the model");
  assert.equal(conv.messages[conv.messages.length - 1].content, "Found it [S1].");
});

test("a second botched call in a row ends the lookups instead of re-prompting again", async () => {
  const marker = content('> \u{1F310} *web search: "x"*');
  const { conv, completions } = await runChat({
    web: true, rounds: [marker, marker, content("I could not look that up.")],
  });
  assert.equal(conv.messages.filter((m) => m.kind === "tool" && m.reason === "format").length, 1);
  assert.ok(conv.messages.some((m) => m.kind === "tool" && m.reason === "unparsed"));
  assert.ok(isFinalRequest(completions[2]));
  assert.equal(completions.length, 3);
});

test("looksLikeWebToolAttempt and stripWebCallText cover the copied marker and every call dialect", () => {
  const { window: w } = loadApp();
  assert.equal(w.looksLikeWebToolAttempt('> \u{1F310} *read page: https://x*'), true);
  assert.equal(w.looksLikeWebToolAttempt("I like the globe emoji \u{1F310} a lot."), false);
  assert.equal(w.looksLikeWebToolAttempt("Contact:\n\u{1F310} Website: https://example.com\n"), false,
    "a globe bullet in an ordinary answer is not a copied marker");
  assert.equal(w.stripWebCallText("Contact:\n\u{1F310} Website: https://example.com"),
    "Contact:\n\u{1F310} Website: https://example.com");
  const text = [
    "Intro.",
    '<tool_call>{"name": "web_search", "args": {"query": "a"}}</tool_call>',
    '<|tool_call>call:web_search{"query": "b"}<tool_call|>',
    '<fetch_url url="https://x"></fetch_url>',
    '```json\n{"name": "web_search", "args": {"query": "c"}}\n```',
    '{"name": "fetch_url", "args": {"url": "https://y"}}',
    '> \u{1F310} *web search: "d"*',
    "```js\nconst keep = 1;\n```",
    "Outro.",
    '<tool_call>{"name": "web_search", "args": {"query": "unclos',
  ].join("\n");
  assert.equal(w.stripWebCallText(text), "Intro.\n\n```js\nconst keep = 1;\n```\nOutro.");
});

test("control notes render as a one-line notice, never as an 'Instruction to the model' card", () => {
  const { window: w } = loadApp();
  const box = w.document.createElement("div");
  w.addToolEventRow(box, w.webNoteEvent("format", "[tool-call format] SECRET model-facing text"));
  w.addToolEventRow(box, w.legacyWebNoteToToolEvent({
    role: "user", web: true,
    content: "[web search limit reached] You have used the maximum web lookups for this " +
             "turn. Stop searching and answer the question now" }));
  w.addToolEventRow(box, w.webCallOutcomeEvent({ name: "web_search", args: { query: "q" } },
    "skipped", "[web request not run] model-facing", { limit: 4 }));
  w.addToolEventRow(box, w.webCallOutcomeEvent({ name: "web_search", args: { query: "q" } },
    "duplicate", "[duplicate web request] model-facing"));
  const text = box.textContent;
  assert.doesNotMatch(text, /Instruction to the model/);
  assert.doesNotMatch(text, /SECRET|Stop searching|model-facing/, "the text the model reads is not shown");
  assert.match(text, /could not be read; the model was asked to send it again/);
  assert.match(text, /Web lookups for this message were stopped/);
  assert.match(text, /limit of 4 web lookups/);
  assert.match(text, /not run again/);
  assert.equal(box.querySelectorAll(".tool-card").length, 2, "only the two calls are cards");
  assert.equal(box.querySelectorAll(".tool-notice-row").length, 2);
});

test("an ordinary answer with a globe bullet is accepted, kept, and re-sent unchanged", async () => {
  const answer = "Contact details:\n\u{1F310} Website: https://example.com\n\u{1F4DE} Phone: 123";
  const { conv, completions } = await runChat({ web: true, rounds: [content(answer)] });
  assert.equal(completions.length, 1, "no format re-prompt for an ordinary answer");
  assert.ok(!conv.messages.some((m) => m.kind === "tool"));
  const last = conv.messages[conv.messages.length - 1];
  assert.equal(last.content, answer);
  assert.ok(!last.webUnfinished);
  const again = await runChat({
    web: true, rounds: [content("ok")],
    history: [{ role: "user", content: "contact?" }, { role: "assistant", content: answer },
              { role: "user", content: "thanks" }],
  });
  assert.equal(again.completions[0].body.messages.find((m) => m.role === "assistant").content, answer,
    "the globe line stays in the history sent to the model");
});

test("a /web result card shows no 'only the first request ran' notice", () => {
  const { window: w } = loadApp();
  const ev = { kind: "tool", tool: "search", status: "done", query: "q", sources: [], chunks: [],
               note: "Using this evidence, answer: q\nCite the source IDs (S1, S2, ...) you relied on." };
  assert.equal(w.toolEventNotice(ev), "");
  const box = w.document.createElement("div");
  w.addToolEventRow(box, ev);
  assert.doesNotMatch(box.textContent, /Only the first web request|Using this evidence/);
});

test("an image turn followed by a botched call keeps user and assistant turns alternating", async () => {
  const history = [{ role: "user", content: [
    { type: "text", text: "what is this?" },
    { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } }] }];
  const { completions } = await runChat({
    web: true, history,
    rounds: [content("<tool_call>{name: web_search}</tool_call>"), content("A cat.")],
  });
  assert.equal(completions.length, 2, "the botched call got the format re-prompt");
  const roles = completions[1].body.messages.filter((m) => m.role !== "system").map((m) => m.role);
  assert.deepEqual(roles, ["user", "assistant", "user"]);
});

test("in the tool-free last completion, text after a stray call is kept", async () => {
  const { conv, completions } = await runChat({
    web: true, webResults: freshResults,
    setup: (w) => runScript(w, "chat.webMaxLookups = 1;"),
    rounds: [
      searchCall("q1"), searchCall("q2"),
      [{ choices: [{ delta: { content: '<tool_call>{"name": "web_search", "args": {"query": "q3"}}</tool_call>\n\n' } }] },
       { choices: [{ delta: { content: "Based on S1, the answer is 42." } }] },
       { choices: [{ delta: {}, finish_reason: "stop" }] }],
    ],
  });
  assert.equal(completions.length, 3);
  const last = conv.messages[conv.messages.length - 1];
  assert.equal(last.content, "Based on S1, the answer is 42.");
  assert.equal(last.webUnfinished, true);
  assert.equal(completions[2].signal.aborted, false, "a single stray call does not cut the last completion");
});

test("in the tool-free last completion, a run of calls is cut where the second one starts", async () => {
  const blk = (q) => `<tool_call>{"name": "web_search", "args": {"query": "${q}"}}</tool_call>`;
  const last = [blk("a") + "\n", "Partial answer.\n", blk("b"), blk("c")]
    .map((c) => ({ choices: [{ delta: { content: c } }] }));
  const { conv, calls, completions } = await runChat({
    web: true, webResults: freshResults,
    setup: (w) => runScript(w, "chat.webMaxLookups = 1;"),
    rounds: [searchCall("q1"), searchCall("q2"), last],
  });
  assert.equal(retrieves(calls), 1);
  assert.equal(completions[2].signal.aborted, true, "the runaway was cut and its request aborted");
  const reply = conv.messages[conv.messages.length - 1];
  assert.equal(reply.content, "Partial answer.");
  assert.equal(reply.webUnfinished, true);
});

test("a call written only in the reasoning is re-sent in canonical form once it ran", async () => {
  const { calls, completions } = await runChat({
    web: true, webResults: freshResults,
    rounds: [
      [{ choices: [{ delta: { reasoning_content:
          'I should search. <tool_call>{"name": "web_search", "args": {"query": "reasoned"}}</tool_call>' } }] },
       { choices: [{ delta: {}, finish_reason: "stop" }] }],
      content("Answer [S1]."),
    ],
  });
  assert.equal(retrieves(calls), 1, "the call written in the reasoning ran");
  const asst = completions[1].body.messages.find((m) => m.role === "assistant");
  assert.equal(asst.content,
    '<tool_call>{"name":"web_search","args":{"query":"reasoned"}}</tool_call>',
    "the model sees the call that produced the results that follow");
});

test("copying a control notice copies the notice, not the text the model reads", async () => {
  const { window: w } = loadApp();
  let copied = null;
  Object.defineProperty(w.navigator, "clipboard",
    { value: { writeText: async (s) => { copied = s; } }, configurable: true });
  const box = w.document.createElement("div");
  w.addToolEventRow(box, w.webNoteEvent("format", "[tool-call format] SECRET model-facing text"));
  await box.querySelector(".copy-btn").onclick();
  assert.match(copied, /could not be read; the model was asked to send it again/);
  assert.doesNotMatch(copied, /SECRET/);
});

test("the webUnfinished flag survives the compaction archive copy", () => {
  const { window: w } = loadApp();
  assert.equal(w.archiveCopy({ role: "assistant", content: "x", webUnfinished: true }).webUnfinished, true);
});

// ---------------------------------------------------------------------------
//  web_search reads the top pages and returns labelled evidence, so the model
//  is told to cite source ids and to state a snippet-only or failed result
//  plainly (WEB-FUNC-005); fetch_url stays for a page the evidence missed.
// ---------------------------------------------------------------------------

test("web ON: the model is told web_search read the pages, to cite source IDs, and to state snippet-only plainly", async () => {
  const { completions } = await runChat({ web: true, rounds: [content("hello")] });
  const sys = systemOf(completions[0]);
  assert.match(sys, /reads the top result pages/,
    "the model must know the pages were already read, or it keeps re-fetching");
  assert.match(sys, /cite the source IDs \(S1, S2, \.\.\.\)/);
  assert.match(sys, /labelled snippet-only or failed, no page could be read/);
  assert.match(sys, /say so plainly instead of implying you read the pages/);
  assert.match(sys, /Use fetch_url only to read a specific page the evidence did not cover/);
  assert.doesNotMatch(sys, /snippets, not page text/,
    "the old snippets-only description would contradict what web_search now returns");
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

test("web ON: a reply is cut at its first <tool_call> block, so only that call is stored and run", async () => {
  const { conv, calls } = await runChat({
    web: true,
    rounds: [twoCalls, content("It is sunny. Source: https://example.com/")],
  });
  assert.equal(calls.filter((c) => c.url === "/api/web/retrieve").length, 1, "the first call ran");
  assert.equal(calls.filter((c) => c.url === "/api/web/fetch").length, 0, "the second never ran");
  const first = conv.messages.find((m) => m.role === "assistant");
  assert.equal((first.content.match(/<tool_call>/g) || []).length, 1,
    "the stored reply holds only the call that ran, so the user sees one lookup");
  assert.doesNotMatch(first.content, /fetch_url/);
});

const twoFencedCalls = content(
  '```json\n{"name": "web_search", "args": {"query": "weather"}}\n```\n' +
  '```json\n{"name": "fetch_url", "args": {"url": "https://example.com/b"}}\n```');

test("web ON: a second tool call in one reply is reported as ignored, not silently dropped", async () => {
  const { window, conv, calls } = await runChat({
    web: true,
    rounds: [twoFencedCalls, content("It is sunny. Source: https://example.com/")],
  });
  assert.equal(calls.filter((c) => c.url === "/api/web/retrieve").length, 1,
    "the first call ran");
  assert.equal(calls.filter((c) => c.url === "/api/web/fetch").length, 0,
    "the second call did NOT run - one call per message is the retained design");
  const note = conv.messages.find(
    (m) => m.kind === "tool" && /only the first tool call ran/.test(m.note || ""));
  assert.ok(note, "the model was never told its second call was ignored");
  assert.match(note.note, /fetch_url/,
    "the notice must name what was ignored, not just that something was");
  assert.equal(note.tool, "search");
  assert.equal(note.status, "done");
  assert.equal(note.reason, "ignored");
  assert.match(window.toolEventNotice(note), /Only the first web request in that reply ran/);
  const body = window.msgText(note);
  assert.match(body, /Results of web_search/,
    "the notice rides on the result event, keeping user/assistant alternation");
  // LM-DA-014: everything inside the fence is DATA the model is told not to obey.
  // A notice that landed in there would be self-defeating - it is our instruction,
  // not fetched content - and "present in the message" cannot tell the two apart.
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
    let body;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = opts.body; }
    calls.push({ url: String(url), body });
    if (String(url) === "/v1/config") return jsonResp({ net_mode: netMode });
    if (String(url) === "/api/web/retrieve") return jsonResp(bundleOf("q", webResults));
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
  assert.equal(calls.filter((c) => c.url === "/api/web/retrieve").length, 1, "approved -> the search ran");
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
  assert.equal(calls.filter((c) => c.url === "/api/web/retrieve").length, 0, "denied -> NO search ran");
  assert.ok(conv.messages.some((m) => m.kind === "tool" && m.status === "denied" &&
    m.query === "weather" && /\[web access denied\]/.test(m.note)),
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
  assert.equal(calls.filter((c) => c.url === "/api/web/retrieve").length, 1, "the search ran without a prompt");
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
//  confirmWebRequest: dismissing the modal (the x, or the backdrop) must
//  settle the promise instead of leaving runCompletion()'s await hanging
//  forever. On unfixed code an unbounded await of a dismissed promise hangs
//  the test runner rather than failing it, so every await below goes through
//  a bounded race instead of a plain await.
// ---------------------------------------------------------------------------

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const race = (p) => Promise.race([p.then((v) => ({ v })), wait(1500).then(() => null)]);

test("confirmWebRequest: dismissing via the shared modal chrome (x/backdrop) resolves false", async () => {
  const { window } = loadApp();
  const modal = window.document.getElementById("modal");

  const p = window.confirmWebRequest({ name: "web_search", args: { query: "x" } });
  assert.notEqual(modal.style.display, "none", "the approval modal opened");
  // Neither button was clicked - simulate the shared chrome's own dismiss,
  // which only ever sets display:none (see helpers.js's modal-close wiring).
  modal.style.display = "none";

  const settled = await race(p);
  assert.ok(settled, "confirmWebRequest never settled after the modal was dismissed");
  assert.strictEqual(settled.v, false);
});

test("confirmWebRequest: dismissing with the remember checkbox ticked must not remember", async () => {
  const { window } = loadApp();
  const modal = window.document.getElementById("modal");

  const p1 = window.confirmWebRequest({ name: "web_search", args: { query: "x" } });
  const cb = modal.querySelector(".web-ask-remember input[type=checkbox]");
  assert.ok(cb, "the remember checkbox is present");
  cb.checked = true;
  // Dismissed via the shared chrome, not a button - ticking the box and then
  // escaping the dialog must not count as having made a choice.
  modal.style.display = "none";

  const s1 = await race(p1);
  assert.ok(s1, "confirmWebRequest never settled after a ticked-then-dismissed modal");
  assert.strictEqual(s1.v, false);

  // The observable that matters: a LATER request must still open a fresh
  // modal. If the dismissal had written webAskSession, this would
  // short-circuit instead of opening one.
  modal.style.display = "none";
  const p2 = window.confirmWebRequest({ name: "web_search", args: { query: "y" } });
  assert.notEqual(modal.style.display, "none",
    "a second modal opened - the ticked-then-dismissed choice was not remembered");

  modal.style.display = "none";
  const s2 = await race(p2);
  assert.ok(s2, "the second prompt never settled");
  assert.strictEqual(s2.v, false);
});

test("confirmWebRequest: Deny and Allow still resolve correctly, remembering only when ticked", async () => {
  // Deny, unticked: resolves false, does not remember - a later call opens a fresh modal.
  {
    const { window } = loadApp();
    const modal = window.document.getElementById("modal");
    const denyBtn = () => [...modal.querySelectorAll("button")].find((b) => b.textContent === "Deny");
    const p1 = window.confirmWebRequest({ name: "web_search", args: { query: "x" } });
    denyBtn().click();
    const s1 = await race(p1);
    assert.ok(s1, "Deny never settled");
    assert.strictEqual(s1.v, false);

    modal.style.display = "none";
    const p2 = window.confirmWebRequest({ name: "web_search", args: { query: "y" } });
    assert.notEqual(modal.style.display, "none", "an unticked Deny does not remember");
    denyBtn().click();
    await race(p2);
  }

  // Allow, unticked: resolves true, does not remember.
  {
    const { window } = loadApp();
    const modal = window.document.getElementById("modal");
    const allowBtn = () => [...modal.querySelectorAll("button")].find((b) => b.textContent === "Allow");
    const p1 = window.confirmWebRequest({ name: "web_search", args: { query: "x" } });
    allowBtn().click();
    const s1 = await race(p1);
    assert.ok(s1, "Allow never settled");
    assert.strictEqual(s1.v, true);

    modal.style.display = "none";
    const p2 = window.confirmWebRequest({ name: "web_search", args: { query: "y" } });
    assert.notEqual(modal.style.display, "none", "an unticked Allow does not remember");
    allowBtn().click();
    await race(p2);
  }

  // Deny, ticked: resolves false AND remembers - a later call auto-denies without reopening.
  {
    const { window } = loadApp();
    const modal = window.document.getElementById("modal");
    const p1 = window.confirmWebRequest({ name: "web_search", args: { query: "x" } });
    modal.querySelector(".web-ask-remember input[type=checkbox]").checked = true;
    [...modal.querySelectorAll("button")].find((b) => b.textContent === "Deny").click();
    const s1 = await race(p1);
    assert.ok(s1, "ticked Deny never settled");
    assert.strictEqual(s1.v, false);

    modal.style.display = "none";
    const p2 = window.confirmWebRequest({ name: "web_search", args: { query: "y" } });
    assert.equal(modal.style.display, "none", "a ticked Deny remembers - no second modal opens");
    const s2 = await race(p2);
    assert.strictEqual(s2.v, false);
  }

  // Allow, ticked: resolves true AND remembers.
  {
    const { window } = loadApp();
    const modal = window.document.getElementById("modal");
    const p1 = window.confirmWebRequest({ name: "web_search", args: { query: "x" } });
    modal.querySelector(".web-ask-remember input[type=checkbox]").checked = true;
    [...modal.querySelectorAll("button")].find((b) => b.textContent === "Allow").click();
    const s1 = await race(p1);
    assert.ok(s1, "ticked Allow never settled");
    assert.strictEqual(s1.v, true);

    modal.style.display = "none";
    const p2 = window.confirmWebRequest({ name: "web_search", args: { query: "y" } });
    assert.equal(modal.style.display, "none", "a ticked Allow remembers - no second modal opens");
    const s2 = await race(p2);
    assert.strictEqual(s2.v, true);
  }
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
  assert.ok(!/<\|tool_call|tool_call\|>|call:/.test(String(asst.content)),
    "the model's raw dialect tokens are NOT re-fed to the model");
  assert.equal(String(asst.content),
    '<tool_call>{"name":"web_search","args":{"query":"privacy"}}</tool_call>',
    "the call that ran is re-sent in the canonical form the tool prompt teaches");
  assert.doesNotMatch(String(asst.content), /\u{1F310}|web search:/u, "no display marker");
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
