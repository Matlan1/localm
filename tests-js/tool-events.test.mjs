// SPDX-License-Identifier: AGPL-3.0-or-later
// First-class web tool events (ADR-0022 Phase 4). A web search, page read,
// approval outcome or control note is stored in conv.messages as
// {kind: "tool", ...} with no role, is drawn as one collapsed activity-and-
// sources card that is neither the user bubble nor the assistant column, is
// rendered to fenced user-role text only while runCompletion assembles the
// request, and a legacy {role: "user", web: true} row is migrated on load.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";
import { bundleOf } from "./web-fixtures.mjs";

function setActiveConv(window, conv) {
  window.__testConv = conv;
  runScript(window, "chat.conversations = [window.__testConv]; chat.activeId = window.__testConv.id;");
}

const jsonResp = (obj) => ({
  ok: true, status: 200, json: async () => obj, text: async () => JSON.stringify(obj),
});

/** A completed search event as requestWebTool returns it for *results*. */
function searchEvent(query, results, extra = {}) {
  const data = bundleOf(query, results);
  return {
    kind: "tool", tool: "search", status: "done", query,
    started_at: 1000, finished_at: 2500,
    provider: data.provider, search_status: data.search_status,
    search_error: data.search_error, grounding: data.grounding,
    grounding_summary: data.grounding_summary, sources: data.sources,
    chunks: data.chunks, prompt_text: data.prompt_text, ...extra,
  };
}

const TWO_SOURCES = [
  { title: "Alpha", url: "https://alpha.example/a", snippet: "alpha snippet", page: "ALPHA PAGE TEXT" },
  { title: "Beta", url: "https://beta.example/b", snippet: "beta snippet" },
];

// ---------------------------------------------------------------------------
//  Rendering: a third kind of row, collapsed, expandable to its sources
// ---------------------------------------------------------------------------

test("renderChat: a tool event is neither a user nor an assistant row and draws a collapsed card", () => {
  const { window } = loadApp();
  setActiveConv(window, {
    id: "c1", title: "t",
    messages: [
      { role: "user", content: "what is alpha", id: "m1" },
      searchEvent("alpha", TWO_SOURCES, { id: "m2" }),
      { role: "assistant", content: "Alpha is a thing [S1].", id: "m3" },
    ],
  });
  window.renderChat();
  const box = window.document.getElementById("chat-messages");
  const rows = [...box.querySelectorAll(".msg-row")];
  assert.equal(rows.length, 3);
  const row = rows[1];
  assert.ok(!row.classList.contains("user"), "not the user bubble");
  assert.ok(!row.classList.contains("assistant"), "not the assistant column");
  assert.ok(row.classList.contains("tool-event"), "its own row class");
  assert.equal(row.querySelector(".msg-role").textContent, "Web");

  const card = row.querySelector("details.tool-card");
  assert.ok(card, "the activity card is a details element");
  assert.equal(card.open, false, "collapsed by default");
  assert.equal(card.dataset.status, "done");
  const summary = card.querySelector("summary");
  assert.match(summary.textContent, /Web search: alpha/);
  assert.match(summary.textContent, /done/);
  assert.match(summary.textContent, /pages read/, "the grounding label is on the summary line");
  assert.match(summary.textContent, /2 sources/);
  assert.match(summary.textContent, /1\.5s/, "elapsed time from started_at/finished_at");

  card.open = true;
  assert.equal(card.open, true, "the card expands");
  const items = [...card.querySelectorAll(".tool-sources li")];
  assert.equal(items.length, 2, "every source is listed");
  assert.match(items[0].textContent, /S1/);
  assert.match(items[0].textContent, /Alpha/);
  assert.match(items[0].textContent, /pages read/);
  assert.match(items[1].textContent, /S2/);
  assert.match(items[1].textContent, /snippets only/);
  assert.match(items[1].textContent, /read failed/, "a failed read is shown honestly");
  const link = items[0].querySelector("a.tool-link");
  assert.equal(link.getAttribute("href"), "https://alpha.example/a");
  assert.equal(link.getAttribute("rel"), "noopener noreferrer");
  assert.match(card.querySelector(".tool-body").textContent, /ALPHA PAGE TEXT/,
    "the evidence excerpts are in the expanded body");
  assert.equal(row.querySelectorAll(".msg-body").length, 0,
    "no message body: the card is the whole row");
});

test("renderChat: a tool event offers no edit/revert/regenerate actions, by construction", () => {
  const { window } = loadApp();
  setActiveConv(window, {
    id: "c1", title: "t",
    messages: [
      { role: "user", content: "hi", id: "m1" },
      searchEvent("q", TWO_SOURCES, { id: "m2" }),
    ],
  });
  window.renderChat();
  const rows = [...window.document.querySelectorAll("#chat-messages .msg-row")];
  const userActions = [...rows[0].querySelectorAll(".msg-meta button")].map((b) => b.textContent);
  assert.ok(userActions.includes("edit") && userActions.includes("revert"),
    "a real user turn still gets edit/revert");
  const toolActions = [...rows[1].querySelectorAll(".msg-meta button")].map((b) => b.textContent);
  assert.ok(!toolActions.some((a) => /edit|revert|regenerate/.test(a)),
    `a tool event gets none of them (got ${JSON.stringify(toolActions)})`);
});

test("renderChat: a source whose URL is not a web URL is shown as text, never as a link", () => {
  const { window } = loadApp();
  const ev = searchEvent("q", [{ title: "Bad", url: "javascript:alert(1)", snippet: "s" }]);
  setActiveConv(window, { id: "c1", title: "t", messages: [ev] });
  window.renderChat();
  const li = window.document.querySelector(".tool-sources li");
  assert.ok(li, "the source is listed");
  assert.equal(li.querySelector("a"), null, "no anchor for a non-http(s) locator");
  assert.match(li.textContent, /Bad/);
});

test("renderChat: remote text in a tool event is set as text, never parsed as markup", () => {
  const { window } = loadApp();
  const ev = searchEvent("q", [
    { title: "<img src=x onerror=alert(1)>", url: "https://x.example/", snippet: "s",
      page: "<b>bold</b> <script>alert(2)</script>" }]);
  setActiveConv(window, { id: "c1", title: "t", messages: [ev] });
  window.renderChat();
  const card = window.document.querySelector(".tool-card");
  assert.equal(card.querySelector("img"), null);
  assert.equal(card.querySelector("script"), null);
  assert.equal(card.querySelector("b"), null);
  assert.match(card.textContent, /<img src=x onerror=alert\(1\)>/);
  assert.match(card.textContent, /<script>alert\(2\)<\/script>/);
});

test("renderChat: running, failed, denied and note events each draw a card with their status and text", () => {
  const { window } = loadApp();
  setActiveConv(window, {
    id: "c1", title: "t",
    messages: [
      { kind: "tool", tool: "search", status: "running", query: "r", started_at: 1, id: "a" },
      { kind: "tool", tool: "fetch", status: "failed", url: "https://f.example/", error: "HTTP 500",
        started_at: 1, finished_at: 2, id: "b" },
      { kind: "tool", tool: "search", status: "denied", query: "d", note: "[web access denied] no.",
        started_at: 1, finished_at: 1, id: "c" },
      { kind: "tool", tool: "note", status: "done", reason: "limit", note: "[web search limit reached] stop.",
        started_at: 1, finished_at: 1, id: "d" },
      { kind: "tool", tool: "fetch", status: "done", url: "https://p.example/",
        page: { url: "https://p.example/final", text: "PAGE BODY", truncated: true },
        started_at: 1, finished_at: 3, id: "e" },
    ],
  });
  window.renderChat();
  const cards = [...window.document.querySelectorAll("details.tool-card")];
  assert.equal(cards.length, 5);
  assert.deepEqual(cards.map((c) => c.dataset.status), ["running", "failed", "denied", "done", "done"]);
  assert.match(cards[0].querySelector("summary").textContent, /Web search: r[\s\S]*in progress/);
  assert.match(cards[1].querySelector("summary").textContent, /Page read: https:\/\/f\.example\/[\s\S]*failed/);
  assert.match(cards[1].textContent, /HTTP 500/);
  assert.match(cards[2].querySelector("summary").textContent, /denied/);
  assert.match(cards[2].textContent, /\[web access denied\] no\./);
  assert.match(cards[3].querySelector("summary").textContent, /Web note/);
  assert.match(cards[3].textContent, /\[web search limit reached\] stop\./);
  assert.match(cards[4].querySelector("summary").textContent, /Page read: https:\/\/p\.example\/final/);
  assert.match(cards[4].textContent, /Page text \(truncated\)[\s\S]*PAGE BODY/);
  for (const c of cards) assert.equal(c.open, false, "every card starts collapsed");
});

// ---------------------------------------------------------------------------
//  Storage vs. the request: no role stored; fenced user-role text on the wire
// ---------------------------------------------------------------------------

/** Drive one runCompletion over *messages* and return the request bodies. */
async function assemble(messages, { web = false } = {}) {
  const calls = [];
  const impl = async (url, opts = {}) => {
    let body;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = opts.body; }
    calls.push({ url: String(url), body });
    return jsonResp({});
  };
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  window.readSSE = async (_r, onData) => {
    onData(JSON.stringify({ choices: [{ delta: { content: "ok" } }] }));
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }));
  };
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = web;
  const conv = { id: "c1", title: "t", messages };
  setActiveConv(window, conv);
  await window.runCompletion(conv);
  const completions = calls.filter((c) => c.url === "/v1/chat/completions");
  return { window, conv, completions };
}

test("runCompletion: a stored tool event has no role; the request carries it as fenced user-role text", async () => {
  const { conv, completions } = await assemble([
    { role: "user", content: "what is alpha", id: "m1" },
    { role: "assistant", content: '<tool_call>{"name": "web_search", "args": {"query": "alpha"}}</tool_call>', id: "m2" },
    searchEvent("alpha", TWO_SOURCES, { id: "m3", note: "[only the first tool call ran] ignored fetch_url" }),
  ]);
  // Stored shape: still a tool event, never a user row.
  const stored = conv.messages[2];
  assert.equal(stored.kind, "tool");
  assert.equal(stored.role, undefined, "no role is ever written onto the stored event");
  assert.equal(stored.content, undefined, "no pre-rendered content is ever written onto it");
  assert.ok(!("role" in stored));
  assert.ok(!conv.messages.some((m) => m.role === "user" && /Results of web_search/.test(String(m.content))),
    "conv.messages never holds a user-role copy of the result");

  // Wire shape: a user-role message rendered from the event at assembly time.
  assert.equal(completions.length, 1);
  const sent = completions[0].body.messages;
  const rendered = sent.find((m) => m.role === "user" && /Results of web_search/.test(m.content));
  assert.ok(rendered, "the event reached the model as a user-role message");
  assert.ok(sent.every((m) => m.kind === undefined && m.tool === undefined),
    "the tool-event shape itself never goes over the wire");
  assert.match(rendered.content, /^\[Results of web_search "alpha"\] \(page-backed: 1 of 2 sources read\)\n/);
  assert.match(rendered.content, /<untrusted_content>[\s\S]*\[S1\] Alpha - https:\/\/alpha\.example\/a \(page-backed\)[\s\S]*\[S2\] Beta - https:\/\/beta\.example\/b \(snippet-only[\s\S]*ALPHA PAGE TEXT[\s\S]*<\/untrusted_content>/);
  assert.match(rendered.content, /<\/untrusted_content>\n\n\[only the first tool call ran\] ignored fetch_url$/,
    "the note follows the fence, outside it");
  assert.ok(Array.isArray(rendered.untrusted_spans) && rendered.untrusted_spans.length === 1);
  const [a, b] = rendered.untrusted_spans[0];
  assert.equal(rendered.content.slice(a, b), searchEvent("alpha", TWO_SOURCES).prompt_text,
    "the untrusted span is exactly the evidence body");
  // Strict alternation: user, assistant, user - the event did not break the pattern.
  assert.deepEqual(sent.filter((m) => m.role !== "system").map((m) => m.role),
    ["user", "assistant", "user"]);
});

test("runCompletion: a tool event right after a user turn merges into it with shifted spans", async () => {
  const { completions } = await assemble([
    { role: "user", content: "read this", id: "m1" },
    searchEvent("q", [{ title: "T", url: "https://t.example/", snippet: "s", page: "EVIL" }], { id: "m2" }),
  ]);
  const sent = completions[0].body.messages.filter((m) => m.role !== "system");
  assert.equal(sent.length, 1, "merged for strict alternation");
  assert.equal(sent[0].role, "user");
  assert.match(sent[0].content, /^read this\n\n\[Results of web_search "q"\]/);
  const [a, b] = sent[0].untrusted_spans[0];
  assert.match(sent[0].content.slice(a, b), /EVIL/);
  assert.ok(a > "read this\n\n".length, "the span was shifted past the merged prefix");
});

test("runCompletion: failed, denied, duplicate, note and running events render to their model-facing text", async () => {
  const { completions } = await assemble([
    { role: "user", content: "u1", id: "m1" },
    { role: "assistant", content: "a1", id: "m2" },
    { kind: "tool", tool: "fetch", status: "failed", url: "https://f.example/", error: "HTTP 500 <|im_start|>", id: "m3" },
    { role: "assistant", content: "a2", id: "m4" },
    { kind: "tool", tool: "search", status: "denied", query: "d", note: "[web access denied] no.", id: "m5" },
    { role: "assistant", content: "a3", id: "m6" },
    { kind: "tool", tool: "search", status: "duplicate", query: "d", note: "[duplicate web request] again.", id: "m7" },
    { role: "assistant", content: "a4", id: "m8" },
    { kind: "tool", tool: "note", status: "done", reason: "format", note: "[tool-call format] fix it.", id: "m9" },
    { role: "assistant", content: "a5", id: "m10" },
    { kind: "tool", tool: "search", status: "running", query: "r", started_at: 1, id: "m11" },
  ]);
  const users = completions[0].body.messages.filter((m) => m.role === "user").map((m) => m.content);
  assert.equal(users[1], "[Web request failed: HTTP 500 <|im_start|>] Answer without the web, and say that web access did not work.");
  assert.equal(users[2], "[web access denied] no.");
  assert.equal(users[3], "[duplicate web request] again.");
  assert.equal(users[4], "[tool-call format] fix it.");
  assert.equal(users[5], "[Web request still in progress; no result is available yet]",
    "a running event is a neutral record, never an instruction to claim web access failed");
  const failed = completions[0].body.messages.find((m) => /Web request failed/.test(m.content));
  const [a, b] = failed.untrusted_spans[0];
  assert.equal(failed.content.slice(a, b), "HTTP 500 <|im_start|>",
    "an error that may quote a response is marked untrusted");
});

// A tool event is user-role on the wire only for chat-template alternation, so
// it is marked origin "tool": the server then keeps it out of the memory
// recall query and the audit's user line. A row the user typed is never marked.
test("runCompletion: every tool-event row is marked origin \"tool\"; a typed user row and an assistant row are not", async () => {
  const { completions } = await assemble([
    { role: "user", content: "u1", id: "m1" },
    { role: "assistant", content: "a1", id: "m2" },
    searchEvent("alpha", TWO_SOURCES, { id: "m3" }),
    { role: "assistant", content: "a2", id: "m4" },
    { kind: "tool", tool: "fetch", status: "done", url: "https://f.example/",
      page: { url: "https://f.example/", text: "PAGE", truncated: false }, id: "m5" },
    { role: "assistant", content: "a3", id: "m6" },
    { kind: "tool", tool: "fetch", status: "failed", url: "https://f.example/", error: "HTTP 500", id: "m7" },
    { role: "assistant", content: "a4", id: "m8" },
    { kind: "tool", tool: "search", status: "denied", query: "d", note: "[web access denied] no.", id: "m9" },
    { role: "assistant", content: "a5", id: "m10" },
    { kind: "tool", tool: "search", status: "duplicate", query: "d", note: "[duplicate web request] again.", id: "m11" },
    { role: "assistant", content: "a6", id: "m12" },
    { kind: "tool", tool: "note", status: "done", reason: "pending", note: "[pending action] do it.", id: "m13" },
    { role: "assistant", content: "a7", id: "m14" },
    { kind: "tool", tool: "note", status: "done", reason: "limit", note: "[web search limit reached] answer.", id: "m15" },
  ]);
  const sent = completions[0].body.messages.filter((m) => m.role !== "system");
  assert.deepEqual(sent.map((m) => [m.role, m.origin]), [
    ["user", undefined],
    ["assistant", undefined], ["user", "tool"],
    ["assistant", undefined], ["user", "tool"],
    ["assistant", undefined], ["user", "tool"],
    ["assistant", undefined], ["user", "tool"],
    ["assistant", undefined], ["user", "tool"],
    ["assistant", undefined], ["user", "tool"],
    ["assistant", undefined], ["user", "tool"],
  ]);
  assert.ok(!("origin" in sent[0]), "a typed user row carries no origin key at all");
  assert.ok(sent.filter((m) => m.role === "assistant").every((m) => !("origin" in m)));
});

test("runCompletion: a merged row keeps origin \"tool\" only when every part of it is a tool event", async () => {
  // User text, then a tool event: merged, and the merged row holds user text.
  let { completions } = await assemble([
    { role: "user", content: "read this", id: "m1" },
    searchEvent("q", [{ title: "T", url: "https://t.example/", snippet: "s" }], { id: "m2" }),
  ]);
  let sent = completions[0].body.messages.filter((m) => m.role !== "system");
  assert.equal(sent.length, 1);
  assert.match(sent[0].content, /^read this\n\n\[Results of web_search "q"\]/);
  assert.ok(!("origin" in sent[0]), "user text first: the merged row is not marked");

  // A tool event, then user text: merged, and still not marked.
  ({ completions } = await assemble([
    { role: "user", content: "u1", id: "m1" },
    { role: "assistant", content: "a1", id: "m2" },
    { kind: "tool", tool: "note", status: "done", reason: "pending", note: "[pending action] do it.", id: "m3" },
    { role: "user", content: "never mind, just answer", id: "m4" },
  ]));
  sent = completions[0].body.messages.filter((m) => m.role !== "system");
  assert.equal(sent.length, 3);
  assert.equal(sent[2].content, "[pending action] do it.\n\nnever mind, just answer");
  assert.ok(!("origin" in sent[2]), "user text last: the merged row is not marked");

  // Two tool events in a row: merged, and nothing in it is user text.
  ({ completions } = await assemble([
    { role: "user", content: "u1", id: "m1" },
    { role: "assistant", content: "a1", id: "m2" },
    searchEvent("q", [{ title: "T", url: "https://t.example/", snippet: "s" }], { id: "m3" }),
    { kind: "tool", tool: "note", status: "done", reason: "format", note: "[tool-call format] fix it.", id: "m4" },
  ]));
  sent = completions[0].body.messages.filter((m) => m.role !== "system");
  assert.equal(sent.length, 3);
  assert.match(sent[2].content, /\n\n\[tool-call format\] fix it\.$/);
  assert.equal(sent[2].origin, "tool", "only tool events merged: the row stays marked");

  // Tool, tool, user: the user text clears the mark the first merge kept.
  ({ completions } = await assemble([
    { role: "user", content: "u1", id: "m1" },
    { role: "assistant", content: "a1", id: "m2" },
    { kind: "tool", tool: "search", status: "denied", query: "d", note: "[web access denied] no.", id: "m3" },
    { kind: "tool", tool: "note", status: "done", reason: "limit", note: "[web search limit reached] answer.", id: "m4" },
    { role: "user", content: "ok", id: "m5" },
  ]));
  sent = completions[0].body.messages.filter((m) => m.role !== "system");
  assert.equal(sent.length, 3);
  assert.match(sent[2].content, /\n\nok$/);
  assert.ok(!("origin" in sent[2]), "a user row merged after two tool events clears the mark");
});

test("lastTurnHasWebResults: only a completed search or read counts, and a migrated row by its text", () => {
  const { window } = loadApp();
  const f = window.lastTurnHasWebResults;
  assert.equal(f({ messages: [{ role: "user", content: "x" }] }), false);
  assert.equal(f({ messages: [searchEvent("q", TWO_SOURCES)] }), true);
  assert.equal(f({ messages: [{ kind: "tool", tool: "fetch", status: "done", page: { url: "u", text: "t" } }] }), true);
  assert.equal(f({ messages: [{ kind: "tool", tool: "search", status: "failed", error: "e" }] }), false);
  assert.equal(f({ messages: [{ kind: "tool", tool: "note", status: "done", reason: "limit", note: "n" }] }), false);
  assert.equal(f({ messages: [{ kind: "tool", tool: "search", status: "done", text: '[Results of web_search "q"] (x)' }] }), true);
  assert.equal(f({ messages: [{ kind: "tool", tool: "note", status: "done", text: "[pending action] x" }] }), false);
});

// ---------------------------------------------------------------------------
//  Migration: legacy {role:"user", web:true} rows, on load, idempotently
// ---------------------------------------------------------------------------

function legacyConv() {
  return {
    id: "L1", title: "legacy", updated_at: 7,
    messages: [
      { role: "user", content: "what's the weather", id: "u1" },
      { role: "assistant", content: '<tool_call>{"name":"web_search","args":{"query":"weather"}}</tool_call>', id: "a1" },
      { role: "user", web: true, id: "w1", untrusted_spans: [[52, 60]],
        content: '[Results of web_search "weather"] (page-backed: 1 of 1 sources read)\nEVIDENCE' },
      { role: "assistant", content: "Cloudy.", id: "a2" },
      { role: "user", web: true, id: "w2", content: "[Content of https://x.example/p] (truncated)\nPAGE" },
      { role: "user", web: true, id: "w3", content: "[Web request failed: boom] Answer without the web." },
      { role: "user", web: true, id: "w4", content: "[duplicate web request] again" },
      { role: "user", web: true, id: "w5", content: "[web access denied] no" },
      { role: "user", web: true, id: "w6", content: "[tool-call format] fix" },
      { role: "user", web: true, id: "w7", content: "[web search limit reached] stop" },
      { role: "user", web: true, id: "w8", content: "[pending action] do it" },
      { role: "user", web: true, id: "w9", content: "[web_search results for \"old\"]\nold style" },
      { role: "user", web: true, id: "w10", content: "something unrecognised" },
      { role: "user", tag: "kb", content: "[Excerpts from \"docs\"]", id: "k1" },
      { role: "user", tag: "doc", content: "[Attached document: a.txt]", id: "d1" },
      { role: "assistant", content: "bridge", bridge: true, id: "b1",
        compacted: [{ role: "user", web: true, id: "cw1", content: "[Content of https://c.example/] archived" }] },
      { kind: "tool", tool: "search", status: "running", query: "stuck", started_at: 5, id: "s1" },
    ],
    branches: [{ parent: "u1", current: 0, tails: [null, [
      { role: "user", web: true, id: "bw1", content: "[Results of web_search \"b\"] (failed: no evidence)" },
      { role: "assistant", content: "branch reply", id: "ba1" },
    ]] }],
    droppedBranches: [[
      { role: "user", web: true, id: "dw1", content: "[web access denied] dropped" },
    ]],
  };
}

test("migrateConversation: every legacy web row becomes a tool event with its text, in place, everything else untouched", () => {
  const { window } = loadApp();
  const conv = legacyConv();
  const before = JSON.parse(JSON.stringify(conv));
  window.migrateConversation(conv);
  const m = conv.messages;
  const by = (id) => m.find((x) => x.id === id);

  assert.equal(m.length, before.messages.length - 1,
    "only the stale running event is removed; nothing else is added or removed");
  for (const id of ["u1", "a1", "a2", "k1", "d1"]) {
    assert.deepEqual(JSON.parse(JSON.stringify(by(id))), before.messages.find((x) => x.id === id),
      `${id} is byte-identical`);
  }
  const w1 = by("w1");
  assert.equal(w1.kind, "tool");
  assert.equal(w1.role, undefined);
  assert.equal(w1.web, undefined);
  assert.equal(w1.content, undefined);
  assert.equal(w1.tool, "search");
  assert.equal(w1.status, "done");
  assert.equal(w1.query, "weather");
  assert.equal(w1.text, before.messages[2].content, "the original text is kept verbatim");
  assert.deepEqual(JSON.parse(JSON.stringify(w1.untrusted_spans)), [[52, 60]], "the spans are kept");
  assert.equal(window.msgText(w1), before.messages[2].content, "and still render as that text");

  assert.deepEqual([by("w2").tool, by("w2").status, by("w2").url], ["fetch", "done", "https://x.example/p"]);
  assert.deepEqual([by("w3").tool, by("w3").status, by("w3").error], ["search", "failed", undefined],
    "a legacy failure keeps its error in the verbatim text only");
  assert.deepEqual([by("w4").tool, by("w4").status], ["search", "duplicate"]);
  assert.deepEqual([by("w5").tool, by("w5").status], ["search", "denied"]);
  assert.deepEqual([by("w6").tool, by("w6").reason], ["note", "format"]);
  assert.deepEqual([by("w7").tool, by("w7").reason], ["note", "limit"]);
  assert.deepEqual([by("w8").tool, by("w8").reason], ["note", "pending"]);
  assert.deepEqual([by("w9").tool, by("w9").status, by("w9").query], ["search", "done", "old"]);
  assert.deepEqual([by("w10").tool, by("w10").status, by("w10").text], ["note", "done", "something unrecognised"]);
  for (const id of ["w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10"]) {
    assert.equal(by(id).role, undefined, `${id} has no role`);
    assert.equal(window.msgText(by(id)), before.messages.find((x) => x.id === id).content,
      `${id} still renders its original text`);
  }
  // A compaction archive nested on a bridge message is walked too.
  assert.equal(by("b1").compacted[0].kind, "tool");
  assert.equal(by("b1").compacted[0].url, "https://c.example/");
  // A tool event still `running` (the load interrupted its call) carries no
  // result and is removed, so the next turn is never told anything about it.
  assert.equal(by("s1"), undefined);
  assert.equal(m[m.length - 1].id, "b1");
  // Parked branch tails and dropped branches.
  assert.equal(conv.branches[0].tails[1][0].kind, "tool");
  assert.equal(conv.branches[0].tails[1][0].status, "done");
  assert.equal(conv.branches[0].tails[1][1].role, "assistant");
  assert.equal(conv.droppedBranches[0][0].kind, "tool");
  assert.equal(conv.droppedBranches[0][0].status, "denied");
});

test("migrateConversation: migrating twice is byte-identical to migrating once", () => {
  const { window } = loadApp();
  const conv = legacyConv();
  window.migrateConversation(conv);
  const once = JSON.stringify(conv);
  window.migrateConversation(conv);
  assert.equal(JSON.stringify(conv), once);
  // And a conversation with nothing legacy in it is untouched entirely.
  const clean = { id: "c", title: "t", messages: [
    { role: "user", content: "hi", id: "1" }, { role: "assistant", content: "yo", id: "2" },
    { kind: "tool", tool: "search", status: "done", query: "q", text: "t", id: "3" },
  ] };
  const cleanBefore = JSON.stringify(clean);
  window.migrateConversation(clean);
  assert.equal(JSON.stringify(clean), cleanBefore);
});

test("migration runs on the cached conversations at boot", () => {
  // A confirmed same-instance cache is the only one init.js paints at boot.
  const impl = async (url) => String(url) === "/v1/config"
    ? jsonResp({ effective_mode: "log", n_ctx_max: 16384, instance_id: "same" })
    : jsonResp({});
  const { window } = loadApp({
    fetchImpl: impl,
    seedLocalStorage: { "localm.instanceId": "same",
                        "localm.conversations": JSON.stringify([legacyConv()]) },
  });
  runScript(window, "window.__loaded = chat.conversations[0];");
  const loaded = window.__loaded;
  assert.equal(loaded.messages[2].kind, "tool");
  assert.equal(loaded.messages[2].role, undefined);
  assert.equal(loaded.branches[0].tails[1][0].kind, "tool");
  assert.ok(!loaded.messages.some((m) => m.web === true), "no legacy marker survives the load");
});

test("migration runs when a server conversation body is hydrated", async () => {
  const impl = async (url) => {
    if (String(url).startsWith("/api/conversations/L1")) {
      const c = legacyConv();
      return jsonResp({ id: "L1", title: "legacy", messages: c.messages, branches: c.branches });
    }
    return jsonResp({});
  };
  const { window } = loadApp({ fetchImpl: impl });
  const placeholder = { id: "L1", title: "legacy", _meta: true, messages: [] };
  setActiveConv(window, placeholder);
  assert.equal(await window.hydrateConversation(placeholder), true);
  assert.equal(placeholder.messages[2].kind, "tool");
  assert.equal(placeholder.messages[2].role, undefined);
  assert.equal(placeholder.messages[2].query, "weather");
  assert.equal(placeholder.branches[0].tails[1][0].kind, "tool");
  assert.ok(!placeholder.messages.some((m) => m.web === true));
});

// ---------------------------------------------------------------------------
//  The live push sites produce the event shape, with timing
// ---------------------------------------------------------------------------

test("runWebCall: pushes a running event, then completes it in place with the result and timing", async () => {
  const calls = [];
  const impl = async (url, opts = {}) => {
    calls.push(String(url));
    if (String(url) === "/api/web/retrieve") {
      return jsonResp(bundleOf(JSON.parse(opts.body).query, TWO_SOURCES));
    }
    return jsonResp({});
  };
  const { window } = loadApp({ fetchImpl: impl });
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  setActiveConv(window, conv);
  const seen = [];
  const saved = [];
  const busyAtRender = [];
  const origRender = window.renderChat;
  const origSave = window.saveConversations;
  window.renderChat = () => {
    seen.push(JSON.stringify(conv.messages[1]));
    runScript(window, "window.__busy = chat.webCall;");
    busyAtRender.push(window.__busy === conv.messages[1]);
    origRender();
  };
  window.saveConversations = (c) => { saved.push(JSON.parse(JSON.stringify(conv.messages[1])).status); origSave(c); };
  await window.runWebCall(conv, { name: "web_search", args: { query: "alpha" } }, "note text");
  assert.equal(calls.filter((u) => u === "/api/web/retrieve").length, 1);
  assert.ok(seen.length >= 2, "rendered while running and again when done");
  assert.equal(JSON.parse(seen[0]).status, "running", "the first render shows the call in progress");
  assert.equal(busyAtRender[0], true, "chat.webCall holds the running event while the call is in flight");
  assert.ok(!saved.includes("running"), "the running state is shown but never saved");
  assert.ok(saved.includes("done"), "the completed event is saved");
  runScript(window, "window.__busy = chat.webCall;");
  assert.equal(window.__busy, null, "chat.webCall is cleared once the call settles");
  const ev = conv.messages[1];
  assert.equal(conv.messages.length, 2, "one event, completed in place");
  assert.equal(ev.kind, "tool");
  assert.equal(ev.role, undefined);
  assert.equal(ev.status, "done");
  assert.equal(ev.query, "alpha");
  assert.equal(ev.sources.length, 2);
  assert.equal(ev.note, "note text");
  assert.ok(typeof ev.started_at === "number" && typeof ev.finished_at === "number" &&
    ev.finished_at >= ev.started_at, "timing is recorded");
});

test("while a web call is in flight, sending, /web and edit/revert are refused; the call still completes", async () => {
  let resolveRetrieve;
  const pending = new Promise((r) => { resolveRetrieve = r; });
  const calls = [];
  const impl = async (url, opts = {}) => {
    calls.push(String(url));
    if (String(url) === "/api/web/retrieve") {
      await pending;
      return jsonResp(bundleOf(JSON.parse(opts.body).query, TWO_SOURCES));
    }
    return jsonResp({});
  };
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  window.readSSE = async (_r, onData) => {
    onData(JSON.stringify({ choices: [{ delta: { content: "answer" } }] }));
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }));
  };
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;

  const running = window.runWebInChat("alpha");          // not awaited: the retrieve is pending
  await new Promise((r) => setTimeout(r, 20));
  const conv = window.currentConv();
  assert.equal(conv.messages.length, 2, "the /web turn and the running event");
  assert.equal(conv.messages[1].status, "running");
  runScript(window, "window.__busy = chat.webCall;");
  assert.equal(window.__busy, conv.messages[1], "the in-flight call is the busy state");

  // A send during the call is refused, with a toast, and pushes nothing.
  doc.getElementById("chat-input").value = "unrelated question";
  await window.sendChat();
  assert.equal(conv.messages.length, 2, "no user row was added during the call");
  assert.match(doc.getElementById("toast").textContent, /web request is still running/i);
  assert.equal(calls.filter((u) => u === "/v1/chat/completions").length, 0,
    "no completion was requested while the call was in flight");
  // So is a second /web, and the user row shows no edit/revert while busy.
  await window.runWebInChat("beta");
  assert.equal(conv.messages.length, 2);
  assert.equal(calls.filter((u) => u === "/api/web/retrieve").length, 1, "beta never ran");
  const userButtons = [...doc.querySelectorAll("#chat-messages .msg-row.user .msg-meta button")]
    .map((b) => b.textContent);
  assert.ok(!userButtons.some((b) => /edit|revert/.test(b)), `no edit/revert while busy: ${userButtons}`);

  resolveRetrieve();
  await running;
  assert.equal(conv.messages[1].status, "done");
  assert.equal(conv.messages[1].sources.length, 2);
  runScript(window, "window.__busy = chat.webCall;");
  assert.equal(window.__busy, null, "busy state cleared after completion");
  assert.equal(calls.filter((u) => u === "/v1/chat/completions").length, 1,
    "exactly one completion, after the result arrived");
  assert.equal(conv.messages[2].role, "assistant");
  assert.equal(conv.messages.length, 3, "user, tool event, one reply: no interleaved turn");
});

test("runWebCall: a refused request completes the event as failed with the error", async () => {
  const impl = async (url) => {
    if (String(url) === "/api/web/retrieve") {
      return { ok: false, status: 403, json: async () => ({ detail: "net_mode=off" }), statusText: "Forbidden" };
    }
    return jsonResp({});
  };
  const { window } = loadApp({ fetchImpl: impl });
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  setActiveConv(window, conv);
  await window.runWebCall(conv, { name: "web_search", args: { query: "alpha" } });
  const ev = conv.messages[1];
  assert.equal(ev.status, "failed");
  assert.equal(ev.error, "net_mode=off");
  assert.equal(ev.role, undefined);
  assert.match(window.msgText(ev), /^\[Web request failed: net_mode=off\] Answer without the web/);
});

// ---------------------------------------------------------------------------
//  Compaction keeps the event whole and labels it WEB in the summariser excerpt
// ---------------------------------------------------------------------------

test("archiveCopy keeps a tool event whole; compaction's excerpt labels it WEB", () => {
  const { window } = loadApp();
  const ev = searchEvent("q", TWO_SOURCES, { id: "e1" });
  const copy = window.archiveCopy(ev);
  assert.deepEqual(JSON.parse(JSON.stringify(copy)), JSON.parse(JSON.stringify(ev)));
  assert.equal(copy.role, undefined);
  assert.match(window.msgText(ev), /^\[Results of web_search "q"\]/);
});
