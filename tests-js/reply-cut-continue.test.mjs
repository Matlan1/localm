// SPDX-License-Identifier: AGPL-3.0-or-later
// A reply stream that ends without a finish reason is kept as cut off, never
// as a finished reply. A message that only asks the model to go on ("continue",
// "weiter", "continúa", ...) is sent as a continuation of the previous reply and
// never runs a knowledge-collection search.
import { test } from "node:test";
import assert from "node:assert/strict";

import { loadApp, runScript } from "./harness.mjs";

function setup(streams, { kb = "" } = {}) {
  const calls = [];
  const impl = async (url, opts = {}) => {
    let body;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = opts.body; }
    calls.push({ url: String(url), body });
    if (String(url).includes("/api/rag/collections/")) {
      return { ok: true, status: 200, headers: { get: () => null }, text: async () => "",
               json: async () => ({ collection: kb, hits: [
                 { source: "x/shell_guard.py", pos: 295, text: "                continue", score: 1 }] }) };
    }
    return { ok: true, status: 200, headers: { get: () => null },
             json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  window.readSSE = async (_r, onData) => {
    const chunks = streams.shift() || [{ choices: [{ delta: {}, finish_reason: "stop" }] }];
    for (const c of chunks) onData(JSON.stringify(c));
  };
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;
  if (kb) {
    const sel = doc.getElementById("p-kb");
    const opt = doc.createElement("option");
    opt.value = kb;
    opt.textContent = kb;
    sel.appendChild(opt);
    sel.value = kb;
  }
  return { window, doc, calls };
}

const content = (text) => ({ choices: [{ delta: { content: text } }] });
const finish = (reason) => ({ choices: [{ delta: {}, finish_reason: reason }] });

function activate(window, conv) {
  window.__testConv = conv;
  runScript(window, "chat.conversations = [window.__testConv]; chat.activeId = window.__testConv.id;");
  return window.currentConv();
}

const chatPosts = (calls) => calls.filter((c) => c.url === "/v1/chat/completions");
const ragPosts = (calls) => calls.filter((c) => c.url.includes("/api/rag/collections/"));

test("a stream that ends without a finish reason is saved as cut off and marked", async () => {
  const { window, doc } = setup([[content("Step one. Step two"), ]]);
  const conv = activate(window, { id: "c1", title: "t", messages: [{ role: "user", content: "explain" }] });
  await window.runCompletion(conv);
  const reply = conv.messages[conv.messages.length - 1];
  assert.equal(reply.role, "assistant");
  assert.equal(reply.content, "Step one. Step two");
  assert.equal(reply.cut, true, "the reply is flagged as cut off");
  const shown = doc.getElementById("chat-messages").textContent;
  assert.match(shown, /cut off/, "the transcript shows the cut-off marker");
});

test("a cut reply holding half a web tool call is not re-prompted or run", async () => {
  const { window, doc, calls } = setup([[content("Let me look. <tool_call>{\"name\": \"web_se")]]);
  doc.getElementById("p-web").checked = true;
  const conv = activate(window, { id: "c1", title: "t", messages: [{ role: "user", content: "news?" }] });
  await window.runCompletion(conv);
  assert.equal(chatPosts(calls).length, 1, "no follow-up completion after a cut stream");
  const last = conv.messages[conv.messages.length - 1];
  assert.equal(last.role, "assistant");
  assert.equal(last.cut, true);
});

test("a cut reply is not read aloud", async () => {
  const { window, doc } = setup([[content("Half of the answer")]]);
  doc.getElementById("p-speak").checked = true;
  const spoken = [];
  window.speak = (text) => spoken.push(text);
  const conv = activate(window, { id: "c1", title: "t", messages: [{ role: "user", content: "explain" }] });
  await window.runCompletion(conv);
  assert.deepEqual(spoken, []);
});

test("a finished reply is still read aloud", async () => {
  const { window, doc } = setup([[content("Whole answer."), finish("stop")]]);
  doc.getElementById("p-speak").checked = true;
  const spoken = [];
  window.speak = (text) => spoken.push(text);
  const conv = activate(window, { id: "c1", title: "t", messages: [{ role: "user", content: "explain" }] });
  await window.runCompletion(conv);
  assert.deepEqual(spoken, ["Whole answer."]);
});

test("a reply that finished with stop carries no cut-off flag", async () => {
  const { window, doc } = setup([[content("All done."), finish("stop")]]);
  const conv = activate(window, { id: "c1", title: "t", messages: [{ role: "user", content: "explain" }] });
  await window.runCompletion(conv);
  const reply = conv.messages[conv.messages.length - 1];
  assert.equal(reply.cut, undefined);
  assert.doesNotMatch(doc.getElementById("chat-messages").textContent, /cut off/);
});

test("a length stop keeps its own max-tokens marker, not the cut-off one", async () => {
  const { window } = setup([[content("partial"), finish("length")]]);
  const conv = activate(window, { id: "c1", title: "t", messages: [{ role: "user", content: "explain" }] });
  await window.runCompletion(conv);
  const reply = conv.messages[conv.messages.length - 1];
  assert.equal(reply.truncated, true);
  assert.equal(reply.cut, undefined);
});

test("an empty stream with no finish reason leaves a marker and saves nothing", async () => {
  const { window, doc } = setup([[]]);
  const conv = activate(window, { id: "c1", title: "t", messages: [{ role: "user", content: "explain" }] });
  await window.runCompletion(conv);
  assert.equal(conv.messages.length, 1, "no assistant row is persisted");
  assert.match(doc.getElementById("chat-messages").textContent, /ended before any text arrived/);
});

test("the cut-off flag survives the compaction archive copy", () => {
  const { window } = setup([]);
  const copy = window.archiveCopy({ role: "assistant", content: "x", cut: true });
  assert.equal(copy.cut, true);
});

for (const phrase of ["continue", "Continue.", "continue please", "please continue!",
                      "Go on", "keep going", "weiter", "Mach weiter", "bitte weiter",
                      "continúa", "continua", "continuez", "ga door", "продолжай", "继续",
                      "続けて", "ok, continue"]) {
  test(`"${phrase}" is a continue request`, () => {
    const { window } = setup([]);
    assert.equal(window.isContinueRequest(phrase), true);
  });
}

for (const phrase of ["go ahead", "yes, go ahead", "ok proceed", "Proceed!", "vas-y", "finish it",
                      "continue the list from item 3", "how does continue work in python",
                      "what does the continue keyword do", "ok", "please", "", "weiter oben steht was?",
                      "explain derived_request"]) {
  test(`"${phrase}" is not a continue request`, () => {
    const { window } = setup([]);
    assert.equal(window.isContinueRequest(phrase), false);
  });
}

test("continue after a reply skips the knowledge search and asks for a continuation", async () => {
  const { window, calls } = setup([[content(" and then the rest."), finish("stop")]], { kb: "localm" });
  const conv = activate(window, { id: "c1", title: "t", messages: [
    { role: "user", content: "how does derived_request work" },
    { role: "assistant", content: "It copies the scope and", cut: true },
  ] });
  await window.dispatchChatTurn(conv, "continue");
  assert.equal(ragPosts(calls).length, 0, "no knowledge query for a continue request");
  assert.equal(conv.messages.filter((m) => m.tag === "kb").length, 0, "no excerpts injected");
  const sent = chatPosts(calls)[0].body.messages;
  const last = sent[sent.length - 1];
  assert.equal(last.role, "user");
  assert.ok(last.content.startsWith("continue\n\n["), "the user's word is kept first");
  assert.match(last.content, /Continue your previous reply from exactly where it ends/);
  assert.equal(sent[sent.length - 2].content, "It copies the scope and",
    "the cut reply is sent as the assistant turn being continued");
  const stored = conv.messages.find((m) => m.role === "user" && m.content === "continue");
  assert.equal(stored.continues, true);
});

test("an ordinary question still searches the knowledge collection", async () => {
  const { window, calls } = setup([[content("answer"), finish("stop")]], { kb: "localm" });
  const conv = activate(window, { id: "c1", title: "t", messages: [
    { role: "user", content: "hi" }, { role: "assistant", content: "hello" },
  ] });
  await window.dispatchChatTurn(conv, "how does derived_request work");
  assert.equal(ragPosts(calls).length, 1);
  const stored = conv.messages.find((m) => m.content === "how does derived_request work");
  assert.equal(stored.continues, undefined);
});

test("continue as the first message is not flagged as a continuation", async () => {
  const { window, calls } = setup([[content("?"), finish("stop")]], { kb: "localm" });
  const conv = activate(window, { id: "c1", title: "t", messages: [] });
  await window.dispatchChatTurn(conv, "continue");
  assert.equal(ragPosts(calls).length, 0, "a bare continue never searches the collection");
  assert.equal(conv.messages[0].continues, undefined);
  const sent = chatPosts(calls)[0].body.messages;
  assert.equal(sent[sent.length - 1].content, "continue");
});

test("continuing a failed reply sends it without its inference-error text", async () => {
  const { window, calls } = setup([[content(" rest"), finish("stop")]]);
  const conv = activate(window, { id: "c1", title: "t", messages: [
    { role: "user", content: "explain" },
    { role: "assistant", failed: true,
      content: "Half of the answer\n[inference error: The reply was cut off: the model was unloaded or replaced while it was being generated.]" },
  ] });
  await window.dispatchChatTurn(conv, "weiter");
  const sent = chatPosts(calls)[0].body.messages;
  assert.equal(sent[sent.length - 2].content, "Half of the answer");
});
