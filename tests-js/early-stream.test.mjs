// SPDX-License-Identifier: AGPL-3.0-or-later
// A chat stream the server opens before it finished preparing (a model load,
// memory recall) shows each phase as a status, carries the response headers
// in a `localm_headers` chunk, and reports a refusal in its terminal chunk's
// `localm_error`, which the chat handles exactly like the HTTP error.
import { test } from "node:test";
import assert from "node:assert/strict";

import { loadApp, runScript } from "./harness.mjs";

const IMG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg";

function setup(streams, { web = false } = {}) {
  const calls = [];
  const impl = async (url, opts = {}) => {
    let body;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = opts.body; }
    calls.push({ url: String(url), body });
    return { ok: true, status: 200, headers: { get: () => null },
             json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  const seenStatus = [];
  window.readSSE = async (_r, onData) => {
    const chunks = streams.shift() || [{ choices: [{ delta: {}, finish_reason: "stop" }] }];
    for (const c of chunks) {
      onData(JSON.stringify(c));
      const label = window.document.querySelector(".msg-status-indicator .status-text");
      if (label) seenStatus.push(label.textContent);
    }
  };
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = web;
  return { window, doc, calls, seenStatus };
}

const status = (text, code) => ({ choices: [{ delta: { status: text, status_code: code } }] });
const content = (text) => ({ choices: [{ delta: { content: text } }] });
const stop = { choices: [{ delta: {}, finish_reason: "stop" }] };
const refusal = (code, detail) => [
  content(detail),
  { choices: [{ delta: {}, finish_reason: "error" }],
    localm_error: { status: code, detail } },
];
function activateConv(window, conv) {
  window.__testConv = conv;
  runScript(window, "chat.conversations = [window.__testConv]; chat.activeId = window.__testConv.id;");
}

const chatPosts = (calls) => calls.filter((c) => c.url === "/v1/chat/completions");

test("loading and recall phases show as localized statuses before the reply", async () => {
  const { window, seenStatus } = setup([[
    status("Loading model...", "loading_model"),
    status("Recalling memories...", "recalling_memory"),
    content("hello"), stop,
  ]]);
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  await window.runCompletion(conv);
  assert.deepEqual(seenStatus, ["Loading model…", "Recalling memories…"]);
  assert.equal(conv.messages[conv.messages.length - 1].content, "hello");
});

test("routing carried in a localm_headers chunk records the answering model", async () => {
  const routing = { resolved: "seer", requested: "plain", routed: true, pinned: false,
                    gaps: { vision: "absent" }, unmet: [] };
  const { window, doc } = setup([[
    status("Loading model...", "loading_model"),
    { choices: [{ delta: {}, finish_reason: null }],
      localm_headers: { "X-Localm-Model-Routing": JSON.stringify(routing) } },
    content("an answer"), stop,
  ]]);
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "what is this?" }] };
  activateConv(window, conv);
  await window.runCompletion(conv);
  const reply = conv.messages[conv.messages.length - 1];
  assert.equal(reply.model, "seer");
  assert.ok(doc.querySelector(".routed-chip"), "the routed reply carries a chip");
});

test("an in-stream refusal is shown once, as an error, and not saved as a reply", async () => {
  const detail = "Failed to load the model: weights missing";
  const { window, doc } = setup([refusal(503, detail)]);
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  await window.runCompletion(conv);
  const body = doc.getElementById("chat-messages").textContent;
  assert.equal(body.split(detail).length - 1, 1, "the detail appears exactly once");
  assert.match(body, /error: 503/);
});

test("an in-stream image refusal drops the image like the HTTP 400 does", async () => {
  const { window } = setup([refusal(400, "This model cannot read images.")]);
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: [
    { type: "text", text: "what is this?" },
    { type: "image_url", image_url: { url: IMG } },
  ] }] };
  await window.runCompletion(conv);
  assert.equal(window.msgImages(conv.messages[0]).length, 0);
  assert.equal(conv.messages.length, 1, "no blank reply was persisted");
});

test("an in-stream grammar refusal retries the turn unconstrained", async () => {
  const detail = "This model cannot constrain generation to a grammar, so the "
    + "requested grammar would be ignored and the reply would not match it.";
  const { window, calls } = setup([refusal(400, detail), [content("2 + 2 = 4."), stop]],
                                  { web: true });
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "2+2?" }] };
  await window.runCompletion(conv);
  const posts = chatPosts(calls);
  assert.equal(posts.length, 2);
  assert.ok(posts[0].body.grammar_lazy);
  assert.ok(!("grammar" in posts[1].body));
  assert.equal(conv.messages[conv.messages.length - 1].content, "2 + 2 = 4.");
});
