// SPDX-License-Identifier: AGPL-3.0-or-later
// The chat request body sent modelSelect.value verbatim with no fallback to the
// model actually active/loaded in the engine (modelCache.active - already used
// elsewhere client-side, e.g. the chat header and the sendChat empty-model
// guard). If the <select> is ever empty or desynced from the real active model
// (e.g. the dropdown has not been (re)populated yet), this sent a literal empty
// string and let the server 400 - which then silently wiped the on-screen error
// (see chat-error-render.test.mjs), producing a completely unrecoverable-looking
// failure. This proves the request now falls back to modelCache.active.
import { test } from "node:test";
import assert from "node:assert/strict";

import { loadApp, runScript } from "./harness.mjs";

function _fetchRecorder() {
  const calls = [];
  const impl = async (url, opts = {}) => {
    let body = null;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch (e) { body = opts.body; }
    calls.push({ url: String(url), body });
    if (String(url) === "/v1/chat/completions") {
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  return { impl, calls };
}

test("a desynced empty model-select falls back to modelCache.active in the request body", async () => {
  const { impl, calls } = _fetchRecorder();
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  window.readSSE = async () => {};   // no tokens needed for this assertion
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;

  // The real active model is known client-side (e.g. REG-471's immediate
  // publish on switchModel), but the <select> itself has no options yet - a
  // real desync, not a contrived one: model-select starts empty in index.html
  // until refreshModels() populates it.
  runScript(window, "modelCache.active = 'my-real-model';");
  assert.equal(doc.getElementById("model-select").value, "",
    "precondition: the dropdown is desynced/empty");

  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  await window.runCompletion(conv);

  const posts = calls.filter((c) => c.url === "/v1/chat/completions");
  assert.equal(posts.length, 1, "the chat request fired");
  assert.equal(posts[0].body.model, "my-real-model",
    "the request must fall back to modelCache.active, never post an empty model");
});

test("a populated model-select is still sent as-is (fallback does not override a real choice)", async () => {
  const { impl, calls } = _fetchRecorder();
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  window.readSSE = async () => {};
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;

  const select = doc.getElementById("model-select");
  const opt = doc.createElement("option");
  opt.value = "picked-model";
  select.appendChild(opt);
  select.value = "picked-model";
  runScript(window, "modelCache.active = 'a-different-model';");

  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  await window.runCompletion(conv);

  const posts = calls.filter((c) => c.url === "/v1/chat/completions");
  assert.equal(posts[0].body.model, "picked-model",
    "an explicit dropdown selection is never overridden by the fallback");
});
