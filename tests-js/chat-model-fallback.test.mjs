// SPDX-License-Identifier: AGPL-3.0-or-later
// The chat request body's model field: modelSelect.value when the dropdown has
// a selection, modelCache.active when it is empty or desynced.
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
  window.readSSE = async () => {};
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;

  // model-select starts empty in index.html until refreshModels() populates it.
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
