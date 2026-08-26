// SPDX-License-Identifier: AGPL-3.0-or-later
// sendChat refuses locally when modelCache.active is "" and nothing is resumable.
import { test } from "node:test";
import assert from "node:assert/strict";

import { loadApp, runScript } from "./harness.mjs";

function _fetchRecorder(window) {
  const calls = [];
  window.fetch = (...a) => {
    calls.push(String(a[0]));
    // A minimal 503 "No model loaded" response.
    return Promise.resolve({
      ok: false, status: 503, body: null,
      json: async () => ({ detail: "No model loaded" }),
      text: async () => "No model loaded",
    });
  };
  return calls;
}

test("sendChat refuses to send a chat request when no model is loaded", async () => {
  const { window } = loadApp();
  runScript(window, "modelCache.active = '';");        // no model loaded
  window.document.getElementById("chat-input").value = "hello there";
  const calls = _fetchRecorder(window);

  await window.sendChat();

  const chatReqs = calls.filter((u) => u.includes("/chat/completions"));
  assert.equal(chatReqs.length, 0,
    "no chat request should fire without a model loaded");
});

test("sendChat proceeds past the guard when a model IS loaded", async () => {
  const { window } = loadApp();
  runScript(window, "modelCache.active = 'my-model';"); // a model is loaded
  window.document.getElementById("chat-input").value = "hello there";
  const calls = _fetchRecorder(window);

  // The streaming path may throw on the stubbed response.
  await window.sendChat().catch(() => {});

  const chatReqs = calls.filter((u) => u.includes("/chat/completions"));
  assert.ok(chatReqs.length >= 1,
    "a chat request fires past the guard when a model is loaded");
});

// --------------------------------------------------------------------------- //
//  modelCache.resumable names a model the server reloads on the next request,  //
//  so active === "" alone does not mean there is nothing to send to.           //
// --------------------------------------------------------------------------- //

test("sendChat SENDS when the model is unloaded but resumable (idle-unload / Unload button)", async () => {
  const { window } = loadApp();
  runScript(window, "modelCache.active = ''; modelCache.resumable = 'my-model';");
  window.document.getElementById("chat-input").value = "hello there";
  const calls = _fetchRecorder(window);

  await window.sendChat().catch(() => {});

  const chatReqs = calls.filter((u) => u.includes("/chat/completions"));
  assert.ok(chatReqs.length >= 1,
    "the request that triggers the promised reload must not be blocked");
});

test("sendChat still refuses when there is nothing to resume either", async () => {
  const { window } = loadApp();
  // No active model and nothing to resume.
  runScript(window, "modelCache.active = ''; modelCache.resumable = '';");
  window.document.getElementById("chat-input").value = "hello there";
  const calls = _fetchRecorder(window);

  await window.sendChat();

  const chatReqs = calls.filter((u) => u.includes("/chat/completions"));
  assert.equal(chatReqs.length, 0,
    "an empty-model request with nothing to resolve to is still a client bug");
});
