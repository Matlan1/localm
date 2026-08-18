// SPDX-License-Identifier: AGPL-3.0-or-later
// The chat composer must not emit an empty-model request the server can only
// answer with a 503 "No model loaded" (Antigravity-audit gui-empty-model): when
// modelCache.active is "" (no model loaded), sendChat should refuse locally and
// tell the user, instead of round-tripping a doomed request.
import { test } from "node:test";
import assert from "node:assert/strict";

import { loadApp, runScript } from "./harness.mjs";

function _fetchRecorder(window) {
  const calls = [];
  window.fetch = (...a) => {
    calls.push(String(a[0]));
    // A minimal 503-ish response; the guard should prevent us from getting here
    // for a chat send with no model.
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

  // The full streaming path may throw on the stubbed response; we only care that
  // the guard did NOT block the send (a request was attempted).
  await window.sendChat().catch(() => {});

  const chatReqs = calls.filter((u) => u.includes("/chat/completions"));
  assert.ok(chatReqs.length >= 1,
    "a chat request fires past the guard when a model is loaded");
});

// --------------------------------------------------------------------------- //
//  ...but "nothing is resident" is NOT the same state as "nothing to load".    //
//                                                                              //
//  An idle-unload, and the sidebar's own Unload button, both leave the server  //
//  with no ACTIVE model while keeping the Engine for lazy reload - and both    //
//  tell the user it "reloads on the next chat request" (the idle-unload log    //
//  line, and the button's tooltip). The server honours that: an unnamed        //
//  request reloads and is served. The guard above used to refuse anyway,       //
//  because modelCache.active is "" in that state too - so the promise was      //
//  false and chat looked permanently broken until a model was re-picked.       //
//                                                                              //
//  /api/models now reports `resumable` for exactly that state, mirroring what  //
//  /health already resolves. The guard must tell the two apart.                //
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
  // The genuine dead end: no active model AND nothing the server could resolve.
  runScript(window, "modelCache.active = ''; modelCache.resumable = '';");
  window.document.getElementById("chat-input").value = "hello there";
  const calls = _fetchRecorder(window);

  await window.sendChat();

  const chatReqs = calls.filter((u) => u.includes("/chat/completions"));
  assert.equal(chatReqs.length, 0,
    "an empty-model request with nothing to resolve to is still a client bug");
});
