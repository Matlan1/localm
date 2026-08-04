// SPDX-License-Identifier: AGPL-3.0-or-later
// A chat send that fails before any token streams (e.g. a 400 "Model parameter
// is required and cannot be empty") renders "*[error: ...]*" into the live
// assistant bubble in the catch block, but the very next lines - a guard meant
// only for the VIS-1 vision-reject case - fired for ANY zero-content failure and
// unconditionally called renderChat(). renderChat() rebuilds #chat-messages
// purely from the persisted conv.messages array, which never received the
// failed turn, so the just-rendered error was silently wiped a moment later.
// The only surviving signal was a toast that auto-dismisses after 3500ms
// (helpers.js). This proves the error now stays visible.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

/** A fetch that 400s /v1/chat/completions before any token streams, exactly
 *  like a real "model is required" rejection from the server. */
function earlyFailFetch(detail) {
  const impl = async (url) => {
    if (String(url) === "/v1/chat/completions") {
      return {
        ok: false, status: 400,
        json: async () => ({ detail }),
        text: async () => JSON.stringify({ detail }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  return impl;
}

function textConv() {
  return { id: "c1", title: "t", messages: [
    { role: "user", content: "hello" },
  ] };
}

async function setup(detail) {
  const { window } = loadApp({ fetchImpl: earlyFailFetch(detail) });
  window.maybeCompactConversation = async () => {};
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;
  return { window };
}

test("a zero-content chat failure leaves a visible error, not silently wiped", async () => {
  const { window } = await setup("Model parameter is required and cannot be empty");
  const conv = textConv();

  await window.runCompletion(conv);

  const box = window.document.getElementById("chat-messages");
  assert.match(box.textContent, /error/i,
    "the rendered error must still be on screen after the send settles, not " +
    "wiped by an unconditional renderChat() rebuild");
  assert.equal(conv.messages.length, 1,
    "no blank assistant turn was persisted for the failed send");
});

test("a long detail is shown parsed, not raw JSON markup or truncated mid-word", async () => {
  // #996-adjacent: settings-perf.js used to read the raw response TEXT and
  // slice it to 300 chars, so the user saw literal `{"detail":"...` markup
  // and lost whatever came after the cutoff - for a VRAM-overflow 503 that is
  // exactly the server's actionable "Options:" list (lower the context,
  // offload fewer layers, ...), discarded right when it mattered most.
  const longDetail = "Context too large for available VRAM: this load needs " +
    "roughly 12.3 GB but this GPU only has 8.0 GB total - freeing other VRAM " +
    "will not help, it cannot fit regardless.\n  Options:\n    - Lower the " +
    "context:  -c 32768  (or smaller)\n    - Offload fewer layers:  -g 24  " +
    "(or -g 0 for CPU-only)\n    - Let localm auto-size GPU offload:  localm " +
    "config n_gpu_layers_auto true\n    - Let localm auto-size the context:  " +
    "localm config ctx_auto true";
  const { window } = await setup(longDetail);
  const conv = textConv();

  await window.runCompletion(conv);

  const box = window.document.getElementById("chat-messages");
  assert.doesNotMatch(box.textContent, /\{"detail"/,
    "the raw JSON body must never reach the user - it must be parsed and " +
    "the detail field used, not the raw response text");
  assert.match(box.textContent, /auto-size the context/,
    "the actionable tail of a long message (past the old 300-char cutoff) " +
    "must not be truncated away");
});

test("a zero-content chat failure does not touch vision-reject recovery", async () => {
  // Guard against a regression that scopes the skip too broadly: a REAL
  // vision-reject must still clear the live bubble via renderChat() (its own
  // toast + stripped-image recovery is the intended UX, not an error bubble).
  const IMG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg";
  const impl = async (url) => {
    if (String(url) === "/v1/chat/completions") {
      return { ok: false, status: 400, json: async () => ({}),
        text: async () => "This model has no vision support; remove the image." };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;

  const conv = { id: "c1", title: "t", messages: [
    { role: "user", content: [
      { type: "text", text: "what is in this picture?" },
      { type: "image_url", image_url: { url: IMG } },
    ] },
  ] };

  await window.runCompletion(conv);

  assert.equal(conv.messages.length, 1, "no blank assistant turn was persisted");
  assert.match(conv.messages[0].content, /\[Image removed/,
    "the vision-reject recovery path still strips the image as before");
});
