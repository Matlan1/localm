// SPDX-License-Identifier: AGPL-3.0-or-later
// VIS-1: attaching an image to a text-only model used to WEDGE the chat - the
// server 400s the image, the client kept it in history and re-sent it every
// turn, so every following assistant reply was empty and unrecoverable. The fix
// drops the rejected image from history on a 400 and persists no blank reply, so
// the chat stays usable (text only). This proves recovery, not just the strip.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const IMG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg";

/** A fetch that 400s the FIRST /v1/chat/completions (the image) then 200s. */
function visionRejectFetch() {
  const calls = [];
  let chatCalls = 0;
  const impl = async (url, opts = {}) => {
    let body = null;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch (e) { body = opts.body; }
    calls.push({ url: String(url), body });
    if (String(url) === "/v1/chat/completions") {
      chatCalls += 1;
      if (chatCalls === 1) {
        return { ok: false, status: 400, json: async () => ({}),
          text: async () => "This model has no vision support; remove the image." };
      }
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  return { impl, calls };
}

function imageConv() {
  return { id: "c1", title: "t", messages: [
    { role: "user", content: [
      { type: "text", text: "what is in this picture?" },
      { type: "image_url", image_url: { url: IMG } },
    ] },
  ] };
}

async function setup() {
  const { impl, calls } = visionRejectFetch();
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  window.readSSE = async (_r, onData) => {
    for (const d of [
      { choices: [{ delta: { content: "I cannot see images, but ask me anything." } }] },
      { choices: [{ delta: {}, finish_reason: "stop" }] },
    ]) onData(JSON.stringify(d));
  };
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;
  return { window, calls };
}

test("VIS-1: a 400 image reject drops the image and saves no blank reply", async () => {
  const { window } = await setup();
  const conv = imageConv();
  await window.runCompletion(conv);

  assert.equal(window.msgImages(conv.messages[0]).length, 0,
    "the rejected image is removed from history");
  assert.equal(typeof conv.messages[0].content, "string");
  assert.match(conv.messages[0].content, /\[Image removed/);
  assert.match(conv.messages[0].content, /what is in this picture\?/,
    "the user's text is preserved when the image is stripped");
  assert.equal(conv.messages.length, 1,
    "no blank assistant turn was persisted (the wedge symptom)");
});

test("VIS-1: the chat is recoverable - the next turn sends no image and answers", async () => {
  const { window, calls } = await setup();
  const conv = imageConv();
  await window.runCompletion(conv);   // image rejected, stripped
  await window.runCompletion(conv);   // a normal follow-up turn

  const posts = calls.filter((c) => c.url === "/v1/chat/completions");
  const last = posts[posts.length - 1];
  const reSentImage = last.body.messages.some((m) => Array.isArray(m.content) &&
    m.content.some((p) => p.type === "image_url"));
  assert.equal(reSentImage, false, "the recovered turn re-sends no image");
  assert.ok(conv.messages.some((m) => m.role === "assistant" &&
    /ask me anything/.test(String(m.content))), "the model's reply is saved");
});
