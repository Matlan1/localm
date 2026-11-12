// SPDX-License-Identifier: AGPL-3.0-or-later
// A refusal never removes an image from the conversation. A text-only model
// that refuses images gets earlier turns' images left out of later requests to
// it; a model that can read images still gets them. An image the server could
// not use is kept on screen and never sent again. No blank reply is saved, so
// the chat stays usable (VIS-1).

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const IMG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg";

/** A fetch that answers the FIRST /v1/chat/completions (the image) with
 *  *status* and *detail*, then 200s. */
function visionRejectFetch(status = 400,
  detail = "This model has no vision support; remove the image.", refusal = "image") {
  const calls = [];
  const server = { active: "text-model" };
  let chatCalls = 0;
  const impl = async (url, opts = {}) => {
    let body;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = opts.body; }
    calls.push({ url: String(url), body });
    if (String(url).startsWith("/api/models")) {
      const data = { models: [{ name: "text-model" }, { name: "vision-model", vision: true }],
                     active: server.active };
      return { ok: true, status: 200, json: async () => data, text: async () => "" };
    }
    if (String(url) === "/v1/chat/completions") {
      chatCalls += 1;
      if (chatCalls === 1) {
        return { ok: false, status, json: async () => ({ detail }), text: async () => detail,
          headers: { get: (k) => (k === "X-Localm-Refusal" ? refusal : null) } };
      }
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  return { impl, calls, server };
}

function imageConv() {
  return { id: "c1", title: "t", messages: [
    { role: "user", content: [
      { type: "text", text: "what is in this picture?" },
      { type: "image_url", image_url: { url: IMG } },
    ] },
  ] };
}

async function setup(status, detail, refusal) {
  const { impl, calls, server } = visionRejectFetch(status, detail, refusal);
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
  const select = doc.getElementById("model-select");
  for (const name of ["text-model", "vision-model"]) {
    const opt = doc.createElement("option");
    opt.value = name;
    select.appendChild(opt);
  }
  select.value = "text-model";
  runScript(window, `modelCache.active = "text-model"; modelCache.models = ` +
    `[{ name: "text-model" }, { name: "vision-model", vision: true }];`);
  return { window, calls, server };
}

const posted = (calls) => calls.filter((c) => c.url === "/v1/chat/completions");
const sentImages = (body) => body.messages.flatMap((m) =>
  Array.isArray(m.content) ? m.content.filter((p) => p.type === "image_url") : []);

test("a 400 image refusal keeps every image in the conversation and saves no blank reply", async () => {
  const { window } = await setup();
  const conv = imageConv();
  await window.runCompletion(conv);

  assert.equal(window.msgImages(conv.messages[0]).length, 1, "the image stays");
  assert.match(window.msgText(conv.messages[0]), /what is in this picture\?/);
  assert.doesNotMatch(JSON.stringify(conv.messages[0].content), /Image removed/);
  assert.equal(conv.messages.length, 1,
    "no blank assistant turn was persisted (the wedge symptom)");
});

test("after a refusal the refusing model gets no image, and the chat answers", async () => {
  const { window, calls } = await setup();
  const conv = imageConv();
  await window.runCompletion(conv);   // image refused by the selected model
  conv.messages.push({ role: "user", content: "then just tell me a joke" });
  await window.runCompletion(conv);   // a normal follow-up turn

  const last = posted(calls).pop();
  assert.equal(sentImages(last.body).length, 0, "the refusing model is not sent the image again");
  assert.match(JSON.stringify(last.body.messages), /not included in this request/,
    "the model is told an image was left out");
  assert.ok(conv.messages.some((m) => m.role === "assistant" &&
    /ask me anything/.test(String(m.content))), "the model's reply is saved");
  assert.equal(window.msgImages(conv.messages[0]).length, 1, "the image is still in the chat");
});

test("after a refusal a model that can read images is sent the earlier image", async () => {
  const { window, calls, server } = await setup();
  const conv = imageConv();
  await window.runCompletion(conv);   // refused by the selected model
  server.active = "vision-model";
  runScript(window, 'modelCache.active = "vision-model";');
  window.document.getElementById("model-select").value = "vision-model";
  conv.messages.push({ role: "user", content: "and now?" });
  await window.runCompletion(conv);

  const last = posted(calls).pop();
  assert.equal(last.body.model, "vision-model");
  assert.equal(sentImages(last.body).length, 1, "the image reaches a model that was not refused");
  assert.equal(JSON.stringify(last.body).includes("refusedBy"), false,
    "display-only fields never go on the wire");
});

test("a 413 image refusal keeps both images and never sends the refused one again", async () => {
  const { window, calls } = await setup(413,
    "The attached image could not be read (UnidentifiedImageError).");
  const conv = { id: "c1", title: "t", messages: [
    { role: "user", content: [
      { type: "text", text: "first picture" },
      { type: "image_url", image_url: { url: IMG + "AAAA" } },
    ] },
    { role: "assistant", content: "A red square." },
    { role: "user", content: [
      { type: "text", text: "second picture" },
      { type: "image_url", image_url: { url: IMG + "BBBB" } },
    ] },
  ] };
  await window.runCompletion(conv);

  assert.equal(window.msgImages(conv.messages[0]).length, 1, "the earlier image stays");
  assert.equal(window.msgImages(conv.messages[2]).length, 1, "the refused image stays on screen");
  assert.equal(conv.messages.length, 3, "no blank assistant turn was persisted");

  conv.messages.push({ role: "user", content: "what about the first one?" });
  await window.runCompletion(conv);
  const last = JSON.stringify(posted(calls).pop().body.messages);
  assert.equal(last.includes("BBBB"), false, "the refused image is not sent again");
  assert.equal(last.includes("AAAA"), true, "the earlier image is still sent");
});

test("a 413 that is not an image refusal keeps the conversation's images", async () => {
  const { window } = await setup(413,
    "The conversation is too long for this model.", "");
  const conv = imageConv();
  await window.runCompletion(conv);

  assert.equal(window.msgImages(conv.messages[0]).length, 1,
    "the image stays in history");
  assert.doesNotMatch(JSON.stringify(conv.messages[0].content), /\[Image removed/);
});

test("a 400 about something other than the image keeps the conversation's images", async () => {
  const { window } = await setup(400, "Invalid grammar: could not parse the rule.", "");
  const conv = imageConv();
  await window.runCompletion(conv);

  assert.equal(window.msgImages(conv.messages[0]).length, 1,
    "the image stays in history");
  assert.doesNotMatch(JSON.stringify(conv.messages[0].content), /\[Image removed/);
});
