// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

// H4: the server now streams the model's reasoning in delta.reasoning_content
// (clean delta.content). The GUI must rebuild <think>...</think> from it so the
// collapsible thoughts block renders and persists exactly as before.

async function run(deltas) {
  const okResponse = { ok: true, status: 200, text: async () => "", json: async () => ({}) };
  const { window } = loadApp({ fetchImpl: () => Promise.resolve(okResponse) });
  window.maybeCompactConversation = async () => {};
  window.readSSE = async (_r, onData) => {
    for (const d of deltas) onData(JSON.stringify(d));
  };
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-web").checked = false;
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  await window.runCompletion(conv);
  return { window, conv };
}

test("H4: reasoning_content deltas render a think block and persist as <think>", async () => {
  const { window, conv } = await run([
    { choices: [{ delta: { reasoning_content: "because X" } }] },
    { choices: [{ delta: { content: "The answer." } }] },
    { choices: [{ delta: {}, finish_reason: "stop" }], usage: { total_tokens: 5 } },
  ]);
  const reply = conv.messages.find((m) => m.role === "assistant");
  assert.ok(reply, "assistant reply persisted");
  // The consumer rebuilds the same content shape as before this change
  // (<think>reasoning</think>answer), so the unchanged splitThink renderer, the
  // localStorage persist, and reload all keep working - that shape is the unit
  // of behaviour my change is responsible for.
  assert.match(reply.content, /<think>[\s\S]*because X[\s\S]*<\/think>/);
  assert.match(reply.content, /The answer\./);
});

test("H4: a content-only reply (no reasoning) persists without a <think> wrapper", async () => {
  const { conv } = await run([
    { choices: [{ delta: { content: "plain answer" } }] },
    { choices: [{ delta: {}, finish_reason: "stop" }] },
  ]);
  const reply = conv.messages.find((m) => m.role === "assistant");
  assert.equal(reply.content, "plain answer");
  assert.ok(!reply.content.includes("<think>"));
});

test("H4 back-compat: an old server inlining <think> in content still renders", async () => {
  const { conv } = await run([
    { choices: [{ delta: { content: "<think>old style</think>answer" } }] },
    { choices: [{ delta: {}, finish_reason: "stop" }] },
  ]);
  const reply = conv.messages.find((m) => m.role === "assistant");
  // reasoning stays empty, so content is passed through unchanged (splitThink
  // still finds the inline block on render).
  assert.equal(reply.content, "<think>old style</think>answer");
});
