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
  const { conv } = await run([
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

// The reasoning bubble must be toggleable WHILE the model is still streaming.
// Regression: renderMarkdown rebuilt the <details> on every token, resetting its
// open/closed state, so a mid-stream collapse was undone by the next chunk.
test("thoughts bubble toggles mid-stream and the choice sticks", () => {
  const { window } = loadApp();
  const target = window.document.createElement("div");
  window.document.body.appendChild(target);

  window.renderMarkdown(target, "<think>step one");
  const det = target.querySelector("details.think-block");
  assert.ok(det, "think block created while streaming");
  assert.equal(det.open, true, "open by default while thinking");
  assert.equal(det.querySelector("summary").textContent, "Thinking…");

  // User collapses it mid-stream (jsdom does not auto-toggle on click, so set
  // the state too; the click marks the manual override).
  det.querySelector("summary").dispatchEvent(new window.Event("click"));
  det.open = false;

  window.renderMarkdown(target, "<think>step one, step two");
  const det2 = target.querySelector("details.think-block");
  assert.equal(det2, det, "same <details> reused, not rebuilt each token");
  assert.equal(det2.open, false, "user collapse survives streaming updates");

  window.renderMarkdown(target, "<think>step one</think>Final answer.");
  assert.equal(det.open, false, "still collapsed after reasoning ends (user owns it)");
  assert.match(target.querySelector(".md-main").textContent, /Final answer/);
});

test("thoughts bubble auto-collapses when done if the user never touched it", () => {
  const { window } = loadApp();
  const target = window.document.createElement("div");
  window.document.body.appendChild(target);
  window.renderMarkdown(target, "<think>reasoning");
  assert.equal(target.querySelector("details.think-block").open, true);
  window.renderMarkdown(target, "<think>reasoning</think>answer");
  assert.equal(target.querySelector("details.think-block").open, false,
    "auto-collapses when reasoning ends and the user never toggled it");
});
