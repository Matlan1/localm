// SPDX-License-Identifier: AGPL-3.0-or-later
// The chat transcript shows a divider where the active model changed between
// turns. A run of turns from the same model shows none, the first model-bearing
// turn establishes the baseline, and messages with no `model` field neither
// crash nor divide.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function setupConv(window, messages) {
  runScript(window, `
    chat.conversations = [{ id: "t1", title: "t", messages: ${JSON.stringify(messages)} }];
    chat.activeId = "t1";
    window.chatState = chat;
  `);
}

test("NEW-1: a divider marks where the model changed mid-conversation", () => {
  const { window } = loadApp();
  setupConv(window, [
    { role: "user", content: "hi" },
    { role: "assistant", content: "a1", model: "gemma-4-12b" },
    { role: "user", content: "more" },
    { role: "assistant", content: "a2", model: "gemma-4-12b" },
    { role: "user", content: "switch" },
    { role: "assistant", content: "a3", model: "mistral-nemo-12b" },
  ]);
  window.renderChat();
  const dividers = window.document.querySelectorAll(".model-switch");
  assert.equal(dividers.length, 1, "exactly one switch divider (gemma -> mistral)");
  assert.match(dividers[0].textContent, /mistral-nemo-12b/);
});

test("NEW-1: no divider when every turn is the same model", () => {
  const { window } = loadApp();
  setupConv(window, [
    { role: "user", content: "hi" },
    { role: "assistant", content: "a1", model: "gemma-4-12b" },
    { role: "user", content: "more" },
    { role: "assistant", content: "a2", model: "gemma-4-12b" },
  ]);
  window.renderChat();
  assert.equal(window.document.querySelectorAll(".model-switch").length, 0);
});

test("NEW-1: legacy messages without a model field never crash or divide", () => {
  const { window } = loadApp();
  setupConv(window, [
    { role: "user", content: "hi" },
    { role: "assistant", content: "old reply" },              // no model field
    { role: "user", content: "more" },
    { role: "assistant", content: "new", model: "gemma-4-12b" },
  ]);
  window.renderChat();
  // The first model-bearing turn establishes the baseline -> no divider.
  assert.equal(window.document.querySelectorAll(".model-switch").length, 0);
});

test("NEW-1: switching back to a prior model still shows a divider each change", () => {
  const { window } = loadApp();
  setupConv(window, [
    { role: "user", content: "1" },
    { role: "assistant", content: "a", model: "gemma-4-12b" },
    { role: "user", content: "2" },
    { role: "assistant", content: "b", model: "mistral-nemo-12b" },
    { role: "user", content: "3" },
    { role: "assistant", content: "c", model: "gemma-4-12b" },
  ]);
  window.renderChat();
  const dividers = window.document.querySelectorAll(".model-switch");
  assert.equal(dividers.length, 2, "gemma->mistral and mistral->gemma");
  assert.match(dividers[0].textContent, /mistral-nemo-12b/);
  assert.match(dividers[1].textContent, /gemma-4-12b/);
});
