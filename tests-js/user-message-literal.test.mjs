// SPDX-License-Identifier: AGPL-3.0-or-later
// CHAT-1: a user's OWN message renders LITERALLY (exactly as typed) - markdown is the
// model's output format, not the user's input. So typed *asterisks*, # hashes, pasted
// code/URLs, and line breaks are shown verbatim, never silently reformatted. The
// assistant's reply still goes through the markdown renderer.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const safe = async () => ({
  ok: true, status: 200,
  json: async () => ({ models: [], active: "", plugins: [] }),
  text: async () => "",
});

test("a user message is rendered literally (no markdown container, text verbatim)", () => {
  const { window } = loadApp({ fetchImpl: safe });
  const box = window.document.createElement("div");
  const raw = "*not italic* and # not a heading\nsecond line  with  spaces";
  window.addMessageRow(box, "user", raw);
  const body = box.querySelector(".msg-row.user .msg-body");
  assert.ok(body, "user row + body rendered");
  assert.ok(body.classList.contains("msg-literal"), "user body is the literal variant");
  assert.equal(body.textContent, raw, "user text appears exactly as typed");
  assert.equal(body.querySelector(".md-main"), null,
    "no markdown container is built for a user message");
});

test("an assistant message still goes through the markdown renderer", () => {
  const { window } = loadApp({ fetchImpl: safe });
  const box = window.document.createElement("div");
  window.addMessageRow(box, "assistant", "*italic*");
  const body = box.querySelector(".msg-row.assistant .msg-body");
  assert.ok(body, "assistant row + body rendered");
  assert.equal(body.classList.contains("msg-literal"), false,
    "assistant is NOT literal");
  assert.ok(body.querySelector(".md-main"),
    "assistant content goes through renderMarkdown (.md-main container)");
});

test("a user message cannot inject HTML (textContent, not innerHTML)", () => {
  const { window } = loadApp({ fetchImpl: safe });
  const box = window.document.createElement("div");
  window.addMessageRow(box, "user", "<img src=x onerror=alert(1)> <b>hi</b>");
  const body = box.querySelector(".msg-row.user .msg-body");
  assert.equal(body.querySelector("img"), null, "no element from user-typed HTML");
  assert.equal(body.querySelector("b"), null, "angle-bracket text stays inert");
  assert.ok(body.textContent.includes("<b>hi</b>"), "shown as literal text");
});
