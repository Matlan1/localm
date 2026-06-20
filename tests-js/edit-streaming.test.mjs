// SPDX-License-Identifier: AGPL-3.0-or-later
// U1: clicking "edit" while a reply is streaming corrupted the branch tree
// (editMessage forks before the streaming reply lands). editMessage must bail
// while chat.abort is set, like switchBranch / regenerate.

import assert from "node:assert";
import { test } from "node:test";

import { loadApp, runScript } from "./harness.mjs";

test("editMessage is a no-op while a reply is streaming", () => {
  const { window } = loadApp();
  runScript(window, `
    chat.abort = new AbortController();   // simulate an active stream
    const conv = { id: "c1", branches: {}, messages: [
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi" },
    ] };
    document.getElementById("chat-input").value = "ORIGINAL";
    try { editMessage(conv, 0); } catch (e) { window.__u1err = String(e); }
    window.__u1val = document.getElementById("chat-input").value;
    window.__u1len = conv.messages.length;
  `);
  // Guard returned early: the composer was not loaded with the message text and
  // the conversation was not forked/truncated. Pre-fix, editMessage set the
  // input to "hello" and forked before returning.
  assert.equal(window.__u1val, "ORIGINAL");
  assert.equal(window.__u1len, 2);
});
