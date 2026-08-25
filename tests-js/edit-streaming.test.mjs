// SPDX-License-Identifier: AGPL-3.0-or-later
// editMessage bails while chat.abort is set, like switchBranch and regenerate.

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
  // The composer is not loaded with the message text and the conversation is
  // neither forked nor truncated.
  assert.equal(window.__u1val, "ORIGINAL");
  assert.equal(window.__u1len, 2);
});
