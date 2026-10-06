// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

// Mistral's [TOOL_CALLS] token, written out as text by some finetunes, must not
// reach the rendered reply.

test("scrubMarkers removes every [TOOL_CALLS] token and keeps the text", () => {
  const { window } = loadApp();
  assert.equal(window.scrubMarkers("[TOOL_CALLS] The grep found no matches"),
    " The grep found no matches");
  assert.equal(window.scrubMarkers("[TOOL_CALLS]".repeat(300)), "");
  assert.equal(window.scrubMarkers("keep [tool_calls] and [TOOL_CALL] and [1]"),
    "keep [tool_calls] and [TOOL_CALL] and [1]");
});

test("a rendered reply carrying the token shows no token", () => {
  const { window } = loadApp();
  const target = window.document.createElement("div");
  window.document.body.appendChild(target);
  window.renderMarkdown(target, "[TOOL_CALLS][TOOL_CALLS] Nothing matched.", { final: true });
  assert.ok(!target.textContent.includes("TOOL_CALLS"), target.textContent);
  assert.match(target.textContent, /Nothing matched\./);
});
