// SPDX-License-Identifier: AGPL-3.0-or-later
// A tool_result lands on the tool card with the same call id, not the oldest
// pending card, so two calls finishing out of order keep their own results.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

function session(window) {
  const feedEl = window.document.createElement("div");
  return { info: { id: "s1" }, feedEl, pendingCards: [], eventLog: [],
           confirmCards: new Map(), inlineBrowserAttempted: true };
}

const call = (tool, call_id) => ({ type: "tool_call", tool, args: {}, call_id });
const result = (call_id, summary, ok = true) =>
  ({ type: "tool_result", tool: "x", ok, summary, call_id });

function stateOf(s, tool) {
  const card = [...s.feedEl.querySelectorAll(".tool-card")]
    .find((c) => c.textContent.includes(tool));
  return card.querySelector(".state").textContent;
}

test("out-of-order results are matched by call id", () => {
  const { window } = loadApp();
  const s = session(window);
  window.handleCoderEvent(s, call("browser_navigate", "c1"));
  window.handleCoderEvent(s, call("browser_read", "c2"));
  window.handleCoderEvent(s, result("c2", "no browser session", false));
  window.handleCoderEvent(s, result("c1", "opened example.com"));
  assert.match(stateOf(s, "browser_navigate"), /opened example\.com/);
  assert.match(stateOf(s, "browser_read"), /no browser session/);
});

test("events without a call id still pair oldest first", () => {
  const { window } = loadApp();
  const s = session(window);
  window.handleCoderEvent(s, call("first_tool", undefined));
  window.handleCoderEvent(s, call("second_tool", undefined));
  window.handleCoderEvent(s, result(undefined, "one"));
  assert.match(stateOf(s, "first_tool"), /one/);
});

test("the markdown export lists each result under its own call", () => {
  const { window } = loadApp();
  const log = [call("browser_navigate", "c1"), call("browser_read", "c2"),
    result("c2", "no browser session", false), result("c1", "opened example.com")];
  const lines = window.coderLogToMarkdown(log, "# h").split("\n");
  const nav = lines.findIndex((l) => l.includes("browser_navigate"));
  const read = lines.findIndex((l) => l.includes("browser_read"));
  assert.match(lines[nav + 1], /opened example\.com/);
  assert.match(lines[read + 1], /no browser session/);
});
