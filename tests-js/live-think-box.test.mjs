// SPDX-License-Identifier: AGPL-3.0-or-later
// While a thinking model reasons, the live reply shows an open "Thinking..." box;
// once the answer starts the box reads "Thoughts" and collapses.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadApp } from "./harness.mjs";

const APP = new URL("../localm/plugins/gui/static/app/", import.meta.url);
const src = (n) => fs.readFileSync(new URL(n, APP), "utf8");

function live(win, reasoning, content) {
  const target = win.document.createElement("div");
  win.document.body.appendChild(target);
  win.renderMarkdown(target, win.liveThinkText(reasoning, content));
  return target;
}

test("reasoning with no answer yet renders an open Thinking box", () => {
  const { window: win } = loadApp();
  const det = live(win, "step one", "").querySelector("details.think-block");
  assert.ok(det, "the box exists while the model reasons");
  assert.equal(det.open, true);
  assert.equal(det.querySelector("summary").textContent, "Thinking…");
});

test("once the answer starts the box reads Thoughts and collapses", () => {
  const { window: win } = loadApp();
  const det = live(win, "step one", "The answer").querySelector("details.think-block");
  assert.equal(det.open, false);
  assert.equal(det.querySelector("summary").textContent, "Thoughts");
});

test("no reasoning leaves the content untouched", () => {
  const { window: win } = loadApp();
  assert.equal(win.liveThinkText("", "hello"), "hello");
  assert.equal(win.liveThinkText("", ""), "");
});

test("the chat and coder live renders both go through liveThinkText", () => {
  for (const f of ["settings-perf.js", "coder.js"]) {
    assert.match(src(f), /liveThinkText\(/, f);
  }
  assert.doesNotMatch(src("settings-perf.js"),
    /renderMarkdown\(liveBody,\s*reasoning \?/);
  assert.doesNotMatch(src("coder.js"), /liveReasoning \?/);
});
