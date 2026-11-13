// SPDX-License-Identifier: AGPL-3.0-or-later
// A reply of about 40 KB of degenerate emphasis marks renders in well under a
// second through the real vendored marked and DOMPurify, and ordinary
// emphasis, thematic breaks and code keep rendering as before.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadApp } from "./harness.mjs";

const VENDOR = new URL("../localm/plugins/gui/static/vendor/", import.meta.url);

function realPipeline() {
  const { window: win } = loadApp();
  for (const f of ["marked.min.js", "purify.min.js"]) {
    const tag = win.document.createElement("script");
    tag.textContent = fs.readFileSync(new URL(f, VENDOR), "utf8");
    win.document.head.appendChild(tag);
  }
  win.marked.setOptions({ breaks: true });
  return win;
}

function render(win, text) {
  const target = win.document.createElement("div");
  win.document.body.appendChild(target);
  const t0 = performance.now();
  win.renderMarkdown(target, text, { final: true });
  return { target, ms: performance.now() - t0 };
}

const RUN = "**".repeat(20000);
const CASES = {
  "marks between two letters": "x" + RUN + "x",
  "marks after a lead-in": "Here is the answer: " + RUN,
  "marks between spaced words": "x " + RUN + " x",
  "underscores between letters": "x" + "_".repeat(40000) + "x",
  "single stars between letters": "x" + "*".repeat(40000) + "x",
};

for (const [name, text] of Object.entries(CASES)) {
  test(`40 KB of ${name} renders in under a second`, () => {
    const win = realPipeline();
    render(win, "warm up *a*");
    const { target, ms } = render(win, text);
    assert.ok(ms < 1000, `${name}: ${ms.toFixed(0)} ms`);
    assert.ok(target.textContent.length > 39000, "the marks are shown as text, not dropped");
  });
}

test("ordinary emphasis, a thematic break and code render as before", () => {
  const win = realPipeline();
  const { target } = render(win,
    "**bold** and *it* and ****nested****\n\n***\n\n`a ******** b`\n\n```\nx ************ y\n```");
  assert.equal(target.querySelector("strong").textContent, "bold");
  assert.equal(target.querySelector("em").textContent, "it");
  assert.ok(target.querySelector("hr"), "a line of marks is still a thematic break");
  const codes = [...target.querySelectorAll("code")].map((c) => c.textContent);
  assert.ok(codes.includes("a ******** b"), "marks inside inline code are untouched");
  assert.ok(codes.some((c) => c.includes("x ************ y")), "marks inside a fence are untouched");
  assert.doesNotMatch(target.textContent, /\\\*/, "no escape backslash is shown");
});
