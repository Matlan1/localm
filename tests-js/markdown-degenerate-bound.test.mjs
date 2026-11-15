// SPDX-License-Identifier: AGPL-3.0-or-later
// A reply of degenerate emphasis marks (one kind or mixed, in runs or spread
// over a block) renders in well under a second through the real vendored
// marked and DOMPurify with its text shown; ordinary emphasis, thematic breaks,
// lists and code keep rendering as before; a renderer that throws leaves the
// reply as plain text, never an empty bubble.
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

// *s* without whitespace and backticks.
const bare = (s) => s.replace(/[\s`]+/g, "");

const RUN = "**".repeat(20000);
const CASES = {
  "marks between two letters": "x" + RUN + "x",
  "marks after a lead-in": "Here is the answer: " + RUN,
  "marks between spaced words": "x " + RUN + " x",
  "underscores between letters": "x" + "_".repeat(40000) + "x",
  "single stars between letters": "x" + "*".repeat(40000) + "x",
  "alternating star and underscore on a line of their own": "*_".repeat(20480),
  "alternating underscore and star between letters": "x" + "_*".repeat(20000) + "x",
  "spaced single stars, then alternating marks": "x *".repeat(10000) + "_*".repeat(15500),
  "unmatched star openers between words": "a *b ".repeat(10000),
  "unmatched underscore openers between words": "a _b ".repeat(10000),
  "unmatched double-star openers between words": "a **b ".repeat(8000),
  "unmatched closers between words": "a* b ".repeat(10000),
  "short mixed runs between words": "a *_*_*_* b ".repeat(4000),
  "marks split by inline triple-backtick spans": "a *b ```c``` ".repeat(4000),
};

for (const [name, text] of Object.entries(CASES)) {
  test(`${(text.length / 1024).toFixed(0)} KB of ${name} renders in under a second with its text shown`, () => {
    const win = realPipeline();
    render(win, "warm up *a*");
    const { target, ms } = render(win, text);
    assert.ok(ms < 1000, `${name}: ${ms.toFixed(0)} ms`);
    const shown = bare(target.textContent);
    assert.ok(shown.length >= bare(text).length - 2, `${name}: the text is shown, not dropped (${shown.length} chars)`);
    assert.doesNotMatch(target.textContent, /\\[*_]/, "no escape backslash is shown");
  });
}

test("a 1 MB line of mixed marks is escaped in linear time", () => {
  const { window: win } = loadApp();
  const inputs = {
    mixed: "*_".repeat(500000),
    "short runs": "*_*_*_* ".repeat(130000),
    backslashes: "\\".repeat(1000000) + "*",
    "unmatched openers": "a *b ".repeat(200000),
    "spaced lines": ("\n" + " ".repeat(1000)).repeat(1000) + "*",
  };
  for (const [name, text] of Object.entries(inputs)) {
    const t0 = performance.now();
    const out = win.escapeDelimiterRuns(text);
    const ms = performance.now() - t0;
    assert.ok(ms < 1000, `${name}: ${ms.toFixed(0)} ms`);
    const strip = (s) => s.replace(/\\(?=[*_])/g, "");
    assert.equal(strip(out), strip(text), `${name}: only escapes were added`);
  }
});

test("a run of eight or more marks in any mix is escaped; a shorter one is not", () => {
  const { window: win } = loadApp();
  assert.equal(win.escapeDelimiterRuns("a *_*_*_*_ b"), "a \\*\\_\\*\\_\\*\\_\\*\\_ b");
  assert.equal(win.escapeDelimiterRuns("a *_*_*_* b"), "a *_*_*_* b");
  assert.equal(win.escapeDelimiterRuns("\\*_*_*_*_*"), "\\*\\_\\*\\_\\*\\_\\*\\_\\*");
  assert.equal(win.escapeDelimiterRuns("`*_*_*_*_` and\n```\n*_*_*_*_\n```"), "`*_*_*_*_` and\n```\n*_*_*_*_\n```");
});

test("ordinary emphasis, thematic breaks and code render as before", () => {
  const win = realPipeline();
  const { target } = render(win,
    "**bold** and *it* and _under_ and ****nested****\n\n***\n\n___\n\n* * *\n\n"
    + "`a ******** b` and `*_*_*_*_*_`\n\n```\nx ************ y\n_*_*_*_*_*_*\n```");
  assert.equal(target.querySelector("strong").textContent, "bold");
  const ems = [...target.querySelectorAll("em")].map((e) => e.textContent);
  assert.ok(ems.includes("it") && ems.includes("under"), `emphasis kept: ${ems}`);
  assert.equal(target.querySelectorAll("hr").length, 3, "lines of one kind of mark are still thematic breaks");
  const codes = [...target.querySelectorAll("code")].map((c) => c.textContent);
  assert.ok(codes.includes("a ******** b"), "marks inside inline code are untouched");
  assert.ok(codes.includes("*_*_*_*_*_"), "mixed marks inside inline code are untouched");
  assert.ok(codes.some((c) => c.includes("x ************ y") && c.includes("_*_*_*_*_*_*")),
    "marks inside a fence are untouched");
  assert.doesNotMatch(target.textContent, /\\[*_]/, "no escape backslash is shown");
});

test("a block keeps its emphasis up to the per-block mark limit and shows marks as text above it", () => {
  const win = realPipeline();
  const cap = win.eval("EMPHASIS_MARKS_PER_BLOCK");
  const bolds = (n) => "**b** ".repeat(n).trim();
  const at = render(win, bolds(cap / 4)).target;
  assert.equal(at.querySelectorAll("strong").length, cap / 4, "a block at the limit keeps its emphasis");
  const over = render(win, bolds(cap / 4 + 1)).target;
  assert.equal(over.querySelectorAll("strong").length, 0, "a block over the limit has no emphasis");
  assert.ok(over.textContent.includes("**b**"), "its marks are shown as text");
  const split = render(win, bolds(cap / 4) + "\n\n" + bolds(cap / 4)).target;
  assert.equal(split.querySelectorAll("strong").length, cap / 2, "the limit counts each block on its own");
  const fenced = render(win, bolds(cap / 4) + "\n```\ncode\n```\n" + bolds(cap / 4)).target;
  assert.equal(fenced.querySelectorAll("strong").length, cap / 2, "a fenced code block ends a block");
  const ruled = render(win, bolds(cap / 4) + "\n***\n" + bolds(cap / 4)).target;
  assert.equal(ruled.querySelectorAll("strong").length, cap / 2, "a thematic break ends a block");
});

test("a bullet list keeps its emphasis past the limit, one item at a time", () => {
  const win = realPipeline();
  const cap = win.eval("EMPHASIS_MARKS_PER_BLOCK");
  const n = cap / 4 + 10;
  for (const marker of ["*", "-", "+"]) {
    const { target } = render(win, `${marker} item **b**\n`.repeat(n));
    assert.equal(target.querySelectorAll("li").length, n, `${marker}: every item is kept`);
    assert.equal(target.querySelectorAll("li strong").length, n, `${marker}: every item keeps its emphasis`);
  }
});

test("lines that do not end a paragraph in marked do not end a block", () => {
  const { window: win } = loadApp();
  const cap = win.eval("EMPHASIS_MARKS_PER_BLOCK");
  const n = cap / 4 + 1;
  const escaped = (s) => /\\\*/.test(win.escapeDelimiterRuns(s));
  assert.ok(!escaped("a *b ".repeat(cap)), "a block at the limit is left alone");
  assert.ok(escaped("a *b\n`c`".repeat(cap + 1)), "a line holding only inline code is not a blank line");
  assert.ok(escaped("intro\n" + "2. a *b\n".repeat(cap + 1)), "a list item numbered other than 1 does not interrupt a paragraph");
  assert.ok(escaped("a *b\n    ***\n".repeat(n)), "a thematic break indented four spaces is paragraph text");
  assert.ok(escaped("a *b\n+ \n".repeat(cap + 1)), "an empty bullet does not interrupt a paragraph");
});

test("many blocks at the per-block limit render in under a second", () => {
  const win = realPipeline();
  const cap = win.eval("EMPHASIS_MARKS_PER_BLOCK");
  const block = "x_*".repeat(cap / 2);
  for (const [name, text] of [["paragraphs", (block + "\n\n").repeat(50)], ["bullet items", ("- " + block + "\n").repeat(50)]]) {
    render(win, "warm up *a*");
    const { target, ms } = render(win, text);
    assert.ok(ms < 1000, `${name}: ${ms.toFixed(0)} ms`);
    assert.ok(target.textContent.length > 10000, `${name}: the text is shown`);
  }
});

test("a renderer that throws leaves the reply as plain text and records why", () => {
  const win = realPipeline();
  win.marked = { parse() { throw new RangeError("Maximum call stack size exceeded"); } };
  const warned = [];
  win.console.warn = (...a) => warned.push(a.map(String).join(" "));
  const text = "Some *reply* <img src=x onerror=alert(1)> with $x$ and more";
  const { target } = render(win, "<think>thinking *hard*</think>" + text);
  const plain = target.querySelector(".md-main .md-plain");
  assert.ok(plain, "the reply body is a plain text box");
  assert.equal(plain.textContent, text);
  assert.equal(target.querySelector("img"), null, "the text is not parsed as HTML");
  assert.equal(target.querySelector(".md-empty"), null, "the reply is not shown as empty");
  assert.equal(target.querySelector(".think-block .md-plain").textContent, "thinking *hard*");
  assert.ok(warned.some((w) => w.includes("Markdown rendering failed")), `console note: ${warned}`);
  const recorded = win.eval("recentGuiErrors()").map((e) => e.text);
  assert.ok(recorded.some((r) => r.includes("Maximum call stack size exceeded")), `recorded: ${recorded}`);
});
