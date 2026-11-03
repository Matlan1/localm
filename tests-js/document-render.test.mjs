// SPDX-License-Identifier: AGPL-3.0-or-later
// A changelog or update note is a document, not a model reply: a literal
// `<think>` in it must not open a "Thinking..." box. Drives the shipped
// renderMarkdown against the real vendored marked and DOMPurify.
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

const CHANGELOG = fs.readFileSync(new URL("../CHANGELOG.md", import.meta.url), "utf8");
const NOTE = "reasoning model's `` `<think>` `` block is not applied. The Ollama API's\n" +
  "`` `options.stop` `` uses it.\n\nLater text.";

function renderVia(win, text, opts) {
  const target = win.document.createElement("div");
  win.document.body.appendChild(target);
  win.renderMarkdown(target, text, opts);
  return target;
}

test("document render: the real changelog has no think block and shows <think> as code", () => {
  const win = realPipeline();
  const t = renderVia(win, CHANGELOG, { document: true });
  assert.equal(t.querySelectorAll("details.think-block").length, 0);
  const codes = [...t.querySelectorAll("code")].map((c) => c.textContent);
  assert.ok(codes.includes("<think>"), "<think> shown as code");
  assert.ok(codes.includes("options.stop"), "options.stop shown as code");
});

test("document render: a note quoting <think> keeps its text and code", () => {
  const win = realPipeline();
  const t = renderVia(win, NOTE, { document: true });
  assert.equal(t.querySelectorAll("details.think-block").length, 0);
  assert.match(t.textContent, /Later text\./);
});

test("chat render: a real leading think block still gets its box", () => {
  const win = realPipeline();
  const t = renderVia(win, "<think>pondering</think>The answer.");
  const d = t.querySelector("details.think-block");
  assert.ok(d, "think block present");
  assert.match(d.textContent, /pondering/);
  assert.match(t.querySelector(".md-main").textContent, /The answer\./);
});

test("chat render: a reply quoting <think> in code is not split", () => {
  const win = realPipeline();
  for (const reply of [NOTE, "Use the `<think>` tag.\n\nMore.", "Example:\n\n```\n<think>x</think>\n```\n\nDone."]) {
    const t = renderVia(win, reply);
    assert.equal(t.querySelectorAll("details.think-block").length, 0, reply);
  }
});

test("chat render: a still-open real think block streams as open", () => {
  const win = realPipeline();
  const t = renderVia(win, "<think>partial `code` so far");
  assert.equal(t.querySelector("details.think-block").open, true);
});
