// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { loadApp, runScript } from "./harness.mjs";

// A settled render whose body renders to nothing shows a "(no reply text)" note.
// These tests run against the real vendored marked, not the harness's identity
// stub, so the rendered HTML matches the browser's.

const STATIC = join(dirname(fileURLToPath(import.meta.url)), "..",
  "localm", "plugins", "gui", "static");

/** Swaps the harness's identity `marked` stub for the real vendored bundle. The
 *  UMD takes its global branch in jsdom, so it sets window.marked. */
function withRealMarked(window) {
  runScript(window, readFileSync(join(STATIC, "vendor", "marked.min.js"), "utf-8"));
  assert.equal(typeof window.marked.parse, "function", "real marked loaded");
}

const EMPTY_FENCE_REPLY = "<think>\nsome reasoning here\n</think>\n```text\n";

test("a reply that renders to an empty code fence shows a note, not a blank bubble", () => {
  const { window } = loadApp();
  withRealMarked(window);
  const target = window.document.createElement("div");
  window.document.body.appendChild(target);

  window.renderMarkdown(target, EMPTY_FENCE_REPLY, { final: true });

  const note = target.querySelector(".md-main .md-empty");
  assert.ok(note, "a placeholder note replaced the blank body");
  assert.match(note.textContent, /no reply/i);
  // The empty code box is replaced, not left beside the note.
  assert.equal(target.querySelector(".md-main pre"), null, "empty code box replaced");
  // The reasoning block is still shown.
  const think = target.querySelector("details.think-block");
  assert.ok(think, "the thoughts bubble is preserved");
  assert.match(think.textContent, /some reasoning here/);
});

test("a normal reply is untouched (no note)", () => {
  const { window } = loadApp();
  withRealMarked(window);
  const target = window.document.createElement("div");
  window.document.body.appendChild(target);

  window.renderMarkdown(target, "The answer is 42.", { final: true });

  assert.equal(target.querySelector(".md-empty"), null, "no placeholder on a real reply");
  assert.match(target.querySelector(".md-main").textContent, /The answer is 42/);
});

test("mid-stream (not final) an empty body shows NO note - never flash a false 'no reply'", () => {
  const { window } = loadApp();
  withRealMarked(window);
  const target = window.document.createElement("div");
  window.document.body.appendChild(target);

  // Reasoning has streamed, the answer has not started, so the body is empty.
  window.renderMarkdown(target, "<think>\nthinking\n</think>\n", { final: false });
  assert.equal(target.querySelector(".md-empty"), null, "no placeholder while streaming");
  // Default (no opts) is also treated as non-final.
  window.renderMarkdown(target, "<think>\nthinking\n</think>\n");
  assert.equal(target.querySelector(".md-empty"), null, "default render adds no placeholder");
});

test("an image-only reply is visible content, not 'empty'", () => {
  const { window } = loadApp();
  withRealMarked(window);
  const target = window.document.createElement("div");
  window.document.body.appendChild(target);

  window.renderMarkdown(target, "![pic](data:image/png;base64,AAAA)", { final: true });
  assert.equal(target.querySelector(".md-empty"), null, "an image counts as visible");
  assert.ok(target.querySelector(".md-main img"), "the image rendered");
});

test("a math-only reply is not falsely flagged empty", () => {
  const { window } = loadApp();
  withRealMarked(window);
  const target = window.document.createElement("div");
  window.document.body.appendChild(target);

  window.renderMarkdown(target, "$x^2 + 1$", { final: true });
  assert.equal(target.querySelector(".md-empty"), null, "math source is visible content");
});

test("addMessageRow: a settled turn gets the note; a fresh streaming shell does not", () => {
  const { window } = loadApp();
  withRealMarked(window);

  // Settled: final:true, so the blank body gets the note.
  const settled = window.document.createElement("div");
  window.addMessageRow(settled, "assistant", EMPTY_FENCE_REPLY, { final: true });
  assert.ok(settled.querySelector(".md-empty"), "settled empty reply shows the note");

  // Fresh live shell: default opts, no final, so no note.
  const shell = window.document.createElement("div");
  window.addMessageRow(shell, "assistant", "");
  assert.equal(shell.querySelector(".md-empty"), null, "a fresh live shell shows no note");
});
