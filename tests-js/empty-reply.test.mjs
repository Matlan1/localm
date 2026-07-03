// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { loadApp, runScript } from "./harness.mjs";

// A tiny model (the reported case: a 1B GLM/Gemma) can produce a reply whose
// <think> works but whose visible answer is only a bare, unterminated ```code
// fence - which marked renders as an EMPTY <pre><code> (a blank box). The chat
// then showed nothing at all. The renderer must never leave a blank reply
// bubble: on a settled render it shows a plain "(no reply text)" note instead.
//
// The jsdom harness stubs marked with an identity parse, which would NOT
// reproduce the empty-code-fence render (the raw string is non-empty). So these
// tests load the REAL vendored marked bundle into the window, so renderMarkdown
// produces the exact HTML the browser does - the faithful reproduction. The
// placeholder therefore only appears if the body genuinely rendered to nothing.

const STATIC = join(dirname(fileURLToPath(import.meta.url)), "..",
  "localm", "plugins", "gui", "static");

/** Swap the harness's identity `marked` stub for the real vendored bundle so the
 *  renderer produces browser-identical HTML. The UMD's global branch runs in
 *  jsdom (no module/exports/define), so it sets window.marked to real marked. */
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
  // The empty code box must be gone, not left sitting next to the note.
  assert.equal(target.querySelector(".md-main pre"), null, "empty code box replaced");
  // The reasoning the model DID produce is still shown - we did not drop the turn.
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

  // A slow model: reasoning has streamed, the answer has not started yet. The
  // wrapper always closes <think>, so the body is momentarily empty.
  window.renderMarkdown(target, "<think>\nthinking\n</think>\n", { final: false });
  assert.equal(target.querySelector(".md-empty"), null, "no placeholder while streaming");
  // Default (no opts) is also treated as non-final - the streaming shell path.
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

  // Settled (from history / renderChat): final:true -> note for the blank body.
  const settled = window.document.createElement("div");
  window.addMessageRow(settled, "assistant", EMPTY_FENCE_REPLY, { final: true });
  assert.ok(settled.querySelector(".md-empty"), "settled empty reply shows the note");

  // Fresh live shell (settings-perf/coder create it with ""): default opts, no
  // final -> no note, so the empty shell never flashes before its first token.
  const shell = window.document.createElement("div");
  window.addMessageRow(shell, "assistant", "");
  assert.equal(shell.querySelector(".md-empty"), null, "a fresh live shell shows no note");
});
