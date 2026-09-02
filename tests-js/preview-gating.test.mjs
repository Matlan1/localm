// SPDX-License-Identifier: AGPL-3.0-or-later
// The preview (artifacts canvas) permission gate: the server decides whether this
// caller is offered the canvas button, and the client draws, withdraws, and
// restores it from that answer.
//
// Structural only, like the rest of the jsdom suite: it pins WHO IS OFFERED the
// button. Whether the sandbox the button opens actually contains the artifact is
// a real-browser question and is answered in tests-e2e/preview-sandbox.spec.mjs.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

function mk(win, lang, text) {
  const pre = win.document.createElement("pre");
  const c = win.document.createElement("code");
  c.dataset.lang = lang;
  c.textContent = text;
  pre.appendChild(c);
  win.document.body.appendChild(pre);
  return pre;
}

/** A fetch stub whose /api/capabilities answers *payload*. */
function capsFetch(payload) {
  return async (url) => ({
    ok: true,
    status: 200,
    json: async () => (String(url).includes("/api/capabilities") ? payload : {}),
    text: async () => "",
  });
}

test("preview starts allowed, so the default install draws the button", () => {
  const { window: win } = loadApp();
  assert.equal(win.isPreviewAllowed(), true);
  const pre = mk(win, "html", "<b>hi</b>");
  win.enhanceCodeBlock(pre);
  assert.ok(pre.querySelector(".canvas-btn"));
});

test("a withdrawn preview means a renderable block gets no canvas button", () => {
  const { window: win } = loadApp();
  win.setPreviewAllowed(false);
  const pre = mk(win, "html", "<b>hi</b>");
  win.enhanceCodeBlock(pre);
  assert.ok(pre.querySelector(".copy-btn"), "copy is unaffected");
  assert.equal(pre.querySelector(".canvas-btn"), null);
});

test("refreshPreviewButtons removes a button drawn before the answer landed", () => {
  const { window: win } = loadApp();
  const pre = mk(win, "html", "<b>hi</b>");
  win.enhanceCodeBlock(pre);
  assert.ok(pre.querySelector(".canvas-btn"), "drawn while still allowed");
  win.setPreviewAllowed(false);
  win.refreshPreviewButtons();
  assert.equal(pre.querySelector(".canvas-btn"), null, "withdrawn on refresh");
});

test("refreshPreviewButtons restores the button when preview is granted again", () => {
  const { window: win } = loadApp();
  const pre = mk(win, "html", "<b>hi</b>");
  win.setPreviewAllowed(false);
  win.enhanceCodeBlock(pre);
  assert.equal(pre.querySelector(".canvas-btn"), null);
  win.setPreviewAllowed(true);
  win.refreshPreviewButtons();
  assert.ok(pre.querySelector(".canvas-btn"), "restored");
  assert.equal(pre.querySelectorAll(".copy-btn").length, 1, "copy not duplicated");
});

test("openArtifact refuses to open the pane when preview is not allowed", () => {
  const { window: win } = loadApp();
  const pane = win.document.getElementById("artifact-pane");
  win.setPreviewAllowed(false);
  win.openArtifact("<svg></svg>", "svg");
  assert.equal(pane.hidden, true, "the pane must stay closed");
  assert.equal(pane.querySelector("iframe"), null);
});

test("withdrawing preview tears down a pane that is already open", () => {
  const { window: win } = loadApp();
  const pane = win.document.getElementById("artifact-pane");
  win.openArtifact("<svg></svg>", "svg");
  assert.equal(pane.hidden, false, "open while allowed");
  win.setPreviewAllowed(false);
  win.refreshPreviewButtons();
  assert.equal(pane.hidden, true);
  assert.equal(pane.querySelector("iframe"), null, "iframe torn down");
});

test("a capabilities answer of preview:false withdraws the button end to end", async () => {
  const { window: win } = loadApp({ fetchImpl: capsFetch({ preview: false }) });
  const pre = mk(win, "html", "<b>hi</b>");
  win.enhanceCodeBlock(pre);
  assert.ok(pre.querySelector(".canvas-btn"), "drawn before the answer lands");
  await win.refreshPluginCommands();
  assert.equal(win.isPreviewAllowed(), false);
  assert.equal(pre.querySelector(".canvas-btn"), null, "withdrawn once the server answers");
});

test("a capabilities answer with no preview key keeps the canvas", async () => {
  const { window: win } = loadApp({ fetchImpl: capsFetch({}) });
  const pre = mk(win, "html", "<b>hi</b>");
  await win.refreshPluginCommands();
  assert.equal(win.isPreviewAllowed(), true, "an older server that omits the key");
  win.enhanceCodeBlock(pre);
  assert.ok(pre.querySelector(".canvas-btn"));
});

test("refreshPreviewButtons leaves a job-log <pre> completely alone", () => {
  const { window: win } = loadApp();
  // index.html ships several log panes: a <pre> with no <code> child. The
  // whole-document walk must not turn them into code blocks.
  const log = win.document.getElementById("pull-log");
  assert.ok(log, "the pull-log pane exists in the shipped shell");
  win.refreshPreviewButtons();
  assert.equal(log.querySelector(".copy-btn"), null, "no copy button on a log pane");
  assert.equal(log.querySelector(".canvas-btn"), null, "no canvas button on a log pane");
  assert.equal(log.childElementCount, 0, "the pane is untouched");
});

test("refreshPreviewButtons does not add a copy button to an unenhanced block", () => {
  const { window: win } = loadApp();
  const pre = mk(win, "html", "<b>hi</b>");
  win.refreshPreviewButtons();
  assert.ok(pre.querySelector(".canvas-btn"), "canvas is its job");
  assert.equal(pre.querySelector(".copy-btn"), null, "copy is renderMarkdown's job");
});
