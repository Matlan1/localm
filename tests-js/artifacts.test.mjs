// SPDX-License-Identifier: AGPL-3.0-or-later
// Artifacts canvas: renders a self-contained HTML/SVG reply block in a
// sandboxed iframe with no same-origin and a CSP that blocks network access.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

function codeEl(win, { lang, cls, text } = {}) {
  const c = win.document.createElement("code");
  if (cls) c.className = cls;
  if (lang) c.dataset.lang = lang;
  c.textContent = text || "";
  return c;
}

test("artifactLang detects html/svg and ignores other languages", () => {
  const { window: win } = loadApp();
  assert.equal(win.artifactLang(codeEl(win, { lang: "html", text: "<div>hi</div>" })), "html");
  assert.equal(win.artifactLang(codeEl(win, { lang: "svg", text: "<svg></svg>" })), "svg");
  assert.equal(win.artifactLang(codeEl(win, { cls: "language-html", text: "x" })), "html");
  // content sniff for an unlabelled block
  assert.equal(win.artifactLang(codeEl(win, { text: "<svg viewBox='0 0 1 1'></svg>" })), "svg");
  assert.equal(win.artifactLang(codeEl(win, { text: "<!doctype html><html></html>" })), "html");
  // not renderable
  assert.equal(win.artifactLang(codeEl(win, { lang: "python", text: "print(1)" })), null);
  assert.equal(win.artifactLang(codeEl(win, { lang: "js", text: "let x = 1" })), null);
  assert.equal(win.artifactLang(null), null);
});

test("artifactSrcdoc injects a network-blocking CSP", () => {
  const { window: win } = loadApp();
  const svg = win.artifactSrcdoc("<svg></svg>", "svg");
  assert.match(svg, /Content-Security-Policy/);
  assert.match(svg, /default-src 'none'/);
  assert.match(svg, /<svg><\/svg>/);
  // full document: the CSP is injected into the existing head
  const full = win.artifactSrcdoc(
    "<!doctype html><html><head><title>t</title></head><body>x</body></html>", "html");
  assert.match(full, /Content-Security-Policy/);
  assert.ok(full.indexOf("Content-Security-Policy") < full.indexOf("</head>"));
  // a fragment is wrapped in a minimal document
  const frag = win.artifactSrcdoc("<button>go</button>", "html");
  assert.match(frag, /<body><button>go<\/button><\/body>/);
});

test("openArtifact shows a hard-sandboxed iframe; close tears it down", () => {
  const { window: win } = loadApp();
  const pane = win.document.getElementById("artifact-pane");
  assert.ok(pane.hidden, "pane starts hidden");
  win.openArtifact("<svg></svg>", "svg");
  assert.equal(pane.hidden, false);
  const frame = pane.querySelector("iframe.artifact-frame");
  assert.ok(frame, "iframe present");
  const sandbox = frame.getAttribute("sandbox");
  assert.equal(sandbox, "allow-scripts");
  assert.ok(!/allow-same-origin/.test(sandbox), "must NOT grant same-origin");
  assert.match(frame.getAttribute("srcdoc") || frame.srcdoc || "", /Content-Security-Policy/);
  assert.match(pane.querySelector(".artifact-title").textContent, /svg/);
  win.closeArtifact();
  assert.equal(pane.hidden, true);
  assert.equal(pane.querySelector("iframe"), null, "iframe removed on close");
});

test("enhanceCodeBlock adds copy always and canvas only for renderable blocks", () => {
  const { window: win } = loadApp();
  const mk = (lang, text) => {
    const pre = win.document.createElement("pre");
    const c = win.document.createElement("code");
    c.dataset.lang = lang;
    c.textContent = text;
    pre.appendChild(c);
    return pre;
  };

  const htmlPre = mk("html", "<b>hi</b>");
  win.enhanceCodeBlock(htmlPre);
  assert.ok(htmlPre.querySelector(".copy-btn"));
  const canvasBtn = htmlPre.querySelector(".canvas-btn");
  assert.ok(canvasBtn, "a renderable block gets a canvas button");
  canvasBtn.onclick();   // opens the pane
  assert.equal(win.document.getElementById("artifact-pane").hidden, false);
  win.closeArtifact();

  const pyPre = mk("python", "print(1)");
  win.enhanceCodeBlock(pyPre);
  assert.ok(pyPre.querySelector(".copy-btn"));
  assert.equal(pyPre.querySelector(".canvas-btn"), null, "non-renderable: no canvas button");

  // a second call does not duplicate buttons
  win.enhanceCodeBlock(htmlPre);
  assert.equal(htmlPre.querySelectorAll(".copy-btn").length, 1);
  assert.equal(htmlPre.querySelectorAll(".canvas-btn").length, 1);
});
