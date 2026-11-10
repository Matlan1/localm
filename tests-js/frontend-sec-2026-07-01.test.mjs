// SPDX-License-Identifier: AGPL-3.0-or-later
// Frontend security checks: the artifact CSP precedes any pre-<head> content,
// KaTeX runs with trust:false, and an explicit accepts_path / accepts_dir flag
// forces a Browse button.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

test("artifactSrcdoc puts the CSP before a pre-head <script> (R41-D4)", () => {
  const { window: win } = loadApp();
  const evil = "<!doctype html><html><script>fetch('//evil')</script>"
    + "<head><title>t</title></head><body>x</body></html>";
  const out = win.artifactSrcdoc(evil, "html");
  assert.match(out, /Content-Security-Policy/);
  assert.ok(
    out.indexOf("Content-Security-Policy") < out.indexOf("<script>"),
    "the CSP meta must be parsed before any executable node the artifact placed "
    + "before its own <head>");
});

// Each artifact places a script somewhere a splice anchored on one of its own
// tags could miss; the parsed srcdoc must hold the CSP meta in <head>, ahead of
// every script in document order.
const SCRIPT_PLACEMENTS = {
  "before <html>": "<script>A()</script><html><body>x</body></html>",
  "before <!doctype>": "<script>A()</script><!doctype html><html><body>x</body></html>",
  "between <html> and <head>":
    "<!doctype html><html lang='en'><script>A()</script><head></head><body>x</body></html>",
  "inside svg in a document":
    "<html><body><svg><script>A()</script></svg></body></html>",
  "after a commented-out <html>":
    "<!-- <html> --><script>A()</script><html><body>x</body></html>",
  "inside an attribute-looking comment":
    "<html a='<!--'><script>A()</script><head></head></html>",
  "after an early </head>":
    "</head><script>A()</script><html><head></head><body>x</body></html>",
  "uppercase tags": "<SCRIPT>A()</SCRIPT><HTML><HEAD></HEAD><BODY>x</BODY></HTML>",
};

for (const [where, code] of Object.entries(SCRIPT_PLACEMENTS)) {
  test(`artifactSrcdoc parses the CSP before a script placed ${where}`, () => {
    const { window: win } = loadApp();
    const out = win.artifactSrcdoc(code, "html");
    const doc = new win.DOMParser().parseFromString(out, "text/html");
    const metas = [...doc.querySelectorAll('meta[http-equiv="Content-Security-Policy"]')];
    assert.equal(metas.length, 1, out);
    const csp = metas[0];
    assert.equal(csp.parentElement, doc.head, "a CSP meta outside <head> is ignored");
    assert.match(csp.getAttribute("content"), /default-src 'none'/);
    const scripts = [...doc.querySelectorAll("script")];
    assert.ok(scripts.length >= 1, `the artifact script was lost: ${out}`);
    for (const s of scripts) {
      assert.ok(
        csp.compareDocumentPosition(s) & win.Node.DOCUMENT_POSITION_FOLLOWING,
        `a script ${where} precedes the CSP: ${out}`);
    }
  });
}

test("artifactSrcdoc keeps the artifact's html attributes and doctype choice", () => {
  const { window: win } = loadApp();
  const out = win.artifactSrcdoc(
    "<!doctype html><html lang='de' class='dark'><head><title>t</title></head>"
    + "<body data-x='1'>x</body></html>", "html");
  const doc = new win.DOMParser().parseFromString(out, "text/html");
  assert.equal(doc.documentElement.getAttribute("lang"), "de");
  assert.equal(doc.documentElement.getAttribute("class"), "dark");
  assert.equal(doc.body.getAttribute("data-x"), "1");
  assert.equal(doc.title, "t");
  assert.equal(doc.compatMode, "CSS1Compat");
  const noDoctype = win.artifactSrcdoc("<html><body>x</body></html>", "html");
  assert.ok(!/<!doctype/i.test(noDoctype), noDoctype);
});

test("artifactSrcdoc still injects a network-blocking CSP for normal docs", () => {
  const { window: win } = loadApp();
  const full = win.artifactSrcdoc(
    "<!doctype html><html><head><title>t</title></head><body>x</body></html>", "html");
  assert.match(full, /default-src 'none'/);
  assert.ok(full.indexOf("Content-Security-Policy") < full.indexOf("</head>"));
});

test("artifactSrcdoc denies form submission, which default-src 'none' does not (R41)", () => {
  // Checked across all three shapes artifactSrcdoc emits, each of which splices
  // the CSP in by a different route.
  const { window: win } = loadApp();
  const shapes = {
    "bare svg": win.artifactSrcdoc("<svg viewBox='0 0 1 1'></svg>", "svg"),
    "full document": win.artifactSrcdoc(
      "<!doctype html><html><head></head><body>x</body></html>", "html"),
    "fragment": win.artifactSrcdoc("<p>hello</p>", "html"),
  };
  for (const [shape, out] of Object.entries(shapes)) {
    assert.match(out, /form-action 'none'/,
      `the ${shape} artifact carries no form-action, so a model-authored `
      + "<form> inside it could submit to any origin");
  }
});

test("renderMarkdown pins renderMathInElement trust:false (R41-D4)", () => {
  const { window: win } = loadApp();
  runScript(win, "renderMathInElement = (el, opts) => { globalThis.__mathOpts = opts; };");
  const target = win.document.createElement("div");
  win.renderMarkdown(target, "inline $x^2$ math");
  const opts = win.__mathOpts;
  assert.ok(opts, "renderMathInElement was called");
  assert.equal(opts.trust, false, "KaTeX trust must be pinned false");
});

test("accepts_path / accepts_dir force a Browse button regardless of key/label (NEW-M-BROWSE)", async () => {
  const SCHEMA = { fields: [
    { key: "source", widget: "text", label: "Source", accepts_path: true,
      group: "X", owner: "core", default: "" },
    { key: "workspace", widget: "text", label: "Workspace", accepts_dir: true,
      group: "X", owner: "core", default: "" },
    { key: "plain", widget: "text", label: "Plain", group: "X", owner: "core", default: "" },
  ]};
  const fetchImpl = async (url) => {
    if (url === "/v1/config/schema")
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  // Host-path fields render only for a host-access caller: let init.js's
  // capabilities fetch settle, then pin host.
  await new Promise((r) => setTimeout(r, 0));
  runScript(win, `caps.fsAccess = "host"; refreshSettingsPage();`);
  await new Promise((r) => setTimeout(r, 0));
  const doc = win.document;
  assert.ok(doc.querySelector('button[data-browse="source"]'),
    "accepts_path field gets a Browse button");
  assert.ok(doc.querySelector('button[data-browse="workspace"]'),
    "accepts_dir field gets a Browse button");
  assert.equal(doc.querySelector('button[data-browse="plain"]'), null,
    "a plain text field with no path tokens/flags gets no Browse button");
});
