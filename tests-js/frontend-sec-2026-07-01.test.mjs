// SPDX-License-Identifier: AGPL-3.0-or-later
// Regression tests for the 2026-07-01 frontend-security backlog cluster.
//   R41-D4       - artifact CSP precedes any pre-<head> content; KaTeX trust:false
//   NEW-M-BROWSE - an explicit accepts_path/accepts_dir flag forces a Browse button
// (R41-D2 = the DOMPurify 3.2.6 vendor bump, verified out-of-band; P1a deferred.)
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

test("artifactSrcdoc still injects a network-blocking CSP for normal docs", () => {
  const { window: win } = loadApp();
  const full = win.artifactSrcdoc(
    "<!doctype html><html><head><title>t</title></head><body>x</body></html>", "html");
  assert.match(full, /default-src 'none'/);
  assert.ok(full.indexOf("Content-Security-Policy") < full.indexOf("</head>"));
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
  runScript(win, "refreshSettingsPage();");
  await new Promise((r) => setTimeout(r, 0));
  const doc = win.document;
  assert.ok(doc.querySelector('button[data-browse="source"]'),
    "accepts_path field gets a Browse button");
  assert.ok(doc.querySelector('button[data-browse="workspace"]'),
    "accepts_dir field gets a Browse button");
  assert.equal(doc.querySelector('button[data-browse="plain"]'), null,
    "a plain text field with no path tokens/flags gets no Browse button");
});
