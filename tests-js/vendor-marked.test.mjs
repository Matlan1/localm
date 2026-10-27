// SPDX-License-Identifier: AGPL-3.0-or-later
// Guards the hand-vendored marked at
// localm/plugins/gui/static/vendor/marked.min.js, which is the release's
// lib/marked.umd.js stored under the file name index.html loads.
//
// The file has no package.json entry, so no dependency tooling sees it. This
// test pins the exact artefact (banner plus content hash), checks that it parses
// real markdown, checks the option set app/helpers.js applies at load, and pins
// the pipeline contract that marked does NOT sanitize: DOMPurify runs on its
// output.
//
// window.marked.version is undefined, so the pin is bytes rather than a runtime
// version floor. A test below asserts the field stays absent, so a release that
// adds one fails the guard and gets upgraded to a floor.
//
// Advisory status for 18.1.2: OSV reports 0 vulns (control queries:
// marked@18.0.1 -> 1, marked@4.0.0 -> 2) and the repository's advisory list has
// no range containing 18.1.2 (GHSA-6v9c-7cg6-27q7 covers 18.0.0 to 18.0.1, fixed
// in 18.0.2).
//
// The hash is taken over CRLF-normalised bytes so a Windows checkout with
// autocrlf and a Linux checkout agree.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import crypto from "node:crypto";
import { JSDOM } from "jsdom";

const VENDOR = new URL("../localm/plugins/gui/static/vendor/", import.meta.url);

// Provenance pin. sha256, base64, over CRLF-normalised bytes. Recorded from the
// npm registry artefact, tarball shasum-verified against the registry's own
// published dist.shasum (358c831ed0c4278f97e517af6769e2f213b048ac) before
// extraction:  npm pack marked@18.1.2  ->  package/lib/marked.umd.js
const VENDORED_VERSION = "18.1.2";
const PINNED_HASH = "nrcw0jYq4ajGIBaY8XAJ4WZxvmlG09BlA93AV1KD23w=";

function readVendored(name) {
  return fs.readFileSync(new URL(name, VENDOR));
}
/** sha256 (base64) over CRLF-normalised bytes. */
function normalisedHash(buf) {
  const lf = Buffer.from(buf.toString("latin1").replace(/\r\n/g, "\n"), "latin1");
  return crypto.createHash("sha256").update(lf).digest("base64");
}

/** Load the REAL vendored marked.min.js as a classic script in a jsdom window,
 *  the same load path index.html uses (<script src="/vendor/marked.min.js">). */
function loadVendoredMarked() {
  const dom = new JSDOM("<!doctype html><html><head></head><body></body></html>",
    { runScripts: "dangerously" });
  const tag = dom.window.document.createElement("script");
  tag.textContent = readVendored("marked.min.js").toString("utf8");
  dom.window.document.head.appendChild(tag);
  return dom.window;
}

test("the vendored marked loads and exposes the entry points helpers.js calls", () => {
  const win = loadVendoredMarked();
  try {
    assert.ok(win.marked, "the vendored file did not define window.marked");
    assert.equal(typeof win.marked.parse, "function",
      "marked.parse is missing; app/helpers.js renders every message with it");
    assert.equal(typeof win.marked.setOptions, "function",
      "marked.setOptions is missing; app/helpers.js calls it at load time, so "
      + "the whole GUI would throw on startup");
  } finally { win.close(); }
});

test("marked still exposes NO runtime version, so the byte pin is the only signal", () => {
  const win = loadVendoredMarked();
  try {
    assert.equal(win.marked.version, undefined,
      "marked now exposes a runtime version. Replace this test with a real "
      + "version-floor assertion and cross-check it against the banner, the way "
      + "vendor-dompurify.test.mjs does.");
  } finally { win.close(); }
});

test("the banner comment says 18.1.2", () => {
  const banner = readVendored("marked.min.js").toString("latin1").slice(0, 300);
  const m = banner.match(/marked v(\d+\.\d+\.\d+)/);
  assert.ok(m, "no version banner found in the first 300 bytes: "
    + JSON.stringify(banner.slice(0, 120)));
  assert.equal(m[1], VENDORED_VERSION,
    `banner says v${m[1]}, VENDORED_VERSION says ${VENDORED_VERSION}`);
});

test("the vendored bytes are the recorded upstream npm artefact", () => {
  const got = normalisedHash(readVendored("marked.min.js"));
  assert.equal(got, PINNED_HASH,
    `marked.min.js does not match the pinned marked@${VENDORED_VERSION} npm `
    + "artefact. If this was a deliberate bump, update VENDORED_VERSION and "
    + "PINNED_HASH together with vendor/README.md, and re-check the advisory "
    + "ranges for the new version; if not, the vendored file has been modified "
    + "or corrupted.");
});

test("the vendored marked actually parses real markdown", () => {
  // Not "the module loads": a guard that only checks the file parses would pass
  // on a truncated or half-replaced drop that cannot tokenise anything. Same
  // shape as the jsQR decode test and the KaTeX render test.
  const win = loadVendoredMarked();
  try {
    const out = win.marked.parse("# Title\n\n- one\n- two\n\n`code` and **bold**");
    assert.match(out, /<h1[^>]*>Title<\/h1>/, "no heading emitted for real markdown");
    assert.match(out, /<li>one<\/li>/, "no list item emitted");
    assert.match(out, /<code>code<\/code>/, "no inline code emitted");
    assert.match(out, /<strong>bold<\/strong>/, "no strong emphasis emitted");
  } finally { win.close(); }
});

test("marked accepts the exact options helpers.js sets at load", () => {
  const win = loadVendoredMarked();
  try {
    win.marked.setOptions({ breaks: true });
    assert.match(win.marked.parse("a\nb"), /<br\s*\/?>/,
      "breaks:true no longer turns a single newline into a <br>, so helpers.js's "
      + "load-time options are not taking effect");
  } finally { win.close(); }
});

test("dropping the legacy mangle/headerIds options changes nothing", () => {
  const win = loadVendoredMarked();
  try {
    win.marked.setOptions({ breaks: true });
    const out = win.marked.parse("# Title\n\n<me@example.com>");
    assert.doesNotMatch(out, /<h1[^>]* id=/, "headings now carry generated ids");
    assert.match(out, /href="mailto:me@example\.com"/,
      "autolinked email addresses are no longer emitted verbatim");
    assert.doesNotMatch(out, /&#x?[0-9a-f]+;/i, "email addresses are entity-obfuscated");
  } finally { win.close(); }
});

test("marked.parse returns a string synchronously", () => {
  const win = loadVendoredMarked();
  try {
    win.marked.setOptions({ breaks: true });
    assert.equal(typeof win.marked.parse("x"), "string",
      "helpers.js assigns the result of marked.parse straight into DOMPurify.sanitize");
    assert.equal(win.marked.parse(""), "");
  } finally { win.close(); }
});

test("marked does NOT sanitize, which is why DOMPurify is load-bearing", () => {
  const win = loadVendoredMarked();
  try {
    const out = win.marked.parse('<img src=x onerror="alert(1)">\n\n[l](javascript:alert(1))');
    assert.match(out, /onerror/,
      "marked now strips event handlers; localm relies on DOMPurify for that");
    assert.match(out, /javascript:/,
      "marked now filters javascript: URLs; localm relies on DOMPurify for that");
  } finally { win.close(); }
});
