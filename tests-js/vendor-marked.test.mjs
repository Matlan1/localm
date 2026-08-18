// SPDX-License-Identifier: AGPL-3.0-or-later
// Guards the HAND-VENDORED marked at
// localm/plugins/gui/static/vendor/marked.min.js.
//
// WHY THIS FILE EXISTS. Same reason as vendor-dompurify.test.mjs,
// vendor-katex.test.mjs, vendor-jsqr.test.mjs and vendor-highlightjs.test.mjs:
// this file has no entry in package.json, so no dependency tooling can ever see
// it and a vulnerable version here produces no PR, no alert and no red check.
// vendor/README.md named marked as the LAST library still without a guard
// ("marked still has no guard. Until it does, check it by hand when you touch
// this directory"); this closes it, so every vendored library is now pinned by
// something that actually runs.
//
// ADVISORY STATUS AT THE TIME OF WRITING, established from the affected-VERSION
// RANGES rather than by reading upstream code for a quoted vulnerable function.
// An advisory documents a bug where it is easiest to explain, and upstream often
// fixes it in the CALLER - so presence of the quoted code is not proof of
// vulnerability, and its absence is not proof of a fix. Two independent sources,
// each with a control query that returned a known-positive, so a zero could not
// be a broken-query artefact:
//   OSV (api.osv.dev)   marked@12.0.2 -> 0 vulns
//                       control: marked@4.0.0 -> 2, dompurify@3.2.6 -> 19
//   GitHub advisory DB  all 18 marked advisories fetched, every affected range
//                       tested against 12.0.2 -> none matches
//                       control: 4.0.0 matches both "< 4.0.10" ReDoS ranges
// Every marked advisory except one is bounded above by 4.0.10 or lower; the one
// later advisory (GHSA-6v9c-7cg6-27q7, OOM via infinite recursion) is
// ">= 18.0.0, <= 18.0.1" and so sits ABOVE this build, not below it.
//
// THERE IS NO RUNTIME VERSION TO ASSERT, and that is measured rather than
// assumed: window.marked.version is undefined on this build (the export carries
// parse/Lexer/Parser/use/... and no version field). So unlike the DOMPurify and
// highlight.js guards, this one CANNOT cross-check a banner against a runtime
// value and cannot express a ">= floor" against the library itself. It pins the
// EXACT artefact instead - banner plus content hash - which is the same
// treatment vendor/README.md already documents for auto-render.min.js,
// katex.min.css and jsQR.js. A test below asserts the version field is still
// absent, so if a future marked starts exposing one this guard fails and gets
// upgraded to a real floor rather than silently continuing to pin bytes.
//
// NO ReDoS TIMING TEST, deliberately, and that is a difference from the
// highlight.js guard rather than an omission. That one anchors on a NAMED
// upstream fix (issue #4362) with a concrete PoC, so its growth-shape threshold
// was derived by measuring a known-bad build against a known-good one. marked
// 12.0.2 has no such known defect to anchor on, so any threshold here would be
// invented - and an invented threshold on a shared, loaded box is how a gate
// starts flaking and then gets disabled, which is worse than no gate because
// everyone assumes it is still running.
//
// THE HASH IS TAKEN OVER CRLF-NORMALISED BYTES, and that is load-bearing rather
// than tidiness (see vendor-katex/jsqr/highlightjs for the same note). This repo
// has core.autocrlf=true and no .gitattributes rule for vendor/, so the checked
// out file differs in size from the git blob purely by line-ending conversion.
// MEASURED for this file: 35485 bytes in a Windows working tree, 35479 after
// normalising - exactly the 6 newlines in its banner comment. A raw-byte hash
// would pass on one platform and fail on the other, and the failure would read
// as a corrupted vendor drop rather than a line-ending artefact.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import crypto from "node:crypto";
import { JSDOM } from "jsdom";

const VENDOR = new URL("../localm/plugins/gui/static/vendor/", import.meta.url);

// Provenance pin. sha256, base64, over CRLF-normalised bytes. Recorded from the
// npm registry artefact, tarball shasum-verified against the registry's own
// published dist.shasum (b31578fe608b599944c69807b00f18edab84647e) before
// extraction:  npm pack marked@12.0.2  ->  package/marked.min.js
// NOTE the path: marked.min.js ships from the PACKAGE ROOT, not from a dist/
// subdirectory, and it is NOT the same artefact as lib/marked.umd.js.
const VENDORED_VERSION = "12.0.2";
const PINNED_HASH = "Ffq85bZYmLMrA/XtJen4kacprUwNbYdxEKd0SqhHqJQ=";

function readVendored(name) {
  return fs.readFileSync(new URL(name, VENDOR));
}
/** sha256 (base64) over CRLF-normalised bytes. */
function normalisedHash(buf) {
  const lf = Buffer.from(buf.toString("latin1").replace(/\r\n/g, "\n"), "latin1");
  return crypto.createHash("sha256").update(lf).digest("base64");
}

/** Load the REAL vendored marked.min.js as a classic script in a jsdom window,
 *  the same load path index.html uses (<script src="/vendor/marked.min.js">).
 *  Evaluating the actual artefact is the point: an npm import would be
 *  structurally incapable of saying anything about the file that ships. */
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
      "marked.parse is missing, and it is the only rendering entry point "
      + "app/helpers.js calls (helpers.js:276 and :290)");
    assert.equal(typeof win.marked.setOptions, "function",
      "marked.setOptions is missing; helpers.js:187 calls it at load time, so "
      + "the whole GUI would throw on startup");
  } finally { win.close(); }
});

test("marked still exposes NO runtime version, so the byte pin is the only signal", () => {
  // Not a nicety. If this ever fails because upstream added a version field, the
  // right response is to UPGRADE this guard to a real version floor (as
  // vendor-dompurify and vendor-highlightjs do), not to delete the assertion.
  // Pinning bytes is strictly weaker than pinning a floor, because it cannot say
  // "this or anything newer is acceptable".
  const win = loadVendoredMarked();
  try {
    assert.equal(win.marked.version, undefined,
      "marked now exposes a runtime version. Replace this test with a real "
      + "version-floor assertion and cross-check it against the banner, the way "
      + "vendor-dompurify.test.mjs does.");
  } finally { win.close(); }
});

test("the banner comment says 12.0.2", () => {
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
  // helpers.js:187 - marked.setOptions({ breaks: true, mangle: false, headerIds: false }).
  // `breaks` is the one with a visible effect and it must survive a bump.
  // mangle/headerIds were removed upstream in v7 and are inert here; harmless,
  // but exactly the sort of thing that silently changes meaning across a bump.
  const win = loadVendoredMarked();
  try {
    win.marked.setOptions({ breaks: true, mangle: false, headerIds: false });
    assert.match(win.marked.parse("a\nb"), /<br\s*\/?>/,
      "breaks:true no longer turns a single newline into a <br>, so helpers.js's "
      + "load-time options are not taking effect");
  } finally { win.close(); }
});

test("marked does NOT sanitize, which is why DOMPurify is load-bearing", () => {
  // THE CONTRACT THIS FILE EXISTS TO PIN. marked's own `sanitize` option was
  // REMOVED upstream in v7; it passes raw HTML through by design and its README
  // tells the caller to sanitize the OUTPUT. So localm's safety at
  // helpers.js:276/290 rests entirely on DOMPurify running AFTER marked, never
  // on marked itself. If a future bump made marked start escaping raw HTML this
  // test goes red - the correct outcome, because the pipeline's rationale would
  // have changed and the next reader should re-derive it rather than inherit a
  // stale comment.
  const win = loadVendoredMarked();
  try {
    const out = win.marked.parse('<img src=x onerror="alert(1)">\n\n[l](javascript:alert(1))');
    assert.match(out, /onerror/,
      "marked now strips event handlers. It did not before, and localm relies on "
      + "DOMPurify for this - re-read helpers.js's pipeline comment before "
      + "changing this test.");
    assert.match(out, /javascript:/,
      "marked now filters javascript: URLs. Same note as above.");
  } finally { win.close(); }
});
