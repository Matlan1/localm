// SPDX-License-Identifier: AGPL-3.0-or-later
// Guards the HAND-VENDORED highlight.js at
// localm/plugins/gui/static/vendor/highlight.min.js. Does NOT cover
// github-dark.min.css, which ships beside it: measured byte-identical
// (CRLF-normalised) between 11.9.0 and 11.12.0, so there is nothing for a
// guard to pin there.
//
// WHY THIS FILE EXISTS. Same reason as vendor-dompurify.test.mjs,
// vendor-katex.test.mjs and vendor-jsqr.test.mjs: this file has no entry in
// package.json, so no dependency tooling can ever see it and a fix here
// produces no PR, no alert and no red check. vendor/README.md named
// highlight.js (with marked) as the two still without a guard; this closes
// highlight.js.
//
// THE DEFECT. highlight.js 11.12.0's own CHANGES.md records two ReDoS fixes
// with NEITHER a CVE NOR a GHSA advisory:
//   "fix(c, cpp) bound the run of type tokens in front of a function name
//    (ReDoS), issue #4362"
//   "fix(xml) remove recursive sublanguage references to prevent ReDoS"
// Because there is no advisory, no scanner anywhere (dependency or
// otherwise) can ever flag the shipped 11.9.0 build. This test is the only
// thing in the repo that can.
//
// REACHABILITY. helpers.js:325 calls hljs.highlightElement(block)
// SYNCHRONOUSLY on the main thread for every `pre code` in a rendered
// message, and `language-c` arrives straight off a model-emitted ```c
// fence. A local model can emit a large C code block with no adversary
// involved, so this is a real, reachable main-thread freeze, not a
// theoretical one.
//
// WHY A GROWTH-SHAPE ASSERTION, NOT A WALL-CLOCK BOUND. This box runs many
// concurrent sessions, so a fixed millisecond threshold flakes under load.
// Instead this times the PoC at two sizes two doublings apart and asserts
// the RATIO: quadratic growth gives ~16x per two doublings, and the fixed
// build measures consistently under 5x (see the threshold comment below).
// The ratio is far less sensitive to absolute machine speed than a wall
// clock bound is, which is why it survives a shared, loaded box.
//
// THE HASH IS TAKEN OVER CRLF-NORMALISED BYTES, and that is load-bearing
// rather than tidiness (see vendor-katex.test.mjs and vendor-jsqr.test.mjs
// for the same note). This repo has core.autocrlf=true and no
// .gitattributes rule for vendor/, so the checked-out file differs in size
// from the git blob purely by line-ending conversion. A raw-byte hash would
// pass on one platform and fail on the other.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import crypto from "node:crypto";
import { JSDOM } from "jsdom";

const VENDOR = new URL("../localm/plugins/gui/static/vendor/", import.meta.url);

// The lowest highlight.js carrying BOTH ReDoS fixes named above.
const MIN_SAFE = [11, 12, 0];

// Provenance pin. sha256, base64, over CRLF-normalised bytes. Recorded from
// the npm registry artefact, tarball shasum-verified against the registry's
// published dist.shasum before extraction:
//   npm pack @highlightjs/cdn-assets@11.12.0  ->  package/highlight.min.js
const VENDORED_VERSION = "11.12.0";
const PINNED_HASH = "ircesJxR9QHl4lFX2c/xAORswpvL/HRNC3RtRR/Kf1M=";

function readVendored(name) {
  return fs.readFileSync(new URL(name, VENDOR));
}
/** sha256 (base64) over CRLF-normalised bytes. */
function normalisedHash(buf) {
  const lf = Buffer.from(buf.toString("latin1").replace(/\r\n/g, "\n"), "latin1");
  return crypto.createHash("sha256").update(lf).digest("base64");
}
function parseVersion(v) {
  const m = String(v).match(/^(\d+)\.(\d+)\.(\d+)/);
  assert.ok(m, `unparseable highlight.js version: ${JSON.stringify(v)}`);
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}
function atLeast(got, min) {
  for (let i = 0; i < 3; i++) {
    if (got[i] > min[i]) return true;
    if (got[i] < min[i]) return false;
  }
  return true;
}

/** Load the REAL vendored highlight.min.js as a classic script in a jsdom
 *  window, the same load path index.html uses
 *  (<script src="/vendor/highlight.min.js">). Evaluating the actual
 *  artefact is the point: an npm import would be structurally incapable of
 *  saying anything about the file that ships. */
function loadVendoredHljs() {
  const dom = new JSDOM("<!doctype html><html><head></head><body></body></html>",
    { runScripts: "dangerously" });
  const tag = dom.window.document.createElement("script");
  tag.textContent = readVendored("highlight.min.js").toString("utf8");
  dom.window.document.head.appendChild(tag);
  return dom.window;
}

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

/** Median wall-clock ms to highlight `"a ".repeat(n)` as C, over `runs`
 *  trials. This is issue #4362's own PoC: a long run of bare identifiers is
 *  what drove the pre-fix "type tokens in front of a function name" regex
 *  quadratic. */
function medianHighlightMs(win, n, runs = 3) {
  const src = "a ".repeat(n);
  const times = [];
  for (let i = 0; i < runs; i++) {
    const t0 = process.hrtime.bigint();
    win.hljs.highlight(src, { language: "c" });
    const t1 = process.hrtime.bigint();
    times.push(Number(t1 - t0) / 1e6);
  }
  return median(times);
}

test("the vendored highlight.js loads and exposes a runtime version", () => {
  const win = loadVendoredHljs();
  try {
    assert.ok(win.hljs, "the vendored file did not define window.hljs");
    assert.equal(typeof win.hljs.versionString, "string",
      "hljs.versionString is missing, so the version floor below would be "
      + "asserting against undefined rather than against the real library");
    assert.equal(typeof win.hljs.highlightElement, "function",
      "hljs.highlightElement is missing, which is the only entry point "
      + "app/helpers.js calls");
  } finally { win.close(); }
});

test("the banner comment and the runtime version agree, and both say 11.12.0", () => {
  // Two independent sources on purpose: a hand-edited banner over old bytes
  // is exactly the half-finished drop this guards against, and a banner
  // that was correctly bumped over a runtime that was not would be the
  // mirror-image mistake.
  const banner = readVendored("highlight.min.js").toString("latin1").slice(0, 200);
  const bannerMatch = banner.match(/Highlight\.js v(\d+\.\d+\.\d+)/);
  assert.ok(bannerMatch, `no version banner found in the first 200 bytes: ${JSON.stringify(banner)}`);
  assert.equal(bannerMatch[1], VENDORED_VERSION,
    `banner says v${bannerMatch[1]}, VENDORED_VERSION says ${VENDORED_VERSION}`);

  const win = loadVendoredHljs();
  try {
    assert.equal(win.hljs.versionString, VENDORED_VERSION,
      `runtime hljs.versionString is ${win.hljs.versionString}, VENDORED_VERSION `
      + `says ${VENDORED_VERSION}. The banner comment and the actual code have `
      + "drifted apart.");
  } finally { win.close(); }
});

test("the vendored highlight.js is at or above the ReDoS fix (11.12.0)", () => {
  const win = loadVendoredHljs();
  try {
    const got = parseVersion(win.hljs.versionString);
    assert.ok(atLeast(got, MIN_SAFE),
      `vendored highlight.js ${win.hljs.versionString} is below `
      + `${MIN_SAFE.join(".")}, which carries the fix for issue #4362 (c/cpp `
      + "ReDoS) and the xml recursive-sublanguage ReDoS fix. Re-vendor "
      + "highlight.min.js from a current @highlightjs/cdn-assets release.");
  } finally { win.close(); }
});

test("the vendored bytes are the recorded upstream artefact", () => {
  const got = normalisedHash(readVendored("highlight.min.js"));
  assert.equal(got, PINNED_HASH,
    `highlight.min.js does not match the pinned @highlightjs/cdn-assets@`
    + `${VENDORED_VERSION} npm artefact. If this was a deliberate bump, `
    + "update VENDORED_VERSION and PINNED_HASH together with vendor/README.md; "
    + "if not, the vendored file has been modified or corrupted.");
});

test("the vendored highlight.js actually highlights real code", () => {
  // Not "the module loads": a guard that only checks the file parses would
  // pass on a truncated or half-replaced drop that cannot tokenise
  // anything. This is the same shape as the jsQR decode test and the KaTeX
  // render test.
  const win = loadVendoredHljs();
  try {
    const { value, language } = win.hljs.highlight(
      "int main(void) { return 0; }", { language: "c" });
    assert.equal(language, "c");
    assert.match(value, /hljs-keyword/, "no keyword span emitted for real C code");
    assert.match(value, /\bint\b/, "the tokenised source text is missing entirely");
  } finally { win.close(); }
});

test("issue #4362's C/C++ ReDoS PoC no longer grows quadratically", () => {
  // Sizes and threshold were chosen by measuring, not copied from a note:
  // n=8000 keeps the FIXED build's own timing comfortably above timer noise
  // (several ms, not sub-ms), and n=32000 is two doublings further out.
  // Measured on this box, repeated trials: the fixed build's ratio holds in
  // roughly 3.3-4.35x; the unfixed 11.9.0 build (see the fires-control
  // instructions in vendor/README.md) measures roughly 14.3-15.7x, close to
  // the ~16x a true O(n^2) algorithm gives for two doublings. A threshold of
  // 8x sits with real margin on both sides of that gap.
  const win = loadVendoredHljs();
  try {
    const small = medianHighlightMs(win, 8000);
    const large = medianHighlightMs(win, 32000);
    const ratio = large / small;
    assert.ok(ratio < 8,
      `growth ratio ${ratio.toFixed(2)}x over two doublings (n=8000: `
      + `${small.toFixed(2)}ms, n=32000: ${large.toFixed(2)}ms) looks quadratic, `
      + "not linear. issue #4362's ReDoS fix may be missing or reverted.");
  } finally { win.close(); }
});
