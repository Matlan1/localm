// SPDX-License-Identifier: AGPL-3.0-or-later
// Guards the HAND-VENDORED DOMPurify at localm/plugins/gui/static/vendor/.
//
// WHY THIS FILE EXISTS. DOMPurify is vendored as a plain file with NO entry in
// package.json, so no dependency tooling (Dependabot or otherwise) can see it,
// and nothing would ever flag a known-vulnerable build sitting there. That is
// exactly how the shipped copy stayed on 3.2.6 while CVE-2026-0540 (rawtext
// SAFE_FOR_XML bypass, fixed in 3.3.2) went unnoticed. This test is the
// replacement signal for DOMPurify. `vendor-katex.test.mjs` is the same thing
// for the KaTeX trio; `marked`, `highlight.js` and `jsQR` still have none.
//
// It deliberately does NOT use harness.mjs. That harness stubs the library out
// entirely (`win.DOMPurify = { sanitize: (s) => s }`), which is right for the
// nav/abort logic it exists to test and structurally incapable of saying
// anything about the vendored file. Every assertion here loads the real file as
// a classic script element, the same way index.html does with
// <script src="/vendor/purify.min.js">.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { JSDOM } from "jsdom";

const VENDOR_URL = new URL(
  "../localm/plugins/gui/static/vendor/purify.min.js", import.meta.url);

// The lowest DOMPurify that carries the CVE-2026-0540 / GHSA-v2wj-7wpq-c8vv fix
// (affected 3.1.3 through 3.3.1; also 2.5.3 through 2.5.8, fixed in 2.5.9).
const MIN_SAFE = [3, 3, 2];

function parseVersion(v) {
  const m = String(v).match(/^(\d+)\.(\d+)\.(\d+)/);
  assert.ok(m, `unparseable DOMPurify version: ${JSON.stringify(v)}`);
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}

function atLeast(got, min) {
  for (let i = 0; i < 3; i++) {
    if (got[i] > min[i]) return true;
    if (got[i] < min[i]) return false;
  }
  return true;
}

/** Load the REAL vendored purify.min.js into a jsdom window, as a classic
 *  script element (index.html's actual load path), and hand back the window
 *  plus the raw source so a caller can also inspect the banner. */
function loadVendoredDOMPurify() {
  const src = fs.readFileSync(VENDOR_URL, "utf8");
  const dom = new JSDOM("<!doctype html><html><head></head><body></body></html>",
    { runScripts: "dangerously" });
  const tag = dom.window.document.createElement("script");
  tag.textContent = src;
  dom.window.document.head.appendChild(tag);
  return { window: dom.window, src };
}

test("the vendored DOMPurify actually loads and exposes a supported sanitizer", () => {
  const { window: win } = loadVendoredDOMPurify();
  try {
    assert.ok(win.DOMPurify, "the vendored file did not define window.DOMPurify");
    assert.equal(win.DOMPurify.isSupported, true);
    // The everyday property, so a wholesale breakage cannot hide behind the
    // version-only assertions below.
    assert.doesNotMatch(win.DOMPurify.sanitize('<img src=x onerror="alert(1)">'),
      /onerror/i);
  } finally { win.close(); }
});

test("the vendored DOMPurify is at or above the CVE-2026-0540 fix (3.3.2)", () => {
  const { window: win } = loadVendoredDOMPurify();
  try {
    const got = parseVersion(win.DOMPurify.version);
    assert.ok(atLeast(got, MIN_SAFE),
      `vendored DOMPurify ${win.DOMPurify.version} is below ${MIN_SAFE.join(".")}, `
      + "which is the fix for CVE-2026-0540 (rawtext SAFE_FOR_XML bypass). "
      + "Re-vendor dist/purify.min.js from a current release.");
  } finally { win.close(); }
});

test("the vendored banner comment agrees with the library's own version", () => {
  // The banner is only a COMMENT, so it proves nothing on its own - but a
  // banner that disagrees with the runtime version means the drop was edited by
  // hand or half-replaced, which is the failure mode a silent vendor swap has.
  const { window: win, src } = loadVendoredDOMPurify();
  try {
    const m = src.slice(0, 400).match(/@license DOMPurify (\d+\.\d+\.\d+)/);
    assert.ok(m, "no parseable '@license DOMPurify <version>' banner");
    assert.equal(m[1], win.DOMPurify.version,
      "the banner version and the library's runtime version disagree");
  } finally { win.close(); }
});

test("CVE-2026-0540: a rawtext close tag in an attribute value is not passed through", () => {
  // The actual security property, asserted on BEHAVIOUR rather than on a version
  // string: a doctored banner cannot satisfy this one. Affected builds left the
  // raw close tag intact in the attribute value, so placing the sanitized output
  // inside that element closed it early and executed the payload.
  const { window: win } = loadVendoredDOMPurify();
  try {
    for (const el of ["noscript", "xmp", "noembed", "noframes", "iframe"]) {
      const out = win.DOMPurify.sanitize(
        `<p title="</${el}><img src=x onerror=alert(1)>">hi</p>`);
      assert.ok(!out.includes(`</${el}>`),
        `sanitized output still carries a raw </${el}> close tag: ${out}`);
      assert.doesNotMatch(out, /onerror/i);
    }
  } finally { win.close(); }
});

test("the SVG profile call settings.js uses still strips script from an SVG", () => {
  // settings.js sanitizes logo SVGs with USE_PROFILES svg + svgFilters. Pinned
  // here so a future vendor bump that changes profile behaviour is visible.
  const { window: win } = loadVendoredDOMPurify();
  try {
    const out = win.DOMPurify.sanitize(
      '<svg><circle r="5"/><script>alert(1)</script></svg>',
      { USE_PROFILES: { svg: true, svgFilters: true } });
    assert.match(out, /<circle/);
    assert.doesNotMatch(out, /<script/i);
  } finally { win.close(); }
});
