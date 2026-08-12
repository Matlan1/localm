// SPDX-License-Identifier: AGPL-3.0-or-later
// Guards the HAND-VENDORED KaTeX trio at localm/plugins/gui/static/vendor/:
// katex.min.js, katex.min.css and contrib/auto-render.min.js.
//
// WHY THIS FILE EXISTS. Same reason as vendor-dompurify.test.mjs: these files
// have no entry in package.json, so no dependency tooling can see them and a
// published advisory against them produces no PR, no alert and no red check.
// The shipped copy sat on 0.16.11 for the whole affected range of
// CVE-2025-23207 / GHSA-cg87-wmx4-v546 (\htmlData does not validate attribute
// names; affected >= 0.12.0 <= 0.16.20, fixed in 0.16.21) and nothing noticed,
// because there was nothing that could.
//
// It deliberately does NOT use harness.mjs. That harness stubs the library out
// (`win.renderMathInElement = () => {};` at harness.mjs:73), which is right for
// the render logic it exists to test and STRUCTURALLY INCAPABLE of saying
// anything about the vendored bytes. Every assertion here loads the real files
// as classic script elements, the same way index.html does.
//
// THREE THINGS KaTeX DOES DIFFERENTLY FROM DOMPurify, and each shapes a test:
//   1. None of the three files carries a banner comment at all (no `@license`,
//      no `/*!`), so the DOMPurify "banner agrees with runtime version" check
//      has nothing to read. The half-replaced-drop failure it exists to catch
//      is covered here by the MATCHED SET test instead, which is strictly
//      stronger: it catches new-JS-with-old-CSS in either direction, which a
//      banner never could.
//   2. auto-render.min.js and katex.min.css carry NO version string of any
//      kind. They are pinned by content hash instead.
//   3. The hash is taken over CRLF-NORMALISED bytes, and that is load-bearing
//      rather than tidiness. This repo has core.autocrlf=true and no
//      .gitattributes rule for vendor/, so a file containing any newline is
//      checked out with different bytes than the blob git stores:
//      katex.min.css is 23335 B in git and 23336 B on a Windows working tree.
//      A raw-byte hash would therefore pass on one platform and fail on the
//      other, and the failure would read as a corrupted vendor drop.
//      katex.min.js and auto-render.min.js happen to contain zero newlines, so
//      for them the normalisation is a no-op; it is applied uniformly anyway so
//      the next file added here cannot reintroduce the trap.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import crypto from "node:crypto";
import { JSDOM } from "jsdom";

const VENDOR = new URL("../localm/plugins/gui/static/vendor/", import.meta.url);

// The lowest KaTeX carrying the CVE-2025-23207 fix. Verified against
// api.github.com/repos/KaTeX/KaTeX/security-advisories: the other four KaTeX
// advisories (CVE-2024-28243/28244/28245/28246) are all patched at or below
// 0.16.10 and do not bear on this floor.
const MIN_SAFE = [0, 16, 21];

// Provenance pin. Each hash is sha256, base64, over the file's CRLF-normalised
// bytes (see the header). Recorded from the npm registry artefact:
//   npm pack katex@0.18.4  ->  package/dist/{katex.min.js,katex.min.css}
//                              package/dist/contrib/auto-render.min.js
// Bumping KaTeX means updating VENDORED_VERSION and all three hashes together.
// That churn is the point: it is what makes a half-bump impossible to land
// quietly.
const VENDORED_VERSION = "0.18.4";
const PINNED = {
  "katex.min.js": "LsWRaUHvQ4PgMU6qvMcSMBsGAB2fto4I11HSuuWieho=",
  "katex.min.css": "GAwtd9Q019pR1mJcUKlk1P1v29ubyHlqCgFsMMSZMfs=",
  "auto-render.min.js": "5TctGZvNrotN5x0PfOunKkuhJ3SifGCm8fd9A7MijuQ=",
};

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
  assert.ok(m, `unparseable KaTeX version: ${JSON.stringify(v)}`);
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}
function atLeast(got, min) {
  for (let i = 0; i < 3; i++) {
    if (got[i] > min[i]) return true;
    if (got[i] < min[i]) return false;
  }
  return true;
}

/** Load the REAL vendored katex.min.js and auto-render.min.js into a jsdom
 *  window as classic script elements, which is index.html's actual load path
 *  (<script src="/vendor/katex.min.js"> then <script src="/vendor/auto-render.min.js">). */
function loadVendoredKatex() {
  const dom = new JSDOM("<!doctype html><html><head></head><body></body></html>",
    { runScripts: "dangerously" });
  for (const f of ["katex.min.js", "auto-render.min.js"]) {
    const tag = dom.window.document.createElement("script");
    tag.textContent = readVendored(f).toString("utf8");
    dom.window.document.head.appendChild(tag);
  }
  return dom.window;
}

/** The exact option bag app/helpers.js passes to renderMathInElement. Kept in
 *  one place so a test cannot accidentally assert a security property under
 *  options the app never uses. */
function appOptions(overrides = {}) {
  return {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
    trust: false,
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    ...overrides,
  };
}

/** A detached div holding one text node, which is what auto-render walks. */
function divWithText(win, s) {
  const d = win.document.createElement("div");
  d.appendChild(win.document.createTextNode(s));
  win.document.body.appendChild(d);
  return d;
}
function markup(win, el) {
  return new win.XMLSerializer().serializeToString(el);
}

test("the vendored KaTeX loads and exposes a runtime version", () => {
  const win = loadVendoredKatex();
  try {
    assert.ok(win.katex, "the vendored file did not define window.katex");
    assert.equal(typeof win.katex.version, "string",
      "window.katex.version is missing, so the version floor below would be "
      + "asserting against undefined rather than against the real library");
    assert.equal(typeof win.renderMathInElement, "function",
      "auto-render.min.js did not define window.renderMathInElement, which is "
      + "the only KaTeX entry point app/helpers.js calls");
  } finally { win.close(); }
});

test("the vendored KaTeX is at or above the CVE-2025-23207 fix (0.16.21)", () => {
  // Read from the library's RUNTIME `version` property, never from a comment:
  // a banner can be doctored or left behind by a half-finished drop, and in
  // KaTeX's case there is no banner to read in the first place.
  const win = loadVendoredKatex();
  try {
    const got = parseVersion(win.katex.version);
    assert.ok(atLeast(got, MIN_SAFE),
      `vendored KaTeX ${win.katex.version} is below ${MIN_SAFE.join(".")}, which `
      + "is the fix for CVE-2025-23207 / GHSA-cg87-wmx4-v546 (\\htmlData does not "
      + "validate attribute names). Re-vendor dist/katex.min.js, dist/katex.min.css "
      + "and dist/contrib/auto-render.min.js from one current release.");
  } finally { win.close(); }
});

test("the vendored KaTeX trio are the recorded upstream bytes", () => {
  // auto-render.min.js and katex.min.css carry NO version string, so a hash is
  // the only thing that can say which upstream build they came from. Recording
  // katex.min.js the same way keeps all three provable from one place.
  for (const [name, expected] of Object.entries(PINNED)) {
    assert.equal(normalisedHash(readVendored(name)), expected,
      `vendor/${name} does not match the recorded hash for KaTeX `
      + `${VENDORED_VERSION}. If this is a deliberate bump, update PINNED and `
      + "VENDORED_VERSION together with vendor/README.md. If it is not, the "
      + "file has drifted from its upstream dist artefact.");
  }
  const win = loadVendoredKatex();
  try {
    assert.equal(win.katex.version, VENDORED_VERSION,
      "the recorded VENDORED_VERSION disagrees with the library's own runtime "
      + "version, so the hashes above are pinning a different release than the "
      + "one this file claims to vendor");
  } finally { win.close(); }
});

test("the vendored auto-render really renders math through the real KaTeX", () => {
  // typeof === "function" only proves a symbol exists. This proves the two
  // vendored files actually work TOGETHER, which is the thing a half-bump
  // breaks, and it covers every delimiter pair helpers.js configures.
  const win = loadVendoredKatex();
  try {
    const el = divWithText(win,
      "before $x^2$ then $$\\sum_{i=1}^{n} i$$ then \\(a+b\\) then \\[c+d\\] after");
    win.renderMathInElement(el, appOptions());
    const html = markup(win, el);
    assert.equal((html.match(/class="katex"/g) || []).length, 4,
      `expected all four delimiter pairs to render, got: ${html.slice(0, 400)}`);
    // Surrounding prose must survive untouched.
    assert.match(el.textContent, /before/);
    assert.match(el.textContent, /after/);

    // ignoredTags is what keeps fenced code blocks out of the math renderer.
    const code = win.document.createElement("code");
    code.appendChild(win.document.createTextNode("$x^2$"));
    const wrap = win.document.createElement("div");
    wrap.appendChild(code);
    win.document.body.appendChild(wrap);
    win.renderMathInElement(wrap, appOptions());
    assert.doesNotMatch(markup(win, wrap), /katex/i,
      "math inside <code> was rendered, so ignoredTags stopped being honoured");
  } finally { win.close(); }
});

test("katex.min.js and katex.min.css are a MATCHED SET, not a half-bump", () => {
  // The real hazard when three files must move together and only one of them
  // carries a version. KaTeX 0.18.0's single breaking change was "prefix css
  // classes": the JS began emitting katex-base / katex-strut / katex-sizing
  // where it used to emit base / strut / sizing, and the CSS moved with it. So
  // a new JS beside an old CSS (or the reverse) renders visibly broken while
  // every version assertion about the file you DID replace still passes.
  //
  // NOT asserted here: that EVERY emitted class is defined in the CSS. That was
  // tried and is wrong - mord, mbin, mrel, mopen, mclose, mop and mtight are
  // JS-internal spacing markers that appear in NEITHER release's stylesheet
  // (measured against both 0.16.11 and 0.18.4), so it fails on correctly
  // matched files.
  const css = readVendored("katex.min.css").toString("utf8");
  const win = loadVendoredKatex();
  try {
    const el = divWithText(win, "$$\\frac{a}{b} + \\sum_{i=1}^{n} x_i^2$$");
    win.renderMathInElement(el, appOptions());
    const emitted = new Set();
    el.querySelectorAll("*").forEach((node) => {
      (node.getAttribute("class") || "").split(/\s+/).filter(Boolean)
        .forEach((c) => emitted.add(c));
    });
    // Guard the guard: an empty or near-empty render would make every
    // assertion below pass vacuously.
    assert.ok(emitted.size >= 10,
      `only ${emitted.size} classes emitted, so this test would pass vacuously`);

    // Breadth: the katex-* family IS the structural family the stylesheet owns.
    const undefinedKatexClasses = [...emitted]
      .filter((c) => c.startsWith("katex") && !css.includes(`.${c}`));
    assert.deepEqual(undefinedKatexClasses, [],
      "katex.min.js emits katex-* classes that katex.min.css does not define, "
      + "so the two files are from different releases (a half-bump). Re-vendor "
      + "katex.min.js, katex.min.css and auto-render.min.js from ONE release.");

    // Both directions, keyed on the exact token upstream's breaking-change note
    // names. Deriving the expected generation FROM THE CSS means this keeps
    // working after the next bump without being edited, and it fails whichever
    // of the two files is the stale one.
    const cssIsPrefixedGeneration = css.includes(".katex-base");
    if (cssIsPrefixedGeneration) {
      assert.ok(emitted.has("katex-base"),
        "katex.min.css is the prefixed generation (>= 0.18.0) but katex.min.js "
        + "still emits the pre-0.18 class names: the JS is the stale file");
      assert.ok(!emitted.has("base"),
        "katex.min.js emits both prefixed and unprefixed structural classes");
    } else {
      assert.ok(emitted.has("base"),
        "katex.min.css is the pre-0.18 generation but katex.min.js emits the "
        + "prefixed class names: the CSS is the stale file");
      assert.ok(!emitted.has("katex-base"),
        "katex.min.js emits prefixed structural classes the CSS cannot style");
    }
  } finally { win.close(); }
});

test("R41-D4: trust:false keeps \\htmlData and \\href from emitting markup", () => {
  // The security property app/helpers.js:315 pins, asserted on BEHAVIOUR so a
  // version bump that changed the default cannot slip past a version check.
  // The trust:true CONTROL is in this same test on purpose: without it, a build
  // where \htmlData silently stopped working at all would satisfy every
  // assertion below while proving nothing about trust.
  const win = loadVendoredKatex();
  try {
    const denied = divWithText(win, "$\\htmlData{foo=bar}{x}$");
    win.renderMathInElement(denied, appOptions());
    assert.doesNotMatch(markup(win, denied), /data-foo/,
      "trust:false still emitted the \\htmlData attribute");

    const allowed = divWithText(win, "$\\htmlData{foo=bar}{x}$");
    win.renderMathInElement(allowed, appOptions({ trust: true }));
    assert.match(markup(win, allowed), /data-foo/,
      "CONTROL FAILED: trust:true did not emit data-foo either, so the "
      + "trust:false assertion above is passing for the wrong reason and "
      + "cannot detect a real regression");

    // \href under trust:false must not produce a link ELEMENT. Note the exact
    // property: KaTeX renders the refused command as red error text and echoes
    // the raw TeX source into the MathML <annotation>, so the literal
    // "javascript:" string IS present in the output and asserting on its
    // absence would fail against correct, safe behaviour.
    const link = divWithText(win, "$\\href{javascript:alert(1)}{click}$");
    win.renderMathInElement(link, appOptions());
    assert.equal(link.querySelectorAll("a").length, 0,
      "trust:false produced an anchor element from \\href");
    assert.doesNotMatch(markup(win, link), /href="/,
      "trust:false produced an href attribute from \\href");
  } finally { win.close(); }
});
