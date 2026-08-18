// SPDX-License-Identifier: AGPL-3.0-or-later
// The client half of the remote-image proxy: renderMarkdown must point every
// REMOTE <img> at /api/image-proxy so the browser never contacts the remote host.
//
// That is the whole privacy claim of the feature, and it lives entirely on this
// side: the server can refuse to proxy, but it cannot stop the browser making a
// direct request for an <img src> that was never rewritten. So this file guards
// the property the server is structurally unable to enforce.
//
// The rewrite is UNCONDITIONAL by design (the server owns the on/off decision),
// so there is no enabled/disabled arm to test here - see the rationale on
// proxyRemoteImages in helpers.js.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadApp } from "./harness.mjs";

const VENDOR = new URL("../localm/plugins/gui/static/vendor/", import.meta.url);

/** loadApp() with the REAL vendored marked + DOMPurify swapped over the stubs.
 *  The stubs would let a payload through unchanged, which would make an
 *  assertion about what survives sanitisation meaningless here. */
function loadReal() {
  const { window: win } = loadApp();
  for (const f of ["marked.min.js", "purify.min.js"]) {
    const tag = win.document.createElement("script");
    tag.textContent = fs.readFileSync(new URL(f, VENDOR), "utf8");
    win.document.head.appendChild(tag);
  }
  win.marked.setOptions({ breaks: true, mangle: false, headerIds: false });
  return win;
}

function render(win, md) {
  const t = win.document.createElement("div");
  win.document.body.appendChild(t);
  win.renderMarkdown(t, md, { final: true });
  return t;
}

const srcOf = (t) => {
  const img = t.querySelector("img");
  return img && img.getAttribute("src");
};

test("a remote image in a reply is rewritten to the proxy", () => {
  const win = loadReal();
  const src = srcOf(render(win, "![a chart](https://example.com/chart.png)"));
  assert.ok(src, "no <img> was rendered at all");
  assert.ok(src.startsWith("/api/image-proxy?url="),
    `remote image was not rewritten, so the browser would fetch it directly: ${src}`);
  assert.ok(!src.includes("example.com/chart.png") || src.includes("%2F"),
    "the target must be URL-ENCODED into the query, not spliced in raw");
  assert.equal(decodeURIComponent(src.split("url=")[1]),
    "https://example.com/chart.png");
});

test("http as well as https is rewritten", () => {
  const win = loadReal();
  const src = srcOf(render(win, "![x](http://example.com/a.png)"));
  assert.ok(src.startsWith("/api/image-proxy?url="), src);
});

test("raw HTML <img> is rewritten too, not just markdown image syntax", () => {
  // A model emits both. Covering only the markdown form would leave the raw-HTML
  // path making direct remote requests, which is the exact leak this prevents.
  const win = loadReal();
  const src = srcOf(render(win, '<img src="https://example.com/raw.png" alt="x">'));
  assert.ok(src && src.startsWith("/api/image-proxy?url="), String(src));
});

test("an image inside a <think> block is covered by the same pass", () => {
  const win = loadReal();
  const t = render(win, "<think>![x](https://example.com/t.png)</think>answer");
  const img = t.querySelector("details.think-block img");
  assert.ok(img, "no image rendered inside the think block");
  assert.ok(img.getAttribute("src").startsWith("/api/image-proxy?url="),
    "the think-block sink was left unproxied");
});

test("data:, blob: and relative sources are left ALONE", () => {
  // They already load under the shell CSP, and routing them through the proxy
  // would be a pointless round trip - and for data: would push the whole payload
  // through a query string.
  const win = loadReal();
  const cases = [
    ["data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
     "data:"],
    ["/uploads/local.png", "/uploads/local.png"],
  ];
  for (const [srcIn, expectPrefix] of cases) {
    const got = srcOf(render(win, `<img src="${srcIn}">`));
    assert.ok(got && got.startsWith(expectPrefix),
      `${srcIn} should have been left alone, got ${got}`);
    assert.ok(!got.startsWith("/api/image-proxy"),
      `${srcIn} was needlessly proxied`);
  }
});

test("a same-origin absolute URL is not detoured through the proxy", () => {
  const win = loadReal();
  const origin = win.location.origin;
  const got = srcOf(render(win, `<img src="${origin}/media/mine.png">`));
  assert.ok(got && !got.startsWith("/api/image-proxy"),
    `our own bytes should not round-trip through the proxy: ${got}`);
});

test("the rewrite is idempotent across a streaming re-render", () => {
  // renderMarkdown runs repeatedly while a reply streams. A second pass must not
  // wrap an already-proxied URL inside another proxy URL.
  const win = loadReal();
  const t = win.document.createElement("div");
  win.document.body.appendChild(t);
  const md = "![x](https://example.com/a.png)";
  win.renderMarkdown(t, md, { final: false });
  win.renderMarkdown(t, md, { final: false });
  win.renderMarkdown(t, md, { final: true });
  const src = t.querySelector("img").getAttribute("src");
  assert.equal((src.match(/\/api\/image-proxy/g) || []).length, 1,
    `nested proxy URL after repeated renders: ${src}`);
  assert.equal(decodeURIComponent(src.split("url=")[1]), "https://example.com/a.png");
});

test("metacharacters in a surviving src are encoded into the proxy query", () => {
  // proxyRemoteImages runs AFTER sanitisation, which is the hazardous position.
  // It must only ever replace an attribute VALUE, never insert markup, and the
  // target must be encoded rather than spliced in raw.
  //
  // The payload is chosen to SURVIVE sanitisation on purpose. The obvious
  // `"><script>` src does not: DOMPurify's SAFE_FOR_XML pass drops any attribute
  // whose value contains `</script`, so the attribute is removed outright and a
  // test built on it would assert against an element that no longer has a src -
  // passing or failing for a reason that has nothing to do with the rewrite.
  const win = loadReal();
  const t = render(win, '<img src="https://example.com/a.png?q=&lt;b&gt;&amp;z=1">');
  assert.equal(win.__x, undefined, "a payload executed during the rewrite");
  assert.equal(t.querySelectorAll("script").length, 0,
    "the rewrite produced a <script> element");
  const src = t.querySelector("img").getAttribute("src");
  assert.ok(src.startsWith("/api/image-proxy?url="), src);
  const q = src.split("url=")[1];
  assert.ok(!/[<>"']/.test(q), `unencoded metacharacters in the proxy URL: ${q}`);
  assert.equal(decodeURIComponent(q), "https://example.com/a.png?q=%3Cb%3E&z=1");
});

test("a src the sanitizer strips is never resurrected by the rewrite", () => {
  // The complement of the test above, and the more important direction: if the
  // sanitizer removed a src, the rewrite must not put one back. Rewriting from a
  // stale/raw value rather than from the sanitized attribute is exactly how a
  // post-sanitize pass reintroduces what sanitisation just removed.
  const win = loadReal();
  const t = render(win,
    '<img src="https://example.com/a.png?q=&quot;&gt;&lt;script&gt;window.__x=1&lt;/script&gt;">');
  const img = t.querySelector("img");
  assert.ok(img, "expected the <img> element itself to survive");
  assert.equal(img.getAttribute("src"), null,
    "DOMPurify no longer strips this src. Re-read whether the rewrite is still "
    + "safe for it before changing this expectation.");
  assert.equal(win.__x, undefined, "a payload executed");
  assert.equal(t.querySelectorAll("script").length, 0);
});
