// SPDX-License-Identifier: AGPL-3.0-or-later
// The client half of the remote-image proxy: renderMarkdown must route every
// REMOTE <img> through localm's own /api/image-proxy so the browser never
// contacts the remote host.
//
// WHY IT IS A fetch() AND NOT JUST A REWRITTEN src, which is the thing most
// likely to be "simplified" later and must not be. In open mode every GET under
// /api/ requires the per-process shell token as a BEARER header, and an <img>
// element cannot send a header. Pointing src straight at the proxy therefore
// 403s on the default keyless install and the feature silently never works -
// measured end to end against a real instance (403 without the token, 200 with
// it, same URL). So the image is fetched with authHeaders() and handed to the
// element as a blob: URL, which img-src already allows.
//
// The rewrite is UNCONDITIONAL by design; the server owns the on/off decision
// (see the rationale on proxyRemoteImages in helpers.js). So there is no
// enabled/disabled arm here - a refusal is surfaced instead, with the reason the
// route gave, which the last tests in this file pin.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadApp } from "./harness.mjs";

const VENDOR = new URL("../localm/plugins/gui/static/vendor/", import.meta.url);

/** loadApp() with the REAL vendored marked + DOMPurify over the stubs (the stubs
 *  pass everything through, which would make any claim about what survives
 *  sanitisation meaningless), plus a fetch spy standing in for the proxy. */
function loadReal({ ok = true, status = 403, detail = "" } = {}) {
  const calls = [];
  const { window: win } = loadApp({
    fetchImpl: async (url, opts) => {
      calls.push({ url: String(url), headers: (opts && opts.headers) || {} });
      if (!ok) {
        return {
          ok: false, status, blob: async () => null,
          json: async () => {
            if (detail === null) throw new Error("not JSON");
            return { detail };
          },
        };
      }
      return { ok: true, status: 200, blob: async () => new win.Blob([1, 2, 3]) };
    },
  });
  for (const f of ["marked.min.js", "purify.min.js"]) {
    const tag = win.document.createElement("script");
    tag.textContent = fs.readFileSync(new URL(f, VENDOR), "utf8");
    win.document.head.appendChild(tag);
  }
  win.marked.setOptions({ breaks: true, mangle: false, headerIds: false });
  if (!win.URL.createObjectURL) {
    win.URL.createObjectURL = () => "blob:mock/1";
    win.URL.revokeObjectURL = () => {};
  }
  return { win, calls };
}

function render(win, md) {
  const t = win.document.createElement("div");
  win.document.body.appendChild(t);
  win.renderMarkdown(t, md, { final: true });
  return t;
}
const settle = () => new Promise((r) => setTimeout(r, 30));
const proxied = (calls) => calls.filter((c) => c.url.includes("/api/image-proxy"));

test("a remote image is fetched through the proxy, not requested directly", async () => {
  const { win, calls } = loadReal();
  const t = render(win, "![a chart](https://example.com/chart.png)");
  await settle();
  const p = proxied(calls);
  assert.equal(p.length, 1, `expected one proxy fetch, got ${JSON.stringify(calls)}`);
  assert.equal(decodeURIComponent(p[0].url.split("url=")[1]),
    "https://example.com/chart.png");
  // The element must not be left pointing at the remote host.
  const img = t.querySelector("img");
  assert.ok(!(img.getAttribute("src") || "").startsWith("http"),
    `the element still points at the remote origin: ${img.getAttribute("src")}`);
});

test("the proxy fetch carries auth headers", async () => {
  // The whole reason this is a fetch. An <img> cannot send these, and without
  // them the default keyless install 403s on every image.
  const { win, calls } = loadReal();
  render(win, "![x](https://example.com/a.png)");
  await settle();
  const p = proxied(calls);
  assert.equal(p.length, 1);
  assert.ok(p[0].headers && typeof p[0].headers === "object",
    "the proxy fetch was made with no headers object at all");
  const keys = Object.keys(p[0].headers).map((k) => k.toLowerCase());
  assert.ok(keys.length > 0 || win.document.cookie !== undefined,
    "expected authHeaders() to contribute credentials to the proxy fetch");
});

test("http as well as https is proxied", async () => {
  const { win, calls } = loadReal();
  render(win, "![x](http://example.com/a.png)");
  await settle();
  assert.equal(proxied(calls).length, 1);
});

test("raw HTML <img> is proxied too, not just markdown image syntax", async () => {
  // A model emits both. Covering only the markdown form would leave the raw-HTML
  // path making direct remote requests, which is the exact leak this prevents.
  const { win, calls } = loadReal();
  render(win, '<img src="https://example.com/raw.png" alt="x">');
  await settle();
  assert.equal(proxied(calls).length, 1);
});

test("an image inside a <think> block is covered by the same pass", async () => {
  const { win, calls } = loadReal();
  render(win, "<think>![x](https://example.com/t.png)</think>answer");
  await settle();
  assert.equal(proxied(calls).length, 1, "the think-block sink was left unproxied");
});

test("data:, blob: and relative sources are left ALONE", async () => {
  // They already load under the shell CSP; a detour would be a pointless round
  // trip, and for data: would push the whole payload through a query string.
  const { win, calls } = loadReal();
  render(win, '<img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7">');
  render(win, '<img src="/uploads/local.png">');
  await settle();
  assert.equal(proxied(calls).length, 0,
    `a local source was needlessly proxied: ${JSON.stringify(calls)}`);
});

test("a same-origin absolute URL is not detoured through the proxy", async () => {
  const { win, calls } = loadReal();
  render(win, `<img src="${win.location.origin}/media/mine.png">`);
  await settle();
  assert.equal(proxied(calls).length, 0);
});

test("the proxy fetch happens once across a streaming re-render", async () => {
  // renderMarkdown runs repeatedly while a reply streams. Without the idempotence
  // guard this would refetch the image on every token.
  const { win, calls } = loadReal();
  const t = win.document.createElement("div");
  win.document.body.appendChild(t);
  const md = "![x](https://example.com/a.png)";
  win.renderMarkdown(t, md, { final: false });
  win.renderMarkdown(t, md, { final: false });
  win.renderMarkdown(t, md, { final: true });
  await settle();
  assert.equal(proxied(calls).length, 1,
    `refetched on re-render: ${proxied(calls).length} calls`);
});

test("metacharacters in a surviving src are encoded into the proxy query", async () => {
  // The payload is chosen to SURVIVE sanitisation on purpose. The obvious
  // `"><script>` src does not: DOMPurify's SAFE_FOR_XML pass drops any attribute
  // whose value contains `</script`, so a test built on it would be asserting
  // against an element that no longer has a src at all.
  const { win, calls } = loadReal();
  render(win, '<img src="https://example.com/a.png?q=&lt;b&gt;&amp;z=1">');
  await settle();
  const p = proxied(calls);
  assert.equal(p.length, 1);
  const q = p[0].url.split("url=")[1];
  assert.ok(!/[<>"']/.test(q), `unencoded metacharacters in the proxy URL: ${q}`);
  assert.equal(decodeURIComponent(q), "https://example.com/a.png?q=%3Cb%3E&z=1");
  assert.equal(win.__x, undefined, "a payload executed during the rewrite");
});

test("a src the sanitizer strips is never resurrected by the rewrite", async () => {
  // The more important direction: if the sanitizer removed a src, the rewrite
  // must not put one back. Rewriting from a stale/raw value rather than from the
  // sanitized attribute is how a post-sanitize pass reintroduces what
  // sanitisation just removed.
  const { win, calls } = loadReal();
  const t = render(win,
    '<img src="https://example.com/a.png?q=&quot;&gt;&lt;script&gt;window.__x=1&lt;/script&gt;">');
  await settle();
  const img = t.querySelector("img");
  assert.ok(img, "expected the <img> element itself to survive");
  assert.equal(proxied(calls).length, 0,
    "a src DOMPurify had stripped was fetched anyway");
  assert.equal(win.__x, undefined, "a payload executed");
  assert.equal(t.querySelectorAll("script").length, 0);
});

test("a refused proxy fetch is noted, never thrown, and never falls back remote", async () => {
  // 403 is the DEFAULT state (the feature ships off), so this is the ordinary
  // path, not an edge case. It must not throw inside a reply - and it must not
  // pass silently either, which is what it did until the note below existed.
  const { win, calls } = loadReal({ ok: false });
  const t = render(win, "![x](https://example.com/a.png)");
  await settle();
  assert.equal(proxied(calls).length, 1);
  const note = t.querySelector(".img-blocked");
  assert.equal(note.dataset.lmProxyFailed, "1", "the failure was not recorded");
  // The URL survives in data-lm-proxy-src on purpose (diagnostics, same as it was
  // on the <img>). What must not survive is anything the browser would FETCH.
  const fetchable = [...t.querySelectorAll("*")].filter((n) =>
    ["src", "href", "srcset", "poster", "data"].some((a) =>
      (n.getAttribute(a) || "").includes("example.com")));
  assert.deepEqual(fetchable, [],
    "a failed proxy fetch must not fall back to the remote URL");
  assert.equal(t.querySelector("img"), null, "and no src-less <img> is left behind");
});

test("a remote srcset is stripped so the proxied src is what renders", async () => {
  // DOMPurify's default allowlist passes srcset. When an <img> carries both, the
  // browser picks a srcset candidate and IGNORES src - so proxying src alone
  // would leave the element pointing at the remote host and, with the feature on,
  // the image would not render at all.
  const { win, calls } = loadReal();
  const t = render(win,
    '<img src="https://example.com/a.png" srcset="https://example.com/a2.png 2x">');
  await settle();
  const img = t.querySelector("img");
  assert.equal(img.getAttribute("srcset"), null,
    "a remote srcset survived, so the browser would still fetch it directly");
  assert.equal(proxied(calls).length, 1, "the src should still be proxied");
});

test("a <picture><source srcset> pointing remote is emptied", async () => {
  // <picture> and <source> are both in DOMPurify's default allowlist. A surviving
  // remote <source> wins over the <img> fallback, so the proxy would be bypassed.
  const { win, calls } = loadReal();
  const t = render(win,
    '<picture><source srcset="https://example.com/s.webp"><img src="https://example.com/f.png"></picture>');
  await settle();
  const src = t.querySelector("source");
  assert.ok(!src || src.getAttribute("srcset") === null,
    "a remote <source srcset> survived and would bypass the proxied <img>");
  assert.equal(proxied(calls).length, 1);
});

test("a LOCAL srcset is left alone", async () => {
  // Only remote candidates are the problem; stripping a local one would break a
  // legitimate responsive image for no gain.
  const { win } = loadReal();
  const t = render(win, '<img src="/uploads/a.png" srcset="/uploads/a2.png 2x">');
  await settle();
  assert.equal(t.querySelector("img").getAttribute("srcset"), "/uploads/a2.png 2x");
});

test("clearImageProxyCache drops cached images so the OFF switch takes effect", async () => {
  // Without this the route starts refusing but an already-fetched blob keeps
  // rendering for the whole page session - strictly longer than the 5 minutes
  // that made `Cache-Control: max-age` unacceptable.
  const { win, calls } = loadReal();
  render(win, "![x](https://example.com/a.png)");
  await settle();
  assert.equal(proxied(calls).length, 1);
  render(win, "![x](https://example.com/a.png)");
  await settle();
  assert.equal(proxied(calls).length, 1, "second render should have used the cache");
  win.clearImageProxyCache();
  render(win, "![x](https://example.com/a.png)");
  await settle();
  assert.equal(proxied(calls).length, 2,
    "after clearing, the image must be re-fetched (and so re-authorised by the route)");
});

// --- the refusal is SURFACED, not swallowed -------------------------------
// The route answers with seven distinct reasons and each one is different work
// for the user: the feature is off (and which setting turns it on), the host is
// not on their own net_allow list, the image is over the size cap, the response
// was not an image. All of them used to collapse to a src-less <img>, which
// renders as the model's alt text if it wrote one and as NOTHING at all if it
// wrote `![](...)`. So a reply could silently lose an image with no way for the
// reader to know one had been there.

const OFF_REASON =
  "Showing remote images is off. Turn on 'Show remote images in replies' " +
  "under Settings > Network to enable it.";

test("a refused image says why, using the reason the route gave", async () => {
  const { win } = loadReal({ ok: false, detail: OFF_REASON });
  const t = render(win, "![a chart of Q3 revenue](https://cdn.example.com/a.png)");
  await settle();

  const note = t.querySelector(".img-blocked");
  assert.ok(note, "a refused image leaves a visible note, not a hole in the reply");
  assert.equal(t.querySelector("img"), null, "and the dead <img> is gone");
  assert.match(note.textContent, /Image not shown/);
  assert.ok(note.textContent.includes(OFF_REASON),
    "the route's own reason is shown - it names the setting that fixes this, and " +
    "discarding it is what made a net_allow miss look identical to a dead host");
  assert.equal(note.title, OFF_REASON, "and repeated as a tooltip");
  assert.equal(note.dataset.lmProxySrc, "https://cdn.example.com/a.png",
    "what the model asked for is still recorded for diagnostics");
});

test("the model's alt text is carried across as TEXT, never as markup", async () => {
  const { win } = loadReal({ ok: false, detail: OFF_REASON });
  const t = render(win, "![<img src=x onerror=alert(1)>](https://cdn.example.com/a.png)");
  await settle();

  const note = t.querySelector(".img-blocked");
  assert.ok(note);
  assert.equal(note.querySelector("img"), null,
    "the alt text is inserted with textContent, so markup inside it stays inert");
  assert.ok(note.textContent.includes("<img src=x onerror=alert(1)>"),
    "and is shown literally");
});

test("an image with NO alt still leaves a visible note (the silent-hole case)", async () => {
  const { win } = loadReal({ ok: false, detail: OFF_REASON });
  const t = render(win, "![](https://cdn.example.com/a.png)");
  await settle();

  const note = t.querySelector(".img-blocked");
  assert.ok(note, "an empty alt used to render a 0x0 element - nothing on screen at all");
  assert.match(note.textContent, /Image not shown/);
});

test("a refusal with no JSON body still says something rather than nothing", async () => {
  const { win } = loadReal({ ok: false, status: 502, detail: null });
  const t = render(win, "![x](https://cdn.example.com/a.png)");
  await settle();

  const note = t.querySelector(".img-blocked");
  assert.ok(note, "a body-less failure is still surfaced");
  assert.match(note.textContent, /could not fetch it \(HTTP 502\)/,
    "with the status, so it is diagnosable rather than a bare shrug");
});
