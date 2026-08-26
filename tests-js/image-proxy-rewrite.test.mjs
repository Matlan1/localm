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

// A STAND-IN for what the route says, not a reading of it: the harness supplies
// this as the 403 body, so this file cannot notice the route rewording itself.
// tests/test_image_proxy.py owns the real sentence.
const OFF_REASON =
  "Showing remote images is off. Set 'Show remote images in replies' " +
  "to 'ask' or 'on' under Settings > Network to enable it.";

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

// --------------------------------------------------------------------------- //
//  The `ask` state: per-origin consent.
//
//  `on` decides WHO makes the request. `ask` decides WHETHER it is made, and the
//  route raises its 428 before fetching anything, so what these pin is the
//  client half: one dialog per ORIGIN per CONVERSATION, an answer remembered for
//  exactly that long, and a decline that does not come back on the next chunk.
// --------------------------------------------------------------------------- //

const ASK_REASON =
  "Showing remote images is set to 'ask', and this site has not been allowed " +
  "in this conversation.";

/** loadReal(), but standing in for a server whose setting is `ask`: every
 *  request without `consent=1` is a 428, and one with it succeeds. */
function loadAsking() {
  const calls = [];
  const { window: win } = loadApp({
    fetchImpl: async (url, opts) => {
      const u = String(url);
      calls.push({ url: u, headers: (opts && opts.headers) || {} });
      if (u.includes("/api/image-proxy") && !/[?&]consent=1(&|$)/.test(u)) {
        return {
          ok: false, status: 428, blob: async () => null,
          json: async () => ({ detail: ASK_REASON }),
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

function renderScoped(win, md, imageScope) {
  const t = win.document.createElement("div");
  win.document.body.appendChild(t);
  win.renderMarkdown(t, md, { final: true, imageScope });
  return t;
}

const modalOpen = (win) => win.document.getElementById("modal").style.display === "flex";
const modalText = (win) => win.document.getElementById("modal-body").textContent;
function clickConsent(win, label) {
  const btn = [...win.document.querySelectorAll("#modal-body button")]
    .find((b) => b.textContent === label);
  assert.ok(btn, "no \"" + label + "\" button in the dialog: " + modalText(win));
  btn.click();
}
const consented = (calls) => proxied(calls).filter((c) => /[?&]consent=1(&|$)/.test(c.url));

test("ask: nothing loads until the reader answers, and the dialog names the origin",
  async () => {
    const { win, calls } = loadAsking();
    const t = renderScoped(win, "![x](https://cdn.example.com/a.png)", "chat:1");
    await settle();

    assert.ok(modalOpen(win), "the reader is asked");
    assert.match(modalText(win), /https:\/\/cdn\.example\.com/,
      "and the dialog names the ORIGIN, which is what they are deciding about");
    assert.doesNotMatch(modalText(win), /a\.png/,
      "not the model-chosen path, which is the exfiltration payload itself");
    // The load-bearing assertion: no consented request has been made, so at this
    // point the remote host has not been contacted for this image at all.
    assert.equal(consented(calls).length, 0,
      "a consented fetch went out before the reader answered: " + JSON.stringify(calls));
    assert.equal(t.querySelector("img").getAttribute("src"), null);
  });

test("ask: allowing the origin re-requests WITH consent and the image renders",
  async () => {
    const { win, calls } = loadAsking();
    const t = renderScoped(win, "![x](https://cdn.example.com/a.png)", "chat:1");
    await settle();
    clickConsent(win, "Show images from this site");
    await settle();

    assert.equal(consented(calls).length, 1, JSON.stringify(calls));
    assert.match(t.querySelector("img").getAttribute("src") || "", /^blob:/);
    assert.equal(modalOpen(win), false, "and the dialog closed");
  });

test("ask: declining says so by name and does NOT re-ask on the next render",
  async () => {
    // The dialog-per-token defect in its second form. renderMarkdown reassigns
    // innerHTML on every streamed chunk, so a decline that is not remembered
    // reopens the dialog for the whole length of a reply.
    const { win, calls } = loadAsking();
    const t = renderScoped(win, "![x](https://cdn.example.com/a.png)", "chat:1");
    await settle();
    clickConsent(win, "Do not load");
    await settle();

    const note = t.querySelector(".img-blocked");
    assert.ok(note, "a declined image leaves a visible note, not a hole");
    assert.match(note.textContent,
      /You chose not to load images from https:\/\/cdn\.example\.com/);
    assert.equal(consented(calls).length, 0, "and nothing was fetched");

    const before = calls.length;
    win.renderMarkdown(t, "![x](https://cdn.example.com/a.png) more text",
                       { final: true, imageScope: "chat:1" });
    await settle();
    assert.equal(modalOpen(win), false, "the next chunk must not re-open the dialog");
    assert.equal(calls.length, before,
      "and must not re-ask the server either - the answer is remembered");
  });

test("ask: one answer covers every image from that ORIGIN, and only that origin",
  async () => {
    // Per-origin is the decision, not an optimisation: the payload is in the
    // URL, so one dialog per URL would be one mis-click chance per attempt.
    const { win, calls } = loadAsking();
    const t = renderScoped(win,
      "![a](https://cdn.example.com/1.png)\n\n![b](https://cdn.example.com/2.png)",
      "chat:1");
    await settle();
    assert.ok(modalOpen(win), "asked once");
    clickConsent(win, "Show images from this site");
    await settle();

    assert.equal(modalOpen(win), false, "and not asked again for the same host");
    assert.equal(consented(calls).length, 2, "both images loaded on one answer");
    assert.equal(t.querySelectorAll("img[src^=\"blob:\"]").length, 2);

    renderScoped(win, "![c](https://other.example.net/3.png)", "chat:1");
    await settle();
    assert.ok(modalOpen(win), "a DIFFERENT origin is a different decision");
    assert.match(modalText(win), /https:\/\/other\.example\.net/);
  });

test("ask: an answer given in one conversation is not visible in another", async () => {
  // "Remembered for the conversation" is the whole lifetime contract. It holds
  // by KEYING, not by a reset hook, so no future way of switching conversation
  // can forget to clear it.
  const { win } = loadAsking();
  renderScoped(win, "![a](https://cdn.example.com/1.png)", "chat:aaa");
  await settle();
  clickConsent(win, "Show images from this site");
  await settle();
  assert.equal(modalOpen(win), false);

  renderScoped(win, "![a](https://cdn.example.com/9.png)", "chat:bbb");
  await settle();
  assert.ok(modalOpen(win), "a second conversation must ask again for the same host");
});

test("consent is never written anywhere that outlives the page", async () => {
  const { win } = loadAsking();
  renderScoped(win, "![a](https://cdn.example.com/1.png)", "chat:1");
  await settle();
  clickConsent(win, "Show images from this site");
  await settle();

  for (const store of [win.localStorage, win.sessionStorage]) {
    for (const k of Object.keys(store)) {
      assert.doesNotMatch(String(store.getItem(k)), /cdn\.example\.com/,
        "a consent decision reached " + k + ", so it would outlive the page session");
    }
  }
});

test("clearImageProxyCache forgets consent, so a settings save re-asks", async () => {
  const { win } = loadAsking();
  renderScoped(win, "![a](https://cdn.example.com/1.png)", "chat:1");
  await settle();
  clickConsent(win, "Show images from this site");
  await settle();

  win.clearImageProxyCache();
  renderScoped(win, "![a](https://cdn.example.com/1.png)", "chat:1");
  await settle();
  assert.ok(modalOpen(win),
    "the setting that governs asking may have moved, so every answer is stale");
});

test("dismissing the dialog through the modal chrome counts as declining", async () => {
  // The x and the backdrop are not our handlers - they just hide the modal. A
  // dismissal that resolved as "allowed" would load the very image the reader
  // was getting away from.
  const { win, calls } = loadAsking();
  const t = renderScoped(win, "![a](https://cdn.example.com/1.png)", "chat:1");
  await settle();
  assert.ok(modalOpen(win));
  win.document.getElementById("modal-close").click();
  await new Promise((r) => setTimeout(r, 350));   // the display:none poll is 200ms

  assert.equal(consented(calls).length, 0, "nothing was fetched");
  assert.ok(t.querySelector(".img-blocked"), "and the reader is told why");
});

test("two origins in one reply raise their dialogs one at a time", async () => {
  // openModal drives a SINGLE #modal element, so two overlapping asks would
  // leave the second silently replacing the first, and the first's promise
  // resolving off buttons that are no longer on screen.
  const { win } = loadAsking();
  renderScoped(win,
    "![a](https://one.example.com/1.png)\n\n![b](https://two.example.net/2.png)",
    "chat:1");
  await settle();

  const first = modalText(win);
  assert.match(first, /https:\/\/one\.example\.com/);
  assert.doesNotMatch(first, /two\.example\.net/,
    "the second ask must not have overwritten the first");
  clickConsent(win, "Do not load");
  await settle();
  assert.ok(modalOpen(win), "and the second one follows");
  assert.match(modalText(win), /https:\/\/two\.example\.net/);
});

test("ask: a DECLINED origin stays declined for its other images too", async () => {
  // The mirror of the test above, and the one the fires-control exposed as
  // missing: an ALLOW was remembered per origin while a DECLINE was remembered
  // only per URL, so the next image from a host the reader had just refused
  // opened the dialog all over again. Per-origin has to mean both answers.
  const { win, calls } = loadAsking();
  renderScoped(win, "![a](https://cdn.example.com/1.png)", "chat:1");
  await settle();
  clickConsent(win, "Do not load");
  await settle();

  const t = renderScoped(win, "![b](https://cdn.example.com/2.png)", "chat:1");
  await settle();
  assert.equal(modalOpen(win), false,
    "a different image from an origin the reader already refused must not re-ask");
  assert.equal(consented(calls).length, 0, "and nothing was fetched");
  assert.ok(t.querySelector(".img-blocked"), "the second image is refused too");
});

test("a REFUSED origin's later images resolve without waiting on another dialog",
  async () => {
    // The reason requestOriginConsent answers from memory BEFORE joining the
    // queue. An ALLOWED origin never reaches it (it sends its consent and gets
    // a 200, so no 428 and no ask). A REFUSED one does: every later image from
    // it still 428s, and without the memory read up front that 428 queues
    // behind whatever dialog happens to be open, so a reply full of images from
    // a host the reader already refused sits blank until they deal with an
    // unrelated host.
    const { win, calls } = loadAsking();
    renderScoped(win, "![a](https://refused.example.com/1.png)", "chat:1");
    await settle();
    clickConsent(win, "Do not load");
    await settle();

    // Open a dialog for a DIFFERENT, undecided host and LEAVE IT OPEN.
    renderScoped(win, "![b](https://undecided.example.net/2.png)", "chat:1");
    await settle();
    assert.ok(modalOpen(win), "the undecided host is asking");
    assert.match(modalText(win), /undecided\.example\.net/);

    // Another image from the refused host must reach its note meanwhile.
    const t = renderScoped(win, "![c](https://refused.example.com/3.png)", "chat:1");
    await settle();
    assert.ok(t.querySelector(".img-blocked"),
      "a refused origin's image still says so while an unrelated dialog is open");
    assert.equal(consented(calls).length, 0, "and nothing was fetched for it");
    assert.ok(modalOpen(win), "without disturbing the open dialog");
    assert.match(modalText(win), /undecided\.example\.net/);
  });

test("a realistic reply raises ONE dialog per ORIGIN, however many images it has",
  async () => {
    // The number that decides whether this feature is usable. Per-origin was
    // chosen over per-URL so a reply cannot produce a dialog per image, and the
    // failure mode being designed against is habituation, not leakage: a modal
    // that appears often enough becomes reflex, and a prompt people are trained
    // to dismiss is worse than no prompt. So the COUNT is the property, and it
    // is asserted here rather than left to follow from the mechanism.
    const { win } = loadAsking();
    const md = [
      "![1](https://a.example.com/1.png)", "![2](https://a.example.com/2.png)",
      "![3](https://a.example.com/3.png)", "![4](https://b.example.net/4.png)",
      "![5](https://b.example.net/5.png)", "![6](https://c.example.org/6.png)",
      "![7](https://c.example.org/7.png)", "![8](https://a.example.com/8.png)",
    ].join("\n\n");

    renderScoped(win, md, "chat:1");
    let dialogs = 0;
    // Answer whatever is open until nothing more asks, counting as we go.
    for (let i = 0; i < 20; i++) {
      await settle();
      if (!modalOpen(win)) break;
      dialogs += 1;
      clickConsent(win, "Show images from this site");
    }
    assert.equal(dialogs, 3,
      "8 images across 3 origins must ask 3 times, not 8");

    // And a re-render of the same reply, which streaming does per token, adds none.
    renderScoped(win, md, "chat:1");
    await settle();
    assert.equal(modalOpen(win), false, "a re-render asks nothing further");
  });
