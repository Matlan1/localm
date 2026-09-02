// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - shared helpers (split from app.js). Classic script: it
   shares the one global lexical environment with the other app/* and
   pages/* scripts, so every cross-section reference resolves by bare
   name exactly as before. */
"use strict";

// --- ES module imports ---
import { t } from "./i18n.js";

export const $ = (id) => document.getElementById(id);

// Read a JSON value from localStorage without letting a CORRUPT entry crash the
// caller. `JSON.parse(getItem(k) || "[]")` collapses two cases: a MISSING key
// (normal first run) and a PRESENT-but-malformed value (truncated write, quota
// loss, manual/extension edit). The `|| "[]"` only covers missing; a corrupt
// value still throws SyntaxError, and at module top level (chat.js state init)
// that aborts the whole ES-module graph, booting a blank shell with no recovery.
// Branch the two cases and surface corruption (rule 5: warn, do not swallow).
export function readStoredJSON(key, fallback) {
  let raw;
  try { raw = localStorage.getItem(key); }
  catch (e) { console.warn(`localm: localStorage unavailable for "${key}":`, e); return fallback; }
  if (raw === null) return fallback;                 // absent - the normal case
  try {
    return JSON.parse(raw);
  } catch (e) {
    console.warn(`localm: ignoring corrupt localStorage["${key}"] (kept a blank default):`, e);
    return fallback;
  }
}

// AUD-INSTANCEID (canonical: see reconcileInstanceId below). localStorage is
// scoped by browser ORIGIN only, never by which backend DATA DIRECTORY runs
// behind it, and localm reuses the default port, so a fresh install can inherit
// a prior instance's origin and its localStorage bucket. Every key below is only
// meaningful for the connected backend, so all are wiped together when its
// instance id (served on /v1/config) does not match this origin's last-confirmed
// one. (localm.theme, localm.logoStyle, and the TTS voice picks are genuine
// device/browser preferences and are deliberately NOT in this list.)
//
// LM-DA-047: this list also doubles as the privacy-mode wipe list (chat.js's
// refreshCtxLimit) - both wipes must cover every key ANY call site writes only
// outside privacy mode, or a trace written before privacy mode was turned on
// survives it. Every such write goes through chat.js's lsSetScoped(key, value),
// which warns loudly (not silently) if `key` is missing here, and
// tests-js/privacy-scoped-keys.test.mjs source-scans every lsSetScoped call
// site and fails if one is missing - so the two lists (this one, and the set
// of actual write-gated call sites) cannot drift apart unnoticed again.
export const INSTANCE_SCOPED_KEYS = [
  "localm.conversations",
  "localm.activeView",
  "localm.coderCwd",
  "localm.kbAddPath",
  "localm.convCollapsed",
  "localm.imgMoveDest",
  "localm.musicMoveDest",
  "localm.videoMoveDest",
  "localm.onboarded",
  "localm.webAccess",
  "localm.speakAloud",
  "localm.studioOpen",
  "localm.coderOpen",
  "localm.backendHintDismissed",
];

const INSTANCE_ID_KEY = "localm.instanceId";

/** True when this browser origin already confirmed the connected backend's id on
 *  an EARLIER /v1/config round trip. Gates every instance-scoped localStorage
 *  read that runs at boot BEFORE this page load's own round trip resolves (see
 *  init.js): a browser that never confirmed pairing with this exact backend must
 *  not trust, render, or upload data left by a different one at the same
 *  origin/port (AUD-INSTANCEID). */
export function instanceCacheTrusted() {
  try { return !!localStorage.getItem(INSTANCE_ID_KEY); }
  catch (e) { return false; }
}

/** Reconcile the cached instance id against the one the connected backend just
 *  reported (cfg.instance_id from /v1/config). Returns one of three states -
 *  callers must not collapse them back into a boolean (that collapse is
 *  exactly what let an "unknown" read authorise an upload meant only for a
 *  "confirmed" one, AUD-INSTANCEID residual 2):
 *   - "confirmed": the cached id matches THIS backend - safe to render, merge
 *     AND upload.
 *   - "mismatched": the cache just belonged to a DIFFERENT backend, or had
 *     never been confirmed for this origin before (a brand-new pairing,
 *     exactly the cross-instance leak scenario) - every instance-scoped key is
 *     wiped before returning.
 *   - "unknown": a missing/falsy *serverInstanceId* (an older server that
 *     predates this field) or an unreadable localStorage means there is
 *     nothing to compare against - existing (optimistic) rendering is
 *     preserved, but callers must NOT treat this as a confirmed match for
 *     anything that writes data back to the backend. */
export function reconcileInstanceId(serverInstanceId) {
  if (!serverInstanceId) return "unknown";
  let cached;
  try { cached = localStorage.getItem(INSTANCE_ID_KEY); }
  catch (e) { return "unknown"; }   // localStorage unavailable - nothing to protect or confirm
  if (cached === serverInstanceId) return "confirmed";
  for (const key of INSTANCE_SCOPED_KEYS) {
    try { localStorage.removeItem(key); } catch (e) { /* best-effort wipe */ }
  }
  try { localStorage.setItem(INSTANCE_ID_KEY, serverInstanceId); }
  catch (e) { /* storage full/blocked - callers still correct in-memory state */ }
  return "mismatched";
}

// S2: the API key is no longer kept in JS-readable localStorage. Open mode uses
// the per-process shell token (global, sent as a bearer HEADER); protected mode
// rides the HttpOnly session cookie (opaque session id, auto-sent same-origin)
// plus a session-DERIVED CSRF token authHeaders() reads from window.__LOCALM_CSRF__
// (fetched from GET /api/session).
export const SHELL_TOKEN = window.__LOCALM_SHELL_TOKEN__ || "";

export function readCookie(name) {
  const m = document.cookie.match(new RegExp("(?:^|; )" + name + "=([^;]*)"));
  if (!m) return "";
  // A cookie value is untrusted input. decodeURIComponent throws a URIError on
  // malformed percent-encoding; if that propagates authHeaders() throws, EVERY
  // fetch rejects unsent, and bootAuthProbe reports a reachable server as
  // "unreachable" (reconnect overlay, no way out). A bad cookie must never brick
  // the client, so decode best-effort and fall back to the raw value on failure.
  try { return decodeURIComponent(m[1]); }
  catch (e) { return m[1]; }
}

export function authHeaders(extra = {}) {
  const h = { "Content-Type": "application/json", ...extra };
  const csrf = window.__LOCALM_CSRF__ || "";
  if (csrf) {
    // Session (cookie) mode: HttpOnly session cookie (auto-sent same-origin) + a
    // CSRF header. The token is DERIVED from the session server-side (fetched from
    // GET /api/session), NOT a readable cookie that could be cleared and desync
    // from the session (which 403'd every write - the reported bug). Do NOT also
    // send the shell-token bearer: the Authorization header wins over the cookie
    // server-side, and the open-mode shell token is rejected once auth is on.
    h["X-CSRF-Token"] = csrf;
  } else if (SHELL_TOKEN) {
    // Open mode: the per-process loopback shell token authorises local management.
    h["Authorization"] = "Bearer " + SHELL_TOKEN;
  }
  return h;
}

/** True when *headers* is a request header set WE built (authHeaders above) that
 *  carries the OPEN-MODE shell token as its credential.
 *
 *  The two auth modes are mutually exclusive by construction in authHeaders: a
 *  session sends `X-CSRF-Token` and deliberately NO bearer, open mode sends the
 *  shell bearer and has no CSRF token to send. So an Authorization header equal
 *  to our own shell token identifies an open-mode request exactly, and comparing
 *  the VALUE (not merely the presence of a bearer) keeps this from firing on some
 *  other caller's hand-built Authorization header.
 *
 *  Used by the 403 handler in init.js to tell "this process rotated the shell
 *  token out from under us" apart from every other reason a request can 403. */
export function sentShellToken(headers) {
  if (!SHELL_TOKEN || !headers) return false;
  const auth = headers instanceof Headers
    ? headers.get("Authorization")
    : (typeof headers === "object" ? headers["Authorization"] : null);
  return auth === "Bearer " + SHELL_TOKEN;
}

// Fetch the CSRF token for the current session and stash it for authHeaders().
// The token is an HMAC of the session computed server-side, so it is always in
// lockstep with the session cookie - there is no separate cookie to fall out of
// sync. Called at boot and by the 403-CSRF self-heal. Returns the token or "".
export async function refreshCsrf() {
  try {
    const r = await fetch("/api/session", { cache: "no-store" });
    if (!r.ok) return "";
    const j = await r.json();
    window.__LOCALM_CSRF__ = (j && j.csrf) || "";
    return window.__LOCALM_CSRF__;
  } catch (e) {
    return "";
  }
}

export function toast(msg, isError = false) {
  const el = $("toast");
  el.textContent = msg;
  el.className = "show" + (isError ? " error" : "");
  clearTimeout(el._t);
  el._t = setTimeout(() => (el.className = ""), 3500);
}

marked.setOptions({ breaks: true, mangle: false, headerIds: false });

/** Split leading <think>…</think> reasoning from the visible reply.
 *  Handles a still-open block during streaming. */
export function splitThink(text) {
  const m = /<think>([\s\S]*?)(?:<\/think>([\s\S]*)|$)/.exec(text || "");
  if (!m) return { think: null, open: false, rest: text || "" };
  const closed = m[2] !== undefined;
  const before = text.slice(0, m.index);
  return {
    think: m[1].trim(),
    open: !closed,
    rest: before + (closed ? m[2] : ""),
  };
}

/** Reply text with reasoning blocks removed - what gets sent back to the
 *  model on later turns. */
export function stripThink(text) {
  return (text || "").replace(/<think>[\s\S]*?(<\/think>|$)/g, "").trim();
}

/** Replace a raw tool-call block with a compact note. Runs in the DISPLAY and on
 *  the assistant history RE-SENT to the model, so a model never sees its own raw
 *  control tokens echoed back (feeding `<|tool_call>` markers back destabilised
 *  some finetunes into repetition - CHAT-TOOL-1). Matches every dialect
 *  `parseWebCall` EXECUTES (`<tool_call>`, the |-piped `<|tool_call|>` wrappers,
 *  the Gemma `call:{...}` prefix); name/query use tolerant regexes (inner JSON is
 *  often single-quoted / trailing-comma'd) so anything that ran is defanged too. */
export function formatToolCalls(text) {
  return (text || "").replace(
    /<\|?\/?tool_call\|?>[\s\S]*?<\|?\/?tool_call\|?>/g,
    (block) => {
      const name = (block.match(/"name"\s*:\s*"(\w+)"/) || [])[1] || "";
      const arg = (block.match(/"(?:query|url)"\s*:\s*"([^"]*)"/) || [])[1] || "";
      const what =
        name === "web_search" ? t("common.toolCall.webSearch", { arg }) :
        name === "fetch_url"  ? t("common.toolCall.readPage", { arg: arg || "" }) :
        arg ? t("common.toolCall.webRequest", { arg }) : t("common.toolCall.webRequestBare");
      return `\n> 🌐 *${what}*\n`;
    });
}

/** Last-resort client scrub of model-internal control markers. The backend
 *  (localm/inference/textnorm.py) normalises these, but a third-party or
 *  plugged-in backend might not, so the GUI never renders raw channel tokens.
 *  Mirrors the server regexes; runs on the full accumulated text so there is no
 *  streaming-boundary concern here. */
export function scrubMarkers(text) {
  return (text || "")
    .replace(/<\|"\|>/g, '"')
    .replace(/<\|?\s*channel\s*\|?>(thought|thinking|analysis|reasoning|commentary|reflection)\n?(<\|?\s*message\s*\|?>)?/g, "<think>\n")
    .replace(/<\s*channel\s*\|>|<\|?\s*channel\s*\|?>final\n?(<\|?\s*message\s*\|?>)?/g, "\n</think>\n")
    .replace(/<\|?\s*channel\s*\|?>|<\s*channel\s*\|>|<\|?\s*message\s*\|?>|<\|start\|>(assistant|user|system)?|<\|return\|>|<\|turn>(user|model|assistant|system)?\n?|<turn\|>|<\|tool>|<tool\|>|<\|think\|>|<think\|>|<unused\d+>?/g, "");
}

/** Point every REMOTE <img> in a rendered reply at localm's own image proxy, so
 *  the browser never contacts the remote host.
 *
 *  The shell's CSP is `img-src 'self' data: blob:`, so a model-linked remote
 *  image simply does not load - the one place localm rendered less than the
 *  comparable UIs do. They close it by letting the browser fetch the image
 *  directly, which hands the remote host the user's IP, User-Agent and referrer.
 *  This closes it without that: /api/image-proxy fetches server-side through the
 *  same netpolicy path as every other outbound request.
 *
 *  DELIBERATELY UNCONDITIONAL - the server decides, not this function. The
 *  feature is off by default and the route 403s until the owner turns it on, so
 *  a default install renders exactly as before (a broken image, same as today).
 *  Reading a config flag here instead would mean baking it into the page at load
 *  and going stale the moment the user toggles the setting, and would put a
 *  security decision in the browser where it cannot be enforced.
 *
 *  THAT HOLDS FOR THE `ask` STATE TOO, which is why the prompt is driven by the
 *  route's 428 rather than by this function reading the mode. The client never
 *  learns what the setting says; it attempts, and the answer it gets IS the
 *  current setting. *scope* identifies the conversation the render belongs to,
 *  and is what keeps one conversation's answers out of another's - see
 *  _imgOriginConsent.
 *
 *  Runs AFTER sanitisation, and only ever REPLACES a src attribute with a
 *  same-origin URL built through encodeURIComponent - it inserts no markup, so
 *  it is not a sanitize-then-modify hazard. data:, blob: and relative/same-origin
 *  sources are left exactly as they are: they already load, and routing them
 *  through the proxy would be a pointless round trip. */
/** remote href -> blob: URL once fetched, or the in-flight Promise for it.
 *
 *  Keyed on the URL, NOT on the element, and that is load-bearing rather than an
 *  optimisation: renderMarkdown reassigns innerHTML on every streamed chunk, so
 *  the <img> is a BRAND NEW element each time and any per-element "already done"
 *  flag is destroyed with its predecessor. Measured before this existed: three
 *  renders of one reply produced three fetches, so a streaming reply would have
 *  re-fetched every image on every token. The cache also removes the flicker of
 *  an image blanking and reloading mid-stream. */
const _imgProxyCache = new Map();
const _IMG_PROXY_CACHE_MAX = 64;

/** WHAT THE READER HAS AGREED TO, for the `ask` state of the setting.
 *
 *  Keyed on (scope, origin); the value is the reader's answer, TRUE OR FALSE.
 *  Both, because per-origin has to mean both: remembering only the yes left a
 *  refused host re-opening the dialog for its every other image.
 *
 *  The scope is the conversation the render belongs to (renderMarkdown's
 *  `imageScope` option), so a decision taken in one conversation is NOT VISIBLE
 *  in another - the
 *  cross-conversation leak is unrepresentable rather than prevented by a reset
 *  that a future seventh way of switching conversation could miss.
 *
 *  THE LIFETIME IS THIS MAP'S OWN: one page session. It is never written to
 *  localStorage, sessionStorage or the server, so a reload asks again, and a
 *  consent decision cannot outlive the context it was given in. It is also
 *  dropped by clearImageProxyCache(), which every settings save calls.
 *
 *  ORIGIN, not URL, and that is the decision rather than an optimisation: the
 *  exfiltration payload is IN the URL, so one prompt per URL would be one
 *  mis-click chance per exfiltration attempt. An origin is also the thing a
 *  reader can actually reason about.
 *
 *  A remembered NO can never keep an image out once the setting is `on`: it is
 *  only ever consulted on the route's 428, and `on` does not raise one. */
const _imgOriginConsent = new Map();
/** Dialogs are serialised through this: openModal drives a SINGLE #modal
 *  element, so two overlapping asks would leave the second silently replacing
 *  the first and the first's promise resolving off the wrong buttons.
 *
 *  It is also what de-duplicates them. Ten images from one host queue ten
 *  thunks, the first opens a dialog, and each of the nine behind it re-reads
 *  _imgOriginConsent on its turn and finds the answer already there. */
let _imgConsentQueue = Promise.resolve();

function _consentKey(scope, origin) {
  // JSON rather than a delimiter: an origin cannot collide with a scope
  // whatever either of them contains, with no separator to reason about.
  return JSON.stringify([scope || "", origin]);
}

/** Drop every cached proxied image, release its object URL, and forget every
 *  per-origin consent.
 *
 *  MUST be called when the remote-image setting may have changed. Without it the
 *  OFF switch does not take effect for anything already on screen: the route
 *  starts refusing, but a cached blob keeps rendering for the REST OF THE PAGE
 *  SESSION, including in a conversation the user has not opened yet. That is the
 *  same staleness the response's `no-store` was added to fix, and strictly worse
 *  - a session outlasts the five minutes that was judged unacceptable there.
 *  Closing the HTTP cache while leaving this one open fixed half the defect.
 *
 *  Consent goes with it, for the same reason and on the same terms: the setting
 *  that governs asking may just have moved, so the safe reading of a save is
 *  that every earlier answer is stale. It costs one re-prompt, and the
 *  alternative is deciding which key moved, which puts a security decision
 *  behind a diff. */
export function clearImageProxyCache() {
  for (const v of _imgProxyCache.values()) {
    if (typeof v === "string") URL.revokeObjectURL(v);
  }
  _imgProxyCache.clear();
  _imgOriginConsent.clear();
}
window.clearImageProxyCache = clearImageProxyCache;

/** Ask the reader whether to load images from ONE origin in ONE conversation.
 *
 *  Resolves true/false, never rejects. Built on openModal like confirmDanger and
 *  promptText rather than window.confirm, which is suppressed outright in some
 *  mobile / PWA browsers (the NET-1 class): there it returns undefined with no
 *  error, which is indistinguishable from the reader declining and would make
 *  the `ask` state silently unusable on a phone.
 *
 *  Dismissing through the shared modal chrome (the x, or the backdrop) is a
 *  DECLINE. Those handlers are not ours, so it is detected by polling for
 *  display:none - the same pattern promptText and the missing-model modal use. */
function askOriginConsent(origin) {
  return new Promise((resolve) => {
    let settled = false;
    let watch = null;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      if (watch) clearInterval(watch);
      $("modal").style.display = "none";
      resolve(value);
    };
    openModal(t("common.imageConsent.title"), (body) => {
      body.appendChild(el("p", "", t("common.imageConsent.body", { origin })));
      body.appendChild(el("p", "", t("common.imageConsent.remember")));
      const row = el("div", "actions");
      const no = el("button", "btn-secondary", t("common.imageConsent.decline"));
      no.onclick = () => finish(false);
      const yes = el("button", "btn-secondary", t("common.imageConsent.allow"));
      yes.onclick = () => finish(true);
      row.appendChild(no);
      row.appendChild(yes);
      body.appendChild(row);
    });
    watch = setInterval(() => {
      if ($("modal").style.display === "none") finish(false);
    }, 200);
  });
}

/** The reader's answer for one scope+origin: the remembered one if there is one,
 *  otherwise a dialog. Serialised so only one is ever open, which is also what
 *  keeps ten images from one host to a single dialog - see _imgConsentQueue.
 *  Both answers are remembered; nothing is remembered if the ask never ran. */
function requestOriginConsent(scope, origin) {
  const key = _consentKey(scope, origin);
  if (_imgOriginConsent.has(key)) return Promise.resolve(_imgOriginConsent.get(key));
  const p = _imgConsentQueue
    .then(() => (_imgOriginConsent.has(key)
      ? _imgOriginConsent.get(key)
      : askOriginConsent(origin).then((ok) => {
        _imgOriginConsent.set(key, ok);
        return ok;
      })))
    .catch(() => false);
  // The QUEUE swallows the answer: a declined ask must not stop the next one.
  _imgConsentQueue = p.then(() => undefined, () => undefined);
  return p;
}

function _rememberProxiedImage(href, objUrl) {
  if (_imgProxyCache.size >= _IMG_PROXY_CACHE_MAX) {
    const oldest = _imgProxyCache.keys().next().value;
    const stale = _imgProxyCache.get(oldest);
    _imgProxyCache.delete(oldest);
    // Only a settled entry holds a revocable URL; an in-flight Promise does not.
    if (typeof stale === "string") URL.revokeObjectURL(stale);
  }
  _imgProxyCache.set(href, objUrl);
}

/** A refusal the PROXY ROUTE issued, carrying the reason it gave. Distinct from a
 *  transport failure, which arrives at the same catch as an ordinary Error. */
class ImageProxyRefused extends Error {
  constructor(status, detail) {
    super(detail || ("HTTP " + status));
    this.status = status;
    this.detail = detail || "";
  }
}

/** The reader answered "do not load" for this origin in this conversation. Not
 *  a failure of anything - the feature working - so it says so in its own
 *  words rather than borrowing the route's. */
class ImageOriginDeclined extends Error {
  constructor(origin) {
    super("declined: " + origin);
    this.origin = origin;
  }
}

/** One sentence saying why an image did not load, for the placeholder below. */
function proxyRefusalText(e) {
  if (e instanceof ImageOriginDeclined) {
    return t("common.imageProxy.declined", { origin: e.origin });
  }
  if (e instanceof ImageProxyRefused) {
    if (e.detail) return e.detail;
    if (e.status === 403) return t("common.imageProxy.refused");
    return t("common.imageProxy.failed", { status: e.status });
  }
  return t("common.imageProxy.unreachable");
}

/** One attempt at the proxy route. *consented* adds the reader's per-origin
 *  answer for the `ask` state; it is ignored by every other state, and it can
 *  only ever reach what `on` would have reached anyway. */
function _requestProxiedImage(href, consented) {
  return fetch("/api/image-proxy?url=" + encodeURIComponent(href)
               + (consented ? "&consent=1" : ""),
               { headers: authHeaders() })
    .then(async (r) => {
      if (r.ok) return r.blob();
      // The route answers with a REASON, and each one is different work for
      // the user: the feature is off (and which setting turns it on), this site
      // has not been allowed in this conversation, the host is not on their own
      // net_allow list, the image is over the size cap, the response was not an
      // image. Read it here, because after this it is gone.
      let detail = "";
      try { detail = (await r.json()).detail || ""; } catch (e) { /* no JSON body */ }
      return Promise.reject(new ImageProxyRefused(r.status, detail));
    });
}

/** Fetch one remote image through the proxy, asking the reader about its ORIGIN
 *  first if the route says the setting requires it.
 *
 *  Resolves to a blob: URL, or to `{failed, reason, declined}` - it never
 *  rejects, because the caller renders the reason either way.
 *
 *  428 means the setting is `ask` and this origin has no answer yet, and it is
 *  raised BEFORE the route fetches anything, so nothing has reached the remote
 *  host at the point the dialog opens. */
function _proxyImage(href, origin, scope) {
  const consented = _imgOriginConsent.get(_consentKey(scope, origin)) === true;
  return _requestProxiedImage(href, consented)
    .catch((e) => {
      if (!(e instanceof ImageProxyRefused) || e.status !== 428) throw e;
      return requestOriginConsent(scope, origin).then((ok) => {
        if (!ok) throw new ImageOriginDeclined(origin);
        return _requestProxiedImage(href, true);
      });
    })
    .then((blob) => {
      const objUrl = URL.createObjectURL(blob);
      _rememberProxiedImage(href, objUrl);
      return objUrl;
    })
    .catch((e) => {
      const declined = e instanceof ImageOriginDeclined;
      // Forget it so a later render may retry - EXCEPT after a decline, which
      // is an answer rather than a failure. Retrying that would re-ask on every
      // streamed chunk, which is the same dialog-per-token defect keying this
      // on the URL instead of the element exists to avoid. The remembered
      // answer is dropped by clearImageProxyCache() on any settings save, and
      // it can never keep an image OUT once the setting is `on`, because `on`
      // never asks in the first place.
      if (!declined) _imgProxyCache.delete(href);
      return { failed: true, declined, reason: proxyRefusalText(e) };
    });
}

/** Replace a remote image that did not load with a visible note saying so.
 *
 *  Builds the node with createElement/textContent and never assigns markup, so
 *  it stays outside the sanitize-then-modify hazard the surrounding function is
 *  careful about. The model's alt text is carried along as TEXT.
 *  See "a refused image says why" in tests-js/image-proxy-rewrite.test.mjs. */
function showBlockedImage(img, reason) {
  const note = el("span", "img-blocked");
  note.dataset.lmProxySrc = img.dataset.lmProxySrc || "";
  note.dataset.lmProxyFailed = "1";
  note.title = reason;
  const alt = (img.getAttribute("alt") || "").trim();
  note.appendChild(el("span", "img-blocked-label",
                      alt ? t("common.image.notShownWithAlt", { alt }) : t("common.image.notShown")));
  note.appendChild(el("span", "img-blocked-why", reason));
  if (img.parentNode) img.parentNode.replaceChild(note, img);
}

function proxyRemoteImages(root, scope) {
  // srcset FIRST, and it is not optional tidying. DOMPurify's default allowlist
  // passes `srcset`, `picture` and `source` (verified against the vendored
  // build), and when an <img> carries both, the browser picks a srcset candidate
  // and IGNORES src - so proxying src alone leaves the element still pointing at
  // the remote host, and with the feature ON the image would not render at all.
  // There is no cheap way to proxy each candidate (they are per-descriptor
  // alternatives), so the remote ones are dropped and the proxied src becomes the
  // single source. A <source> inside <picture> is emptied for the same reason,
  // which makes the browser fall through to the <img> this function does proxy.
  root.querySelectorAll("img[srcset], source[srcset]").forEach((node) => {
    const set = node.getAttribute("srcset") || "";
    if (/(^|[\s,])https?:\/\//i.test(set)) node.removeAttribute("srcset");
  });
  root.querySelectorAll("img[src]").forEach((img) => {
    const raw = img.getAttribute("src") || "";
    if (!/^https?:\/\//i.test(raw)) return;          // data:/blob:/relative: already fine
    let u;
    try { u = new URL(raw, window.location.href); } catch (e) { return; }
    if (u.origin === window.location.origin) return; // our own bytes, no detour
    img.dataset.lmProxySrc = u.href;                 // what the model asked for
    // Drop the remote src so no broken load stays pending. The browser has not
    // reached that host regardless - `img-src 'self' data: blob:` refused it the
    // moment innerHTML created the element, which is what actually guarantees the
    // privacy property here. If a future change ever adds a remote origin to
    // img-src, that guarantee moves to this line's timing and becomes a race.
    img.removeAttribute("src");

    const settle = (o) => {
      if (typeof o === "string") img.src = o;
      else if (o && o.failed) showBlockedImage(img, o.reason);
    };

    const cached = _imgProxyCache.get(u.href);
    if (typeof cached === "string") { img.src = cached; return; }
    // In flight, or a settled refusal held on purpose (a declined origin). The
    // handler is `settle`, NOT a bare `if (o) img.src = o` - a failure resolves
    // to an OBJECT, which is truthy, so that shape set src to "[object Object]"
    // and the element got no note at all. Reached whenever a second render
    // arrives before the first fetch answers, which streaming does constantly.
    if (cached) { cached.then(settle); return; }

    // MUST be fetch(), not a bare src=. In open mode every GET under /api/ needs
    // the per-process shell token as a BEARER header, and an <img> element cannot
    // send a header - so pointing src straight at the proxy 403s on the default
    // keyless install and the feature silently never works. Measured end to end:
    // 403 without the token, 200 with it, on the same URL.
    const pending = _proxyImage(u.href, u.origin, scope);
    _imgProxyCache.set(u.href, pending);
    pending.then(settle);
  });
}

/** Every `<a href>` under *root* opens in a new tab/window instead of the
 *  current one. */
function secureExternalLinks(root) {
  root.querySelectorAll("a[href]").forEach((a) => {
    a.target = "_blank";
    a.rel = "noopener noreferrer";
  });
}

/** True if the main reply body rendered to something the user can actually see.
 *  A reply can be a non-empty STRING yet render to nothing: a tiny model that
 *  emits only an unterminated / empty ```code fence produces an empty <pre><code>
 *  (blank box) - text-content is whitespace and there is no media. Whitespace-only
 *  and empty code fences count as NOT visible; text, images, tables, rules, math
 *  source etc. count as visible. Runs BEFORE KaTeX, so math is caught via its
 *  source ($x$ has non-empty text) rather than a rendered .katex node. */
function mainHasVisibleContent(main) {
  if ((main.textContent || "").trim() !== "") return true;
  return main.querySelector("img, svg, canvas, video, audio, iframe, table, hr, input") !== null;
}

/** Render one model reply into *target*.
 *
 *  opts.final          this is a SETTLED render (a reload or a post-stream
 *                      rebuild), not a live streaming shell.
 *  opts.imageScope     which conversation this render belongs to. Remote-image
 *                      consent is remembered per origin WITHIN this string, so
 *                      a caller that renders separate conversations must pass a
 *                      different one for each. Omitting it puts the render in
 *                      one shared unnamed scope - correct only where there is a
 *                      single stream of content per page session. */
export function renderMarkdown(target, text, opts = {}) {
  const { think, open, rest: rawRest } = splitThink(scrubMarkers(text));
  const rest = formatToolCalls(rawRest);

  // Think block: update IN PLACE rather than rebuild every token. Recreating the
  // <details> per chunk reset its open/closed state each tick, so the reasoning
  // bubble could not be toggled mid-stream; keeping the same element makes a
  // user toggle stick. Default: open while thinking, collapse once done - until
  // the user clicks it (data-userset), after which their choice is left alone.
  let det = target.querySelector("details.think-block");
  if (think) {
    if (!det) {
      det = document.createElement("details");
      det.className = "think-block";
      const sum = document.createElement("summary");
      sum.addEventListener("click", () => { det.dataset.userset = "1"; });
      det.appendChild(sum);
      det.appendChild(document.createElement("div"));
      target.insertBefore(det, target.firstChild);
    }
    det.querySelector("summary").textContent = open ? t("common.reasoning.thinking") : t("common.reasoning.thoughts");
    det.querySelector("div").innerHTML = DOMPurify.sanitize(marked.parse(think));
    if (!det.dataset.userset) det.open = open;
  } else if (det) {
    det.remove();
  }

  // Main content lives in its own container so refreshing it never disturbs the
  // think block (and its toggle state) above it.
  let main = target.querySelector(".md-main");
  if (!main) {
    main = document.createElement("div");
    main.className = "md-main";
    target.appendChild(main);
  }
  main.innerHTML = DOMPurify.sanitize(marked.parse(rest || ""));
  // On `target`, not `main`, so the think block's sink is covered by the same
  // call. Idempotent across a streaming re-render: an already-proxied src is
  // same-origin, so the second pass leaves it alone.
  secureExternalLinks(target);
  proxyRemoteImages(target, opts.imageScope);
  // Never leave a blank reply bubble. On a SETTLED render (opts.final - a reload
  // or post-stream renderChat, never a mid-stream shell) a body that rendered to
  // nothing visible gets a plain note instead of an empty box (real case: a 1B
  // model whose <think> works but whose answer is a bare empty ```code fence).
  // Gated on final so a slow model is never flashed a false "no reply" early.
  if (opts.final && !mainHasVisibleContent(main)) {
    main.replaceChildren(el("div", "md-empty", t("common.reply.empty")));
  }
  // LaTeX math: $...$, $$...$$, \(...\), \[...\]. KaTeX only rewrites text
  // nodes after sanitisation, so this stays XSS-safe.
  if (typeof renderMathInElement !== "undefined") {
    try {
      renderMathInElement(target, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true },
        ],
        throwOnError: false,
        // Pin trust:false explicitly (R41-D4): KaTeX's \htmlData / \href etc.
        // can emit raw HTML/URLs when trust is enabled; keep it off so the math
        // renderer cannot become an HTML-injection sink, independent of any
        // future KaTeX default change.
        trust: false,
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
      });
    } catch (e) { /* malformed TeX mid-stream - final render fixes it */ }
  }
  target.querySelectorAll("pre code").forEach((block) => {
    // Record the source language BEFORE hljs rewrites the class list, so the
    // artifact detector still knows this block was ```html / ```svg.
    const m = (block.className || "").match(/language-([\w-]+)/);
    if (m && block.dataset) block.dataset.lang = m[1];
    try { hljs.highlightElement(block); } catch (e) { /* unknown lang */ }
  });
  target.querySelectorAll("pre").forEach(enhanceCodeBlock);
}

/* Artifacts canvas (A3): a self-contained HTML/SVG reply block rendered live in
 * a side pane, HARD-sandboxed - an <iframe sandbox="allow-scripts"> (NO
 * allow-same-origin, so no access to this app's origin/cookies/storage) whose
 * srcdoc carries a CSP that blocks ALL network. Interactive yet cannot phone
 * home or read the app (privacy contract / "do not hide problems"). */

/** The artifact language for a <code> element, or null if it is not a
 *  renderable self-contained block. Reads the captured data-lang first, then
 *  sniffs the content (so an unlabelled <svg>/<!doctype html> still works). */
export function artifactLang(codeEl) {
  if (!codeEl) return null;
  const cls = (codeEl.className || "").match(/language-([\w-]+)/);
  const lang = ((codeEl.dataset && codeEl.dataset.lang) || (cls && cls[1]) || "").toLowerCase();
  const text = codeEl.innerText || codeEl.textContent || "";
  if (lang === "svg" || /^\s*<svg[\s>]/i.test(text)) return "svg";
  if (lang === "html" || lang === "xhtml") return "html";
  if (lang === "xml" && /^\s*<svg[\s>]/i.test(text)) return "svg";
  if (!lang && /<!doctype\s+html|<html[\s>]/i.test(text)) return "html";
  return null;
}

/** Stamp this document's CSP nonce onto every <script> in artifact *code* that
 *  does not already carry one.
 *
 *  MEASURED, and it is why this exists: a srcdoc document INHERITS the embedding
 *  document's CSP, and its own <meta> CSP cannot loosen what the inherited
 *  policy forbids. Once the shell's policy enforces `script-src 'self'
 *  'nonce-X'`, an artifact's inline <script> is BLOCKED - proven against a real
 *  browser, with the same iframe carrying the parent nonce running fine, so the
 *  nonce is the only variable. Without this the artifacts canvas renders markup
 *  but nothing interactive ever runs.
 *
 *  This does not widen what an artifact may do. The frame is
 *  sandbox="allow-scripts" with NO allow-same-origin, so it is an opaque origin
 *  that cannot touch this app's origin, cookies or storage, and the meta CSP
 *  below still denies it all network. The sandbox is the boundary; the inherited
 *  policy was only ever collateral damage. Artifact scripts were always meant to
 *  execute - that is the whole feature.
 *
 *  Deliberately a targeted rewrite of the <script> OPEN TAG rather than a
 *  DOMParser round trip: the caller's three shapes (bare SVG, full document,
 *  fragment) are spliced as strings, and re-serializing a full document through
 *  a parser would move nodes and could defeat the R41-D4 ordering guarantee
 *  below. A <script that is not a real tag is not valid HTML source anyway. */
function stampArtifactNonce(code) {
  const n = window.__LOCALM_CSP_NONCE__;
  if (!n) return code;                       // no enforcing policy in play
  return String(code).replace(
    /<script\b(?![^>]*\bnonce=)([^>]*)>/gi,
    '<script nonce="' + n + '"$1>');
}

/** Build the iframe srcdoc for an artifact, injecting a strict CSP that blocks
 *  network access. Inline script/style are allowed (the artifact runs), data:
 *  images are allowed, everything else is denied. */
export function artifactSrcdoc(code, lang) {
  code = stampArtifactNonce(code);
  // form-action 'none' is NOT redundant with default-src 'none', and leaving it
  // out made this function's own "blocks ALL network" claim untrue. form-action
  // is a NAVIGATION directive: it has no default-src fallback, so an unset
  // form-action allows submission to ANY origin. An artifact is model-authored
  // HTML, so <form action="https://elsewhere/"> was a way for the pane to send
  // whatever a user typed into it off the machine - no script, so the sandbox
  // and the nonce were never in that path. Measured on the shell's own policy
  // 2026-08-18 (same defect, fixed alongside in http_server.py's _CSP_SUFFIX).
  const csp = '<meta http-equiv="Content-Security-Policy" content="'
    + "default-src 'none'; img-src data: blob:; media-src data: blob:; "
    + "style-src 'unsafe-inline'; script-src 'unsafe-inline'; font-src data:; "
    + "form-action 'none';\">";
  if (lang === "svg" || /^\s*<svg[\s>]/i.test(code)) {
    return "<!doctype html><html><head><meta charset=\"utf-8\">" + csp
      + "<style>html,body{margin:0;height:100%}svg{max-width:100%;height:auto;display:block}</style>"
      + "</head><body>" + code + "</body></html>";
  }
  if (/<!doctype\s+html/i.test(code) || /<html[\s>]/i.test(code)) {
    // Full document: the CSP meta must be parsed BEFORE any executable node, or
    // a <script> the artifact placed before its own <head> runs pre-CSP and can
    // still hit the network (R41-D4). Anchor on <html> FIRST: inject our own
    // <head> carrying the CSP immediately after the <html ...> tag, so the CSP
    // precedes anything between <html> and the artifact's own <head>. Only fall
    // back to splicing an existing <head> (no <html> tag) or prepending.
    if (/<html[^>]*>/i.test(code)) return code.replace(/<html([^>]*)>/i, "<html$1><head>" + csp + "</head>");
    if (/<head[\s>]/i.test(code)) return code.replace(/<head([^>]*)>/i, "<head$1>" + csp);
    return csp + code;
  }
  // A fragment: wrap it in a minimal document.
  return "<!doctype html><html><head><meta charset=\"utf-8\">" + csp + "</head><body>"
    + code + "</body></html>";
}

/* Whether this caller is offered the canvas button. Starts true so the default
 * install renders it before /api/capabilities lands; the server's answer
 * overwrites it and refreshPreviewButtons() retracts buttons already drawn. */
let _previewAllowed = true;

/** Record whether the artifact canvas is offered to this caller. */
export function setPreviewAllowed(allowed) {
  _previewAllowed = allowed !== false;
}

/** Whether the artifact canvas is currently offered to this caller. */
export function isPreviewAllowed() {
  return _previewAllowed;
}

/** Add or remove *pre*'s canvas button for the current preview permission. Only
 *  ever touches the canvas button, never the copy button: a <pre> with no <code>
 *  child (a job-log pane) is left completely alone. */
function applyCanvasButton(pre) {
  const codeEl = pre.querySelector("code");
  const lang = _previewAllowed ? artifactLang(codeEl) : null;
  const existing = pre.querySelector(".canvas-btn");
  if (!lang) {
    if (existing) existing.remove();
    return;
  }
  if (existing) return;
  const cbtn = document.createElement("button");
  cbtn.className = "canvas-btn";
  cbtn.textContent = t("common.codeBlock.canvas");
  cbtn.title = t("common.codeBlock.canvasTitle", { lang: lang.toUpperCase() });
  cbtn.onclick = () => openArtifact(codeEl?.innerText || codeEl?.textContent || "", lang);
  pre.appendChild(cbtn);
}

/** Re-apply the current preview permission to every rendered code block under
 *  *root* (default: the document): adds a missing canvas button when allowed,
 *  removes one when not. Closes an open pane when no longer allowed. */
export function refreshPreviewButtons(root) {
  (root || document).querySelectorAll("pre").forEach(applyCanvasButton);
  if (!_previewAllowed) closeArtifact();
}

/** Open the artifact pane and render *code* in the hard-sandboxed iframe. */
export function openArtifact(code, lang) {
  if (!_previewAllowed) return;
  const pane = document.getElementById("artifact-pane");
  if (!pane) return;
  const body = pane.querySelector(".artifact-body");
  body.replaceChildren();
  const frame = document.createElement("iframe");
  frame.className = "artifact-frame";
  frame.setAttribute("sandbox", "allow-scripts");   // NO allow-same-origin
  frame.setAttribute("referrerpolicy", "no-referrer");
  frame.srcdoc = artifactSrcdoc(code, lang);
  body.appendChild(frame);
  const title = pane.querySelector(".artifact-title");
  if (title) title.textContent = t("common.artifact.title", { lang });
  pane.hidden = false;
  // Wire the controls lazily (idempotent): the GUI init does not run under the
  // test harness, and this keeps them working regardless of load order.
  const closeBtn = pane.querySelector("#artifact-close");
  if (closeBtn) closeBtn.onclick = closeArtifact;
  const refreshBtn = pane.querySelector("#artifact-refresh");
  if (refreshBtn) refreshBtn.onclick = () => openArtifact(code, lang);
}

/** Close the artifact pane and tear down the iframe (stops any running script). */
export function closeArtifact() {
  const pane = document.getElementById("artifact-pane");
  if (!pane) return;
  pane.hidden = true;
  const body = pane.querySelector(".artifact-body");
  if (body) body.replaceChildren();
}

/** Add the copy button (and, for a renderable block, an "open canvas" button)
 *  to a <pre>. Idempotent. */
export function enhanceCodeBlock(pre) {
  if (!pre.querySelector(".copy-btn")) {
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = t("common.codeBlock.copy");
    btn.onclick = () => {
      navigator.clipboard.writeText(pre.querySelector("code")?.innerText || pre.innerText);
      btn.textContent = t("common.codeBlock.copied");
      setTimeout(() => (btn.textContent = t("common.codeBlock.copy")), 1200);
    };
    pre.appendChild(btn);
  }
  applyCanvasButton(pre);
}

/** Create an element with class and (safe) text content. */
export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Matches settings_schema.py's _AVATAR_DATA_URI_RE (the mime alternation),
 * plus a base64-alphabet check on the payload the server does not need
 * (config.py never renders the value into markup). Returns the value
 * REBUILT from the captured mime and payload groups, never the original
 * string, so nothing reaches an <img src> that was not read back out of this
 * exact character set. Returns null for anything else - a glyph, an empty
 * value, or a near-miss like "data:text/html,...". See
 * test_safeAvatarImageSrc_rebuilds_from_validated_groups_or_returns_null. */
const AVATAR_DATA_URI_RE = /^data:image\/(png|jpeg|gif|webp);base64,([A-Za-z0-9+/]*={0,2})$/i;
export function safeAvatarImageSrc(value) {
  const m = typeof value === "string" ? value.match(AVATAR_DATA_URI_RE) : null;
  return m ? "data:image/" + m[1].toLowerCase() + ";base64," + m[2] : null;
}

/** Reads an image file client-side and resolves a data:image/png;base64,...
 * URI downscaled so its longest edge is at most maxSize - kept well under
 * settings_schema.py's _AVATAR_MAX_DATA_URI_LEN regardless of the source
 * file's size, and read entirely in-browser (no server round trip). */
export function fileToAvatarDataUri(file, maxSize = 128) {
  return new Promise((resolve, reject) => {
    if (!file.type || !file.type.startsWith("image/")) {
      reject(new Error(t("common.imageFile.chooseImage")));
      return;
    }
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(t("common.imageFile.readFailed")));
    reader.onload = () => {
      const img = new Image();
      img.onerror = () => reject(new Error(t("common.imageFile.decodeFailed")));
      img.onload = () => {
        const scale = Math.min(1, maxSize / Math.max(img.width, img.height));
        const w = Math.max(1, Math.round(img.width * scale));
        const h = Math.max(1, Math.round(img.height * scale));
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        canvas.getContext("2d").drawImage(img, 0, 0, w, h);
        resolve(canvas.toDataURL("image/png"));
      };
      img.src = reader.result;
    };
    reader.readAsDataURL(file);
  });
}

/** Reads an image file client-side and resolves a data:image/jpeg;base64,...
 * URI downscaled so its longest edge is at most maxSize, encoded as JPEG
 * (not PNG - a downscaled photo is far smaller as JPEG at this resolution)
 * at the given quality. Kept well under settings_schema.py's
 * _BACKGROUND_MAX_DATA_URI_LEN regardless of the source file's size, and read
 * entirely in-browser (no server round trip). Same shape as
 * fileToAvatarDataUri, kept separate rather than parameterized: the two
 * pickers want different output formats and default sizes. */
export function fileToBackgroundDataUri(file, maxSize = 1920, quality = 0.85) {
  return new Promise((resolve, reject) => {
    if (!file.type || !file.type.startsWith("image/")) {
      reject(new Error(t("common.imageFile.chooseImage")));
      return;
    }
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(t("common.imageFile.readFailed")));
    reader.onload = () => {
      const img = new Image();
      img.onerror = () => reject(new Error(t("common.imageFile.decodeFailed")));
      img.onload = () => {
        const scale = Math.min(1, maxSize / Math.max(img.width, img.height));
        const w = Math.max(1, Math.round(img.width * scale));
        const h = Math.max(1, Math.round(img.height * scale));
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        canvas.getContext("2d").drawImage(img, 0, 0, w, h);
        resolve(canvas.toDataURL("image/jpeg", quality));
      };
      img.src = reader.result;
    };
    reader.readAsDataURL(file);
  });
}

/** Apply (or clear) the chat wallpaper: sets the --chat-bg-image custom
 * property #chat-messages reads (style.css). value is "" or a data:image/...
 * URI; anything safeAvatarImageSrc will not rebuild (an unset field, or a
 * near-miss value) clears the wallpaper rather than reaching a background-image
 * url(). */
export function applyChatBackground(value) {
  const src = safeAvatarImageSrc(value);
  document.documentElement.style.setProperty("--chat-bg-image", src ? `url("${src}")` : "none");
}

export function autoGrow(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 220) + "px";
}

/* ---------------- the "Advanced" disclosure (.adv-fold) ---------------- */
/* Chat's #params drawer and the three Studio forms fold their rarely-touched
   knobs into a <details class="adv-fold">. Folding creates one hazard that did
   not exist while everything was visible: a value can be SET and UNSEEN. These
   two helpers are the whole answer to it. */

/** Count the fields inside `fold` that currently hold a value. A checkbox counts
 *  when checked; everything else when its trimmed value is non-empty, so a
 *  <select> resting on its blank "None" option does not count. */
function foldFilledCount(fold) {
  let n = 0;
  for (const f of fold.querySelectorAll("input, select, textarea")) {
    const set = (f.type === "checkbox" || f.type === "radio")
      ? f.checked : String(f.value ?? "").trim() !== "";
    if (set) n++;
  }
  return n;
}

/** Refresh one fold's "n set" badge and return that count. A collapsed fold
 *  holding live overrides would otherwise read as empty, which is the hidden
 *  problem AGENTS.md rule 5 forbids; the badge says so without opening it. */
export function updateAdvancedCount(fold) {
  const n = foldFilledCount(fold);
  const badge = fold.querySelector("summary .adv-fold-count");
  if (badge) {
    badge.textContent = n ? t("common.advancedFold.setCount", { n }) : "";
    badge.hidden = !n;
  }
  return n;
}

/** Open every `.adv-fold` under `root` that holds a value, and refresh all their
 *  badges. Returns how many folds were opened.
 *
 *  CALL THIS AFTER ANY PROGRAMMATIC WRITE INTO FORM FIELDS. Two flows restore
 *  values into fields that now live behind a fold - "reuse settings" on an image
 *  history entry, and applyPersona (also reached by /persona) - and both then
 *  report success. Without this they would report "Settings restored" while most
 *  of what they restored sits behind a closed triangle.
 *
 *  It keys on the VALUE rather than on a list of ids on purpose: a writer that
 *  later learns a new field is covered automatically, and a writer that set
 *  nothing (a persona carrying only a temperature, a history entry whose
 *  advanced values are all null) correctly leaves the fold shut. */
export function revealFilledAdvanced(root) {
  let opened = 0;
  for (const fold of (root || document).querySelectorAll("details.adv-fold")) {
    if (updateAdvancedCount(fold) && !fold.open) { fold.open = true; opened++; }
  }
  return opened;
}

// Typing into a folded field updates its badge, so collapsing afterwards still
// tells the truth. Delegated from the document rather than bound per fold: the
// folds are static markup in index.html, and this way a fold added later needs
// no wiring. Guarded because helpers.js is also loaded in non-DOM contexts.
if (typeof document !== "undefined" && document.addEventListener) {
  const onFieldEdit = (e) => {
    const fold = e.target && e.target.closest && e.target.closest("details.adv-fold");
    if (fold) updateAdvancedCount(fold);
  };
  document.addEventListener("input", onFieldEdit);
  document.addEventListener("change", onFieldEdit);
}

export function nearBottom(elm) {
  return elm.scrollHeight - elm.scrollTop - elm.clientHeight < 80;
}

/** Parse an SSE byte stream from fetch(), invoking onData per `data:` payload.
 *  onAnyFrame, when given, fires once per parsed frame (any blank-line-terminated
 *  block) regardless of whether it carries a `data:` payload - so a bare comment
 *  such as a keepalive still reaches it. */
export async function readSSE(response, onData, onAnyFrame) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      if (onAnyFrame) onAnyFrame();
      for (const line of frame.split("\n")) {
        if (line.startsWith("data: ")) onData(line.slice(6));
      }
    }
  }
}

/** Stream a background job's events. onLine gets text lines; the optional
 *  onProgress gets {downloaded,total,pct,phase} events. Resolves with end. */
// Ask the server to cancel a running job (model pull, media generation). The
// job's worker stops cooperatively (media gen interrupts ComfyUI mid-render),
// so streamJob's "end" event arrives with status "cancelled".
export async function cancelJob(jobId) {
  try {
    await fetch(`/api/jobs/${jobId}/cancel`,
                { method: "POST", headers: authHeaders() });
  } catch (e) { /* best-effort - the stream will still end */ }
}

// Reconnect tuning for streamJob, below - overridable by a test so it does not
// have to wait out the real delay. Same 1500ms shape as coder.js's
// streamSession, the proven precedent for this exact reconnect pattern.
export let JOB_RECONNECT_DELAY_MS = 1500;
export let JOB_RECONNECT_MAX_ATTEMPTS = 20;   // ~30s of reconnect budget

export async function streamJob(jobId, onLine, onProgress) {
  // GET /api/jobs/{id}/events can end WITHOUT an "end" frame (a network
  // blip, laptop sleep/wake, a backgrounded tab) while the job keeps running
  // server-side regardless of whether anyone is subscribed (JobManager.Job.
  // push() fans out to whoever is listening; it does not know or care).
  // Verified live 2026-08-05: aborting the SSE connection ~5ms after opening
  // it produced exactly this - the reader either threw (AbortError) or
  // resolved {done:true} with no "end" event - while the job went on to
  // finish successfully several seconds later. The OLD code here returned
  // {status:"failed"} (or let the exception propagate to the caller's own
  // try/catch, which renders "Pull failed: <message>") for BOTH shapes,
  // telling the user their model download failed when it had not - a
  // transport-level disconnect rendered as an application-level outcome.
  //
  // Reconnect instead, matching coder.js's streamSession (the proven shape
  // for exactly this problem: 1500ms backoff, retry until the operation is
  // confirmed over). GET /api/jobs/{id}/events has no "since"/replay=false
  // mode (unlike the coder session route) - job.subscribe() always replays
  // the FULL history - so `seen` tracks how many events this call has
  // already delivered to onLine/onProgress and skips that many on every
  // reconnect, or a resumed stream would double-print every line already
  // shown. Only a genuinely exhausted retry budget, or a 404 (the job is
  // provably gone, not just unreachable), returns a distinct "disconnected"
  // status - never "failed", which must stay reserved for the job's OWN
  // end event saying so.
  let seen = 0;
  for (let attempt = 0; attempt < JOB_RECONNECT_MAX_ATTEMPTS; attempt++) {
    let endEvent = null;
    try {
      const r = await fetch(`/api/jobs/${jobId}/events`, { headers: authHeaders() });
      if (r.status === 404) return { status: "disconnected", reason: "job not found" };
      if (!r.ok) throw new Error(r.statusText);
      let indexThisAttempt = 0;
      await readSSE(r, (payload) => {
        let ev;
        try { ev = JSON.parse(payload); } catch { return; }
        const isNew = indexThisAttempt >= seen;
        indexThisAttempt++;
        if (isNew) seen++;
        if (!isNew) return;
        if (ev.type === "line" && onLine) onLine(ev.text);
        if (ev.type === "progress" && onProgress) onProgress(ev);
        if (ev.type === "end") endEvent = ev;
      });
      if (endEvent) return endEvent;
      // Stream ended with no "end" frame - lost connection, not a job
      // outcome. Fall through to retry below.
    } catch (e) {
      // Thrown network/abort error - the other shape of the same lost
      // connection. Same treatment: retry rather than claim failure.
    }
    if (attempt < JOB_RECONNECT_MAX_ATTEMPTS - 1) {
      await new Promise((res) => setTimeout(res, JOB_RECONNECT_DELAY_MS));
    }
  }
  return { status: "disconnected" };
}

/** A streamJob() end status, worded for dropping into a "<Operation> " +
 *  jobStatusWord(status) style message (images.js/music.js/video.js/
 *  knowledge.js/slash.js all build their non-"done" message this way).
 *  "cancelled"/"failed" already read naturally as the bare status word;
 *  "disconnected" (streamJob gave up reconnecting, or the job was already
 *  gone - never a job OUTCOME) does not - "Generation disconnected" reads
 *  as jargon and invites the same "so did it fail?" question the status is
 *  trying to avoid. "interrupted" drops into the same sentence shapes
 *  without claiming an outcome that was never observed. */
export function jobStatusWord(status) {
  if (status === "disconnected") return "interrupted";
  return status;
}

// Sizes are shown in binary units (GiB/MiB/KiB) but labelled GB/MB/KB - the
// GPU/LLM convention: matches the VRAM printed on the card, llama.cpp's logs,
// and HuggingFace quant tables. (The driver reports e.g. 16 GiB; showing
// decimal GB would read a confusing 17.2 for the same card.)
export const GIB = 1024 ** 3, MIB = 1024 ** 2, KIB = 1024;

export function fmtBytes(n) {
  if (n == null) return "";
  if (n >= GIB) return (n / GIB).toFixed(2) + " GB";
  if (n >= MIB) return (n / MIB).toFixed(1) + " MB";
  if (n >= KIB) return (n / KIB).toFixed(0) + " KB";
  return n + " B";
}

/** Smoothed download rate + ETA from a rolling window of {t, downloaded}
 *  samples (ms timestamps, oldest first). Averaging over the whole window
 *  (first..last) instead of the last chunk damps per-chunk jitter so the
 *  readout does not flicker. Returns {bytesPerSec, etaSec}; either is null when
 *  it cannot be computed - needs >=2 samples, a positive time span, and forward
 *  progress; etaSec also needs a known total >= the bytes so far. */
export function downloadRate(samples, total) {
  const out = { bytesPerSec: null, etaSec: null };
  if (!samples || samples.length < 2) return out;
  const first = samples[0], last = samples[samples.length - 1];
  const dt = (last.t - first.t) / 1000;
  const db = last.downloaded - first.downloaded;
  if (dt <= 0 || db <= 0) return out;
  out.bytesPerSec = db / dt;
  if (total != null && total >= last.downloaded) {
    out.etaSec = (total - last.downloaded) / out.bytesPerSec;
  }
  return out;
}

/** Seconds -> a compact ETA string: "1h 01m" / "1m 30s" / "45s". Empty for a
 *  missing / non-finite / negative value. */
export function fmtDuration(sec) {
  if (sec == null || !isFinite(sec) || sec < 0) return "";
  sec = Math.round(sec);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

/** Fetch an auth-protected image into an object URL. */
export async function fetchImageURL(path) {
  const r = await fetch(path, { headers: authHeaders() });
  if (!r.ok) throw new Error(r.statusText);
  return URL.createObjectURL(await r.blob());
}

/* ---- modal ---- */

export function openModal(title, bodyBuilder) {
  $("modal-title").textContent = title;
  const body = $("modal-body");
  body.innerHTML = "";
  bodyBuilder(body);
  $("modal").style.display = "flex";
}
$("modal-close").onclick = () => ($("modal").style.display = "none");
$("modal").onclick = (e) => { if (e.target === $("modal")) $("modal").style.display = "none"; };

/** Confirm a destructive action with the in-page modal. window.confirm() is
 *  suppressed in some mobile / PWA browsers (the NET-1 prompt() class of bug),
 *  so we render our own Cancel / <confirm> dialog. */
export function confirmDanger(title, message, confirmLabel, onConfirm) {
  openModal(title, (body) => {
    body.appendChild(el("p", "", message));
    const row = el("div", "actions");
    const cancel = el("button", "btn-secondary", t("common.modal.cancel"));
    cancel.onclick = () => ($("modal").style.display = "none");
    const ok = el("button", "btn-secondary btn-danger", confirmLabel);
    ok.onclick = () => { $("modal").style.display = "none"; onConfirm(); };
    row.appendChild(cancel);
    row.appendChild(ok);
    body.appendChild(row);
  });
}

/** In-page text-input equivalent of confirmDanger, for the free-text half of
 *  the same NET-1 class: window.prompt() is suppressed in the same mobile/PWA
 *  browsers confirmDanger's own comment names, so a raw prompt() call goes
 *  silent with no error and no toast (indistinguishable from the user
 *  cancelling). Resolves to the entered text (untrimmed, exactly like
 *  prompt()'s own return value - callers already trim/validate the same way
 *  they did with prompt()), or null if cancelled. Cancelling and submitting an
 *  emptied field resolve differently on purpose: some callers (e.g. the
 *  conversation folder prompt) treat an empty submit as "clear the value" but
 *  a cancel as "leave it alone". */
export function promptText(title, defaultValue) {
  return new Promise((resolve) => {
    let settled = false;
    let watch = null;
    let input;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      if (watch) clearInterval(watch);
      $("modal").style.display = "none";
      resolve(value);
    };
    openModal(title, (body) => {
      input = el("input");
      input.type = "text";
      input.value = defaultValue || "";
      body.appendChild(input);
      const row = el("div", "actions");
      const cancel = el("button", "btn-secondary", t("common.modal.cancel"));
      cancel.onclick = () => finish(null);
      const ok = el("button", "btn-secondary", t("common.modal.ok"));
      ok.onclick = () => finish(input.value);
      row.appendChild(cancel);
      row.appendChild(ok);
      body.appendChild(row);
      input.onkeydown = (e) => {
        if (e.key === "Enter") { e.preventDefault(); finish(input.value); }
        else if (e.key === "Escape") { e.preventDefault(); finish(null); }
      };
    });
    input.focus();
    input.select();
    // Dismissing via the shared modal chrome (x / backdrop) sets display:none;
    // poll for it and treat as cancel - those handlers are not ours (same
    // pattern _offerModelDownload's missing-model modal uses below, for the
    // same reason).
    watch = setInterval(() => {
      if ($("modal").style.display === "none") finish(null);
    }, 200);
  });
}

/** Offer to download ONE curated missing model (repo/file/size shown in full,
 *  a real Download button - never a silent auto-pull). Resolves true whether
 *  the user downloads or skips (either way the caller proceeds to its real
 *  preflight-gated generate call, which is the authoritative check); resolves
 *  false only if the download itself failed after the user asked for it, so
 *  the caller can decide whether to keep going.  *log*, when given, gets the
 *  download's streamed progress lines appended (same log panel the page
 *  already uses for the generation job itself).  *plugin* ("image"/"music"/
 *  "video"), when given, tells the server which plugin's own ComfyUI folder
 *  to download into (NEW-COMFY-DOWNLOAD-DEST-IGNORES-PLUGIN-WORKDIR) - without
 *  it the server falls back to the legacy shared comfy_workdir only. */
function _offerModelDownload(missingModel, log, plugin) {
  const { filename, source } = missingModel;
  return new Promise((resolve) => {
    let settled = false;
    let watch = null;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      if (watch) clearInterval(watch);
      $("modal").style.display = "none";
      resolve(value);
    };
    openModal(t("common.modelDownload.title", { filename }), (body) => {
      body.appendChild(el("p", "",
        t("common.modelDownload.body", { filename, size: fmtBytes(source.size_bytes) })));
      body.appendChild(el("p", "",
        t("common.modelDownload.source", { repo: source.repo, file: source.file })));
      const row = el("div", "actions");
      const skip = el("button", "btn-secondary", t("common.modelDownload.notNow"));
      skip.onclick = () => finish(true);
      const dl = el("button", "btn-secondary", t("common.modelDownload.download"));
      dl.onclick = async () => {
        dl.disabled = true; skip.disabled = true; dl.textContent = t("common.modelDownload.starting");
        try {
          const r = await fetch("/api/models/pull-comfy-source", {
            method: "POST", headers: authHeaders(),
            body: JSON.stringify({ filename, plugin: plugin || null }),
          });
          const data = await r.json();
          if (!r.ok) throw new Error(data.detail || r.statusText);
          if (watch) clearInterval(watch);
          $("modal").style.display = "none";
          if (log) {
            log.style.display = "block";
            log.textContent += t("common.modelDownload.progress", { filename, repo: source.repo }) + "\n";
          }
          const end = await streamJob(data.job_id, (line) => {
            if (log) { log.textContent += line + "\n"; log.scrollTop = log.scrollHeight; }
          });
          if (end.status !== "done") toast(`Download ${end.status}: ${filename}`, true);
          resolve(true);
        } catch (e) {
          toast(t("common.modelDownload.failed", { message: e.message }), true);
          resolve(false);
        }
      };
      row.appendChild(skip);
      row.appendChild(dl);
      body.appendChild(row);
    });
    // Dismissing via the shared modal chrome (x / backdrop) sets display:none;
    // poll for it and treat as "not now" - those handlers are not ours (same
    // pattern as picker.js's pickPath, for the same reason).
    watch = setInterval(() => {
      if ($("modal").style.display === "none") finish(true);
    }, 200);
  });
}

/** Report ONE missing model that has NO curated download source: an honest,
 *  distinct state instead of vanishing silently behind checkModelsBeforeGenerate's
 *  curated-only filter. Generic over class_type/input_name - not LoRA-specific -
 *  so the same message covers any future non-curated model type (a checkpoint or
 *  VAE outside the pinned few also hits this, not just a LoRA). Never blocks:
 *  the real generate call's own preflight_models() gate remains authoritative. */
function _reportUncuratedMiss(missingModel, log) {
  const { filename, class_type, input_name } = missingModel;
  const msg = t("common.modelDownload.uncuratedMissing", { filename, class_type, input_name });
  toast(msg, true);
  if (log) {
    log.style.display = "block";
    log.textContent += msg + "\n";
  }
}

/** Pre-generate model-existence check: calls the read-only preflight endpoint
 *  for *kind* ("image" | "video" | "music"). A missing model WITH a curated
 *  download source is offered via _offerModelDownload; one WITHOUT gets an
 *  honest _reportUncuratedMiss instead of disappearing - the user learns what's
 *  missing and where to put it before submitting, not only from the real
 *  generate call's later preflight_models() failure. Always resolves true
 *  (proceed) - neither path blocks generation on its own account.
 *  Best-effort: any failure to reach the pre-check itself also resolves true. */
export async function checkModelsBeforeGenerate(kind, log, overrides = {}) {
  let data;
  try {
    const r = await fetch(`/api/media/${kind}/preflight`, {
      method: "POST", headers: authHeaders(), body: JSON.stringify(overrides),
    });
    if (!r.ok) return true;
    data = await r.json();
  } catch (e) {
    return true;
  }
  const missing = (data && data.missing) || [];
  for (const m of missing) {
    if (m.source) await _offerModelDownload(m, log, kind);
    else _reportUncuratedMiss(m, log);
  }
  return true;
}

