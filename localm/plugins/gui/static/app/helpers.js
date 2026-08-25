// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - shared helpers. */
"use strict";

export const $ = (id) => document.getElementById(id);

// Read a JSON value from localStorage. Returns `fallback` when the key is
// absent, unreadable, or holds malformed JSON, warning on corruption.
export function readStoredJSON(key, fallback) {
  let raw;
  try { raw = localStorage.getItem(key); }
  catch (e) { console.warn(`localm: localStorage unavailable for "${key}":`, e); return fallback; }
  if (raw === null) return fallback;                 // absent
  try {
    return JSON.parse(raw);
  } catch (e) {
    console.warn(`localm: ignoring corrupt localStorage["${key}"] (kept a blank default):`, e);
    return fallback;
  }
}

// localStorage keys that belong to the connected backend. All are wiped
// together when the backend's instance id (served on /v1/config) does not match
// this origin's last-confirmed one, and again by the privacy-mode wipe in
// chat.js's refreshCtxLimit. Device/browser preferences (localm.theme,
// localm.logoStyle, the TTS voice picks) are not listed here.
//
// Every key written through chat.js's lsSetScoped(key, value) must appear in
// this list; lsSetScoped warns when it does not.
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
  "localm.backendHintDismissed",
];

const INSTANCE_ID_KEY = "localm.instanceId";

/** True when this browser origin already confirmed the connected backend's id on
 *  an earlier /v1/config round trip. Gates every instance-scoped localStorage
 *  read that runs at boot before this page load's own round trip resolves. */
export function instanceCacheTrusted() {
  try { return !!localStorage.getItem(INSTANCE_ID_KEY); }
  catch (e) { return false; }
}

/** Reconcile the cached instance id against the one the connected backend just
 *  reported (cfg.instance_id from /v1/config). Returns one of three states,
 *  which callers must keep distinct rather than collapsing into a boolean:
 *   - "confirmed": the cached id matches this backend - safe to render, merge
 *     and upload.
 *   - "mismatched": the cache belonged to a different backend, or had never
 *     been confirmed for this origin - every instance-scoped key is wiped
 *     before returning.
 *   - "unknown": a missing/falsy *serverInstanceId* or an unreadable
 *     localStorage leaves nothing to compare against. Existing rendering is
 *     preserved, but callers must not treat this as a confirmed match for
 *     anything that writes data back to the backend. */
export function reconcileInstanceId(serverInstanceId) {
  if (!serverInstanceId) return "unknown";
  let cached;
  try { cached = localStorage.getItem(INSTANCE_ID_KEY); }
  catch (e) { return "unknown"; }   // localStorage unavailable
  if (cached === serverInstanceId) return "confirmed";
  for (const key of INSTANCE_SCOPED_KEYS) {
    try { localStorage.removeItem(key); } catch (e) { /* best-effort wipe */ }
  }
  try { localStorage.setItem(INSTANCE_ID_KEY, serverInstanceId); }
  catch (e) { /* storage full or blocked */ }
  return "mismatched";
}

// Open mode's credential: the per-process shell token, sent as a bearer header.
// Protected mode instead uses the HttpOnly session cookie plus the CSRF token
// authHeaders() reads from window.__LOCALM_CSRF__ (fetched from GET /api/session).
export const SHELL_TOKEN = window.__LOCALM_SHELL_TOKEN__ || "";

export function readCookie(name) {
  const m = document.cookie.match(new RegExp("(?:^|; )" + name + "=([^;]*)"));
  if (!m) return "";
  // Decode best-effort: malformed percent-encoding falls back to the raw value.
  try { return decodeURIComponent(m[1]); }
  catch (e) { return m[1]; }
}

export function authHeaders(extra = {}) {
  const h = { "Content-Type": "application/json", ...extra };
  const csrf = window.__LOCALM_CSRF__ || "";
  if (csrf) {
    // Session (cookie) mode: the auto-sent HttpOnly session cookie plus this
    // CSRF header, and no bearer - the Authorization header would win over the
    // cookie server-side.
    h["X-CSRF-Token"] = csrf;
  } else if (SHELL_TOKEN) {
    // Open mode: the per-process loopback shell token authorises local management.
    h["Authorization"] = "Bearer " + SHELL_TOKEN;
  }
  return h;
}

/** True when *headers* carries our own open-mode shell token as its credential:
 *  an Authorization header whose value equals this process's shell bearer. */
export function sentShellToken(headers) {
  if (!SHELL_TOKEN || !headers) return false;
  const auth = headers instanceof Headers
    ? headers.get("Authorization")
    : (typeof headers === "object" ? headers["Authorization"] : null);
  return auth === "Bearer " + SHELL_TOKEN;
}

// Fetch the CSRF token for the current session and stash it for authHeaders().
// Called at boot and by the 403-CSRF self-heal. Returns the token or "".
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

/** Replace a raw tool-call block with a compact note. Runs on the display text
 *  and on the assistant history re-sent to the model. Matches every dialect
 *  `parseWebCall` executes (`<tool_call>`, the |-piped `<|tool_call|>` wrappers,
 *  the Gemma `call:{...}` prefix); name/query use tolerant regexes so
 *  single-quoted or trailing-comma inner JSON still matches. */
export function formatToolCalls(text) {
  return (text || "").replace(
    /<\|?\/?tool_call\|?>[\s\S]*?<\|?\/?tool_call\|?>/g,
    (block) => {
      const name = (block.match(/"name"\s*:\s*"(\w+)"/) || [])[1] || "";
      const arg = (block.match(/"(?:query|url)"\s*:\s*"([^"]*)"/) || [])[1] || "";
      const what =
        name === "web_search" ? `web search: "${arg}"` :
        name === "fetch_url"  ? `read page: ${arg || ""}` :
        arg ? `web request: ${arg}` : "web request";
      return `\n> 🌐 *${what}*\n`;
    });
}

/** Strip model-internal control markers from *text*, mirroring the server-side
 *  normalisation in localm/inference/textnorm.py. Runs on the full accumulated
 *  text. */
export function scrubMarkers(text) {
  return (text || "")
    .replace(/<\|"\|>/g, '"')
    .replace(/<\|?\s*channel\s*\|?>(thought|thinking|analysis|reasoning|commentary|reflection)\n?(<\|?\s*message\s*\|?>)?/g, "<think>\n")
    .replace(/<\s*channel\s*\|>|<\|?\s*channel\s*\|?>final\n?(<\|?\s*message\s*\|?>)?/g, "\n</think>\n")
    .replace(/<\|?\s*channel\s*\|?>|<\s*channel\s*\|>|<\|?\s*message\s*\|?>|<\|start\|>(assistant|user|system)?|<\|return\|>|<\|turn>(user|model|assistant|system)?\n?|<turn\|>|<\|tool>|<tool\|>|<\|think\|>|<think\|>|<unused\d+>?/g, "");
}

/** Point every remote <img> in a rendered reply at localm's own image proxy
 *  (/api/image-proxy), which fetches server-side. Runs after sanitisation and
 *  only ever replaces a src attribute with a same-origin URL - it inserts no
 *  markup. data:, blob: and relative/same-origin sources are left alone.
 *  Unconditional: the route itself 403s while the feature is off. */
/** remote href -> blob: URL once fetched, or the in-flight Promise for it.
 *
 *  Keyed on the URL rather than on the element: renderMarkdown reassigns
 *  innerHTML on every streamed chunk, so the <img> is a new element each time
 *  and any per-element flag is destroyed with its predecessor. */
const _imgProxyCache = new Map();
const _IMG_PROXY_CACHE_MAX = 64;

/** Drop every cached proxied image and release its object URL. Must be called
 *  whenever the remote-image setting may have changed, or cached blobs keep
 *  rendering for the rest of the page session. */
export function clearImageProxyCache() {
  for (const v of _imgProxyCache.values()) {
    if (typeof v === "string") URL.revokeObjectURL(v);
  }
  _imgProxyCache.clear();
}
window.clearImageProxyCache = clearImageProxyCache;

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

function proxyRemoteImages(root) {
  // srcset first: a remote srcset wins over src, so drop it on <img> and on any
  // <source> inside a <picture>, leaving the proxied src as the single source.
  root.querySelectorAll("img[srcset], source[srcset]").forEach((node) => {
    const set = node.getAttribute("srcset") || "";
    if (/(^|[\s,])https?:\/\//i.test(set)) node.removeAttribute("srcset");
  });
  root.querySelectorAll("img[src]").forEach((img) => {
    const raw = img.getAttribute("src") || "";
    if (!/^https?:\/\//i.test(raw)) return;          // data:/blob:/relative
    let u;
    try { u = new URL(raw, window.location.href); } catch (e) { return; }
    if (u.origin === window.location.origin) return; // same origin
    img.dataset.lmProxySrc = u.href;                 // what the model asked for
    // Drop the remote src so no broken load stays pending.
    img.removeAttribute("src");

    const cached = _imgProxyCache.get(u.href);
    if (typeof cached === "string") { img.src = cached; return; }
    if (cached) { cached.then((o) => { if (o) img.src = o; }); return; }  // in flight

    // fetch(), not a bare src=: /api/ needs the shell token as a bearer header
    // in open mode, and an <img> element cannot send one.
    const pending = fetch("/api/image-proxy?url=" + encodeURIComponent(u.href),
                          { headers: authHeaders() })
      .then((r) => (r.ok ? r.blob() : Promise.reject(new Error("HTTP " + r.status))))
      .then((blob) => {
        const objUrl = URL.createObjectURL(blob);
        _rememberProxiedImage(u.href, objUrl);
        return objUrl;
      })
      .catch(() => {
        // Off (403), refused by the network policy, or the host is unreachable.
        // Forget it so a later render may retry, and leave the image blank.
        _imgProxyCache.delete(u.href);
        return null;
      });
    _imgProxyCache.set(u.href, pending);
    pending.then((o) => {
      if (o) img.src = o;
      else img.dataset.lmProxyFailed = "1";
    });
  });
}

/** True if the main reply body rendered to something visible. Whitespace-only
 *  content and empty code fences count as not visible; text, images, tables,
 *  rules and math source count as visible. Runs before KaTeX, so math is caught
 *  via its source rather than a rendered .katex node. */
function mainHasVisibleContent(main) {
  if ((main.textContent || "").trim() !== "") return true;
  return main.querySelector("img, svg, canvas, video, audio, iframe, table, hr, input") !== null;
}

export function renderMarkdown(target, text, opts = {}) {
  const { think, open, rest: rawRest } = splitThink(scrubMarkers(text));
  const rest = formatToolCalls(rawRest);

  // Think block: updated in place so a user toggle survives each streamed chunk.
  // Open while thinking, collapsed once done, until the user clicks it
  // (data-userset), after which their choice is left alone.
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
    det.querySelector("summary").textContent = open ? "Thinking…" : "Thoughts";
    det.querySelector("div").innerHTML = DOMPurify.sanitize(marked.parse(think));
    if (!det.dataset.userset) det.open = open;
  } else if (det) {
    det.remove();
  }

  // Main content lives in its own container, so refreshing it leaves the think
  // block above it untouched.
  let main = target.querySelector(".md-main");
  if (!main) {
    main = document.createElement("div");
    main.className = "md-main";
    target.appendChild(main);
  }
  main.innerHTML = DOMPurify.sanitize(marked.parse(rest || ""));
  // On `target`, not `main`, so the think block is covered too. Idempotent
  // across a streaming re-render: an already-proxied src is same-origin.
  proxyRemoteImages(target);
  // On a settled render (opts.final), a body that rendered to nothing visible
  // gets a plain note instead of an empty bubble.
  if (opts.final && !mainHasVisibleContent(main)) {
    main.replaceChildren(el("div", "md-empty", "(no reply text)"));
  }
  // LaTeX math: $...$, $$...$$, \(...\), \[...\]. Runs after sanitisation.
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
        // Pinned off: with trust enabled KaTeX's \htmlData / \href can emit raw
        // HTML and URLs.
        trust: false,
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
      });
    } catch (e) { /* malformed TeX mid-stream */ }
  }
  target.querySelectorAll("pre code").forEach((block) => {
    // Record the source language before hljs rewrites the class list.
    const m = (block.className || "").match(/language-([\w-]+)/);
    if (m && block.dataset) block.dataset.lang = m[1];
    try { hljs.highlightElement(block); } catch (e) { /* unknown lang */ }
  });
  target.querySelectorAll("pre").forEach(enhanceCodeBlock);
}

/* Artifacts canvas: a self-contained HTML/SVG reply block rendered in a side
 * pane inside an <iframe sandbox="allow-scripts"> (no allow-same-origin, so no
 * access to this app's origin, cookies or storage) whose srcdoc carries a CSP
 * that blocks all network. */

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
 *  does not already carry one. A srcdoc document inherits the embedding
 *  document's CSP, so without the nonce an artifact's inline <script> is blocked
 *  by the shell's `script-src 'self' 'nonce-X'`.
 *
 *  Rewrites the <script> open tag as a string rather than round-tripping through
 *  a parser, which would move nodes and break the ordering guarantee below. */
function stampArtifactNonce(code) {
  const n = window.__LOCALM_CSP_NONCE__;
  if (!n) return code;                       // no enforcing policy
  return String(code).replace(
    /<script\b(?![^>]*\bnonce=)([^>]*)>/gi,
    '<script nonce="' + n + '"$1>');
}

/** Build the iframe srcdoc for an artifact, injecting a strict CSP that blocks
 *  network access. Inline script/style are allowed (the artifact runs), data:
 *  images are allowed, everything else is denied. */
export function artifactSrcdoc(code, lang) {
  code = stampArtifactNonce(code);
  // form-action has no default-src fallback, so it is set explicitly: unset, it
  // would allow form submission to any origin.
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
    // Full document: the CSP meta must be parsed before any executable node.
    // Anchor on <html> first, injecting a <head> carrying the CSP immediately
    // after the <html ...> tag, so it precedes anything between <html> and the
    // artifact's own <head>. Otherwise splice an existing <head>, or prepend.
    if (/<html[^>]*>/i.test(code)) return code.replace(/<html([^>]*)>/i, "<html$1><head>" + csp + "</head>");
    if (/<head[\s>]/i.test(code)) return code.replace(/<head([^>]*)>/i, "<head$1>" + csp);
    return csp + code;
  }
  // A fragment: wrap it in a minimal document.
  return "<!doctype html><html><head><meta charset=\"utf-8\">" + csp + "</head><body>"
    + code + "</body></html>";
}

/** Open the artifact pane and render *code* in the hard-sandboxed iframe. */
export function openArtifact(code, lang) {
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
  if (title) title.textContent = "Artifact (" + lang + ")";
  pane.hidden = false;
  // Wire the controls lazily, independent of load order. Idempotent.
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
    btn.textContent = "copy";
    btn.onclick = () => {
      navigator.clipboard.writeText(pre.querySelector("code")?.innerText || pre.innerText);
      btn.textContent = "copied";
      setTimeout(() => (btn.textContent = "copy"), 1200);
    };
    pre.appendChild(btn);
  }
  const codeEl = pre.querySelector("code");
  const lang = artifactLang(codeEl);
  if (lang && !pre.querySelector(".canvas-btn")) {
    const cbtn = document.createElement("button");
    cbtn.className = "canvas-btn";
    cbtn.textContent = "canvas";
    cbtn.title = "Render this " + lang.toUpperCase() + " in a sandboxed canvas";
    cbtn.onclick = () => openArtifact(codeEl?.innerText || codeEl?.textContent || "", lang);
    pre.appendChild(cbtn);
  }
}

/** Create an element with class and (safe) text content. */
export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function autoGrow(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 220) + "px";
}

/* ---------------- the "Advanced" disclosure (.adv-fold) ---------------- */
/* Chat's #params drawer and the three Studio forms fold their rarely-touched
   knobs into a <details class="adv-fold">. */

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

/** Refresh one fold's "n set" badge and return that count, so a collapsed fold
 *  still shows how many values it holds. */
export function updateAdvancedCount(fold) {
  const n = foldFilledCount(fold);
  const badge = fold.querySelector("summary .adv-fold-count");
  if (badge) {
    badge.textContent = n ? n + " set" : "";
    badge.hidden = !n;
  }
  return n;
}

/** Open every `.adv-fold` under `root` that holds a value, and refresh all their
 *  badges. Returns how many folds were opened. Call after any programmatic write
 *  into form fields. Keys on the field values, not on a list of ids, so a fold
 *  whose fields were all left empty stays shut. */
export function revealFilledAdvanced(root) {
  let opened = 0;
  for (const fold of (root || document).querySelectorAll("details.adv-fold")) {
    if (updateAdvancedCount(fold) && !fold.open) { fold.open = true; opened++; }
  }
  return opened;
}

// Typing into a folded field updates its badge. Delegated from the document, so
// a fold added later needs no wiring. Guarded for non-DOM contexts.
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

/** Parse an SSE byte stream from fetch(), invoking onData per `data:` payload. */
export async function readSSE(response, onData) {
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
      for (const line of frame.split("\n")) {
        if (line.startsWith("data: ")) onData(line.slice(6));
      }
    }
  }
}

/** Stream a background job's events. onLine gets text lines; the optional
 *  onProgress gets {downloaded,total,pct,phase} events. Resolves with end. */
// Ask the server to cancel a running job (model pull, media generation). The
// worker stops cooperatively, so streamJob's "end" event arrives with status
// "cancelled".
export async function cancelJob(jobId) {
  try {
    await fetch(`/api/jobs/${jobId}/cancel`,
                { method: "POST", headers: authHeaders() });
  } catch (e) { /* best-effort - the stream will still end */ }
}

// Reconnect tuning for streamJob, below. Overridable.
export let JOB_RECONNECT_DELAY_MS = 1500;
export let JOB_RECONNECT_MAX_ATTEMPTS = 20;   // ~30s of reconnect budget

export async function streamJob(jobId, onLine, onProgress) {
  // GET /api/jobs/{id}/events can end without an "end" frame (a network blip,
  // sleep/wake, a backgrounded tab) while the job keeps running server-side.
  // Both shapes of that - a thrown error and a clean end with no "end" event -
  // reconnect rather than report an outcome.
  //
  // job.subscribe() always replays the full history, so `seen` counts how many
  // events this call has already delivered to onLine/onProgress and skips that
  // many on each reconnect. An exhausted retry budget, or a 404, returns
  // "disconnected"; "failed" only ever comes from the job's own end event.
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
      // Stream ended with no "end" frame. Fall through to retry below.
    } catch (e) {
      // Network or abort error. Retry as well.
    }
    if (attempt < JOB_RECONNECT_MAX_ATTEMPTS - 1) {
      await new Promise((res) => setTimeout(res, JOB_RECONNECT_DELAY_MS));
    }
  }
  return { status: "disconnected" };
}

/** A streamJob() end status, worded for a "<Operation> " + jobStatusWord(status)
 *  message. "cancelled" and "failed" pass through; "disconnected" becomes
 *  "interrupted". */
export function jobStatusWord(status) {
  if (status === "disconnected") return "interrupted";
  return status;
}

// Sizes are computed in binary units (GiB/MiB/KiB) and labelled GB/MB/KB.
export const GIB = 1024 ** 3, MIB = 1024 ** 2, KIB = 1024;

export function fmtBytes(n) {
  if (n == null) return "";
  if (n >= GIB) return (n / GIB).toFixed(2) + " GB";
  if (n >= MIB) return (n / MIB).toFixed(1) + " MB";
  if (n >= KIB) return (n / KIB).toFixed(0) + " KB";
  return n + " B";
}

/** Smoothed download rate + ETA from a rolling window of {t, downloaded}
 *  samples (ms timestamps, oldest first), averaged over the whole window.
 *  Returns {bytesPerSec, etaSec}; either is null when it cannot be computed -
 *  needs >=2 samples, a positive time span, and forward progress; etaSec also
 *  needs a known total >= the bytes so far. */
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

/** Confirm a destructive action with the in-page modal: a Cancel / <confirm>
 *  dialog rendered in the page rather than window.confirm(), which some mobile
 *  and PWA browsers suppress. */
export function confirmDanger(title, message, confirmLabel, onConfirm) {
  openModal(title, (body) => {
    body.appendChild(el("p", "", message));
    const row = el("div", "actions");
    const cancel = el("button", "btn-secondary", "Cancel");
    cancel.onclick = () => ($("modal").style.display = "none");
    const ok = el("button", "btn-secondary btn-danger", confirmLabel);
    ok.onclick = () => { $("modal").style.display = "none"; onConfirm(); };
    row.appendChild(cancel);
    row.appendChild(ok);
    body.appendChild(row);
  });
}

/** In-page text-input equivalent of confirmDanger, replacing window.prompt().
 *  Resolves to the entered text, untrimmed, or null if cancelled - cancelling
 *  and submitting an emptied field resolve differently, and callers rely on the
 *  distinction. */
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
      const cancel = el("button", "btn-secondary", "Cancel");
      cancel.onclick = () => finish(null);
      const ok = el("button", "btn-secondary", "OK");
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
    // poll for it and treat it as cancel.
    watch = setInterval(() => {
      if ($("modal").style.display === "none") finish(null);
    }, 200);
  });
}

/** Offer to download one curated missing model, showing repo, file and size
 *  behind a Download button. Resolves true whether the user downloads or skips,
 *  and false only when a requested download failed. *log*, when given, gets the
 *  download's streamed progress lines appended. *plugin* ("image"/"music"/
 *  "video"), when given, tells the server which plugin's own ComfyUI folder to
 *  download into; without it the server uses the shared comfy_workdir. */
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
    openModal(`Missing model: ${filename}`, (body) => {
      body.appendChild(el("p", "",
        `This workflow needs '${filename}' (${fmtBytes(source.size_bytes)}), `
        + "which isn't installed."));
      body.appendChild(el("p", "", `Source: ${source.repo} / ${source.file}`));
      const row = el("div", "actions");
      const skip = el("button", "btn-secondary", "Not now");
      skip.onclick = () => finish(true);
      const dl = el("button", "btn-secondary", "Download");
      dl.onclick = async () => {
        dl.disabled = true; skip.disabled = true; dl.textContent = "Starting…";
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
            log.textContent += `Downloading ${filename} from ${source.repo}…\n`;
          }
          const end = await streamJob(data.job_id, (line) => {
            if (log) { log.textContent += line + "\n"; log.scrollTop = log.scrollHeight; }
          });
          if (end.status !== "done") toast(`Download ${end.status}: ${filename}`, true);
          resolve(true);
        } catch (e) {
          toast("Download failed: " + e.message, true);
          resolve(false);
        }
      };
      row.appendChild(skip);
      row.appendChild(dl);
      body.appendChild(row);
    });
    // Dismissing via the shared modal chrome (x / backdrop) sets display:none;
    // poll for it and treat it as "not now".
    watch = setInterval(() => {
      if ($("modal").style.display === "none") finish(true);
    }, 200);
  });
}

/** Report one missing model that has no curated download source, naming the
 *  class_type/input_name that needs it. Does not block: the generate call's own
 *  preflight_models() gate remains authoritative. */
function _reportUncuratedMiss(missingModel, log) {
  const { filename, class_type, input_name } = missingModel;
  const msg = `'${filename}' is missing (needed by ${class_type}.${input_name}) - `
    + "localm has no automatic download source for it. Add it to your ComfyUI "
    + "installation's matching models folder, then try again.";
  toast(msg, true);
  if (log) {
    log.style.display = "block";
    log.textContent += msg + "\n";
  }
}

/** Pre-generate model-existence check: calls the read-only preflight endpoint
 *  for *kind* ("image" | "video" | "music"). A missing model with a curated
 *  download source is offered via _offerModelDownload; one without goes to
 *  _reportUncuratedMiss. Always resolves true (proceed), including when the
 *  pre-check itself cannot be reached. */
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

