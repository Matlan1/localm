// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - shared helpers (split from app.js). Classic script: it
   shares the one global lexical environment with the other app/* and
   pages/* scripts, so every cross-section reference resolves by bare
   name exactly as before. */
"use strict";

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
 *  reported (cfg.instance_id from /v1/config). Returns true when the
 *  instance-scoped cache is CONFIRMED to belong to THIS backend (safe to
 *  render/merge/upload); false when it was just wiped because it either
 *  belonged to a DIFFERENT backend or had never been confirmed for this origin
 *  before - a brand-new pairing, exactly the cross-instance leak scenario
 *  (AUD-INSTANCEID). A missing/falsy *serverInstanceId* (an older server that
 *  predates this field) is a no-op: there is nothing to compare against, so
 *  existing behaviour is preserved rather than wiping on missing information. */
export function reconcileInstanceId(serverInstanceId) {
  if (!serverInstanceId) return true;
  let cached;
  try { cached = localStorage.getItem(INSTANCE_ID_KEY); }
  catch (e) { return true; }   // localStorage unavailable - nothing to protect
  const matched = cached === serverInstanceId;
  if (!matched) {
    for (const key of INSTANCE_SCOPED_KEYS) {
      try { localStorage.removeItem(key); } catch (e) { /* best-effort wipe */ }
    }
    try { localStorage.setItem(INSTANCE_ID_KEY, serverInstanceId); }
    catch (e) { /* storage full/blocked - callers still correct in-memory state */ }
  }
  return matched;
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
        name === "web_search" ? `web search: "${arg}"` :
        name === "fetch_url"  ? `read page: ${arg || ""}` :
        arg ? `web request: ${arg}` : "web request";
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
    det.querySelector("summary").textContent = open ? "Thinking…" : "Thoughts";
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
  // Never leave a blank reply bubble. On a SETTLED render (opts.final - a reload
  // or post-stream renderChat, never a mid-stream shell) a body that rendered to
  // nothing visible gets a plain note instead of an empty box (real case: a 1B
  // model whose <think> works but whose answer is a bare empty ```code fence).
  // Gated on final so a slow model is never flashed a false "no reply" early.
  if (opts.final && !mainHasVisibleContent(main)) {
    main.replaceChildren(el("div", "md-empty", "(no reply text)"));
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

/** Build the iframe srcdoc for an artifact, injecting a strict CSP that blocks
 *  network access. Inline script/style are allowed (the artifact runs), data:
 *  images are allowed, everything else is denied. */
export function artifactSrcdoc(code, lang) {
  const csp = '<meta http-equiv="Content-Security-Policy" content="'
    + "default-src 'none'; img-src data: blob:; media-src data: blob:; "
    + "style-src 'unsafe-inline'; script-src 'unsafe-inline'; font-src data:;\">";
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
// job's worker stops cooperatively (media gen interrupts ComfyUI mid-render),
// so streamJob's "end" event arrives with status "cancelled".
export async function cancelJob(jobId) {
  try {
    await fetch(`/api/jobs/${jobId}/cancel`,
                { method: "POST", headers: authHeaders() });
  } catch (e) { /* best-effort - the stream will still end */ }
}

export async function streamJob(jobId, onLine, onProgress) {
  const r = await fetch(`/api/jobs/${jobId}/events`, { headers: authHeaders() });
  if (!r.ok) throw new Error(r.statusText);
  let endEvent = null;
  await readSSE(r, (payload) => {
    let ev;
    try { ev = JSON.parse(payload); } catch { return; }
    if (ev.type === "line" && onLine) onLine(ev.text);
    if (ev.type === "progress" && onProgress) onProgress(ev);
    if (ev.type === "end") endEvent = ev;
  });
  return endEvent || { status: "failed" };
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
    const cancel = el("button", "btn-secondary", "Cancel");
    cancel.onclick = () => ($("modal").style.display = "none");
    const ok = el("button", "btn-secondary btn-danger", confirmLabel);
    ok.onclick = () => { $("modal").style.display = "none"; onConfirm(); };
    row.appendChild(cancel);
    row.appendChild(ok);
    body.appendChild(row);
  });
}

/** Offer to download ONE curated missing model (repo/file/size shown in full,
 *  a real Download button - never a silent auto-pull). Resolves true whether
 *  the user downloads or skips (either way the caller proceeds to its real
 *  preflight-gated generate call, which is the authoritative check); resolves
 *  false only if the download itself failed after the user asked for it, so
 *  the caller can decide whether to keep going.  *log*, when given, gets the
 *  download's streamed progress lines appended (same log panel the page
 *  already uses for the generation job itself). */
function _offerModelDownload(missingModel, log) {
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
            body: JSON.stringify({ filename }),
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
    // poll for it and treat as "not now" - those handlers are not ours (same
    // pattern as picker.js's pickPath, for the same reason).
    watch = setInterval(() => {
      if ($("modal").style.display === "none") finish(true);
    }, 200);
  });
}

/** Pre-generate model-existence check: calls the read-only preflight endpoint
 *  for *kind* ("image" | "video" | "music") and, for each missing model that
 *  has a curated download source, offers it via _offerModelDownload before
 *  the caller submits its real generate request. Always resolves true
 *  (proceed) - a missing model with NO curated source, or one the user chose
 *  not to download, falls through unchanged to the real generate call's own
 *  preflight_models() gate, which fails with today's exact existing message.
 *  Best-effort: any failure to reach the pre-check itself also resolves true,
 *  so this can never block generation on its own account. */
export async function checkModelsBeforeGenerate(kind, log) {
  let data;
  try {
    const r = await fetch(`/api/media/${kind}/preflight`, {
      method: "POST", headers: authHeaders(), body: JSON.stringify({}),
    });
    if (!r.ok) return true;
    data = await r.json();
  } catch (e) {
    return true;
  }
  const curated = ((data && data.missing) || []).filter((m) => m.source);
  for (const m of curated) {
    await _offerModelDownload(m, log);
  }
  return true;
}

