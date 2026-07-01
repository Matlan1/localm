// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - shared helpers (split from app.js). Classic script: it
   shares the one global lexical environment with the other app/* and
   pages/* scripts, so every cross-section reference resolves by bare
   name exactly as before. */
"use strict";

/* ================================================================ */
/*  Shared helpers                                                   */
/* ================================================================ */

export const $ = (id) => document.getElementById(id);

// S2: the API key is no longer kept in JS-readable localStorage. Open-mode
// management uses the per-process shell token (injected as a global, sent as a
// bearer HEADER); protected mode rides the HttpOnly session cookie set at login
// or loopback auto-seed (auto-sent same-origin) with a double-submit CSRF token.
export const SHELL_TOKEN = window.__LOCALM_SHELL_TOKEN__ || "";

export function readCookie(name) {
  const m = document.cookie.match(new RegExp("(?:^|; )" + name + "=([^;]*)"));
  if (!m) return "";
  // A cookie value is untrusted input. decodeURIComponent throws a URIError on a
  // malformed percent-encoding; letting that propagate makes authHeaders() throw,
  // so EVERY `fetch(url, {headers: authHeaders()})` rejects before the request is
  // even sent - and bootAuthProbe then reports a perfectly reachable server as
  // "unreachable" and shows the reconnect overlay with no way out. A bad cookie
  // must never brick the client, so decode best-effort and fall back to the raw
  // value (what the server stored) on failure.
  try { return decodeURIComponent(m[1]); }
  catch (e) { return m[1]; }
}

export function authHeaders(extra = {}) {
  const h = { "Content-Type": "application/json", ...extra };
  if (SHELL_TOKEN) h["Authorization"] = "Bearer " + SHELL_TOKEN;
  const csrf = readCookie("localm_csrf");
  if (csrf) h["X-CSRF-Token"] = csrf;
  return h;
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

/** Replace raw <tool_call> JSON blocks with a compact human-readable note -
 *  shown while the web-access loop executes the request. */
export function formatToolCalls(text) {
  return (text || "").replace(
    /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g,
    (m, body) => {
      try {
        const call = JSON.parse(body);
        const a = call.args || call.arguments || {};
        const what =
          call.name === "web_search" ? `web search: "${a.query || ""}"` :
          call.name === "fetch_url"  ? `read page: ${a.url || ""}` :
          String(call.name || "request");
        return `\n> 🌐 *${what}*\n`;
      } catch (e) {
        return "\n> 🌐 *web request*\n";
      }
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

export function renderMarkdown(target, text) {
  const { think, open, rest: rawRest } = splitThink(scrubMarkers(text));
  const rest = formatToolCalls(rawRest);

  // Think block: update IN PLACE rather than rebuilding it every token. The old
  // code wiped target.innerHTML on each streamed chunk and recreated the
  // <details>, which reset its open/closed state every tick - so the reasoning
  // bubble could not be toggled WHILE the model was working. Keeping the same
  // element lets the user open/collapse it mid-stream and have that stick.
  // Default: open while still thinking, collapse once done - until the user
  // clicks it (data-userset), after which their choice is left alone.
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

/* ---- Artifacts canvas (A3) --------------------------------------------- *
 * A self-contained HTML/SVG block in a reply can be rendered live in a side
 * pane. The render is HARD-sandboxed: an <iframe sandbox="allow-scripts">
 * (NO allow-same-origin, so no access to this app's origin/cookies/storage)
 * whose srcdoc carries a Content-Security-Policy that blocks ALL network. So
 * an artifact can be interactive yet cannot phone home or read the app -
 * consistent with the privacy contract / "do not hide problems".            */

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

