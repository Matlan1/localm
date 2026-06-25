// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - vanilla JS, no build step.
   Talks to the localm FastAPI server: /v1 (OpenAI-compatible) + /api (GUI).
   All model/agent-originating strings go through textContent or DOMPurify -
   never raw innerHTML. pages.js builds on the helpers defined here. */

"use strict";

/* ================================================================ */
/*  Shared helpers                                                   */
/* ================================================================ */

const $ = (id) => document.getElementById(id);

// S2: the API key is no longer kept in JS-readable localStorage. Open-mode
// management uses the per-process shell token (injected as a global, sent as a
// bearer HEADER); protected mode rides the HttpOnly session cookie set at login
// or loopback auto-seed (auto-sent same-origin) with a double-submit CSRF token.
const SHELL_TOKEN = window.__LOCALM_SHELL_TOKEN__ || "";

function readCookie(name) {
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

function authHeaders(extra = {}) {
  const h = { "Content-Type": "application/json", ...extra };
  if (SHELL_TOKEN) h["Authorization"] = "Bearer " + SHELL_TOKEN;
  const csrf = readCookie("localm_csrf");
  if (csrf) h["X-CSRF-Token"] = csrf;
  return h;
}

function toast(msg, isError = false) {
  const el = $("toast");
  el.textContent = msg;
  el.className = "show" + (isError ? " error" : "");
  clearTimeout(el._t);
  el._t = setTimeout(() => (el.className = ""), 3500);
}

marked.setOptions({ breaks: true, mangle: false, headerIds: false });

/** Split leading <think>…</think> reasoning from the visible reply.
 *  Handles a still-open block during streaming. */
function splitThink(text) {
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
function stripThink(text) {
  return (text || "").replace(/<think>[\s\S]*?(<\/think>|$)/g, "").trim();
}

/** Replace raw <tool_call> JSON blocks with a compact human-readable note -
 *  shown while the web-access loop executes the request. */
function formatToolCalls(text) {
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
function scrubMarkers(text) {
  return (text || "")
    .replace(/<\|"\|>/g, '"')
    .replace(/<\|?\s*channel\s*\|?>(thought|thinking|analysis|reasoning|commentary|reflection)\n?(<\|?\s*message\s*\|?>)?/g, "<think>\n")
    .replace(/<\s*channel\s*\|>|<\|?\s*channel\s*\|?>final\n?(<\|?\s*message\s*\|?>)?/g, "\n</think>\n")
    .replace(/<\|?\s*channel\s*\|?>|<\s*channel\s*\|>|<\|?\s*message\s*\|?>|<\|start\|>(assistant|user|system)?|<\|return\|>|<\|turn>(user|model|assistant|system)?\n?|<turn\|>|<\|tool>|<tool\|>|<\|think\|>|<think\|>|<unused\d+>?/g, "");
}

function renderMarkdown(target, text) {
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
function artifactLang(codeEl) {
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
function artifactSrcdoc(code, lang) {
  const csp = '<meta http-equiv="Content-Security-Policy" content="'
    + "default-src 'none'; img-src data: blob:; media-src data: blob:; "
    + "style-src 'unsafe-inline'; script-src 'unsafe-inline'; font-src data:;\">";
  if (lang === "svg" || /^\s*<svg[\s>]/i.test(code)) {
    return "<!doctype html><html><head><meta charset=\"utf-8\">" + csp
      + "<style>html,body{margin:0;height:100%}svg{max-width:100%;height:auto;display:block}</style>"
      + "</head><body>" + code + "</body></html>";
  }
  if (/<!doctype\s+html/i.test(code) || /<html[\s>]/i.test(code)) {
    // Full document: inject the CSP as early as possible so it governs all loads.
    if (/<head[\s>]/i.test(code)) return code.replace(/<head([^>]*)>/i, "<head$1>" + csp);
    if (/<html[^>]*>/i.test(code)) return code.replace(/<html([^>]*)>/i, "<html$1><head>" + csp + "</head>");
    return csp + code;
  }
  // A fragment: wrap it in a minimal document.
  return "<!doctype html><html><head><meta charset=\"utf-8\">" + csp + "</head><body>"
    + code + "</body></html>";
}

/** Open the artifact pane and render *code* in the hard-sandboxed iframe. */
function openArtifact(code, lang) {
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
function closeArtifact() {
  const pane = document.getElementById("artifact-pane");
  if (!pane) return;
  pane.hidden = true;
  const body = pane.querySelector(".artifact-body");
  if (body) body.replaceChildren();
}

/** Add the copy button (and, for a renderable block, an "open canvas" button)
 *  to a <pre>. Idempotent. */
function enhanceCodeBlock(pre) {
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
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function autoGrow(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 220) + "px";
}

function nearBottom(elm) {
  return elm.scrollHeight - elm.scrollTop - elm.clientHeight < 80;
}

/** Parse an SSE byte stream from fetch(), invoking onData per `data:` payload. */
async function readSSE(response, onData) {
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
async function cancelJob(jobId) {
  try {
    await fetch(`/api/jobs/${jobId}/cancel`,
                { method: "POST", headers: authHeaders() });
  } catch (e) { /* best-effort - the stream will still end */ }
}

async function streamJob(jobId, onLine, onProgress) {
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
const GIB = 1024 ** 3, MIB = 1024 ** 2, KIB = 1024;

function fmtBytes(n) {
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
function downloadRate(samples, total) {
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
function fmtDuration(sec) {
  if (sec == null || !isFinite(sec) || sec < 0) return "";
  sec = Math.round(sec);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

/** Fetch an auth-protected image into an object URL. */
async function fetchImageURL(path) {
  const r = await fetch(path, { headers: authHeaders() });
  if (!r.ok) throw new Error(r.statusText);
  return URL.createObjectURL(await r.blob());
}

/* ---- modal ---- */

function openModal(title, bodyBuilder) {
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
function confirmDanger(title, message, confirmLabel, onConfirm) {
  openModal(title, (body) => {
    body.appendChild(el("p", "", message));
    const row = el("div", "actions");
    const cancel = el("button", "btn-quiet", "Cancel");
    cancel.onclick = () => ($("modal").style.display = "none");
    const ok = el("button", "btn-quiet btn-danger", confirmLabel);
    ok.onclick = () => { $("modal").style.display = "none"; onConfirm(); };
    row.appendChild(cancel);
    row.appendChild(ok);
    body.appendChild(row);
  });
}

/* ================================================================ */
/*  Theme                                                            */
/* ================================================================ */

function applyTheme(name) {
  document.documentElement.dataset.theme = name;
  localStorage.setItem("localm.theme", name);
}
applyTheme(localStorage.getItem("localm.theme") || "dark");
$("theme-toggle").onclick = () =>
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

/* ================================================================ */
/*  Logo style                                                       */
/* ================================================================ */

// The sidebar wordmark can be drawn three ways. The choice is SHARED via server
// config (logo_style), so the web GUI and the desktop launcher agree; the
// localStorage copy is only a no-flash cache for the next page load. The blue
// half always goes in the <span> (#logo span / .logo-tile span = var(--accent));
// the rest inherits the white/ink text colour. The console command, app icon,
// and desktop shortcut are fixed and unaffected.
const LOGO_STYLES = [
  { id: "local-m", white: "LocaL", blue: "M"  },   // default: single blue M, matches the icon (L white, M blue)
  { id: "loca-lm", white: "Loca",  blue: "LM" },
  { id: "localm",  white: "local", blue: "m"  },
];
const LOGO_DEFAULT = LOGO_STYLES[0].id;

// Draw a wordmark into el as white text + an accent-coloured <span>. The parts
// are constant strings, but build via DOM nodes (no innerHTML) to stay clear of
// the no-raw-HTML house style.
function drawWordmark(el, style) {
  el.textContent = style.white;
  const span = document.createElement("span");
  span.textContent = style.blue;
  el.appendChild(span);
}

// Render the wordmark, reflect the active picker tile, and cache the choice
// locally so the next load paints instantly. Does NOT touch the server.
function applyLogoStyle(id) {
  const style = LOGO_STYLES.find((s) => s.id === id) || LOGO_STYLES[0];
  drawWordmark($("logo"), style);
  localStorage.setItem("localm.logoStyle", style.id);
  for (const tile of document.querySelectorAll("#logo-style-picker .logo-tile")) {
    tile.classList.toggle("active", tile.dataset.style === style.id);
  }
  return style.id;
}

// Apply a pick locally, then persist it to the shared server config so the
// launcher (and other browsers) follow. Offline: the cached style still shows.
async function setLogoStyle(id) {
  const applied = applyLogoStyle(id);
  try {
    await fetch("/v1/config", {
      method: "PATCH", headers: authHeaders(),
      body: JSON.stringify({ logo_style: applied }),
    });
  } catch (e) { /* server unreachable - the cached style still applies */ }
}

// Reconcile the cached wordmark with the shared server truth on load (the
// launcher or another browser may have changed it). Best-effort.
async function syncLogoStyleFromConfig() {
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (!r.ok) return;
    const cfg = await r.json();
    if (cfg && typeof cfg.logo_style === "string") applyLogoStyle(cfg.logo_style);
  } catch (e) { /* server unreachable - keep the cached style */ }
}

// Render the three preview tiles into the Settings -> GUI card. Each tile shows
// the wordmark in its own style; clicking one applies + persists it.
function renderLogoPicker() {
  const wrap = $("logo-style-picker");
  if (!wrap) return;
  wrap.textContent = "";
  const current = localStorage.getItem("localm.logoStyle") || LOGO_DEFAULT;
  for (const style of LOGO_STYLES) {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.className = "logo-tile" + (style.id === current ? " active" : "");
    tile.dataset.style = style.id;
    tile.title = "Use the " + style.white + style.blue + " wordmark";
    drawWordmark(tile, style);
    tile.onclick = () => setLogoStyle(style.id);
    wrap.appendChild(tile);
  }
}

applyLogoStyle(localStorage.getItem("localm.logoStyle") || LOGO_DEFAULT);
renderLogoPicker();

/* ================================================================ */
/*  Tabs                                                             */
/* ================================================================ */

// Kernel pages are always present; plugin views (coder, images, music, video,
// knowledge) are added to VIEWS by renderNav() while their plugin is active.
const CORE_VIEWS = ["chat", "models", "plugins", "settings"];
let VIEWS = [...CORE_VIEWS];

// Toggle the .active class on the view sections + nav buttons. Split out of
// showView so the nav rebuild (reconcileActiveView) can re-assert the highlight
// on freshly-created buttons WITHOUT re-running onViewShown - re-firing
// onViewShown for chat/coder calls refreshPluginCommands, which calls renderNav
// -> reconcileActiveView -> showView -> onViewShown, an infinite /api/plugins
// loop.
function _applyActiveClasses(name) {
  for (const v of VIEWS) {
    const view = $("view-" + v), nav = $("nav-" + v);
    if (view) view.classList.toggle("active", v === name);
    if (nav) nav.classList.toggle("active", v === name);
  }
}

/** R09/R10: is the Settings view currently the active one? */
function isSettingsView() {
  const v = document.querySelector(".view.active");
  return !!v && v.id === "view-settings";
}
window.isSettingsView = isSettingsView;

function showView(name) {
  // Fall back to chat for an unknown name OR a view whose section is gone
  // (e.g. a remembered tab whose plugin was since uninstalled). Tolerating a
  // missing nav/view element is what lets the nav rail be rebuilt at runtime.
  if (!$("view-" + name)) name = "chat";
  // R10: leaving Settings with unsaved edits warns first (returning to Settings
  // re-renders the form from server state, so the edits would be silently lost).
  if (name !== "settings" && isSettingsView() &&
      window.settingsDirty && window.settingsDirty()) {
    if (!confirm("You have unsaved settings changes. Leave without saving?")) return;
  }
  _applyActiveClasses(name);
  // Remembered across reloads - but never in privacy mode (no traces).
  if (!chat.privacy) localStorage.setItem("localm.activeView", name);
  // On a phone the sidebar is an off-canvas drawer; navigating closes it.
  closeNav();
  // Lazy page refreshes live in pages.js
  if (window.onViewShown) window.onViewShown(name);
}
// Kernel nav buttons are static; plugin nav buttons get their handler in
// renderNav() as they are (re)created from the active-plugin set.
for (const v of CORE_VIEWS) $("nav-" + v).onclick = () => showView(v);

// --- mobile sidebar drawer ------------------------------------------------
// On a narrow screen the sidebar is off-canvas; the hamburger in the top bar
// toggles it, the backdrop or any navigation closes it. No-ops on desktop,
// where the sidebar is always visible and the toggle/backdrop are hidden.
function setNavOpen(open) {
  const app = $("app");
  if (app) app.classList.toggle("nav-open", open);
  const toggle = $("nav-toggle");
  if (toggle) toggle.setAttribute("aria-expanded", open ? "true" : "false");
}
function closeNav() { setNavOpen(false); }
if ($("nav-toggle")) {
  $("nav-toggle").onclick = () => {
    const app = $("app");
    setNavOpen(!(app && app.classList.contains("nav-open")));
  };
}
if ($("sidebar-backdrop")) $("sidebar-backdrop").onclick = closeNav;
// Close the drawer if the viewport grows back to desktop width (e.g. rotate).
window.addEventListener("resize", () => {
  if (window.innerWidth > 760) closeNav();
});

/* ================================================================ */
/*  Models (sidebar selector)                                        */
/* ================================================================ */

const modelSelect = $("model-select");

function setStatus(state, text) {
  $("status-dot").className = "dot " + state;
  $("status-text").textContent = text;
}

// ---- live hardware monitor in the status bar (CPU/RAM/VRAM/GPU) ----------
// Renders whatever /api/stats reports; any section the box can't measure is
// simply absent (e.g. no psutil -> no CPU/RAM; AMD -> no GPU%). VRAM shows
// used/total when free is known, otherwise just total.
function renderHwStats(data) {
  const el = $("hw-stats");
  if (!el) return;
  const gib = (b) => (b / GIB).toFixed(1);
  const parts = [];
  if (data && data.cpu && typeof data.cpu.percent === "number")
    parts.push(`CPU ${Math.round(data.cpu.percent)}%`);
  if (data && data.ram && typeof data.ram.percent === "number")
    parts.push(`RAM ${Math.round(data.ram.percent)}%`);
  if (data && data.vram && data.vram.total) {
    const v = data.vram;
    parts.push(v.used != null
      ? `VRAM ${gib(v.used)}/${gib(v.total)} GB`
      : `VRAM ${gib(v.total)} GB`);
  }
  if (data && data.gpu && typeof data.gpu.percent === "number")
    parts.push(`GPU ${Math.round(data.gpu.percent)}%`);
  if (parts.length) {
    el.textContent = parts.join(" · ");
    el.hidden = false;
  } else {
    el.hidden = true;
  }
}

async function pollHwStats() {
  if (typeof document !== "undefined" && document.hidden) return;
  try {
    const r = await fetch("/api/stats", { headers: authHeaders() });
    if (!r.ok) return;
    renderHwStats(await r.json());
  } catch (e) { /* transient - keep the last reading */ }
}

let _hwStatsTimer = null;
function startHwStats(intervalMs = 2500) {
  pollHwStats();
  if (_hwStatsTimer) clearInterval(_hwStatsTimer);
  _hwStatsTimer = setInterval(pollHwStats, intervalMs);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) pollHwStats();   // refresh promptly on tab focus
  });
}

// In-page API-key gate. Shown when an authed boot request returns 401 and this
// browser has no working key - the network/phone case, where the loopback key
// is never auto-seeded. Replaces window.prompt() (suppressed by mobile/PWA
// browsers, the NET-1 white-page cause). Idempotent: safe to call repeatedly.
function showKeyGate(message) {
  const gate = $("key-gate");
  if (!gate) return;
  if (message) { const m = $("key-gate-msg"); if (m) m.textContent = message; }
  gate.style.display = "flex";
  // Show/hide the one-tap "Install certificate" step (see updateKeyGateCertStep):
  // only when the local CA is genuinely NOT trusted yet, so a returning trusted
  // device is never told to "reinstall the certificate" every time the gate
  // appears (the SEAMLESS fix).
  updateKeyGateCertStep();
  // Offer "Scan QR code" wherever the browser can open a camera (a secure
  // context). Decoding uses the native BarcodeDetector when present, else the
  // bundled jsQR fallback, so it is not limited to Android Chrome.
  const scan = $("key-gate-scan");
  if (scan) scan.style.display = scanSupported() ? "inline-block" : "none";
  const input = $("key-gate-input");
  if (input) {
    input.value = "";   // HttpOnly key is unreadable; the gate only shows unauthed
    input.focus();
  }
}

// Decide whether the key gate should offer "Install certificate". Over HTTPS,
// offer it UNLESS the local CA is CONFIRMED trusted. The trust signal: a service
// worker registers only in a real secure context, so once SW registration
// succeeds (__swFailed === false) the CA is trusted and the step stays hidden - a
// returning trusted user is never told to "reinstall the certificate" (the
// SEAMLESS goal). But when trust is UNKNOWN - __swFailed still undefined, e.g.
// Firefox on a phone, or a clicked-through self-signed cert where SW registration
// never cleanly resolved - we MUST still offer it: gating on `=== true` (#201) hid
// the cert from exactly those mobile users, leaving them on a self-signed cert
// with no way to download it. So the test is `!== false` (show unless proven
// trusted). Re-run from index.html's renderInstall when SW registration resolves.
function updateKeyGateCertStep() {
  const cert = $("key-gate-cert");
  if (!cert) return;
  const onHttps = location.protocol === "https:";
  const untrusted = onHttps && window.__swFailed !== false;
  cert.style.display = untrusted ? "block" : "none";
  if (untrusted) {
    // Pin the download to an absolute https URL so it never resolves to http on
    // the TLS port, where portmux answers with a 308 + HTML catch page that an
    // <a download> would save as the "cert" (J2).
    const certLink = $("key-gate-cert-link");
    if (certLink) certLink.href = "https://" + location.host + "/localm-ca.crt";
  }
}
window.updateKeyGateCertStep = updateKeyGateCertStep;

// POST the entered key to /api/session so the server sets the HttpOnly auth
// cookie (the key never lives in JS), then reload so the boot re-runs
// authenticated. The CSRF cookie set alongside it is read by authHeaders().
async function loginWithKey(key) {
  try {
    const r = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    });
    return r.ok;
  } catch (e) { return false; }
}

// Submit the gate: log in with the entered key (trimmed) then reload. An empty
// entry just reloads (still unauthenticated -> the gate shows again).
function submitKeyGate() {
  const input = $("key-gate-input");
  const key = (input ? input.value : "").trim();
  if (key) {
    loginWithKey(key).then((ok) => {
      // Mark a SUCCESSFUL login so a still-401 boot after the reload self-heals a
      // stale shell instead of looping the gate (AUTH-1b). A failed login (wrong
      // key / server down) sets nothing, so the gate just shows again.
      if (ok) { try { sessionStorage.setItem("localm.loginOk", "1"); } catch (e) { /* private mode */ } }
      location.reload();
    });
  } else {
    location.reload();
  }
}

// Add a show/hide reveal toggle to a masked API-key input (AUTH-2), like the
// "show password" eye on a login form, so the user can verify what they typed.
// Idempotent: wraps the input in a flex row once and appends a small toggle.
function addRevealToggle(input) {
  if (!input || input.dataset.revealWired) return;
  input.dataset.revealWired = "1";
  const wrap = el("div", "input-reveal");
  if (input.parentNode) input.parentNode.insertBefore(wrap, input);
  wrap.appendChild(input);
  const btn = el("button", "reveal-btn", "show");
  btn.type = "button";          // never submit a surrounding form
  btn.setAttribute("aria-label", "Show or hide the key");
  btn.onclick = () => {
    const hidden = input.type === "password";
    input.type = hidden ? "text" : "password";
    btn.textContent = hidden ? "hide" : "show";
  };
  wrap.appendChild(btn);
}
window.addRevealToggle = addRevealToggle;

// --- Pairing QR scanner (phone) -------------------------------------------
// Reads the key QR shown in the computer's Settings (Companion app) with the
// camera and saves the key without typing. Decoding prefers the native
// BarcodeDetector and falls back to the bundled jsQR, so it works on browsers
// that lack BarcodeDetector (Firefox, Brave, Opera, iOS Safari, ...) - any
// browser that can open a camera in a secure context.
function scanSupported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

// Lazily load the bundled jsQR decoder (only fetched when a scan starts on a
// browser without BarcodeDetector). Sets window.jsQR (UMD). Cached after first
// load; the service worker caches the file for later offline pairing.
let _jsqrPromise = null;
function loadJsQR() {
  if (window.jsQR) return Promise.resolve(window.jsQR);
  if (_jsqrPromise) return _jsqrPromise;
  _jsqrPromise = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "/vendor/jsQR.min.js";
    s.onload = () => resolve(window.jsQR);
    s.onerror = () => { _jsqrPromise = null; reject(new Error("jsQR failed to load")); };
    document.head.appendChild(s);
  });
  return _jsqrPromise;
}

// Decode a scanned payload -> save the key + reload. Returns true when it was a
// localm pairing QR (so the scanner can stop). Factored out for unit testing.
function handleScannedKey(text) {
  const prefix = "localm-key:";
  if (typeof text !== "string" || !text.startsWith(prefix)) return false;
  const key = text.slice(prefix.length).trim();
  if (!key) return false;
  loginWithKey(key).then((ok) => {
    if (ok) { try { sessionStorage.setItem("localm.loginOk", "1"); } catch (e) { /* private mode */ } }
    location.reload();
  });
  return true;
}

let _qrStream = null;
let _qrTimer = null;
function stopQrScan() {
  if (_qrTimer) { clearInterval(_qrTimer); _qrTimer = null; }
  if (_qrStream) { _qrStream.getTracks().forEach((t) => t.stop()); _qrStream = null; }
  const v = $("qr-video"); if (v) v.srcObject = null;
  const s = $("qr-scanner"); if (s) s.style.display = "none";
}

async function startQrScan() {
  const overlay = $("qr-scanner"), video = $("qr-video"), status = $("qr-scan-status");
  if (!overlay || !video) return;
  overlay.style.display = "flex";
  if (status) status.textContent = "Starting camera…";

  // Prefer the native BarcodeDetector; otherwise load the bundled jsQR decoder.
  let detector = null;
  if ("BarcodeDetector" in window) {
    try { detector = new window.BarcodeDetector({ formats: ["qr_code"] }); }
    catch (e) { detector = null; }
  }
  let jsqr = null;
  if (!detector) {
    try { jsqr = await loadJsQR(); }
    catch (e) { if (status) status.textContent = "QR scanning is not available here."; return; }
  }

  try {
    _qrStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" }, audio: false });
  } catch (e) {
    if (status) status.textContent = "Could not open the camera (permission denied?).";
    return;
  }
  video.srcObject = _qrStream;
  try { await video.play(); } catch (e) { /* autoplay guard */ }
  if (status) status.textContent = "Point at the QR in the computer's Settings.";

  // Decode one video frame -> the QR text, or null. BarcodeDetector reads the
  // <video> directly; jsQR needs the pixels sampled onto a canvas first.
  const canvas = document.createElement("canvas");
  const decodeFrame = async () => {
    if (detector) {
      const codes = await detector.detect(video);
      return codes.length ? codes[0].rawValue : null;
    }
    const w = video.videoWidth, h = video.videoHeight;
    if (!w || !h) return null;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(video, 0, 0, w, h);
    const img = ctx.getImageData(0, 0, w, h);
    const res = jsqr(img.data, w, h, { inversionAttempts: "dontInvert" });
    return res ? res.data : null;
  };

  let busy = false;
  _qrTimer = setInterval(async () => {
    if (busy || !_qrStream) return;
    busy = true;
    try {
      const value = await decodeFrame();
      if (value && handleScannedKey(value)) { stopQrScan(); return; }
    } catch (e) { /* transient detect error - keep scanning */ }
    finally { busy = false; }
  }, detector ? 300 : 200);
}

// --- PWA install affordance (P2c) -----------------------------------------
// The browser "Install app" path differs per platform, so the Settings card
// shows the right thing: Android/desktop Chrome fire `beforeinstallprompt` and
// get a real Install button; iOS Safari fires nothing (the only path is the
// Share sheet -> Add to Home Screen), so it gets written steps; an already
// installed launch (standalone display) just confirms it. applyInstallUI() is
// the single decision point and is unit-tested by branch.

function pwaDisplayMode() {
  try {
    if (window.matchMedia && window.matchMedia("(display-mode: standalone)").matches) {
      return "standalone";
    }
  } catch (e) { /* matchMedia absent (older/headless) */ }
  // iOS records an installed PWA on navigator.standalone, not display-mode.
  if (window.navigator && window.navigator.standalone === true) return "standalone";
  return "browser";
}

function isIOSSafari() {
  const nav = window.navigator || {};
  const ua = nav.userAgent || "";
  // iPadOS 13+ reports itself as a Mac, so add the touch check for that case.
  return /iPad|iPhone|iPod/.test(ua)
    || (nav.platform === "MacIntel" && (nav.maxTouchPoints || 0) > 1);
}

// Show exactly one of: installed confirmation, the native Install button, the
// iOS Add-to-Home-Screen steps, or the generic hint. env = {standalone, ios,
// canPrompt}; missing fields are treated as false.
function applyInstallUI(env) {
  env = env || {};
  const btn = document.getElementById("install-app");
  const hint = document.getElementById("install-hint");
  const ios = document.getElementById("install-ios");
  if (btn) btn.style.display = "none";
  if (ios) ios.style.display = "none";
  if (!hint) return;
  hint.style.display = "none";
  if (env.standalone) {
    hint.textContent = "Running as an installed app.";
    hint.style.display = "";
    return;
  }
  if (env.canPrompt && btn) { btn.style.display = ""; return; }
  if (env.ios && ios) { ios.style.display = ""; return; }
  if (env.certNeeded) {
    // On HTTPS the service worker only registers behind a TRUSTED certificate;
    // without it the browser can only make a plain shortcut, not an installed
    // app. Point the user at the certificate step instead of the generic hint.
    hint.textContent = "To install localm as an app, trust this device's "
      + "certificate first (the “Install certificate” step above), then reload.";
    hint.style.display = "";
    return;
  }
  hint.style.display = "";   // generic browser-menu hint (default)
}

// --- Onboarding install gate (mobile, P-mobile) ----------------------------
// After auth, a phone that has not installed localm yet lands on a one-time
// install screen first (Install on Android / Add-to-Home-Screen steps on iOS),
// then taps "Continue" to enter the app. Desktop, an already-installed launch,
// and a return visit skip it. This is the "land on a setup page, reach localm
// via Continue" flow - the install affordance was previously buried in Settings.
function shouldShowInstallGate() {
  if (pwaDisplayMode() === "standalone") return false;       // already installed
  try { if (localStorage.getItem("localm.onboarded") === "1") return false; }
  catch (e) { /* storage blocked - treat as not onboarded */ }
  // Phones/tablets only: a touch device with a coarse pointer. A desktop
  // `localm gui` (fine pointer) opens straight into the app.
  return (navigator.maxTouchPoints || 0) > 0
    || !!(window.matchMedia && window.matchMedia("(pointer: coarse)").matches);
}

// Render the gate's install affordance - mirror of applyInstallUI for the gate's
// own elements. env = {ios, canPrompt}; missing fields treated as false.
function applyInstallGateUI(env) {
  env = env || {};
  const btn = $("install-gate-install");
  const ios = $("install-gate-ios");
  const hint = $("install-gate-hint");
  if (btn) btn.style.display = env.canPrompt ? "" : "none";
  if (ios) ios.style.display = (!env.canPrompt && env.ios) ? "" : "none";
  if (hint) hint.style.display = (!env.canPrompt && !env.ios) ? "" : "none";
}

function showInstallGate() {
  const gate = $("install-gate");
  if (!gate) return;
  const app = $("app");
  if (app) app.style.display = "none";   // the landing fully replaces the app
  applyInstallGateUI({ ios: isIOSSafari(), canPrompt: !!window.__deferredInstall });
  gate.style.display = "flex";
}

// Leave the gate and enter the app; remember it so we do not nag on return -
// but never in privacy mode (no localStorage traces). refreshCtxLimit also wipes
// the flag if privacy is detected after this runs, so the contract holds even if
// Continue is tapped before the privacy state is known.
function dismissInstallGate() {
  const gate = $("install-gate");
  if (gate) gate.style.display = "none";
  const app = $("app");
  if (app) app.style.display = "";
  if (!chat.privacy) {
    try { localStorage.setItem("localm.onboarded", "1"); } catch (e) { /* ignore */ }
  }
}

// Let a late-arriving `beforeinstallprompt` (Chrome fires it after engagement)
// upgrade the gate's generic hint to a real Install button while it is open.
window.refreshInstallGateIfOpen = function () {
  const gate = $("install-gate");
  if (gate && gate.style.display !== "none") {
    applyInstallGateUI({ ios: isIOSSafari(), canPrompt: !!window.__deferredInstall });
  }
};

let modelCache = { models: [], active: "" };

async function refreshModels() {
  try {
    const r = await fetch("/api/models", { headers: authHeaders() });
    if (r.status === 401) {
      // No working key (e.g. a network bind, where the loopback key is never
      // auto-seeded). Show the in-page key gate instead of window.prompt() -
      // mobile/PWA browsers suppress prompt(), which left a phone/LAN client on
      // a blank page (NET-1). The gate stores the key and reloads on submit.
      showKeyGate("This LocaLM server requires an API key.");
      return;
    }
    const data = await r.json();
    // Tolerate a malformed or empty payload (an old server or a proxy returning
    // {}). Without this, iterating an undefined model list throws and the model
    // dropdown silently breaks - fall back to an empty list instead.
    modelCache = (data && Array.isArray(data.models))
      ? data
      : { models: [], active: (data && data.active) || "" };
    // Don't rebuild the select while the user has it open
    if (document.activeElement !== modelSelect) {
      modelSelect.innerHTML = "";
      for (const m of modelCache.models) {
        const opt = document.createElement("option");
        opt.value = m.name;
        const size = m.size_bytes ? ` (${(m.size_bytes / GIB).toFixed(1)} GB)` : "";
        opt.textContent = m.name + size;
        if (m.active) opt.selected = true;
        modelSelect.appendChild(opt);
      }
    }
    setStatus("ok", data.active || "no model");
  } catch (e) {
    setStatus("err", "server unreachable");
  }
}

async function switchModel(model) {
  setStatus("busy", "loading " + model + "…");
  const r = await fetch("/api/models/load", {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ model }),
  });
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  setStatus("ok", model);
}

modelSelect.onchange = async () => {
  const model = modelSelect.value;
  try {
    await switchModel(model);
    toast("Model switched to " + model);
  } catch (e) {
    setStatus("err", "load failed");
    toast("Model load failed: " + e.message, true);
    refreshModels();
  }
};

/* ================================================================ */
/*  Chat                                                             */
/* ================================================================ */

const chat = {
  conversations: JSON.parse(localStorage.getItem("localm.conversations") || "[]"),
  activeId: null,
  abort: null,
  attachments: [],   // image attachments: {name, dataUri}
  docs: [],          // document attachments: {name, text, chars, truncated}
  ctxMax: 16384,     // context ceiling - refreshed from /v1/config
  privacy: false,    // server in privacy mode → conversations not persisted
  persist: false,    // non-privacy: conversations sync to the server store
  stick: true,       // R31: follow the stream to the bottom until the user scrolls up
};

// Conversation compaction mirrors localm/inference/compact.py: once the estimate
// passes 70% of the ceiling, summarise the OLDEST turns and keep the recent ones
// verbatim. R44: keep recent turns by token budget (not a flat last-4), feed the
// summariser whole messages truncated at word boundaries with reasoning stripped,
// and sanitise the returned summary so half-words / <think> never re-enter
// context. Never blocks chat.
const COMPACT_RATIO = 0.7;
const COMPACT_KEEP = 4;            // floor: always keep at least this many recent turns
const COMPACT_TARGET = 0.5;        // R44: keep recent turns verbatim up to ~50% of the ceiling

/** R44: rough token estimate - ~4 chars/token plus a small per-message overhead
 *  for the role and delimiters. Coarse, but less skewed than a bare length/4. */
function estimateTokens(text) {
  return Math.ceil((text || "").length / 4) + 4;
}

function estimateConvTokens(conv) {
  let total = estimateTokens($("p-system").value || "");
  for (const m of conv.messages) {
    total += estimateTokens(msgText(m)) + msgImages(m).length * 750;
  }
  return total;
}

/** R44: truncate at a word boundary with an explicit marker, instead of a raw
 *  mid-word slice that fed half-words (e.g. "REA") into the summariser. */
function truncateAtWord(text, max) {
  if (!text || text.length <= max) return text || "";
  const cut = text.slice(0, max);
  const sp = cut.lastIndexOf(" ");
  return (sp > max * 0.6 ? cut.slice(0, sp) : cut).trimEnd() + " ...[truncated]";
}

async function compactConversation(conv) {
  if (conv.messages.length <= COMPACT_KEEP) return false;
  // R44: keep as many of the most-recent turns verbatim as fit in COMPACT_TARGET
  // of the ceiling (at least COMPACT_KEEP), summarising only what is older -
  // instead of always discarding everything but the last 4 turns.
  const budget = (chat.ctxMax && chat.ctxMax > 0) ? COMPACT_TARGET * chat.ctxMax : 0;
  let keepCount = 0, used = 0;
  for (let i = conv.messages.length - 1; i >= 0; i--) {
    const t = estimateTokens(msgText(conv.messages[i])) +
              msgImages(conv.messages[i]).length * 750;
    const overBudget = budget > 0 ? (used + t > budget) : true;
    if (keepCount >= COMPACT_KEEP && overBudget) break;
    used += t;
    keepCount++;
  }
  keepCount = Math.max(COMPACT_KEEP, Math.min(keepCount, conv.messages.length - 1));
  const older = conv.messages.slice(0, -keepCount);
  const recent = conv.messages.slice(-keepCount);
  if (!older.length) return false;

  // R44: feed whole messages truncated at a word boundary, with reasoning blocks
  // stripped, so the summariser never sees half-words or display-only <think>.
  const excerpt = older.map((m) =>
    `${m.role.toUpperCase()}: ${truncateAtWord(stripThink(msgText(m)), 1200)}`).join("\n\n");

  let summary = "";
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model: modelSelect.value,
        messages: [{
          role: "user",
          content: "Summarise the following conversation in under 200 words. " +
            "Keep facts, names, decisions, and anything the user asked to " +
            "remember. Reply with the summary only.\n\n" + excerpt,
        }],
        max_tokens: 400,
        temperature: 0.3,
        stream: false,
      }),
    });
    if (r.ok) {
      const data = await r.json();
      summary = (data.choices?.[0]?.message?.content || "").trim();
    }
  } catch (e) { /* summarisation unavailable - fall back to a note below */ }
  // R44: sanitise the summary so leaked <think>/markers never re-enter context.
  summary = stripThink(scrubMarkers(summary)).trim();

  const bridge = summary
    ? [{ role: "user", content: "[Conversation summary]\n" + summary },
       { role: "assistant", content: "Understood. Continuing from this summary." }]
    : [{ role: "user", content:
         "[Earlier conversation was trimmed to fit the context window; " +
         "the recent messages below are intact.]" },
       { role: "assistant", content: "Understood." }];

  conv.messages = [...bridge, ...recent];
  pruneBranches(conv);   // forks anchored in the summarised-away region die
  saveConversations(conv);
  renderChat();
  toast(summary ? "Older messages summarised to free context"
                : "Older messages trimmed (summarisation unavailable)");
  return true;
}

async function maybeCompactConversation(conv) {
  if (!chat.ctxMax || chat.ctxMax <= 0) return;
  if (estimateConvTokens(conv) >= COMPACT_RATIO * chat.ctxMax) {
    await compactConversation(conv);
  }
}

async function refreshCtxLimit() {
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (r.ok) {
      const cfg = await r.json();
      // Prefer the resolved ceiling (VRAM-derived under ctx_auto) over the
      // static config value - compaction should track what the model can
      // actually hold.
      chat.ctxMax = cfg.effective_ctx_max ?? cfg.n_ctx_max ?? 16384;
      // Privacy mode: conversations live in memory only - wipe anything a
      // previous non-privacy session left behind and show the hint.
      chat.privacy = cfg.effective_mode === "privacy";
      if (chat.privacy) {
        localStorage.removeItem("localm.conversations");
        localStorage.removeItem("localm.activeView");
        localStorage.removeItem("localm.coderCwd");
        localStorage.removeItem("localm.kbAddPath");
        localStorage.removeItem("localm.convCollapsed");
        localStorage.removeItem("localm.imgMoveDest");
        localStorage.removeItem("localm.musicMoveDest");
        localStorage.removeItem("localm.videoMoveDest");
        localStorage.removeItem("localm.onboarded");
        localStorage.removeItem("localm.webAccess");   // R34: no trace in privacy
        localStorage.removeItem("localm.speakAloud");
        webAskSession = null;                          // R27: forget the session choice
        const h = document.querySelector("#conversations h3");
        if (h && !document.getElementById("privacy-hint")) {
          const hint = document.createElement("div");
          hint.id = "privacy-hint";
          hint.className = "privacy-hint";
          hint.textContent = "privacy mode - this session only";
          hint.title = "The server runs in privacy mode: conversations are " +
            "not saved (here or on disk) and vanish on reload. Export still works.";
          h.after(hint);
        }
      }
      hydrateChatToggles(cfg);   // R34: reflect saved choice / global net policy
    }
  } catch (e) { /* keep default */ }
}

// R34: the per-chat Web-access and Speak-aloud toggles used to reset to OFF on
// every load, so the user had to re-enable them in every session. Reflect the
// user's saved choice; for web, when there is no saved choice fall back to the
// global net policy (net_mode=allow auto-enables web; ask/off leave it off so
// consent still applies). Writes are gated on privacy mode (no traces there).
function hydrateChatToggles(cfg) {
  const webEl = $("p-web"), speakEl = $("p-speak");
  if (!webEl || !speakEl) return;
  const lsGet = (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } };
  const savedSpeak = chat.privacy ? null : lsGet("localm.speakAloud");
  if (savedSpeak !== null) speakEl.checked = savedSpeak === "1";
  const savedWeb = chat.privacy ? null : lsGet("localm.webAccess");
  if (savedWeb !== null) webEl.checked = savedWeb === "1";
  else if (cfg && cfg.net_mode === "allow") webEl.checked = true;
}
window.hydrateChatToggles = hydrateChatToggles;

function saveConversations(changed) {
  if (chat.privacy) return;   // privacy mode: no traces, not even localStorage
  // R40: do NOT cache server index-only rows (_meta) - they carry no messages, so
  // caching them would shadow a real local copy and show empty conversations
  // offline. Only full conversations are cached locally.
  const cacheable = chat.conversations.filter((c) => !c._meta);
  try {
    localStorage.setItem("localm.conversations",
      JSON.stringify(cacheable.slice(0, 50)));
  } catch (e) {
    // Quota: drop image-heavy older conversations and retry once
    const slim = cacheable.slice(0, 10);
    try { localStorage.setItem("localm.conversations", JSON.stringify(slim)); } catch {}
  }
  if (changed) pushConversation(changed);
}

/* ---- server-side conversation store (non-privacy modes only) ---- */

const _convPushTimers = new Map();

/** Debounced upsert of one conversation to the server store. */
function pushConversation(conv) {
  if (!chat.persist) return;
  // Brand-new conversations with nothing in them yet aren't worth a file.
  if (!conv.messages.length && conv.title === "New chat") return;
  conv.updated_at = Date.now();
  clearTimeout(_convPushTimers.get(conv.id));
  _convPushTimers.set(conv.id, setTimeout(async () => {
    _convPushTimers.delete(conv.id);
    try {
      await fetch("/api/conversations/" + encodeURIComponent(conv.id), {
        method: "PUT",
        headers: authHeaders(),
        body: JSON.stringify({ title: conv.title,
                               updated_at: conv.updated_at,
                               pinned: !!conv.pinned,
                               folder: conv.folder || null,
                               branches: conv.branches || [],
                               messages: conv.messages }),
      });
    } catch (e) { /* offline - localStorage still has the copy */ }
  }, 600));
}

function deleteConversationRemote(convId) {
  if (!chat.persist) return;
  clearTimeout(_convPushTimers.get(convId));
  _convPushTimers.delete(convId);
  fetch("/api/conversations/" + encodeURIComponent(convId), {
    method: "DELETE", headers: authHeaders(),
  }).catch(() => {});
}

/** R40: fetch a full conversation's body on demand and replace its index-only
 *  ("_meta") placeholder in chat.conversations. Returns true on success. */
async function hydrateConversation(conv) {
  if (!conv || !conv._meta) return true;
  try {
    const r = await fetch("/api/conversations/" + encodeURIComponent(conv.id),
                          { headers: authHeaders() });
    if (!r.ok) return false;
    const data = await r.json();
    conv.messages = data.messages || [];
    conv.branches = data.branches || [];
    if (data.title != null) conv.title = data.title;
    conv.pinned = !!data.pinned;
    conv.folder = data.folder || null;
    delete conv._meta;
    saveConversations();   // cache the now-full conversation locally
    return true;
  } catch (e) {
    return false;   // offline - keep the placeholder; renderChat shows the hint
  }
}
window.hydrateConversation = hydrateConversation;

/** Load the server store and merge with the localStorage cache: the newer copy
 *  of each conversation wins; local-only ones are uploaded. R40: the server
 *  list is a lightweight index (no message bodies); each conversation's messages
 *  load lazily on open via hydrateConversation. */
async function initServerConversations() {
  if (chat.privacy) return;
  try {
    // R40: lightweight, paginated index - no message bodies or data-URI images.
    const r = await fetch("/api/conversations?meta=true&limit=200&offset=0",
                          { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    if (!data.enabled) return;
    chat.persist = true;
    // Remote rows are index-only placeholders until opened.
    const byId = new Map((data.conversations || []).map(
      (c) => [c.id, { ...c, _meta: true, messages: [] }]));
    for (const local of chat.conversations) {
      const remote = byId.get(local.id);
      // Keep the local FULL copy unless the server has a strictly newer version
      // (>= keeps it on a tie, so an already-cached conversation is not demoted to
      // an index-only placeholder that would need a needless re-fetch / break offline).
      if (!remote || (local.updated_at || 0) >= (remote.updated_at || 0)) {
        byId.set(local.id, local);
        pushConversation(local);
      }
    }
    chat.conversations = [...byId.values()]
      .sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
    if (!chat.activeId && chat.conversations.length) {
      chat.activeId = chat.conversations[0].id;
    }
    await hydrateConversation(currentConv());   // load the active conversation's body
    saveConversations();
    renderConvList();
    renderChat();
    const h = document.querySelector("#conversations h3");
    if (h && !document.getElementById("persist-hint")) {
      const hint = document.createElement("div");
      hint.id = "persist-hint";
      hint.className = "privacy-hint";
      hint.textContent = "history saved on this machine";
      hint.title = "Conversations are stored in the localm data directory " +
        "(chats/) because the server runs in log or full mode. They survive " +
        "browser reloads and profile wipes.";
      h.after(hint);
    }
  } catch (e) { /* store unavailable - localStorage keeps working */ }
}

function currentConv() {
  return chat.conversations.find((c) => c.id === chat.activeId) || null;
}

function newConversation() {
  const conv = { id: Date.now().toString(36), title: "New chat", messages: [] };
  chat.conversations.unshift(conv);
  chat.activeId = conv.id;
  saveConversations(conv);
  renderConvList();
  renderChat();
}

/* message content helpers - content is a string or OpenAI multipart list */
function msgText(m) {
  if (typeof m.content === "string") return m.content;
  return (m.content || []).filter((p) => p.type === "text").map((p) => p.text).join("");
}
function msgImages(m) {
  if (typeof m.content === "string") return [];
  return (m.content || []).filter((p) => p.type === "image_url")
    .map((p) => p.image_url?.url).filter(Boolean);
}

/** VIS-1: replace every user-attached image (a data: URI) in the conversation
 *  with a short text note, so a model that rejected the image is never asked for
 *  it again. Re-sending a rejected image 400s on every turn and wedges the chat
 *  in endless empty assistant replies; dropping it keeps the chat usable.
 *  Server-generated /api/ images are display-only (already mapped to text before
 *  sending), so they are left alone. Returns the number of images dropped. */
function stripUserImages(conv) {
  let dropped = 0;
  for (const m of conv.messages) {
    if (m.role !== "user" || !Array.isArray(m.content)) continue;
    const imgs = m.content.filter(
      (p) => p.type === "image_url" && !p.image_url?.url?.startsWith("/api/"));
    if (!imgs.length) continue;
    dropped += imgs.length;
    const text = m.content.filter((p) => p.type === "text")
      .map((p) => p.text).join("");
    m.content = (text ? text + "\n" : "") +
      "[Image removed: this model cannot read images.]";
  }
  return dropped;
}

/* ---- conversation list: search, pin, folders ---- */

const convUI = {
  search: "",
  collapsed: new Set(JSON.parse(
    localStorage.getItem("localm.convCollapsed") || "[]")),
};

function saveCollapsed() {
  if (chat.privacy) return;   // folder names are conversation-derived
  localStorage.setItem("localm.convCollapsed",
    JSON.stringify([...convUI.collapsed]));
}

/** Short excerpt around the first content match, or null (no match). */
function searchSnippet(conv, term) {
  for (const m of conv.messages) {
    const text = msgText(m);
    const idx = text.toLowerCase().indexOf(term);
    if (idx !== -1) {
      const start = Math.max(0, idx - 18);
      return (start > 0 ? "…" : "") +
        text.slice(start, idx + 50).replace(/\s+/g, " ");
    }
  }
  return null;
}

function buildConvItem(conv, snippet) {
  const item = el("div", "conv-item" + (conv.id === chat.activeId ? " active" : ""));
  const title = el("span", "title");
  title.appendChild(document.createTextNode(conv.title));
  if (snippet) title.appendChild(el("span", "snippet", snippet));
  title.ondblclick = (e) => {
    e.stopPropagation();
    const input = document.createElement("input");
    input.value = conv.title;
    const commit = () => {
      conv.title = input.value.trim() || conv.title;
      saveConversations(conv);
      renderConvList();
    };
    input.onblur = commit;
    input.onkeydown = (ke) => { if (ke.key === "Enter") input.blur(); };
    item.replaceChild(input, title);
    input.focus();
    input.select();
  };
  item.appendChild(title);

  const pin = el("button", "del" + (conv.pinned ? " pinned-btn" : ""), "📌");
  pin.title = conv.pinned ? "Unpin" : "Pin to top";
  pin.onclick = (e) => {
    e.stopPropagation();
    conv.pinned = !conv.pinned;
    if (!conv.pinned) delete conv.pinned;
    saveConversations(conv);
    renderConvList();
  };
  item.appendChild(pin);

  const fold = el("button", "del", "📁");
  fold.title = conv.folder ? `Folder: ${conv.folder} (click to change)`
                           : "Move to a folder";
  fold.onclick = (e) => {
    e.stopPropagation();
    const name = prompt("Folder name (empty removes it from the folder):",
                        conv.folder || "");
    if (name === null) return;
    if (name.trim()) conv.folder = name.trim();
    else delete conv.folder;
    saveConversations(conv);
    renderConvList();
  };
  item.appendChild(fold);

  const del = el("button", "del", "×");
  del.title = "Delete conversation";
  del.onclick = (e) => {
    e.stopPropagation();
    chat.conversations = chat.conversations.filter((c) => c.id !== conv.id);
    if (chat.activeId === conv.id) chat.activeId = chat.conversations[0]?.id || null;
    deleteConversationRemote(conv.id);
    saveConversations();
    renderConvList();
    renderChat();
  };
  item.appendChild(del);

  item.onclick = async () => {
    chat.activeId = conv.id;
    renderConvList();
    if (conv._meta) await hydrateConversation(conv);   // R40: load the body on open
    renderChat();
    showView("chat");
  };
  return item;
}

function renderConvList() {
  const list = $("conv-list");
  list.replaceChildren();
  const term = convUI.search.trim().toLowerCase();

  // Filter: title match shows plain; content match shows a snippet
  const visible = [];
  for (const conv of chat.conversations) {
    if (!term) {
      visible.push({ conv, snippet: null });
    } else if ((conv.title || "").toLowerCase().includes(term)) {
      visible.push({ conv, snippet: null });
    } else {
      const snippet = searchSnippet(conv, term);
      if (snippet !== null) visible.push({ conv, snippet });
    }
  }

  // Group: pinned on top, then folders (A-Z), then the rest
  const pinned = visible.filter((v) => v.conv.pinned);
  const folders = new Map();
  const loose = [];
  for (const v of visible) {
    if (v.conv.pinned) continue;
    if (v.conv.folder) {
      if (!folders.has(v.conv.folder)) folders.set(v.conv.folder, []);
      folders.get(v.conv.folder).push(v);
    } else {
      loose.push(v);
    }
  }

  const addGroup = (label, key, items) => {
    if (!items.length) return;
    if (label) {
      // While searching, groups stay expanded so matches are never hidden
      const collapsed = !term && convUI.collapsed.has(key);
      const head = el("div", "conv-group", (collapsed ? "▸ " : "▾ ") + label);
      head.onclick = () => {
        if (convUI.collapsed.has(key)) convUI.collapsed.delete(key);
        else convUI.collapsed.add(key);
        saveCollapsed();
        renderConvList();
      };
      list.appendChild(head);
      if (collapsed) return;
    }
    for (const v of items) list.appendChild(buildConvItem(v.conv, v.snippet));
  };

  addGroup(pinned.length ? "📌 pinned" : "", "::pinned", pinned);
  for (const name of [...folders.keys()].sort()) {
    addGroup("📁 " + name, "f:" + name, folders.get(name));
  }
  addGroup((pinned.length || folders.size) && loose.length ? "chats" : "",
           "::chats", loose);

  if (term && !visible.length) {
    list.appendChild(el("div", "privacy-hint", "no matching chats"));
  }
}

$("conv-search").addEventListener("input", (e) => {
  convUI.search = e.target.value;
  renderConvList();
});

// A sensible download name for a chat image (VIS-2): from the data: URI's mime,
// or the /api path's basename, falling back to localm-image.png.
function imageFilename(url) {
  try {
    if (url.startsWith("data:")) {
      const m = url.match(/^data:image\/([a-z0-9.+-]+)/i);
      return "localm-image." + (m ? m[1].replace("jpeg", "jpg") : "png");
    }
    const base = new URL(url, location.origin).pathname.split("/").pop();
    return base && base.includes(".") ? base : "localm-image.png";
  } catch (e) { return "localm-image.png"; }
}

// Copy an image (by its resolved src) to the clipboard. Returns true on success.
// Not every browser/context can write an image to the clipboard, so the caller
// surfaces a fallback instead of silently failing (RULE 5).
async function copyImageSrc(src) {
  try {
    if (!window.ClipboardItem || !navigator.clipboard || !navigator.clipboard.write)
      return false;
    const blob = await (await fetch(src)).blob();
    await navigator.clipboard.write([new window.ClipboardItem({ [blob.type]: blob })]);
    return true;
  } catch (e) { return false; }
}
window.copyImageSrc = copyImageSrc;

// Full-view image lightbox (VIS-2): a click on a chat image opens it large with
// Save (download to disk) and Copy controls. Closes on the backdrop, the Close
// button, or Escape.
function openImageLightbox(src, name) {
  if (!src) return;
  const back = el("div", "img-lightbox");
  const panel = el("div", "img-lightbox-panel");
  const full = document.createElement("img");
  full.className = "img-lightbox-img";
  full.src = src;
  full.alt = name || "image";
  const bar = el("div", "img-lightbox-bar");
  const dismiss = () => {
    back.remove();
    document.removeEventListener("keydown", onKey);
  };
  function onKey(e) { if (e.key === "Escape") dismiss(); }
  const save = el("button", "btn-quiet", "Save");
  save.onclick = () => {
    const a = document.createElement("a");
    a.href = src;
    a.download = name || "localm-image.png";
    document.body.appendChild(a);
    a.click();
    a.remove();
  };
  const copy = el("button", "btn-quiet", "Copy image");
  copy.onclick = async () => {
    const ok = await copyImageSrc(src);
    toast(ok ? "Image copied" : "Could not copy the image - use Save instead", !ok);
  };
  const close = el("button", "btn-quiet", "Close");
  close.onclick = dismiss;
  bar.appendChild(save);
  bar.appendChild(copy);
  bar.appendChild(close);
  panel.appendChild(full);
  panel.appendChild(bar);
  back.appendChild(panel);
  back.addEventListener("click", (e) => { if (e.target === back) dismiss(); });
  document.addEventListener("keydown", onKey);
  document.body.appendChild(back);
}
window.openImageLightbox = openImageLightbox;

function addMessageRow(container, role, text, opts = {}) {
  const row = el("div", "msg-row " + role + (opts.cls ? " " + opts.cls : ""));
  row.appendChild(el("div", "msg-role",
    opts.label || (role === "user" ? "You" : "Model")));
  const body = el("div", "msg-body");
  if (role === "user") {
    // CHAT-1: a user's OWN message renders LITERALLY (exactly as typed). Markdown is
    // the model's output format, not the user's input - so typed *asterisks*, # hashes,
    // or pasted code/URLs are never silently reformatted. pre-wrap (.msg-literal)
    // preserves their line breaks; content is set via textContent so it is inert.
    body.classList.add("msg-literal");
    body.textContent = text;
  } else {
    renderMarkdown(body, text);
  }
  for (const url of opts.images || []) {
    const img = document.createElement("img");
    img.className = "msg-img";
    if (url.startsWith("/api/")) {
      // server-side generated image - fetch with auth headers
      fetchImageURL(url).then((u) => (img.src = u)).catch(() => img.remove());
    } else {
      img.src = url;   // data: URI from the user's own attachment
    }
    // Interactable (VIS-2): click to open a full-view lightbox with Save / Copy.
    // imageFilename uses the ORIGINAL url (a data: URI or /api path) for a sane
    // download name, while img.src is the resolved displayable source.
    img.style.cursor = "zoom-in";
    img.title = "Click to view, save, or copy";
    img.addEventListener("click", () => openImageLightbox(img.src, imageFilename(url)));
    body.appendChild(img);
  }
  for (const url of opts.audio || []) {
    const player = document.createElement("audio");
    player.controls = true;
    player.style.width = "100%";
    // bearer-protected file - fetch as a blob with auth headers
    fetchImageURL(url).then((u) => (player.src = u)).catch(() => player.remove());
    body.appendChild(player);
  }
  for (const url of opts.video || []) {
    const player = document.createElement("video");
    player.controls = true;
    player.style.width = "100%";
    player.style.borderRadius = "8px";
    fetchImageURL(url).then((u) => (player.src = u)).catch(() => player.remove());
    body.appendChild(player);
  }
  row.appendChild(body);
  const meta = el("div", "msg-meta");
  const copy = el("button", "copy-btn", "copy");
  copy.onclick = async () => {
    const plain = stripThink(text) || text;
    const firstImg = body.querySelector(".msg-img");
    // Image-only message (e.g. a bare attachment): copy the IMAGE, not empty
    // text - the "copy copied the prompt, not the image" report (VIS-2). For a
    // text+image message the text copy is kept; the image is in the lightbox.
    if (!plain && firstImg && firstImg.src) {
      const ok = await copyImageSrc(firstImg.src);
      copy.textContent = ok ? "copied" : "copy";
      if (!ok) toast("Could not copy the image - open it and use Save", true);
      else setTimeout(() => (copy.textContent = "copy"), 1200);
      return;
    }
    navigator.clipboard.writeText(plain);
    copy.textContent = "copied";
    setTimeout(() => (copy.textContent = "copy"), 1200);
  };
  meta.appendChild(copy);
  if (opts.variant) {
    const nav = el("span", "variant");
    const prev = el("button", "action", "‹");
    prev.title = "Previous variant";
    prev.onclick = opts.variant.prev;
    const next = el("button", "action", "›");
    next.title = "Next variant";
    next.onclick = opts.variant.next;
    nav.appendChild(prev);
    nav.appendChild(el("span", "k", `${opts.variant.k}/${opts.variant.n}`));
    nav.appendChild(next);
    meta.appendChild(nav);
  }
  for (const [label, fn] of opts.actions || []) {
    const btn = el("button", "action", label);
    btn.onclick = fn;
    meta.appendChild(btn);
  }
  row.appendChild(meta);
  container.appendChild(row);
  return { row, body, meta };
}

function buildEmptyHint() {
  const div = el("div", "empty-hint");
  const big = el("div", "big");
  big.appendChild(document.createTextNode("local"));
  const accent = el("span", "", "m");
  accent.style.color = "var(--accent)";
  big.appendChild(accent);
  div.appendChild(big);
  div.appendChild(document.createTextNode(
    "Chat with your local model. Everything stays on this machine."));
  const tip = el("div", "", "Type / for commands - /generate-image creates images locally.");
  tip.style.marginTop = "10px";
  tip.style.fontSize = "13px";
  div.appendChild(tip);
  return div;
}

function renderChat() {
  const box = $("chat-messages");
  box.innerHTML = "";
  const conv = currentConv();
  // R40: a not-yet-loaded conversation (server index row) hydrates its body on
  // first render, then re-renders. Try once per row (a failed/offline load sets
  // _hydrateFailed so we never spin).
  if (conv && conv._meta && !conv._hydrating && !conv._hydrateFailed) {
    conv._hydrating = true;
    hydrateConversation(conv).then((ok) => {
      conv._hydrating = false;
      if (ok) renderChat();
      else conv._hydrateFailed = true;
    });
  }
  if (!conv || conv.messages.length === 0) {
    box.appendChild(buildEmptyHint());
    return;
  }
  const NOTE_LABELS = { web: "Web", doc: "Doc", kb: "Sources" };
  // NEW-1 model-switch-indication: track the model of the previous assistant turn
  // so a small divider marks where the active model changed mid-conversation.
  let lastAssistantModel = null;
  conv.messages.forEach((m, i) => {
    if (m.role === "assistant" && m.model) {
      if (lastAssistantModel && m.model !== lastAssistantModel) {
        box.appendChild(el("div", "model-switch", "switched to " + m.model));
      }
      lastAssistantModel = m.model;
    }
    const tag = m.tag || (m.web ? "web" : null);
    const actions = [];
    if (m.role === "user" && !tag && !chat.abort) {
      actions.push(["edit", () => editMessage(conv, i)]);
      actions.push(["revert", () => revertTo(conv, i)]);
    }
    if (m.role === "assistant" && !tag) {
      actions.push(["🔊", () => speak(msgText(m), { toggle: true })]);
    }
    if (m.role === "assistant" && i === conv.messages.length - 1 && !chat.abort) {
      actions.push(["regenerate", () => regenerate(conv)]);
    }
    // ‹ k/N › on the first message of a fork point with siblings
    let variant = null;
    const pid = i > 0 ? conv.messages[i - 1].id : "root";
    const rec = (conv.branches || []).find((b) => b.parent === pid);
    if (rec && rec.tails.length > 1) {
      variant = {
        k: rec.current + 1,
        n: rec.tails.length,
        prev: () => switchBranch(conv, i, -1),
        next: () => switchBranch(conv, i, +1),
      };
    }
    const noteSuffix = m.truncated
      ? "\n\n*[stopped at the max-tokens limit - raise “Max tokens” in ⚙ parameters, or reply “continue”]*"
      : "";
    addMessageRow(box, m.role, msgText(m) + noteSuffix, {
      images: msgImages(m),
      audio: m.audio ? [m.audio] : [],
      video: m.video ? [m.video] : [],
      actions,
      variant,
      cls: tag ? "web-note" : "",
      label: tag ? NOTE_LABELS[tag] : undefined,
    });
  });
  // R31: only re-pin to the bottom when the user has not scrolled up. This tail
  // runs on every re-render (incl mid-stream web/finalize), so an unconditional
  // scroll here used to yank a reader back down while a reply was still streaming.
  if (chat.stick) box.scrollTop = box.scrollHeight;
}

/* ---- message branching ----
   conv.messages is always the LIVE linear branch (so compaction, retrieval
   injection, export, and the API mapping stay untouched). Alternative
   timelines are parked at fork points:
     conv.branches = [{parent: <msg id or "root">, tails: [[msg…]…], current}]
   The slot at `current` belongs to the live tail and is only written back
   when switching away. Editing a message or regenerating a reply parks the
   old tail as a sibling instead of destroying it; ‹ k/N › in the message
   meta row navigates between siblings. */

let _msgIdCounter = 0;

function msgId(m) {
  if (!m.id) m.id = Date.now().toString(36) + "-" + (_msgIdCounter++);
  return m.id;
}

function parentIdAt(conv, index) {
  return index > 0 ? msgId(conv.messages[index - 1]) : "root";
}

function forkRecord(conv, parentId, create) {
  conv.branches = conv.branches || [];
  let rec = conv.branches.find((b) => b.parent === parentId);
  if (!rec && create) {
    rec = { parent: parentId, tails: [], current: 0 };
    conv.branches.push(rec);
  }
  return rec;
}

/** Park the live tail from *index* as a sibling and open a fresh timeline. */
function forkAt(conv, index) {
  const rec = forkRecord(conv, parentIdAt(conv, index), true);
  if (!rec.tails.length) {
    rec.tails.push(null);          // slot for the pre-existing timeline
    rec.current = 0;
  }
  rec.tails[rec.current] = conv.messages.slice(index);
  rec.tails.push(null);            // slot owned by the new live timeline
  rec.current = rec.tails.length - 1;
  conv.messages = conv.messages.slice(0, index);
}

/** Switch the fork at *index* one sibling left/right (dir = ±1). */
function switchBranch(conv, index, dir) {
  if (chat.abort) { toast("Wait for the current reply to finish", true); return; }
  const rec = forkRecord(conv, parentIdAt(conv, index), false);
  if (!rec || rec.tails.length < 2) return;
  const n = rec.tails.length;
  const next = (rec.current + dir + n) % n;
  if (next === rec.current) return;
  rec.tails[rec.current] = conv.messages.slice(index);   // park live tail
  conv.messages = conv.messages.slice(0, index).concat(rec.tails[next] || []);
  rec.current = next;
  saveConversations(conv);
  renderChat();
}

/** Drop fork records whose parent message no longer exists anywhere
 *  (active branch or any parked tail) - called after compaction rewrites
 *  old history. */
function pruneBranches(conv) {
  if (!conv.branches || !conv.branches.length) return;
  const ids = new Set(["root"]);
  for (const m of conv.messages) if (m.id) ids.add(m.id);
  for (const rec of conv.branches) {
    for (const tail of rec.tails) {
      for (const m of tail || []) if (m.id) ids.add(m.id);
    }
  }
  conv.branches = conv.branches.filter((b) => ids.has(b.parent));
}

function editMessage(conv, index) {
  // Editing forks the branch tree (forkAt); doing that mid-stream parks the
  // messages before the streaming reply has landed, corrupting the branch
  // state. Bail while a reply streams, like switchBranch / regenerate do.
  if (chat.abort) { toast("Wait for the current reply to finish", true); return; }
  const m = conv.messages[index];
  $("chat-input").value = msgText(m);
  autoGrow($("chat-input"));
  // Fork instead of destroy: the old timeline (this message and everything
  // after it) stays reachable via ‹ › once the edited version is sent.
  forkAt(conv, index);
  saveConversations(conv);
  renderChat();
  $("chat-input").focus();
}

function regenerate(conv) {
  if (chat.abort) return;
  const last = conv.messages.length - 1;
  if (conv.messages[last]?.role !== "assistant") return;
  forkAt(conv, last);                // park the old reply as a sibling
  saveConversations(conv);
  renderChat();
  runCompletion(conv);
}

/** Count the sibling timelines a revert to *index* would permanently destroy:
 *  every fork point at or after the revert point keeps its alternatives in the
 *  region being removed. Used to warn before reverting *past* a branch point. */
function branchesLostByRevert(conv, index) {
  let lost = 0;
  for (let i = index; i < conv.messages.length; i++) {
    const pid = i > 0 ? conv.messages[i - 1].id : "root";
    const rec = (conv.branches || []).find((b) => b.parent === pid);
    if (rec && rec.tails.length > 1) {
      lost += rec.tails.filter((t, ti) => ti !== rec.current && t && t.length).length;
    }
  }
  return lost;
}

/** Revert the conversation to *index*: drop this message and everything after it
 *  DESTRUCTIVELY (unlike editMessage, which forks a sibling), and drop the
 *  clicked message back into the composer to modify and resend. Stays in the
 *  SAME branch. Reverting past a fork point destroys the sibling branches in the
 *  removed region, so confirm first when that would happen (the safeguard). */
function revertTo(conv, index) {
  if (chat.abort) { toast("Wait for the current reply to finish", true); return; }
  if (index < 0 || index >= conv.messages.length) return;
  const text = msgText(conv.messages[index]);
  const lost = branchesLostByRevert(conv, index);

  const apply = () => {
    const orig = conv.messages;
    // Keep only forks that diverge STRICTLY before the revert point. A fork at
    // or after it (parent inside the removed region, or the live tail being
    // reverted) is destroyed; a fork whose parent is not on the live branch
    // (nested in a surviving parked tail) is left for pruneBranches to judge.
    conv.branches = (conv.branches || []).filter((rec) => {
      if (rec.parent === "root") return index > 0;
      const p = orig.findIndex((x) => x.id === rec.parent);
      if (p === -1) return true;
      return p + 1 < index;
    });
    conv.messages = orig.slice(0, index);
    pruneBranches(conv);
    $("chat-input").value = text;
    autoGrow($("chat-input"));
    saveConversations(conv);
    renderChat();
    $("chat-input").focus();
  };

  if (lost > 0) {
    confirmDanger(
      "Revert conversation",
      `This permanently deletes ${lost} alternative branch` +
      `${lost === 1 ? "" : "es"} and everything after this message - it ` +
      "can't be undone. Revert anyway?",
      "Revert", apply);
  } else {
    apply();
  }
}

function chatParams() {
  const num = (id) => {
    const v = $(id).value.trim();
    return v === "" ? null : Number(v);
  };
  return {
    temperature: num("p-temperature"),
    top_p: num("p-top-p"),
    top_k: num("p-top-k"),
    repeat_penalty: num("p-repeat-penalty"),
    max_tokens: num("p-max-tokens"),
    seed: num("p-seed"),
    system: $("p-system").value.trim() || null,
    grammar: $("p-grammar").value.trim() || null,
  };
}

/* attachments */

function renderAttachChips() {
  const box = $("attach-chips");
  box.replaceChildren();
  chat.attachments.forEach((att, i) => {
    const chip = el("span", "chip");
    const img = document.createElement("img");
    img.src = att.dataUri;
    chip.appendChild(img);
    chip.appendChild(el("span", "", att.name));
    const rm = el("button", "", "×");
    rm.onclick = () => { chat.attachments.splice(i, 1); renderAttachChips(); };
    chip.appendChild(rm);
    box.appendChild(chip);
  });
  chat.docs.forEach((doc, i) => {
    const chip = el("span", "chip");
    chip.appendChild(el("span", "", "📄 " + doc.name +
      ` (${(doc.chars / 1000).toFixed(1)}k chars${doc.truncated ? ", trimmed" : ""})`));
    const rm = el("button", "", "×");
    rm.onclick = () => { chat.docs.splice(i, 1); renderAttachChips(); };
    chip.appendChild(rm);
    box.appendChild(chip);
  });
}

/** Document attachment → text via /api/rag/extract. The file is converted
 *  in memory on the server and never written to disk (privacy-clean). */
async function attachDocument(file) {
  const b64 = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(new Error("could not read file"));
    reader.readAsDataURL(file);
  });
  const r = await fetch("/api/rag/extract", {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ filename: file.name, content_b64: b64 }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || r.statusText);
  chat.docs.push({ name: data.filename, text: data.text,
                   chars: data.chars, truncated: data.truncated });
  renderAttachChips();
}

/** Ingest files as chat attachments: images inline (data URI), every other
 *  type extracted to text server-side. Shared by the file picker and the
 *  drag-and-drop zone so both behave identically. */
function addAttachedFiles(files) {
  for (const file of files) {
    if (file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = () => {
        chat.attachments.push({ name: file.name, dataUri: reader.result });
        renderAttachChips();
      };
      reader.readAsDataURL(file);
    } else {
      attachDocument(file).catch((err) =>
        toast(`${file.name}: ${err.message}`, true));
    }
  }
}

$("chat-attach").onclick = () => $("chat-file").click();
$("chat-file").addEventListener("change", (e) => {
  addAttachedFiles(e.target.files);
  e.target.value = "";
});

// Dedicated camera button (mobile): opens the camera directly to attach a photo.
// The OS file picker on the attach button already offers camera + gallery, but a
// one-tap "take a photo" makes the phone feel like a real input, not a website.
{
  const cam = $("chat-camera"), camFile = $("chat-camera-file");
  if (cam && camFile) {
    cam.onclick = () => camFile.click();
    camFile.addEventListener("change", (e) => {
      addAttachedFiles(e.target.files);
      e.target.value = "";
    });
  }
}

/** Ingest images shared INTO localm from the phone's share sheet (PWA share
 *  target). The server stashed them; we pull them as chat attachments and then
 *  clear the server inbox. Text/links shared in drop into the composer. */
async function ingestSharedFiles() {
  let items;
  try {
    const r = await fetch("/api/share/pending", { headers: authHeaders() });
    if (!r.ok) return;
    items = (await r.json()).items || [];
  } catch (e) { return; }
  if (!items.length) return;
  const ids = [];
  let imgs = 0;
  for (const it of items) {
    ids.push(it.id);
    if ((it.type || "").startsWith("image/")) {
      chat.attachments.push({ name: it.name, dataUri: it.data_uri });
      imgs++;
    } else if (it.data_uri) {
      try {
        const txt = decodeURIComponent(escape(atob(it.data_uri.split(",", 2)[1] || "")));
        const ta = $("chat-input");
        if (ta) { ta.value = (ta.value ? ta.value + "\n" : "") + txt; autoGrow(ta); }
      } catch (e) { /* not decodable text */ }
    }
  }
  renderAttachChips();
  // We have the data client-side now; clear the server inbox.
  fetch("/api/share/clear", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  }).catch(() => {});
  if (imgs) toast(`${imgs} image${imgs > 1 ? "s" : ""} shared into chat`);
}

/** Drag-and-drop files anywhere on the chat view to attach them, with a
 *  highlight while a file is hovering. Only reacts to file drags (not text
 *  selections), and preventDefault stops the browser opening the dropped file. */
function setupChatDropZone() {
  const zone = $("view-chat");
  if (!zone) return;
  const isFileDrag = (e) =>
    e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
  zone.addEventListener("dragover", (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    zone.classList.add("drag-over");
  });
  zone.addEventListener("dragleave", (e) => {
    if (!zone.contains(e.relatedTarget)) zone.classList.remove("drag-over");
  });
  zone.addEventListener("drop", (e) => {
    if (!e.dataTransfer || !e.dataTransfer.files || !e.dataTransfer.files.length) return;
    e.preventDefault();
    zone.classList.remove("drag-over");
    addAttachedFiles(e.dataTransfer.files);
  });
}
setupChatDropZone();

/* ================================================================ */
/*  Command palette (Ctrl/Cmd+K)                                     */
/* ================================================================ */

// Built fresh on open so runtime-added plugin views are included. The view
// labels are taken from the live nav buttons so they match exactly.
function cmdkCommands() {
  const cmds = [];
  for (const v of VIEWS) {
    if (!$("view-" + v)) continue;
    const nav = $("nav-" + v);
    const label = ((nav ? nav.textContent : v) || v).trim() || v;
    cmds.push({ label: "Go to " + label, run: () => showView(v) });
  }
  cmds.push({ label: "New chat", run: () => { newConversation(); showView("chat"); } });
  cmds.push({ label: "Toggle light/dark theme", run: () => $("theme-toggle").click() });
  cmds.push({ label: "Export conversation", run: () => exportConversation() });
  // Direct jump to the owner-only Keys & devices manager (it lives in a Settings
  // sub-section that is otherwise easy to miss). Offered only when this key may
  // actually manage keys - i.e. the panel is not gated-hidden for it.
  const keysCard = $("keys-card");
  if (keysCard && !keysCard.classList.contains("sec-hidden")) {
    cmds.push({ label: "Manage keys & devices", run: () => {
      showView("settings");
      if (typeof gotoSettingsSection === "function") gotoSettingsSection("keys-card");
    } });
  }
  return cmds;
}

let _cmdkAll = [], _cmdkShown = [], _cmdkSel = 0;

function cmdkFilter(query) {
  const q = (query || "").trim().toLowerCase();
  return q ? _cmdkAll.filter((c) => c.label.toLowerCase().includes(q)) : _cmdkAll.slice();
}

function renderCmdk(query) {
  _cmdkShown = cmdkFilter(query);
  if (_cmdkSel >= _cmdkShown.length) _cmdkSel = Math.max(0, _cmdkShown.length - 1);
  const list = $("cmdk-list");
  list.replaceChildren();
  _cmdkShown.forEach((c, i) => {
    const item = el("div", "cmdk-item" + (i === _cmdkSel ? " sel" : ""), c.label);
    item.onclick = () => runCmdk(i);
    list.appendChild(item);
  });
}

function cmdkIsOpen() {
  const m = $("cmdk");
  return !!m && m.style.display !== "none";
}

function openCommandPalette() {
  _cmdkAll = cmdkCommands();
  _cmdkSel = 0;
  $("cmdk-input").value = "";
  renderCmdk("");
  $("cmdk").style.display = "flex";
  $("cmdk-input").focus();
}

function closeCommandPalette() {
  $("cmdk").style.display = "none";
}

function runCmdk(index) {
  const cmd = _cmdkShown[index];
  closeCommandPalette();
  if (cmd) cmd.run();
}

$("cmdk-input").addEventListener("input", (e) => { _cmdkSel = 0; renderCmdk(e.target.value); });
$("cmdk").addEventListener("click", (e) => { if (e.target === $("cmdk")) closeCommandPalette(); });
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && (e.key === "k" || e.key === "K")) {
    e.preventDefault();
    cmdkIsOpen() ? closeCommandPalette() : openCommandPalette();
    return;
  }
  // R09: Ctrl/Cmd+S saves the active Settings section. Only on the Settings page,
  // so every other view keeps the browser's native "Save page".
  if ((e.ctrlKey || e.metaKey) && (e.key === "s" || e.key === "S")) {
    if (isSettingsView()) {
      e.preventDefault();
      if (window.saveActiveSettingsSection) window.saveActiveSettingsSection();
    }
    return;
  }
  if (!cmdkIsOpen()) return;
  if (e.key === "Escape") { e.preventDefault(); closeCommandPalette(); }
  else if (e.key === "ArrowDown") {
    e.preventDefault(); _cmdkSel = Math.min(_cmdkSel + 1, _cmdkShown.length - 1);
    renderCmdk($("cmdk-input").value);
  } else if (e.key === "ArrowUp") {
    e.preventDefault(); _cmdkSel = Math.max(_cmdkSel - 1, 0);
    renderCmdk($("cmdk-input").value);
  } else if (e.key === "Enter") {
    e.preventDefault(); runCmdk(_cmdkSel);
  }
});

// R10: the browser's native unsaved-changes prompt on tab close / reload while
// Settings has unsaved edits. An empty returnValue is what triggers it; browsers
// ignore any custom message, so we set none.
window.addEventListener("beforeunload", (e) => {
  if (window.settingsDirty && window.settingsDirty()) {
    e.preventDefault();
    e.returnValue = "";
  }
});

/* ================================================================ */
/*  Settings: performance sliders (GPU layers + context) + VRAM est  */
/* ================================================================ */

function _perfGiB(b) { return (Number(b) / GIB).toFixed(1); }
function perfGlLabel(v) { return Number(v) >= 99 ? "all" : String(v); }

let _perfEstTimer = null;
async function refreshPerfEstimate() {
  const gl = $("perf-gpu-layers"), ctx = $("perf-ctx"), out = $("perf-estimate");
  if (!gl || !ctx || !out) return;
  try {
    const q = new URLSearchParams({ n_ctx: ctx.value, n_gpu_layers: gl.value });
    const r = await fetch("/api/vram-estimate?" + q.toString(), { headers: authHeaders() });
    if (!r.ok) { out.textContent = "estimate unavailable"; return; }
    const d = await r.json();
    let text = `~${_perfGiB(d.needed)} GB needed `
      + `(weights ${_perfGiB(d.weights)} · context ${_perfGiB(d.kv_cache)} `
      + `· overhead ${_perfGiB(d.overhead)})`;
    if (typeof d.free === "number")
      text += ` · ${_perfGiB(d.free)} GB free - ` + (d.fits ? "fits ✓" : "may not fit ⚠");
    else
      text += " · free VRAM unknown";
    out.textContent = text;
    out.classList.toggle("perf-warn", d.fits === false);
  } catch (e) {
    out.textContent = "estimate unavailable";
  }
}

function setupPerfCard() {
  const gl = $("perf-gpu-layers"), ctx = $("perf-ctx");
  if (!gl || !ctx) return;
  const sync = () => {
    $("perf-gl-val").textContent = perfGlLabel(gl.value);
    $("perf-ctx-val").textContent = ctx.value;
  };
  const onInput = () => {
    sync();
    clearTimeout(_perfEstTimer);
    _perfEstTimer = setTimeout(refreshPerfEstimate, 150);   // debounce while dragging
  };
  gl.addEventListener("input", onInput);
  ctx.addEventListener("input", onInput);
  const apply = $("perf-apply");
  if (apply) apply.onclick = async () => {
    try {
      const r = await fetch("/v1/config", {
        method: "PATCH", headers: authHeaders(),
        body: JSON.stringify({ n_gpu_layers: Number(gl.value), n_ctx: Number(ctx.value) }),
      });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
      toast("Saved - applies on the next model load");
    } catch (e) { toast("Could not save: " + e.message, true); }
  };
  // Seed slider positions from the current config, then estimate.
  fetch("/v1/config", { headers: authHeaders() })
    .then((r) => (r.ok ? r.json() : {}))
    .then((cfg) => {
      if (typeof cfg.n_gpu_layers === "number")
        gl.value = cfg.n_gpu_layers < 0 ? 99 : Math.min(99, cfg.n_gpu_layers);
      if (typeof cfg.n_ctx === "number")
        ctx.value = Math.min(Number(ctx.max), Math.max(Number(ctx.min), cfg.n_ctx));
      sync();
      refreshPerfEstimate();
    })
    .catch(() => { sync(); refreshPerfEstimate(); });
}

/* ---- web access (model-initiated, via the params-drawer toggle) ---- */

const WEB_MAX_ROUNDS = 3;

// R27: a remembered "don't ask again this session" choice. null = ask each time;
// true = allow all this session; false = deny all this session. In-memory only
// (so it resets on reload = a new session) and leaves no persisted trace.
let webAskSession = null;

// net_mode = ask means the GUI must APPROVE each model-initiated web request
// before it runs (the settings promise: "ask = approve each request"). Read it
// fresh from /v1/config so a change in Settings takes effect without a reload;
// the cost is one small GET per model-initiated round (bounded by
// WEB_MAX_ROUNDS). Unknown / unreachable -> do not block (the per-conversation
// toggle is the standing consent; only "off", enforced server-side, blocks).
async function webModeIsAsk() {
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (r.ok) {
      const cfg = await r.json();
      return !!(cfg && cfg.net_mode === "ask");
    }
  } catch (e) { /* server unreachable - fall through to "do not block" */ }
  return false;
}
window.webModeIsAsk = webModeIsAsk;

// Approval dialog for a model-initiated web request under net_mode=ask. Returns
// a promise<boolean>. Uses the in-page modal (window.confirm/prompt are
// suppressed in some PWA/mobile browsers, the NET-1 class of bug). Overridable
// in tests.
function confirmWebRequest(call) {
  // R27: a remembered choice short-circuits the modal for the rest of the session.
  if (webAskSession !== null) return Promise.resolve(webAskSession);
  return new Promise((resolve) => {
    const args = (call && call.args) || {};
    const target = args.query || args.url || "";
    const verb = call && call.name === "fetch_url" ? "fetch a web page" : "search the web";
    openModal("Allow web access?", (body) => {
      body.appendChild(el("p", "", "The model wants to " + verb + " for:"));
      body.appendChild(el("p", "web-ask-target", target));
      // R27: "don't ask again this session" - remember Allow/Deny so the popup
      // does not fire on every model-initiated request for the rest of the session.
      const remember = el("label", "web-ask-remember");
      const cb = el("input");
      cb.type = "checkbox";
      remember.appendChild(cb);
      remember.appendChild(document.createTextNode(" Don't ask again this session"));
      body.appendChild(remember);
      const row = el("div", "actions");
      const deny = el("button", "btn-quiet", "Deny");
      deny.onclick = () => {
        if (cb.checked) webAskSession = false;
        $("modal").style.display = "none";
        resolve(false);
      };
      const allow = el("button", "btn-quiet btn-primary", "Allow");
      allow.onclick = () => {
        if (cb.checked) webAskSession = true;
        $("modal").style.display = "none";
        resolve(true);
      };
      row.appendChild(deny);
      row.appendChild(allow);
      body.appendChild(row);
    });
  });
}
window.confirmWebRequest = confirmWebRequest;

const WEB_TOOL_PROMPT =
  "You can access the internet through tools. When the answer depends on " +
  "current, real-time, or external information you cannot be certain of " +
  "(news, prices, software versions, documentation, anything after your " +
  "training cutoff), get it from the web instead of guessing. Reply with " +
  "ONLY a tool call block and nothing else:\n" +
  '<tool_call>{"name": "web_search", "args": {"query": "..."}}</tool_call>\n' +
  "To read a specific page:\n" +
  '<tool_call>{"name": "fetch_url", "args": {"url": "https://..."}}</tool_call>\n' +
  "The results arrive in the next message; then answer and cite the source " +
  "URLs you used.\n" +
  "HONESTY: never invent search results, URLs, or page contents, and never " +
  "say you searched or read a page unless you actually emitted a tool call " +
  "and received its result. If a search fails or finds nothing useful, say " +
  "so plainly instead of making something up.";

const NO_WEB_PROMPT =
  "You are offline with NO internet access in this conversation. Do not " +
  "present guessed or invented information as verified fact: current events, " +
  "news, prices, live data, software versions, or anything you cannot confirm " +
  "from this conversation. Never claim you looked something up, searched the " +
  "web, or read a page, because you cannot. If the user needs current or " +
  "external information, say plainly that you cannot verify it offline and " +
  "that they can enable \"Web access\" with the 🌐 toggle in the " +
  "parameters drawer (⚙). Saying \"I do not know\" is better than " +
  "stating something false.";

// Used when web results were just injected (the explicit /web command, or a
// model-initiated search) but the standing toggle is off: the model HAS fresh
// results in hand, so the offline-denial floor would contradict them. Tell it
// to use and cite the provided results, and not to fabricate beyond them.
const WEB_GROUNDED_PROMPT =
  "Web search results have been provided to you in this conversation. Use them " +
  "to answer, and cite the source URLs you relied on. Stay within what the " +
  "results actually support: do not invent facts, URLs, or details beyond them, " +
  "and if they do not answer the question, say so plainly.";

/** True when the most recent message is freshly injected web grounding (search
 *  results or fetched page content), as opposed to a repair note or a failure
 *  note. Used so an explicit /web run is not told it is offline. */
function lastTurnHasWebResults(conv) {
  const last = conv.messages[conv.messages.length - 1];
  if (!last || !last.web) return false;
  const text = typeof last.content === "string" ? last.content : "";
  return /Results of web_search|Content of /.test(text);
}

// Tool-call wrappers a local model may emit. We accept the canonical
// <tool_call> tags plus the mangled finetune dialects (<|tool_call|>, closing
// as <tool_call|>) and ```tool_call / ```json / bare ``` fences, mirroring the
// coder's lenient parser so a slightly-off call still runs instead of being
// silently dropped (which let the model's un-grounded answer through).
const _WEB_TOOLS = new Set(["web_search", "fetch_url"]);

/** Lenient JSON parse for the mangles local finetunes produce (single-quoted
 *  keys, trailing commas). Returns the parsed object, or null. */
function _lenientJSON(body) {
  const tries = [
    (s) => s,
    (s) => s.replace(/'([^']+)'\s*:/g, '"$1":'),       // single-quoted keys
    (s) => s.replace(/,(\s*[}\]])/g, "$1"),            // trailing commas
    (s) => s.replace(/'([^']+)'\s*:/g, '"$1":').replace(/,(\s*[}\]])/g, "$1"),
  ];
  for (const fix of tries) {
    try {
      const obj = JSON.parse(fix(body));
      if (obj && typeof obj === "object") return obj;
    } catch (e) { /* try the next recovery layer */ }
  }
  return null;
}

/** Yield every brace-balanced top-level {...} region in text. String literals
 *  are tracked so braces inside them do not confuse the depth count. */
function* _topLevelObjects(text) {
  for (let i = 0; i < text.length; i++) {
    if (text[i] !== "{") continue;
    let depth = 0, inStr = false, esc = false;
    for (let j = i; j < text.length; j++) {
      const c = text[j];
      if (inStr) {
        if (esc) esc = false;
        else if (c === "\\") esc = true;
        else if (c === '"') inStr = false;
      } else if (c === '"') { inStr = true; }
      else if (c === "{") { depth++; }
      else if (c === "}") {
        depth--;
        if (depth === 0) { yield text.slice(i, j + 1); i = j; break; }
      }
    }
  }
}

/** Normalise a parsed object to a {name, args} web call, or null. Accepts the
 *  OpenAI "arguments" alias for "args". */
function _asWebCall(obj) {
  if (!obj || typeof obj.name !== "string" || !_WEB_TOOLS.has(obj.name)) return null;
  const args = (obj.args && typeof obj.args === "object") ? obj.args
             : (obj.arguments && typeof obj.arguments === "object") ? obj.arguments
             : {};
  return { name: obj.name, args };
}

/** First web tool call in a reply, or null. Tolerates the wrapper and JSON
 *  mangles local models emit so a real attempt is not silently dropped. */
function parseWebCall(text) {
  const clean = stripThink(text);
  // Candidate {prefixName, body} pairs, in priority order: explicit wrappers
  // first (the name may live in a "call:NAME" prefix, Gemma-style), then
  // fences, then any bare top-level JSON object naming a web tool.
  const bodies = [];
  const wrap = /<\|?\/?tool_call\|?>\s*(?:call:(\w+)\s*)?([\s\S]*?)\s*<\|?\/?tool_call\|?>/g;
  for (const mm of clean.matchAll(wrap)) bodies.push({ name: mm[1], body: mm[2] });
  const fence = /```[ \t]*[A-Za-z_]*[ \t]*\r?\n([\s\S]*?)\r?\n[ \t]*```/g;
  for (const mm of clean.matchAll(fence)) bodies.push({ name: undefined, body: mm[1] });
  for (const { name: prefixName, body } of bodies) {
    const trimmed = body.trim().replace(/<\|"\|>/g, '"');   // Gemma quote tokens
    const obj = _lenientJSON(trimmed);
    let call = _asWebCall(obj);
    // Args-only body with the tool named in the wrapper prefix (Gemma native).
    if (!call && obj && _WEB_TOOLS.has(prefixName)) {
      call = { name: prefixName, args: obj };
    }
    if (call) return call;
  }
  // Last resort: a bare {...} object anywhere in the reply naming a web tool.
  for (const chunk of _topLevelObjects(clean)) {
    const call = _asWebCall(_lenientJSON(chunk));
    if (call) return call;
  }
  return null;
}

/** True when a reply looks like a botched web tool call we could not parse: a
 *  tool-call wrapper/fence, or a JSON object that mentions a web tool by name.
 *  Lets the caller ask the model to re-emit it cleanly instead of accepting an
 *  un-grounded answer. */
function looksLikeWebToolAttempt(text) {
  const clean = stripThink(text);
  if (/<\|?\/?tool_call\|?>/.test(clean) || /```[ \t]*tool_call\b/.test(clean)) return true;
  return /"name"\s*:/.test(clean) && /web_search|fetch_url/.test(clean);
}

/** Run a web tool call through the policy-enforced server endpoints. */
async function requestWebTool(call) {
  const a = call.args || call.arguments || {};
  if (call.name === "web_search") {
    const r = await fetch("/api/web/search", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ query: a.query || "", max_results: 5 }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const lines = data.results.map((res, i) =>
      `${i + 1}. ${res.title}\n   ${res.url}` +
      (res.snippet ? `\n   ${res.snippet}` : ""));
    return `[Results of web_search "${a.query}"]\n` + lines.join("\n");
  }
  if (call.name === "fetch_url") {
    const r = await fetch("/api/web/fetch", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ url: a.url || "", max_chars: 6000 }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    return `[Content of ${data.url}]` +
      (data.truncated ? " (truncated)" : "") + `\n${data.text}`;
  }
  throw new Error("Unknown web tool: " + call.name);
}

/** Run one model-requested web call, injecting the result (or the failure,
 *  so the model can adapt) as a dimmed "Web" message. */
async function runWebCall(conv, call) {
  let note;
  try {
    note = await requestWebTool(call);
  } catch (e) {
    note = `[Web request failed: ${e.message}] Answer without the web, ` +
           "and say that web access did not work.";
    toast("Web request failed: " + e.message, true);
  }
  conv.messages.push({ role: "user", content: note, web: true });
  saveConversations(conv);
  renderChat();
}

/* ---- voice: mic (Whisper STT) + read-aloud (browser TTS) ---- */

const voice = { rec: null, chunks: [], available: true, reason: "",
                modelCached: true, model: "" };

/** Grey out the mic up front when the server lacks the [voice] extra,
 *  instead of letting the user record and only then failing. */
async function refreshVoiceStatus() {
  try {
    const r = await fetch("/api/voice/status", { headers: authHeaders() });
    if (!r.ok) return;   // old server without the endpoint - leave enabled
    const data = await r.json();
    voice.available = data.available;
    voice.reason = data.reason || "";
    voice.modelCached = data.model_cached !== false;
    voice.model = data.model || "";
    const btn = $("chat-mic");
    btn.classList.toggle("unavailable", !data.available);
    if (!data.available) btn.title = data.reason;
  } catch (e) { /* server unreachable - status refreshes on next load */ }
}

function blobToB64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(new Error("could not read recording"));
    reader.readAsDataURL(blob);
  });
}

async function toggleMic() {
  const btn = $("chat-mic");
  if (voice.rec) {           // second click stops and transcribes
    voice.rec.stop();
    return;
  }
  if (!voice.available) {
    toast(voice.reason || "Speech-to-text is not installed on the server", true);
    return;
  }
  if (!voice.modelCached) {
    // Transcription is fully local, but the FIRST use fetches the Whisper
    // model from HuggingFace - make that one network access explicit.
    if (!confirm(
        `First use downloads the Whisper "${voice.model}" speech model ` +
        "from HuggingFace (one-time). Transcription itself runs fully " +
        "offline afterwards. Download now?")) return;
    voice.modelCached = true;   // consent given - don't re-ask this session
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    toast("This browser does not support audio recording", true);
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    toast("Microphone unavailable: " + e.message, true);
    return;
  }
  voice.chunks = [];
  voice.rec = new MediaRecorder(stream);
  voice.rec.ondataavailable = (e) => { if (e.data.size) voice.chunks.push(e.data); };
  voice.rec.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    btn.classList.remove("recording");
    const blob = new Blob(voice.chunks, { type: voice.rec.mimeType || "audio/webm" });
    voice.rec = null;
    btn.disabled = true;
    btn.textContent = "…";
    try {
      const r = await fetch("/api/voice/transcribe", {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ audio_b64: await blobToB64(blob) }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || r.statusText);
      const input = $("chat-input");
      input.value = (input.value ? input.value.trimEnd() + " " : "") + data.text;
      autoGrow(input);
      input.focus();
    } catch (e) {
      toast("Transcription failed: " + e.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "🎤";
    }
  };
  voice.rec.start();
  btn.classList.add("recording");
  toast("Recording - click 🎤 again to stop");
}

$("chat-mic").onclick = toggleMic;

/* ---- text-to-speech ----
 *  A client plugin (the `tts` plugin) may install a neural provider via
 *  registerTTS(); otherwise we fall back to the browser's built-in offline
 *  voices. The browser fallback can only reach robotic local voices on Windows
 *  (the good Win11 voices are Narrator-only or cloud), which is exactly why the
 *  tts plugin exists. */
let ttsProvider = null;   // {name, voices(), getVoice(), setVoice(id),
                          //  speaking(), ready(), speak(text, opts), stop()}

/** Install (or clear, with null) the active TTS provider, then refresh the
 *  voice picker. Called by a client plugin's register(ctx). */
function registerTTS(provider) {
  ttsProvider = provider;
  populateVoicePicker();
}

/** The browser SpeechSynthesisVoice the user picked for the fallback, if any. */
function selectedBrowserVoice() {
  if (!window.speechSynthesis) return null;
  const want = localStorage.getItem("localm.ttsVoiceBrowser");
  if (!want) return null;
  return speechSynthesis.getVoices().find((v) => v.name === want) || null;
}

/** Read text aloud. With toggle: true (the 🔊 button) a second call stops
 *  instead; auto-speak replaces the current utterance. */
function speak(text, opts = {}) {
  const clean = stripThink(text).replace(/[*_`#>\[\]()]/g, " ").trim();
  if (ttsProvider) {
    if (ttsProvider.speaking()) {
      ttsProvider.stop();
      if (opts.toggle) return;
    }
    if (clean) ttsProvider.speak(clean, opts);
    return;
  }
  if (!window.speechSynthesis) {
    toast("This browser has no speech synthesis", true);
    return;
  }
  if (speechSynthesis.speaking) {
    speechSynthesis.cancel();
    if (opts.toggle) return;
  }
  if (clean) {
    const u = new SpeechSynthesisUtterance(clean);
    const v = selectedBrowserVoice();
    if (v) u.voice = v;
    speechSynthesis.speak(u);
  }
}

/** Fill the voice picker from the active provider (or, with no provider, the
 *  browser's LOCAL voices) and remember the choice. Hidden when there is
 *  nothing to choose. */
function populateVoicePicker() {
  const sel = $("p-voice");
  const row = $("voice-row");
  if (!sel) return;
  let opts = [];
  let current = "";
  if (ttsProvider) {
    opts = ttsProvider.voices();
    current = localStorage.getItem("localm.ttsVoice") || ttsProvider.getVoice();
    if (current) ttsProvider.setVoice(current);
  } else if (window.speechSynthesis) {
    // getVoices() is async-populated; filter to localService so we never offer
    // a cloud voice that would send text off the machine.
    opts = speechSynthesis
      .getVoices()
      .filter((v) => v.localService)
      .map((v) => ({ id: v.name, label: `${v.name} (${v.lang})` }));
    current = localStorage.getItem("localm.ttsVoiceBrowser") || "";
  }
  sel.replaceChildren();
  if (!opts.length) {
    if (row) row.style.display = "none";
    return;
  }
  if (row) row.style.display = "";
  for (const o of opts) {
    const el = document.createElement("option");
    el.value = o.id;
    el.textContent = o.label;
    sel.appendChild(el);
  }
  if (current) sel.value = current;
}

/** Persist the picked voice and apply it to the active provider. */
function onVoicePick() {
  const id = $("p-voice").value;
  if (ttsProvider) {
    ttsProvider.setVoice(id);
    localStorage.setItem("localm.ttsVoice", id);
  } else {
    localStorage.setItem("localm.ttsVoiceBrowser", id);
  }
}

/** Load client-side plugin modules: for each ACTIVE plugin that ships a
 *  client_entry, import it and call register(ctx). Failures are isolated so a
 *  broken plugin module never breaks chat. */
async function loadClientPlugins() {
  let plugins = [];
  try {
    // /api/capabilities (not /api/plugins) so client-entry modules load for a
    // scoped key too, and ONLY the plugins this key's scopes grant are imported.
    const r = await fetch("/api/capabilities", { headers: authHeaders() });
    if (r.ok) plugins = (await r.json()).plugins || [];
  } catch {
    return; // server unreachable; the built-in browser voice still works
  }
  const ctx = { registerTTS, toast, authHeaders, voicesChanged: populateVoicePicker };
  for (const p of plugins) {
    if (!p.active || !p.client_entry) continue;
    const base = p.assets_base || `/plugins/${p.name}`;
    try {
      const mod = await import(`${base}/${p.client_entry}`);
      if (mod && typeof mod.register === "function") await mod.register(ctx);
    } catch (e) {
      console.error(`client plugin ${p.name} failed to load`, e);
    }
  }
}

/* ---- first-party plugin command catalog ---- */
// Map of slash-command verb -> { plugin, active } across the first-party
// catalog, so a command that belongs to a known-but-inactive plugin (e.g.
// /generate-image with the image plugin off) gets a "needs the X plugin" hint
// instead of a confusing 404 or "unknown command". Populated from /api/plugins;
// stays empty (silent, current behaviour) until loaded or if the server is
// unreachable. `suggest` mirrors the suggest_plugins config toggle.
const pluginCommands = { map: {}, suggest: true };

// R50: signal other same-origin tabs that the installed/enabled plugin set
// changed (a new value is required for the storage event to fire, so use the
// clock). The writing tab refreshes itself directly; other tabs react to the
// storage event wired near the focus listener.
function bumpPluginsRev() {
  try { localStorage.setItem("localm.pluginsRev", String(Date.now())); }
  catch (e) { /* storage blocked / full - cross-tab sync degrades to focus only */ }
}
window.bumpPluginsRev = bumpPluginsRev;

async function refreshPluginCommands() {
  try {
    // /api/capabilities returns ONLY what THIS key may use (scope-filtered) and
    // the core-tab flags, so the nav shows just the usable tabs without needing
    // plugins:read. The Plugins management page still uses /api/plugins.
    const r = await fetch("/api/capabilities", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    const map = {};
    for (const p of data.plugins || []) {
      for (const c of p.commands || []) {
        map[c] = { plugin: p.name, active: !!p.active };
      }
    }
    pluginCommands.map = map;
    pluginCommands.suggest = data.suggest_plugins !== false;
    pluginState = data.plugins || [];
    if (data.core) applyCoreTabVisibility(data.core);
    renderNav();
  } catch { /* server unreachable; fall back to plain unknown-command */ }
}

/** A "/cmd needs the X plugin" hint when *cmd* belongs to a known first-party
 *  plugin that is not active, else null (handle it normally). */
function pluginSuggestion(cmd) {
  if (!pluginCommands.suggest) return null;
  const hit = pluginCommands.map[cmd];
  if (!hit || hit.active) return null;
  return `/${cmd} needs the ${hit.plugin} plugin - install or enable it on the Plugins page.`;
}

/* ---- dynamic nav rail (tabs follow the active plugins) ---- */
// The most recent /api/plugins entries, refreshed alongside the command cache.
let pluginState = [];

// Each plugin's manifest icon name -> the nav emoji. Kernel buttons keep their
// own emoji in index.html; "studio" is the media parent.
const NAV_ICON = { chat: "💬", code: "⚙️", image: "🖼️", music: "🎵", video: "🎬", book: "📚", clock: "⏰" };
// Canonical rail order of first-party plugin tabs (stable so the rail does not
// reshuffle as plugins toggle); "studio" is the media slot (image/music/video).
const NAV_TAB_ORDER = ["coder", "studio", "knowledge"];

function _navButton(id, icon, label, onClick, cls) {
  const b = el("button", cls || "", `${icon} ${label}`);
  b.id = id;
  b.onclick = onClick;
  return b;
}

/** Rebuild the plugin portion of the nav rail from the active-with-a-tab
 *  plugins, then re-derive VIEWS and re-assert the active tab. */
/* Show only the core tabs the current key's scopes grant. chat is the baseline
 * anchor and is NEVER hidden (chatting needs no scope); models/plugins/settings
 * render only when the key holds models:read / plugins:read / config:read. A tab
 * the key lacks is not shown at all (no show-then-"no access"). Driven by
 * /api/capabilities .core. If the active view becomes hidden (e.g. a remembered
 * Settings tab on a key without config:read), fall back to chat so the user is
 * never parked on an inaccessible view. */
function applyCoreTabVisibility(core) {
  if (!core) return;
  const activeView = (document.querySelector(".view.active") || {}).id;
  let activeHidden = false;
  for (const view of ["models", "plugins", "settings"]) {
    const nav = $("nav-" + view);
    if (!nav) continue;
    const show = core[view] !== false;        // default to showing if unknown
    nav.style.display = show ? "" : "none";
    if (!show && activeView === "view-" + view) activeHidden = true;
  }
  if (activeHidden) showView("chat");
}
window.applyCoreTabVisibility = applyCoreTabVisibility;

function renderNav() {
  const slot = $("nav-plugin-slot");
  if (!slot) return;
  slot.replaceChildren();
  // chat owns a tab but is the static kernel anchor; never render it (or any
  // plugin claiming a kernel view) as a dynamic tab.
  const active = pluginState.filter(
    (p) => p.active && p.tab && !CORE_VIEWS.includes(p.tab));
  const studio = active.filter((p) => p.group === "studio");
  const flat = active.filter((p) => p.group !== "studio");
  const byTab = {};
  for (const p of flat) byTab[p.tab] = p;

  const renderFlat = (p) => slot.appendChild(_navButton(
    "nav-" + p.tab, NAV_ICON[p.icon] || "•", p.label || p.name, () => showView(p.tab)));

  const done = new Set();
  for (const key of NAV_TAB_ORDER) {
    if (key === "studio") { renderStudioGroup(slot, studio); continue; }
    if (byTab[key]) { renderFlat(byTab[key]); done.add(key); }
  }
  // any other active plugin tab not in the canonical order, in catalog order.
  // Iterate the tab-deduped map (not raw `flat`) so two plugins claiming the
  // same tab cannot emit duplicate id="nav-<tab>" nodes (LBUG-1).
  for (const p of Object.values(byTab)) if (!done.has(p.tab)) renderFlat(p);

  rebuildViews();
  reconcileActiveView();
}

/** Studio hybrid grouping: nothing for 0 media plugins, a single flat tab for
 *  exactly 1, and one stable-position "Studio" parent expanding to the active
 *  children for 2+. */
function renderStudioGroup(slot, studio) {
  if (studio.length === 0) return;
  if (studio.length === 1) {
    const p = studio[0];
    slot.appendChild(_navButton(
      "nav-" + p.tab, NAV_ICON[p.icon] || "•", p.label || p.name, () => showView(p.tab)));
    return;
  }
  const order = ["images", "music", "video"];
  const known = order.map((t) => studio.find((p) => p.tab === t)).filter(Boolean);
  // include any studio plugin with a non-canonical tab so a third-party media
  // plugin is not counted toward the group yet silently never rendered (LGAP-1)
  const extra = studio.filter((p) => !order.includes(p.tab));
  const kids = [...known, ...extra];
  const activeView = (document.querySelector(".view.active") || { id: "view-chat" })
    .id.replace("view-", "");
  const hasActiveKid = kids.some((p) => p.tab === activeView);
  const open = hasActiveKid ||
    (chat.privacy ? true : localStorage.getItem("localm.studioOpen") !== "0");

  const wrap = el("div", "nav-group");
  const parent = el("button", "nav-group-parent" + (open ? " open" : ""), "🎨 Studio");
  const children = el("div", "nav-children");
  children.style.display = open ? "block" : "none";
  parent.onclick = () => {
    const nowOpen = children.style.display === "none";
    children.style.display = nowOpen ? "block" : "none";
    parent.classList.toggle("open", nowOpen);
    if (!chat.privacy) localStorage.setItem("localm.studioOpen", nowOpen ? "1" : "0");
  };
  for (const p of kids) {
    children.appendChild(_navButton(
      "nav-" + p.tab, NAV_ICON[p.icon] || "•", p.label || p.name,
      () => showView(p.tab), "nav-child"));
  }
  wrap.appendChild(parent);
  wrap.appendChild(children);
  slot.appendChild(wrap);
}

function rebuildViews() {
  const tabs = pluginState
    .filter((p) => p.active && p.tab && !CORE_VIEWS.includes(p.tab))
    .map((p) => p.tab);
  VIEWS = ["chat", ...tabs, "models", "plugins", "settings"];
}

// After the rail is rebuilt, keep the shown view reachable: if its plugin was
// just disabled/uninstalled, fall back to chat; otherwise re-assert the active
// highlight on the (possibly freshly created) nav button.
function reconcileActiveView() {
  const cur = document.querySelector(".view.active");
  const name = cur ? cur.id.replace("view-", "") : "chat";
  const ok = CORE_VIEWS.includes(name) ||
             pluginState.some((p) => p.active && p.tab === name);
  if (ok) {
    // The shown view is still valid: just re-assert the highlight on the
    // (possibly freshly-created) nav button. Do NOT call showView(name) here -
    // it re-fires onViewShown, and for chat/coder onViewShown re-enters
    // refreshPluginCommands -> renderNav -> reconcileActiveView, a runaway
    // /api/plugins loop (and it double-renders whatever page is open).
    _applyActiveClasses(name);
  } else {
    // The shown view's plugin was uninstalled - fall back to chat (a real
    // view switch, page refresh included).
    showView("chat");
  }
}

/* ---- assistant memory ---- */

const memory = { text: "", writable: false };

async function refreshMemory() {
  try {
    const r = await fetch("/api/memory", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    memory.text = data.text || "";
    memory.writable = !!data.writable;
  } catch (e) { /* server unreachable */ }
}

async function rememberFact(fact) {
  if (!fact) { toast("Usage: /remember <fact>", true); return; }
  try {
    const r = await fetch("/api/memory/append", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ text: fact }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    await refreshMemory();
    toast("Remembered ✓");
  } catch (e) {
    toast("Could not save: " + e.message, true);
  }
}

function openMemoryModal() {
  openModal("Memory - what the model knows about you", (body) => {
    body.appendChild(el("div", "sub", memory.writable
      ? "Injected into the system prompt while the 🧠 toggle is on. " +
        "Edit freely - it's a plain markdown file in the localm data directory."
      : "Read-only: privacy mode blocks memory writes (no new traces). " +
        "Existing memory is still injected while the 🧠 toggle is on."));
    const ta = document.createElement("textarea");
    ta.value = memory.text;
    ta.rows = 14;
    ta.style.width = "100%";
    ta.readOnly = !memory.writable;
    body.appendChild(ta);
    if (memory.writable) {
      const save = el("button", "btn-primary", "Save");
      save.style.marginTop = "10px";
      save.onclick = async () => {
        try {
          const r = await fetch("/api/memory", {
            method: "PUT", headers: authHeaders(),
            body: JSON.stringify({ text: ta.value }),
          });
          const data = await r.json();
          if (!r.ok) throw new Error(data.detail || r.statusText);
          await refreshMemory();
          toast("Memory saved");
          $("modal").style.display = "none";
        } catch (e) {
          toast("Save failed: " + e.message, true);
        }
      };
      body.appendChild(save);
    }
  });
}

/* ---- prompt library (personas) ---- */

const PERSONA_PARAM_IDS = {
  temperature: "p-temperature",
  top_p: "p-top-p",
  top_k: "p-top-k",
  repeat_penalty: "p-repeat-penalty",
  max_tokens: "p-max-tokens",
};

let personaCache = [];

async function refreshPersonas() {
  try {
    const r = await fetch("/api/prompts", { headers: authHeaders() });
    if (!r.ok) return;
    personaCache = (await r.json()).prompts;
    const sel = $("p-persona");
    const current = sel.value;
    sel.replaceChildren();
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "(none)";
    sel.appendChild(none);
    for (const p of personaCache) {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.name;
      sel.appendChild(opt);
    }
    if ([...sel.options].some((o) => o.value === current)) sel.value = current;
  } catch (e) { /* server unreachable */ }
}

function applyPersona(name) {
  const p = personaCache.find((x) => x.name === name);
  if (!p) { toast("No such persona: " + name, true); return false; }
  $("p-system").value = p.system || "";
  for (const [key, id] of Object.entries(PERSONA_PARAM_IDS)) {
    $(id).value = p.params?.[key] ?? "";
  }
  $("p-persona").value = name;
  toast(`Persona '${name}' applied`);
  return true;
}

$("p-persona").onchange = () => {
  const name = $("p-persona").value;
  if (name) applyPersona(name);
};

$("persona-save").onclick = async () => {
  const name = prompt("Persona name:", $("p-persona").value || "");
  if (!name || !name.trim()) return;
  const params = {};
  for (const [key, id] of Object.entries(PERSONA_PARAM_IDS)) {
    const v = $(id).value.trim();
    if (v !== "" && !Number.isNaN(Number(v))) params[key] = Number(v);
  }
  try {
    const r = await fetch("/api/prompts/" + encodeURIComponent(name.trim()), {
      method: "PUT", headers: authHeaders(),
      body: JSON.stringify({ system: $("p-system").value, params }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    await refreshPersonas();
    $("p-persona").value = name.trim();
    toast(`Persona '${name.trim()}' saved`);
  } catch (e) {
    toast("Save failed: " + e.message, true);
  }
};

$("persona-delete").onclick = async () => {
  const name = $("p-persona").value;
  if (!name) { toast("Select a persona first", true); return; }
  if (!confirm(`Delete persona '${name}'? The drawer values stay as they are.`)) return;
  try {
    const r = await fetch("/api/prompts/" + encodeURIComponent(name), {
      method: "DELETE", headers: authHeaders() });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    await refreshPersonas();
    $("p-persona").value = "";
    toast(`Persona '${name}' deleted`);
  } catch (e) {
    toast("Delete failed: " + e.message, true);
  }
};

/* sending */

async function runCompletion(conv, webDepth = 0, web = null) {
  // R36: per-send web state. `seen` dedupes already-issued queries so the model
  // cannot loop on the same search; `ask` caches the net policy so a transient
  // /v1/config blip mid-loop cannot silently flip approval off; `forced` ensures
  // we only inject the "limit reached, answer now" nudge once per send.
  if (!web) web = { seen: new Set(), ask: null, forced: false };
  await maybeCompactConversation(conv);
  const params = chatParams();
  const webEnabled = $("p-web").checked;
  const messages = [];
  let sysText = params.system || "";
  if ($("p-memory").checked && memory.text.trim()) {
    sysText = (sysText ? sysText + "\n\n" : "") +
      "Long-term memory - things to remember about the user:\n" +
      memory.text.trim();
  }
  // Always give the model an honesty floor:
  //  - web ON  -> teach the tools so it searches instead of guessing.
  //  - results just injected (explicit /web, toggle off) -> tell it to use and
  //    cite them; the offline-denial floor would contradict results in hand.
  //  - web OFF, no results -> tell it plainly it is offline and must not
  //    fabricate current facts or claim it looked anything up. This is what
  //    stops the model hallucinating instead of admitting it cannot reach the net.
  let webFloor;
  if (webEnabled) webFloor = WEB_TOOL_PROMPT;
  else if (lastTurnHasWebResults(conv)) webFloor = WEB_GROUNDED_PROMPT;
  else webFloor = NO_WEB_PROMPT;
  sysText = (sysText ? sysText + "\n\n" : "") + webFloor;
  if (sysText) messages.push({ role: "system", content: sysText });
  // Server-generated images (/api/ URLs from /generate-image) must not be sent
  // to the model as image parts - replace those messages with a text note.
  const mapped = conv.messages.map((m) => {
    if (Array.isArray(m.content) &&
        m.content.some((p) => p.type === "image_url" &&
                              p.image_url?.url?.startsWith("/api/"))) {
      return { role: m.role,
               content: msgText(m) + "\n[An image was generated and shown to the user.]" };
    }
    // Reasoning blocks are display-only - never resend them as context.
    if (m.role === "assistant" && typeof m.content === "string") {
      return { role: m.role, content: stripThink(m.content) };
    }
    return { role: m.role, content: m.content };
  });
  // Attached documents and knowledge excerpts are stored as separate user
  // rows; some chat templates require strict user/assistant alternation,
  // so consecutive plain-text same-role messages are merged before sending.
  for (const m of mapped) {
    const prev = messages[messages.length - 1];
    if (prev && prev.role === m.role && prev.role !== "system" &&
        typeof prev.content === "string" && typeof m.content === "string") {
      prev.content += "\n\n" + m.content;
    } else {
      messages.push({ role: m.role, content: m.content });
    }
  }

  const body = { model: modelSelect.value, messages, stream: true };
  for (const k of ["temperature", "top_p", "top_k", "repeat_penalty",
                   "max_tokens", "seed"]) {
    if (params[k] !== null && !Number.isNaN(params[k])) body[k] = params[k];
  }
  if (params.grammar) body.grammar = params.grammar;

  const box = $("chat-messages");
  const { body: liveBody } = addMessageRow(box, "assistant", "");
  chat.stick = true;   // R31: a fresh send re-arms autoscroll (follow the reply)
  box.scrollTop = box.scrollHeight;

  const sendBtn = $("chat-send");
  sendBtn.classList.add("stop");
  sendBtn.textContent = "■";
  chat.abort = new AbortController();

  // VIS-1: did this request carry a user-attached image? If a text-only model
  // rejects it (400), we must drop the image so the chat is not wedged.
  const sentImage = messages.some((m) => Array.isArray(m.content) &&
    m.content.some((p) => p.type === "image_url"));

  let full = "";
  let reasoning = "";   // H4: <think> reasoning now streams in delta.reasoning_content
  let usage = null;
  let finishReason = null;
  let aborted = false;
  let visionRejected = false;
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(body),
      signal: chat.abort.signal,
    });
    if (!r.ok) {
      const detail = await r.text();
      const err = new Error(`${r.status}: ${detail.slice(0, 300)}`);
      err.status = r.status;   // so the catch can recover an image-reject 400
      throw err;
    }
    await readSSE(r, (payload) => {
      if (payload === "[DONE]") return;
      let chunk;
      try { chunk = JSON.parse(payload); } catch { return; }
      if (chunk.usage) usage = chunk.usage;
      if (chunk.choices?.[0]?.finish_reason) finishReason = chunk.choices[0].finish_reason;
      const d = chunk.choices?.[0]?.delta || {};
      const cDelta = d.content || "";
      const rDelta = d.reasoning_content || "";   // H4: reasoning streamed apart
      if (cDelta || rDelta) {
        full += cDelta;
        reasoning += rDelta;
        // Rebuild <think> from the reasoning stream so splitThink renders the
        // collapsible block exactly as before. Back-compat: an older server that
        // still inlines <think> in content also renders (reasoning stays empty).
        renderMarkdown(liveBody,
          reasoning ? "<think>\n" + reasoning + "\n</think>\n" + full : full);
        // R31: follow the stream ONLY while the user is at the bottom. chat.stick
        // is latched by the scroll listener, so scrolling up reliably pauses
        // autoscroll (recomputing nearBottom per token fought the user instead).
        if (chat.stick) box.scrollTop = box.scrollHeight;
      }
    });
  } catch (e) {
    if (e.name === "AbortError") {
      aborted = true;
    } else if (sentImage && e.status === 400 && !full.trim()) {
      // VIS-1: a text-only model rejected the image. Drop it from history so the
      // next turn is text-only and the chat stays usable, instead of re-sending
      // the image every turn (which 400s forever -> all-blank assistant replies).
      visionRejected = true;
      const n = stripUserImages(conv);
      saveConversations(conv);
      toast(n
        ? "This model cannot read images - removed the image. You can keep " +
          "chatting (text only)."
        : "Chat request failed: " + e.message, true);
    } else {
      renderMarkdown(liveBody, full + "\n\n*[error: " + e.message + "]*");
      toast("Chat request failed: " + e.message, true);
    }
  } finally {
    chat.abort = null;
    sendBtn.classList.remove("stop");
    sendBtn.textContent = "➤";
  }

  // User pressed Stop: leave the partial text on screen but do NOT persist it,
  // read it aloud, or fire the web loop / recurse on a partial reply (BUG-13).
  // U-STOP: make the stop unmistakable - mark the partial as stopped and halt any
  // speech already playing - so a stopped reply is never silently treated as live.
  if (aborted) {
    renderMarkdown(liveBody,
      (reasoning ? "<think>\n" + reasoning + "\n</think>\n" + full : full) +
      (full || reasoning ? "\n\n" : "") + "*[stopped]*");
    try { window.speechSynthesis && window.speechSynthesis.cancel(); } catch (e) { /* no TTS */ }
    return;
  }

  // VIS-1: a vision reject (or any failure that streamed nothing) must NOT
  // persist an empty assistant turn - a blank reply saved every send is the
  // "every turn after the image is empty" wedge. Re-render from real history
  // (which now has the image stripped) and stop here so the chat recovers.
  if (visionRejected || (!full.trim() && !reasoning.trim())) {
    renderChat();
    return;
  }

  // Persist content with <think> rebuilt (same shape as before this change), so
  // reload + splitThink re-render the collapsible block and TTS/visibleText are
  // unaffected.
  const reply = {
    role: "assistant",
    content: reasoning ? "<think>\n" + reasoning + "\n</think>\n" + full : full,
    // NEW-1: record which model produced this turn so the transcript can show a
    // divider when the active model changes between turns (model-switch-indication).
    model: modelSelect.value || undefined,
  };
  if (finishReason === "length") {
    // The reply was cut by the max-tokens budget, not finished by the model.
    reply.truncated = true;
    toast("Reply hit the max-tokens limit - raise “Max tokens” in ⚙ parameters, or reply “continue”", true);
  }
  conv.messages.push(reply);
  saveConversations(conv);
  if (usage) {
    const bits = [`${usage.total_tokens} tok`];
    if (usage.ttft_ms != null) bits.push(`TTFT ${usage.ttft_ms} ms`);
    if (usage.tokens_per_sec != null) bits.push(`${usage.tokens_per_sec} tok/s`);
    $("chat-usage").textContent = bits.join(" · ");
  }
  renderChat();

  // Web-access loop: when the model requested a search/page and the toggle
  // is on, run it and let the model continue - bounded rounds per send.
  const canWeb = webEnabled && webDepth < WEB_MAX_ROUNDS;
  const nextCall = canWeb ? parseWebCall(full) : null;
  if (nextCall) {
    // R36: dedupe - the model re-issuing a search it already ran this send is the
    // loop. Do not repeat it; tell the model the results are already in hand and
    // to answer from them, and end the web rounds for this send.
    const key = nextCall.name + ":" + String(
      (nextCall.args && (nextCall.args.query || nextCall.args.url)) || "")
      .trim().toLowerCase();
    if (web.seen.has(key)) {
      conv.messages.push({
        role: "user", web: true,
        content:
          "[duplicate web request] You already ran that exact search this turn; " +
          "its results are above. Do not search again - answer now from those " +
          "results and cite the sources, or say plainly if they are insufficient.",
      });
      saveConversations(conv);
      renderChat();
      await runCompletion(conv, WEB_MAX_ROUNDS, web);   // stop web; force an answer
      return;
    }
    // net_mode=ask: approve each MODEL-INITIATED request before it runs (WEB-ask).
    // The explicit /web command is direct consent and is NOT routed through here.
    // Cache the policy for this send so a mid-loop /v1/config blip cannot flip it.
    if (web.ask === null) web.ask = await webModeIsAsk();
    const approved = web.ask ? await confirmWebRequest(nextCall) : true;
    if (!approved) {
      conv.messages.push({
        role: "user", web: true,
        content:
          "[web access denied] The user declined this web request. Do not claim " +
          "you searched or browsed; answer from what you already know, or say " +
          "plainly that you could not look it up.",
      });
      saveConversations(conv);
      renderChat();
      await runCompletion(conv, WEB_MAX_ROUNDS, web);   // no further web rounds this send
      return;
    }
    web.seen.add(key);
    await runWebCall(conv, nextCall);
    await runCompletion(conv, webDepth + 1, web);
  } else if (canWeb && looksLikeWebToolAttempt(full)) {
    // The model tried to call a web tool but emitted a block we could not
    // parse. Re-prompt for the exact format instead of letting the un-grounded
    // reply stand (it would otherwise read as a confident, un-searched answer).
    conv.messages.push({
      role: "user", web: true,
      content:
        "[tool-call format] That looked like a web tool call, but I could not " +
        "parse it. Re-emit it EXACTLY like this and nothing else:\n" +
        '<tool_call>{"name": "web_search", "args": {"query": "..."}}</tool_call>\n' +
        "If you did not mean to search, answer in plain text and do not claim " +
        "you accessed the web.",
    });
    saveConversations(conv);
    renderChat();
    await runCompletion(conv, webDepth + 1, web);
  } else if (webEnabled && webDepth === WEB_MAX_ROUNDS && !web.forced &&
             (parseWebCall(full) || looksLikeWebToolAttempt(full))) {
    // R36: web rounds are used up but the model is STILL trying to search instead
    // of answering (the "never synthesizes an answer" symptom). Force exactly one
    // synthesizing turn from the results already gathered, then accept its answer.
    web.forced = true;
    conv.messages.push({
      role: "user", web: true,
      content:
        "[web search limit reached] You have used the maximum web lookups for " +
        "this turn. Stop searching and answer the question now using the results " +
        "already provided above, citing the sources; if they are insufficient, " +
        "say so plainly.",
    });
    saveConversations(conv);
    renderChat();
    await runCompletion(conv, WEB_MAX_ROUNDS + 1, web);
    return;
  } else if ($("p-speak").checked && full) {
    speak(full);   // read the finished reply aloud (offline browser voices)
  }
}

/** Query the selected knowledge collection and inject cited excerpts. */
async function retrieveKnowledge(conv, query) {
  const kb = $("p-kb").value;
  if (!kb || !query) return;
  try {
    const r = await fetch(
      `/api/rag/collections/${encodeURIComponent(kb)}/query`, {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ query, k: 4 }),
      });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    if (!data.hits.length) return;
    const basename = (p) => p.split(/[\\/]/).pop();
    const lines = data.hits.map((h, i) =>
      `[${i + 1}] ${basename(h.source)}:${h.pos}\n${h.text.slice(0, 900)}`);
    conv.messages.push({
      role: "user", tag: "kb",
      content:
        `[Excerpts from the "${kb}" collection relevant to: ` +
        `${query.slice(0, 120)}]\n\n` + lines.join("\n\n") +
        "\n\nUse these excerpts where relevant and cite them as [1], [2]… " +
        "If they don't answer the question, say so before answering from " +
        "general knowledge.",
    });
    saveConversations(conv);
    renderChat();
  } catch (e) {
    toast("Knowledge retrieval failed: " + e.message, true);
  }
}

/** Populate the params-drawer knowledge selector. pages.js calls this after
 *  collections change on the Knowledge page. */
async function refreshKbSelect() {
  try {
    const r = await fetch("/api/rag/collections", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    const sel = $("p-kb");
    const current = sel.value;
    sel.replaceChildren();
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "(none)";
    sel.appendChild(none);
    for (const c of data.collections) {
      const opt = document.createElement("option");
      opt.value = c.name;
      opt.textContent = `${c.name} (${c.n_docs} docs, ${c.n_chunks} chunks)`;
      sel.appendChild(opt);
    }
    if ([...sel.options].some((o) => o.value === current)) sel.value = current;
  } catch (e) { /* server unreachable - selector stays as-is */ }
}

async function sendChat() {
  const input = $("chat-input");
  const text = input.value.trim();
  if (!text && chat.attachments.length === 0 && chat.docs.length === 0) return;
  if (chat.abort) {
    // A reply is still streaming. Tell the user how to act instead of silently
    // swallowing the send (the send button is a Stop control while streaming).
    toast("Reply still streaming - press the stop button to interrupt", true);
    return;
  }

  if (text.startsWith("/")) {
    input.value = "";
    autoGrow(input);
    handleSlashSubmit(text, execChatCommand);
    return;
  }

  if (!currentConv()) newConversation();
  const conv = currentConv();
  const isFirstMessage = conv.messages.length === 0;

  // Attached documents come first so the model reads them before the question
  for (const doc of chat.docs) {
    conv.messages.push({
      role: "user", tag: "doc",
      content: `[Attached document: ${doc.name}` +
        (doc.truncated ? " (truncated)" : "") + `]\n${doc.text}`,
    });
  }
  chat.docs = [];

  let content;
  if (chat.attachments.length) {
    content = [{ type: "text", text }];
    for (const att of chat.attachments) {
      content.push({ type: "image_url", image_url: { url: att.dataUri } });
    }
  } else {
    content = text || "Please read the attached document(s).";
  }
  conv.messages.push({ role: "user", content });
  chat.attachments = [];
  renderAttachChips();

  if (isFirstMessage) {
    conv.title = text.slice(0, 42) + (text.length > 42 ? "…" : "") || "Document chat";
    renderConvList();
  }
  saveConversations(conv);   // user message persists even if the reply dies
  input.value = "";
  autoGrow(input);
  renderChat();
  await retrieveKnowledge(conv, text);
  await runCompletion(conv);
}

function exportConversation() {
  const conv = currentConv();
  if (!conv || !conv.messages.length) { toast("Nothing to export", true); return; }
  const lines = [`# ${conv.title}`, ""];
  for (const m of conv.messages) {
    lines.push(`**${m.role === "user" ? "You" : "Model"}:**`, "", msgText(m), "");
    if (msgImages(m).length) lines.push(`*[${msgImages(m).length} image(s) attached]*`, "");
  }
  const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = conv.title.replace(/[^\w\- ]+/g, "").trim().replace(/\s+/g, "_") + ".md";
  a.click();
  URL.revokeObjectURL(a.href);
}

$("chat-send").onclick = () => {
  if (chat.abort) { chat.abort.abort(); return; }
  sendChat();
};
/** Enter sends, Shift+Enter inserts a newline, Ctrl/Cmd+Enter also sends.
 *  Skipped while an IME composition or the slash-command menu is active
 *  (the menu's own keydown handler picks the highlighted command). */
function composerEnterToSend(e, send) {
  if (e.key !== "Enter" || e.isComposing) return;
  if (e.shiftKey) return;   // newline - the textarea's default behaviour
  const menu = e.target.closest(".composer-wrap")?.querySelector(".slash-menu");
  if (menu && menu.style.display !== "none") return;
  e.preventDefault();
  send();
}

$("chat-input").addEventListener("keydown", (e) => composerEnterToSend(e, sendChat));
$("chat-input").addEventListener("input", (e) => autoGrow(e.target));
// R31: latch autoscroll on the user's actual scroll position. Scrolling up sets
// chat.stick=false (the streaming loop then leaves the viewport alone); returning
// to the bottom re-arms it. A programmatic scroll-to-bottom lands near the bottom
// and so keeps it armed, so this never fights itself.
$("chat-messages").addEventListener("scroll", () => {
  chat.stick = nearBottom($("chat-messages"));
});
// R34: persist the Web-access and Speak-aloud toggles so they survive a reload
// (privacy mode leaves no trace). hydrateChatToggles restores them on boot.
$("p-web").addEventListener("change", () => {
  if (!chat.privacy) { try { localStorage.setItem("localm.webAccess", $("p-web").checked ? "1" : "0"); } catch (e) { /* storage full/blocked */ } }
});
$("p-speak").addEventListener("change", () => {
  if (!chat.privacy) { try { localStorage.setItem("localm.speakAloud", $("p-speak").checked ? "1" : "0"); } catch (e) { /* storage full/blocked */ } }
});
$("toggle-params").onclick = () => $("params").classList.toggle("open");
$("export-conv").onclick = exportConversation;
$("compact-conv").onclick = async () => {
  const conv = currentConv();
  if (!conv || conv.messages.length <= COMPACT_KEEP) {
    toast("Nothing to compact yet", true);
    return;
  }
  if (chat.abort) { toast("Wait for the current reply to finish", true); return; }
  await compactConversation(conv);
};
$("new-conv").onclick = () => { newConversation(); showView("chat"); };
// Mobile top-bar new-chat button mirrors the sidebar +; also closes the drawer
// if it happened to be open. Guarded for the headless/jsdom DOM.
if ($("mtb-new")) {
  $("mtb-new").onclick = () => { newConversation(); showView("chat"); closeNav(); };
}

/* ================================================================ */
/*  Coder - multi-session                                            */
/* ================================================================ */

const coder = {
  sessions: new Map(),   // id → {info, feedEl, busy, liveBody, liveText, pendingCards, gen}
  activeId: null,
  lastActiveId: null,    // session to return to when leaving setup mode
  docs: [],              // file attachments: {name, text, chars, truncated}
};

function activeSession() {
  return coder.activeId ? coder.sessions.get(coder.activeId) : null;
}

function sessionLabel(info) {
  const dir = info.cwd.split(/[\\/]/).filter(Boolean).pop() || info.cwd;
  return `${dir} (${info.id.slice(0, 6)})`;
}

function renderSessionSelect() {
  const sel = $("session-select");
  sel.replaceChildren();
  // In setup mode (no active session) a placeholder holds the selection, so
  // picking any real session fires onchange - even when only one exists.
  if (!coder.activeId && coder.sessions.size) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(new session)";
    opt.selected = true;
    sel.appendChild(opt);
  }
  for (const [id, s] of coder.sessions) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = sessionLabel(s.info) + (s.busy ? " ⏳" : "");
    if (id === coder.activeId) opt.selected = true;
    sel.appendChild(opt);
  }
  renderCoderSessionList();   // R17: keep the right-side rail in lockstep with the dropdown
}

// R17: the coder's right-side open-sessions rail (mirrors the chat conversation
// list). The #session-select dropdown stays as the mobile fallback.
function renderCoderSessionList() {
  const list = $("coder-session-list");
  if (!list) return;
  list.replaceChildren();
  if (!coder.sessions.size) {
    list.appendChild(el("div", "coder-session-empty", "No open sessions"));
    return;
  }
  for (const [id, s] of coder.sessions) {
    const item = el("div", "coder-session-item" + (id === coder.activeId ? " active" : ""));
    item.appendChild(el("span", "title", sessionLabel(s.info)));
    if (s.busy) item.appendChild(el("span", "badge", "⏳"));
    item.onclick = () => activateSession(id);
    list.appendChild(item);
  }
}

function showCoderUI(hasSession) {
  $("coder-setup").style.display = hasSession ? "none" : "block";
  $("coder-composer").style.display = hasSession ? "block" : "none";
  // Keep the bar while other sessions exist so they stay reachable
  $("coder-bar").classList.toggle("open", hasSession || coder.sessions.size > 0);
  if (!hasSession) {
    // Setup mode: park every session feed and clear the session labels -
    // the form must not render on top of a previous session's transcript.
    // Remember where we came from so "back to session" can return there.
    if (coder.activeId && coder.sessions.has(coder.activeId)) {
      coder.lastActiveId = coder.activeId;
    }
    for (const [, s] of coder.sessions) s.feedEl.classList.remove("active");
    coder.activeId = null;
    $("coder-cwd").textContent = "";
    $("coder-state").textContent = "";
    $("coder-usage").textContent = "";
    renderSessionSelect();
    refreshResumable();   // reveal "Continue last session" if the cwd has one (CODER-2)
  }
  $("setup-cancel").style.display =
    !hasSession && coder.sessions.size > 0 ? "" : "none";
}

function activateSession(id) {
  coder.activeId = id;
  for (const [sid, s] of coder.sessions) {
    s.feedEl.classList.toggle("active", sid === id);
  }
  const s = coder.sessions.get(id);
  if (s) {
    $("coder-cwd").textContent = s.info.cwd;
    $("coder-state").textContent = s.busy ? "working…" : "idle";
    $("coder-usage").textContent = s.info.total_tokens
      ? `${s.info.total_tokens} tok · turn ${s.info.turns}` : "";
  }
  renderSessionSelect();
  showCoderUI(!!s);
}

function registerSession(info, { replay }) {
  const feedEl = el("div", "coder-feed");
  $("coder-feeds").appendChild(feedEl);
  const s = {
    info,
    feedEl,
    busy: info.busy || false,
    liveBody: null,
    liveText: "",
    pendingCards: [],
    confirmCards: new Map(),   // confirm_id → {card, title, buttons, tool}
    closed: false,
  };
  coder.sessions.set(info.id, s);
  streamSession(s, replay);
  return s;
}

/* per-session feed helpers */

function feedAppend(s, node) {
  const stick = nearBottom(s.feedEl);
  s.feedEl.appendChild(node);
  if (stick) s.feedEl.scrollTop = s.feedEl.scrollHeight;
}

function startAssistantBlock(s) {
  if (s.liveBody) return;
  const { body } = addMessageRow(s.feedEl, "assistant", "");
  s.liveBody = body;
  s.liveText = "";
}

function flushAssistantBlock(s) {
  // CODER-EMPTY-MODEL: when the assistant turn produced no VISIBLE text (it emitted
  // only a tool call, or its text scrubbed to nothing), drop the empty "Model" row
  // instead of leaving a blank bubble stacked above the tool card.
  if (s.liveBody && !s.liveBody.textContent.trim()) {
    const row = s.liveBody.closest(".msg-row");
    if (row) row.remove();
  }
  s.liveBody = null;
  s.liveText = "";
}

function renderDiff(text) {
  const pre = el("pre", "diff");
  for (const line of (text || "").split("\n")) {
    let cls = "";
    if (line.startsWith("+") && !line.startsWith("+++")) cls = "add";
    else if (line.startsWith("-") && !line.startsWith("---")) cls = "del";
    else if (line.startsWith("@@")) cls = "hunk";
    pre.appendChild(el("span", cls, line + "\n"));
  }
  return pre;
}

/** Args worth showing next to a diff - the bulky text fields ARE the diff. */
function slimArgs(args) {
  const slim = {};
  for (const [k, v] of Object.entries(args || {})) {
    if (k === "content" || k === "old" || k === "new" || k === "diff") continue;
    slim[k] = v;
  }
  return slim;
}

function buildToolCard(ev) {
  const card = el("div", "tool-card");
  card.dataset.t0 = String(Date.now());
  const inner = el("div", "inner");
  const head = el("div", "head");
  head.appendChild(el("span", "name", ev.tool));
  const hintVal = ev.args?.path || ev.args?.command || ev.args?.pattern || ev.args?.url || "";
  head.appendChild(el("span", "hint", String(hintVal).slice(0, 120)));
  head.appendChild(el("span", "state", "…"));
  const body = el("div", "body");
  if (ev.diff) {
    const rest = slimArgs(ev.args);
    if (Object.keys(rest).length) {
      body.appendChild(el("pre", "args", JSON.stringify(rest, null, 2)));
    }
    body.appendChild(renderDiff(ev.diff));
  } else {
    body.textContent = JSON.stringify(ev.args, null, 2);
  }
  head.onclick = () => card.classList.toggle("open");
  inner.appendChild(head);
  inner.appendChild(body);
  card.appendChild(inner);
  return card;
}

/** Mark a confirm card as resolved. Idempotent - fed both by the local
 *  button click and by the confirm_resolved event from the server (which is
 *  also what replay sends for already-answered confirmations). */
function resolveConfirmCard(s, confirmId, approved, timedOut) {
  const entry = s.confirmCards.get(confirmId);
  if (!entry || entry.card.classList.contains("answered")) return;
  entry.card.classList.add("answered");
  entry.title.textContent = timedOut
    ? "✗ Timed out - rejected " + entry.tool
    : (approved ? "✓ Approved " : "✗ Rejected ") + entry.tool;
}

function buildConfirmCard(s, ev) {
  const card = el("div", "confirm-card");
  const inner = el("div", "inner");
  const title = el("div", "title");
  title.appendChild(document.createTextNode("Approve "));
  title.appendChild(el("span", "name", ev.tool));
  title.appendChild(document.createTextNode("?"));
  inner.appendChild(title);
  if (ev.diff) {
    inner.appendChild(renderDiff(ev.diff));
  } else {
    inner.appendChild(el("pre", "diff", JSON.stringify(ev.args, null, 2)));
  }
  const buttons = el("div", "buttons");
  const yes = el("button", "btn-approve", "Approve");
  const no = el("button", "btn-reject", "Reject");
  // "always allow" lives inside .buttons so the answered-state CSS hides it
  const allowCb = document.createElement("input");
  allowCb.type = "checkbox";
  const allowLabel = el("label", "always-allow");
  allowLabel.appendChild(allowCb);
  allowLabel.appendChild(document.createTextNode(
    ` always allow ${ev.tool} this session`));
  const answer = async (approved) => {
    try {
      const r = await fetch(`/api/coder/sessions/${s.info.id}/confirm`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          confirm_id: ev.confirm_id, approved,
          always_allow: approved && allowCb.checked,
        }),
      });
      if (!r.ok) {
        // Already answered elsewhere (another tab) or timed out server-side -
        // the confirm_resolved event carries the real outcome.
        toast("Confirmation was no longer pending", true);
        return;
      }
      if (approved && allowCb.checked) {
        toast(`${ev.tool} auto-approved for the rest of this session`);
      }
      resolveConfirmCard(s, ev.confirm_id, approved, false);
    } catch (e) {
      toast("Failed to answer confirmation: " + e.message, true);
    }
  };
  yes.onclick = () => answer(true);
  no.onclick = () => answer(false);
  buttons.appendChild(yes);
  buttons.appendChild(no);
  buttons.appendChild(allowLabel);
  inner.appendChild(buttons);
  card.appendChild(inner);
  s.confirmCards.set(ev.confirm_id, { card, title, tool: ev.tool });
  return card;
}

function handleCoderEvent(s, ev) {
  // Keep a light event log (no token spam) so "export" can rebuild the
  // session as markdown without another server round-trip.
  if (ev.type !== "token") {
    (s.eventLog = s.eventLog || []).push(ev);
  }
  switch (ev.type) {
    case "token": {
      startAssistantBlock(s);
      s.liveText += ev.text;
      const stick = nearBottom(s.feedEl);
      renderMarkdown(s.liveBody, s.liveText);
      if (stick) s.feedEl.scrollTop = s.feedEl.scrollHeight;
      break;
    }
    case "turn": {
      flushAssistantBlock(s);
      s.busy = true;
      s.info.turns = ev.turn;
      s.info.total_tokens = ev.total_tokens;
      if (s.info.id === coder.activeId) {
        $("coder-state").textContent = "working…";
        const ctx = ev.ctx_ratio ? ` · ctx ${Math.round(ev.ctx_ratio * 100)}%` : "";
        if (ev.total_tokens)
          $("coder-usage").textContent = `${ev.total_tokens} tok · turn ${ev.turn}${ctx}`;
      }
      break;
    }
    case "tool_call": {
      flushAssistantBlock(s);
      const card = buildToolCard(ev);
      feedAppend(s, card);
      s.pendingCards.push(card);
      break;
    }
    case "tool_result": {
      const card = s.pendingCards.shift();
      if (card) {
        const state = card.querySelector(".state");
        const t0 = Number(card.dataset.t0 || 0);
        const took = t0 ? ` · ${((Date.now() - t0) / 1000).toFixed(1)}s` : "";
        state.textContent = (ev.summary || (ev.ok ? "ok" : "failed")) + took;
        state.className = "state " + (ev.ok ? "ok" : "fail");
        if (ev.output && !card.querySelector(".body .diff")) {
          card.querySelector(".body").textContent = ev.output;
        }
      }
      break;
    }
    case "confirm_request": {
      flushAssistantBlock(s);
      feedAppend(s, buildConfirmCard(s, ev));
      break;
    }
    case "confirm_resolved": {
      resolveConfirmCard(s, ev.confirm_id, ev.approved, ev.timed_out);
      break;
    }
    case "user": {
      // replayed user message (emitted client-side on send; replay rebuilds it)
      flushAssistantBlock(s);
      addMessageRow(s.feedEl, "user", ev.text,
        ev.queued ? { cls: "web-note", label: "Queued" } : {});
      break;
    }
    case "info": {
      flushAssistantBlock(s);
      feedAppend(s, el("div", "feed-info", ev.text));
      break;
    }
    case "history": {
      // A recap row replayed when a past session is resumed (CODER-2): plain,
      // role-styled text, no streaming.
      flushAssistantBlock(s);
      addMessageRow(s.feedEl, ev.role === "assistant" ? "assistant" : "user",
                    ev.text || "");
      break;
    }
    case "replay_done": {
      flushAssistantBlock(s);
      s.feedEl.scrollTop = s.feedEl.scrollHeight;
      break;
    }
    case "final": {
      flushAssistantBlock(s);
      s.busy = false;
      s.info.turns = ev.turns;
      s.info.total_tokens = ev.total_tokens;
      if (s.info.id === coder.activeId) $("coder-state").textContent = "idle";
      renderSessionSelect();
      let finalLine = (ev.ok ? "Task finished" : "Task ended") +
        ` - ${ev.turns} turns, ${ev.total_tokens} tokens`;
      if (ev.changed_files?.length) {
        finalLine += ` · ${ev.changed_files.length} file(s) changed (see "files")`;
      }
      feedAppend(s, el("div", "feed-final", finalLine));
      break;
    }
    case "error": {
      flushAssistantBlock(s);
      s.busy = false;
      if (s.info.id === coder.activeId) $("coder-state").textContent = "error";
      toast("Agent error: " + ev.text, true);
      break;
    }
    case "closed": {
      s.busy = false;
      s.closed = true;
      break;
    }
  }
}

async function streamSession(s, replay) {
  while (coder.sessions.has(s.info.id) && !s.closed) {
    try {
      const r = await fetch(
        `/api/coder/sessions/${s.info.id}/events${replay ? "?replay=true" : ""}`,
        { headers: authHeaders() });
      if (r.status === 404) { s.closed = true; break; }
      if (!r.ok) throw new Error(r.statusText);
      replay = false;   // only the first connection replays
      await readSSE(r, (payload) => {
        let ev;
        try { ev = JSON.parse(payload); } catch { return; }
        if (coder.sessions.has(s.info.id)) handleCoderEvent(s, ev);
      });
    } catch (e) {
      if (!coder.sessions.has(s.info.id) || s.closed) return;
      await new Promise((res) => setTimeout(res, 1500));
    }
  }
}

/* session lifecycle */

function populateSetupModels() {
  const sel = $("setup-model");
  sel.innerHTML = "";
  const current = document.createElement("option");
  current.value = "";
  current.textContent = "active model (" + (modelCache.active || "?") + ")";
  sel.appendChild(current);
  for (const m of modelCache.models || []) {
    if (m.active) continue;
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = m.name;
    sel.appendChild(opt);
  }
}

async function startCoderSession(opts = {}) {
  const resume = !!opts.resume;
  const cwd = $("setup-cwd").value.trim();
  if (!cwd) { toast("Enter a project directory", true); return; }
  $("setup-start").disabled = true;
  try {
    const body = {
      cwd,
      auto_approve: $("setup-auto").checked,
      dry_run: $("setup-dry").checked,
      mode: $("setup-mode").value,
      max_turns: Number($("setup-max-turns").value) || 40,
      resume,
    };
    const model = $("setup-model").value;
    if (model) body.model = model;
    const temp = $("setup-temperature").value.trim();
    if (temp !== "") body.temperature = Number(temp);
    const scope = $("setup-scope").value.trim();
    if (scope) body.scope = scope;

    const r = await fetch("/api/coder/sessions", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const info = await r.json();
    if (!chat.privacy) localStorage.setItem("localm.coderCwd", cwd);
    // A resumed session replays its restored recap from the server; a fresh one
    // has no history to replay (CODER-2).
    registerSession(info, { replay: !!info.resumed });
    activateSession(info.id);
    if (info.resumed) toast("Resumed your last session in this folder");
    else if (resume) toast("No saved session to resume - started fresh");
    refreshModels();
  } catch (e) {
    toast("Failed to start session: " + e.message, true);
  } finally {
    $("setup-start").disabled = false;
  }
}

/* Resume (CODER-2): a dynamically-created "Continue last session" button in the
 * setup panel, revealed when the chosen directory has a saved conversation. Built
 * in JS so it needs no index.html change. */
let _coderContinueBtn = null;
function coderContinueButton() {
  if (_coderContinueBtn) return _coderContinueBtn;
  const btn = el("button", "btn-quiet coder-continue", "Continue last session");
  btn.style.display = "none";
  btn.onclick = () => startCoderSession({ resume: true });
  const start = $("setup-start");
  if (start && start.parentNode) start.parentNode.insertBefore(btn, start.nextSibling);
  _coderContinueBtn = btn;
  return btn;
}

async function refreshResumable() {
  const btn = coderContinueButton();
  const cwd = ($("setup-cwd")?.value || "").trim();
  if (!cwd) { btn.style.display = "none"; return; }
  try {
    const r = await fetch("/api/coder/resumable?cwd=" + encodeURIComponent(cwd),
                          { headers: authHeaders() });
    const d = await r.json();
    if (r.ok && d.resumable) {
      const when = d.interrupted_at
        ? new Date(d.interrupted_at).toLocaleString() : "earlier";
      btn.textContent = `Continue last session (${d.turns} turns, ${when})`;
      btn.style.display = "";
    } else {
      btn.style.display = "none";
    }
  } catch { btn.style.display = "none"; }
}

async function reattachSessions() {
  try {
    const r = await fetch("/api/coder/sessions", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    for (const info of data.sessions) {
      if (!coder.sessions.has(info.id)) {
        registerSession(info, { replay: true });
      }
    }
    if (!coder.activeId && data.sessions.length) {
      activateSession(data.sessions[data.sessions.length - 1].id);
      toast("Reattached to a running coder session");
    } else {
      // Sessions may exist without one being activated (e.g. the host did not
      // auto-open a session). Still surface them in the selector + bar so the
      // host sees the same session list the mobile view does, without having to
      // enter a session first (CODER-3).
      renderSessionSelect();
      if (coder.sessions.size > 0) $("coder-bar").classList.add("open");
    }
  } catch (e) { /* server unreachable; startup poller will retry models anyway */ }
}

/* coder file attachments - extracted to text server-side (same in-memory
 * /api/rag/extract path as chat docs) and prepended to the task message,
 * so the agent sees the content without needing the file inside cwd. */

function renderCoderAttachChips() {
  const box = $("coder-attach-chips");
  box.replaceChildren();
  coder.docs.forEach((doc, i) => {
    const chip = el("span", "chip");
    chip.appendChild(el("span", "", "📄 " + doc.name +
      ` (${(doc.chars / 1000).toFixed(1)}k chars${doc.truncated ? ", trimmed" : ""})`));
    const rm = el("button", "", "×");
    rm.onclick = () => { coder.docs.splice(i, 1); renderCoderAttachChips(); };
    chip.appendChild(rm);
    box.appendChild(chip);
  });
}

async function attachCoderDocument(file) {
  const b64 = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(new Error("could not read file"));
    reader.readAsDataURL(file);
  });
  const r = await fetch("/api/rag/extract", {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ filename: file.name, content_b64: b64 }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || r.statusText);
  coder.docs.push({ name: data.filename, text: data.text,
                    chars: data.chars, truncated: data.truncated });
  renderCoderAttachChips();
}

$("coder-attach").onclick = () => $("coder-file").click();
$("coder-file").addEventListener("change", (e) => {
  for (const file of e.target.files) {
    attachCoderDocument(file).catch((err) =>
      toast(`${file.name}: ${err.message}`, true));
  }
  e.target.value = "";
});

async function sendCoderTask() {
  const s = activeSession();
  const input = $("coder-input");
  const text = input.value.trim();
  if ((!text && coder.docs.length === 0) || !s) return;

  if (text.startsWith("/")) {
    input.value = "";
    autoGrow(input);
    handleSlashSubmit(text, (c) => execCoderCommand(c));
    return;
  }

  // Attached file contents go first so the agent reads them before the task
  let payload = text || "Read the attached file(s).";
  if (coder.docs.length) {
    const blocks = coder.docs.map((d) =>
      `[Attached file: ${d.name}${d.truncated ? " (truncated)" : ""}]\n${d.text}`);
    payload = blocks.join("\n\n") + "\n\n" + payload;
  }

  try {
    const r = await fetch(`/api/coder/sessions/${s.info.id}/message`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ text: payload }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    if (data.status === "queued") {
      // Mid-task steering: the agent reads it at the next turn boundary
      toast("Queued - the agent picks it up at the next turn");
    } else {
      s.busy = true;
      $("coder-state").textContent = "working…";
      renderSessionSelect();
    }
    // The user message arrives back through the event stream (so replay
    // works after a page reload) - no client-side row here.
    input.value = "";
    autoGrow(input);
    coder.docs = [];
    renderCoderAttachChips();
  } catch (e) {
    toast("Failed to send task: " + e.message, true);
  }
}

async function endCoderSession() {
  const s = activeSession();
  if (!s) return;
  try {
    await fetch(`/api/coder/sessions/${s.info.id}`, {
      method: "DELETE", headers: authHeaders() });
  } catch (e) { /* server may already be gone */ }
  s.closed = true;
  s.feedEl.remove();
  coder.sessions.delete(s.info.id);
  const remaining = [...coder.sessions.keys()];
  coder.activeId = remaining[remaining.length - 1] || null;
  activateSession(coder.activeId);
}

/* coder bar buttons */

$("session-select").onchange = () => activateSession($("session-select").value);
$("session-new").onclick = () => {
  populateSetupModels();
  showCoderUI(false);
  $("coder-bar").classList.add("open");   // keep the bar so sessions stay reachable
};
// R17: the open-sessions rail's "+" mirrors the bar's "+ new".
if ($("coder-new-session")) $("coder-new-session").onclick = () => $("session-new").click();
renderCoderSessionList();   // R17: show the empty-state rail on first load
// Arrow wrapper: a bare `.onclick = startCoderSession` would pass the click
// Event as opts, making opts.resume truthy and always resuming (CODER-2).
$("setup-start").onclick = () => startCoderSession();
// Probe for a resumable checkpoint as the directory changes (debounced).
let _resumeProbeTimer = null;
$("setup-cwd").addEventListener("input", () => {
  clearTimeout(_resumeProbeTimer);
  _resumeProbeTimer = setTimeout(refreshResumable, 350);
});
$("setup-cancel").onclick = () => {
  // Return to the session we left (or any remaining one) without starting
  const id = coder.sessions.has(coder.lastActiveId)
    ? coder.lastActiveId
    : [...coder.sessions.keys()].pop();
  if (id) activateSession(id);
};

/* ---- directory picker (browse… on the setup form) ---- */

async function fetchDirs(path) {
  const r = await fetch("/api/fs/dirs?path=" + encodeURIComponent(path || ""),
                        { headers: authHeaders() });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

/** Modal directory browser. Resolves with the chosen path, or null when the
 *  modal is dismissed. Used by the coder setup form and the media pages. */
function pickDirectory(title, startPath = "") {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      $("modal").style.display = "none";
      resolve(value);
    };
    openModal(title || "Pick a directory", (body) => {
      const pathEl = el("div", "dir-picker-path");
      const listEl = el("div", "dir-picker-list");
      const actions = el("div", "actions");
      const useBtn = el("button", "btn-primary", "Use this directory");
      actions.appendChild(useBtn);
      body.append(pathEl, listEl, actions);
      let current = "";

      useBtn.onclick = () => { if (current) finish(current); };
      // Dismissing the modal (×, backdrop) resolves null - poll visibility
      // since the close handlers are owned by the shared modal chrome.
      const watch = setInterval(() => {
        if ($("modal").style.display === "none") {
          clearInterval(watch);
          finish(null);
        }
      }, 200);

      async function show(path) {
        let data;
        try {
          data = await fetchDirs(path);
        } catch (e) {
          toast("Cannot open: " + e.message, true);
          if (path) { show(""); return; }   // fall back to the drive list
          throw e;
        }
        current = data.path;
        pathEl.textContent = current || "Drives";
        useBtn.disabled = !current;
        listEl.replaceChildren();
        if (data.parent !== null && current) {
          const up = el("div", "dir-picker-item up", "↑ ..");
          up.onclick = () => show(data.parent);
          listEl.appendChild(up);
        }
        for (const name of data.dirs) {
          const item = el("div", "dir-picker-item", "📁 " + name);
          // "/" joins fine on Windows too (Python Path accepts both
          // separators); the server resolves and echoes the native form.
          item.onclick = () =>
            show(current ? current.replace(/[\\/]+$/, "") + "/" + name : name);
          listEl.appendChild(item);
        }
        if (!data.dirs.length) {
          listEl.appendChild(el("div", "dir-picker-empty", "no subdirectories"));
        }
      }
      show(startPath).catch(() => {});
    });
  });
}

$("setup-browse").onclick = async () => {
  const dir = await pickDirectory("Pick a project directory",
                                  $("setup-cwd").value.trim());
  if (dir) {
    $("setup-cwd").value = dir;
    localStorage.setItem("localm.coderCwd", dir);
    refreshResumable();   // setting .value does not fire 'input' (CODER-2)
  }
};
$("coder-send").onclick = sendCoderTask;
$("coder-input").addEventListener("keydown", (e) => composerEnterToSend(e, sendCoderTask));
$("coder-input").addEventListener("input", (e) => autoGrow(e.target));

$("coder-stop").onclick = async () => {
  const s = activeSession();
  if (!s) return;
  await fetch(`/api/coder/sessions/${s.info.id}/stop`, {
    method: "POST", headers: authHeaders() });
  toast("Stop requested - agent halts at the next safe point");
};
$("coder-end").onclick = endCoderSession;

$("coder-undo").onclick = async () => {
  const s = activeSession();
  if (!s) return;
  const r = await fetch(`/api/coder/sessions/${s.info.id}/undo`, {
    method: "POST", headers: authHeaders() });
  const data = await r.json();
  if (r.ok) {
    toast(data.summary);
    feedAppend(s, el("div", "feed-info", data.summary));
  } else {
    toast(data.detail || "Nothing to undo", true);
  }
};

$("coder-compact").onclick = async () => {
  const s = activeSession();
  if (!s) return;
  const r = await fetch(`/api/coder/sessions/${s.info.id}/compact`, {
    method: "POST", headers: authHeaders() });
  const data = await r.json();
  if (r.ok) {
    toast("History compacted");
    feedAppend(s, el("div", "feed-info", "Conversation history compacted."));
  } else {
    toast(data.detail || "Nothing to compact", true);
  }
};

/** Audit-entry modal shared by the live-session log and past-session history.
 *  A filter box narrows entries by substring (type, turn, or payload). */
function showAuditModal(title, data) {
  openModal(title, (body) => {
    body.appendChild(el("div", "sub", data.path));
    const filter = document.createElement("input");
    filter.type = "text";
    filter.placeholder = "filter entries… (tool name, text, type)";
    filter.className = "log-filter";
    filter.spellcheck = false;
    body.appendChild(filter);
    const rows = [];
    for (const entry of data.entries) {
      const row = el("div", "log-entry");
      const ts = new Date(entry.t).toLocaleTimeString();
      const label = `${ts} #${entry.turn} ${entry.type}`;
      const payload = JSON.stringify(entry.data);
      row.appendChild(el("span", "t", label));
      row.appendChild(document.createTextNode(payload));
      body.appendChild(row);
      rows.push({ row, text: (label + " " + payload).toLowerCase() });
    }
    filter.addEventListener("input", () => {
      const q = filter.value.trim().toLowerCase();
      for (const r of rows) {
        r.row.style.display = !q || r.text.includes(q) ? "" : "none";
      }
    });
    if (!data.entries.length) body.appendChild(el("div", "sub", "(empty)"));
  });
}

/** Files the agent changed this session, with per-file and full-session diffs. */
async function openFilesModal() {
  const s = activeSession();
  if (!s) return;
  let data;
  try {
    const r = await fetch(`/api/coder/sessions/${s.info.id}/files`,
                          { headers: authHeaders() });
    data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
  } catch (e) {
    toast("Could not load changed files: " + e.message, true);
    return;
  }
  openModal("Files changed - " + sessionLabel(s.info), (body) => {
    if (!data.files.length) {
      body.appendChild(el("div", "sub", "No files changed this session."));
      return;
    }
    const diffBox = el("div", "files-diff");
    const showDiff = async (path) => {
      diffBox.replaceChildren(el("div", "sub", "loading diff…"));
      try {
        const r = await fetch(
          `/api/coder/sessions/${s.info.id}/files/diff?path=` +
          encodeURIComponent(path || ""), { headers: authHeaders() });
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || r.statusText);
        diffBox.replaceChildren(
          d.diff ? renderDiff(d.diff) : el("div", "sub", "(no difference)"));
      } catch (e) {
        diffBox.replaceChildren(el("div", "sub", "diff failed: " + e.message));
      }
    };
    for (const f of data.files) {
      const row = el("div", "log-entry clickable");
      row.appendChild(el("span", "t", f.created ? "new" : "edit"));
      row.appendChild(document.createTextNode(
        `${f.path} - ${f.writes} write(s)` + (f.exists ? "" : " (deleted since)")));
      row.onclick = () => showDiff(f.path);
      // Download the file itself (pull coder output onto this device / phone).
      // Only for files that still exist on disk.
      if (f.exists) {
        const dl = el("button", "btn-quiet file-dl", "download");
        dl.title = "Download this file to your device";
        dl.onclick = (ev) => { ev.stopPropagation(); downloadCoderFile(s, f.path); };
        row.appendChild(dl);
      }
      body.appendChild(row);
    }
    const all = el("button", "btn-quiet", "full session diff");
    all.onclick = () => showDiff("");
    body.appendChild(all);
    body.appendChild(diffBox);
  });
}

/** Download one coder-created/changed file to this device (a phone, say). Fetched
 *  with auth so it works behind a key; saved via a blob so the OS "save file"
 *  flow runs. The server confines the download to tracked, in-root files. */
async function downloadCoderFile(s, path) {
  try {
    const r = await fetch(
      `/api/coder/sessions/${s.info.id}/files/download?path=` +
      encodeURIComponent(path), { headers: authHeaders() });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || r.statusText);
    }
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = path.split(/[\\/]/).pop() || "file";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
  } catch (e) {
    toast("Download failed: " + e.message, true);
  }
}

/** Download the active session's feed as markdown (explicit user action -
 *  works in privacy mode too, same contract as chat /export). */
function exportCoderSession() {
  const s = activeSession();
  const log = s?.eventLog || [];
  if (!log.length) { toast("Nothing to export yet", true); return; }
  const lines = [`# Coder session - ${sessionLabel(s.info)}`, ""];
  for (const ev of log) {
    if (ev.type === "user") {
      lines.push(`**You${ev.queued ? " (queued)" : ""}**: ${ev.text}`, "");
    } else if (ev.type === "tool_call") {
      lines.push(`- \`${ev.tool}\` ` +
        JSON.stringify(slimArgs(ev.args)).slice(0, 200));
    } else if (ev.type === "tool_result") {
      lines.push(`  - ${ev.ok ? "ok" : "FAILED"}` +
        (ev.summary ? `: ${ev.summary}` : ""));
    } else if (ev.type === "info") {
      lines.push(`> ${ev.text}`, "");
    } else if (ev.type === "final") {
      lines.push("", `**Agent**: ${ev.text}`, "");
    }
  }
  const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `coder-session-${s.info.id}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
}

$("coder-files").onclick = openFilesModal;
$("coder-export").onclick = exportCoderSession;

$("coder-log").onclick = async () => {
  const s = activeSession();
  if (!s) return;
  const r = await fetch(`/api/coder/sessions/${s.info.id}/log`, {
    headers: authHeaders() });
  const data = await r.json();
  if (!r.ok) { toast(data.detail || "No log available", true); return; }
  showAuditModal("Audit log - " + sessionLabel(s.info), data);
};

/** Past coder sessions: audit logs left behind by log/full-mode sessions,
 *  including ones from before a server restart. */
async function openSessionHistory() {
  let data = null;
  try {
    const r = await fetch("/api/coder/history", { headers: authHeaders() });
    if (r.ok) data = await r.json();
  } catch (e) { /* handled below */ }
  if (!data) { toast("Could not load session history", true); return; }
  openModal("Past coder sessions", (body) => {
    if (data.authorized === false) {
      body.appendChild(el("div", "sub",
        "Past coder sessions are private to the server owner. Sign in with the " +
        "owner API key on this device to view them."));
      return;
    }
    if (!data.enabled) {
      body.appendChild(el("div", "sub",
        "New sessions are not being recorded (privacy mode). Anything below " +
        "is from earlier log/full-mode sessions."));
    }
    if (!data.logs.length) {
      body.appendChild(el("div", "sub",
        "No session logs yet - start a session with persistence set to " +
        "log or full, and its audit trail will appear here."));
      return;
    }
    for (const item of data.logs) {
      const row = el("div", "log-entry clickable");
      const when = new Date(item.mtime * 1000).toLocaleString();
      const kb = (item.size_bytes / 1024).toFixed(1);
      row.appendChild(el("span", "t", when));
      row.appendChild(document.createTextNode(`${item.name} (${kb} KB)`));
      row.onclick = async () => {
        try {
          const r = await fetch(
            "/api/coder/history/" + encodeURIComponent(item.name),
            { headers: authHeaders() });
          const entries = await r.json();
          if (!r.ok) throw new Error(entries.detail || r.statusText);
          showAuditModal("Session - " + item.name, entries);
        } catch (e) {
          toast("Could not open log: " + e.message, true);
        }
      };
      body.appendChild(row);
    }
  });
}

$("coder-history").onclick = openSessionHistory;
$("setup-history").onclick = openSessionHistory;

/* ================================================================ */
/*  Slash commands                                                   */
/* ================================================================ */

const CHAT_COMMANDS = [
  { cmd: "generate-image", hint: "generate an image with FLUX", args: "<prompt>" },
  { cmd: "generate-music", hint: "generate a music track (ACE-Step, 120s instrumental)", args: "<style tags>" },
  { cmd: "generate-video", hint: "generate a short video clip (Wan, ~5s - slow)", args: "<prompt>" },
  { cmd: "web", hint: "search the web, then answer with sources", args: "<query>" },
  { cmd: "clear", hint: "clear this conversation" },
  { cmd: "compact", hint: "summarise older messages to free context" },
  { cmd: "export", hint: "download this conversation as markdown" },
  { cmd: "rename", hint: "rename this conversation", args: "<title>" },
  { cmd: "persona", hint: "apply a saved persona (system prompt + params)", args: "<name>" },
  { cmd: "remember", hint: "add a fact to the model's long-term memory", args: "<fact>" },
  { cmd: "memory", hint: "view or edit the memory file" },
  { cmd: "pin", hint: "pin/unpin this conversation" },
  { cmd: "folder", hint: "move this conversation to a folder (empty = remove)", args: "<name>" },
  { cmd: "system", hint: "edit the system prompt" },
  { cmd: "new", hint: "start a new conversation" },
];

const CODER_COMMANDS = [
  { cmd: "undo", hint: "revert the last file write" },
  { cmd: "files", hint: "files changed this session, with diffs" },
  { cmd: "compact", hint: "summarise older turns" },
  { cmd: "export", hint: "download this session's feed as markdown" },
  { cmd: "log", hint: "open the audit log" },
  { cmd: "stop", hint: "interrupt the current task" },
  { cmd: "end", hint: "end this session" },
  { cmd: "help", hint: "list available commands" },
];

async function runImagineInChat(promptText) {
  if (!promptText) { toast("Usage: /generate-image <prompt>", true); return; }
  if (!currentConv()) newConversation();
  const conv = currentConv();
  conv.messages.push({ role: "user", content: "/generate-image " + promptText });
  saveConversations(conv);
  renderChat();
  const box = $("chat-messages");
  const { body } = addMessageRow(box, "assistant", "");
  body.textContent = "Generating image…";
  box.scrollTop = box.scrollHeight;
  try {
    const r = await fetch("/api/imagine", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ prompt: promptText }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      body.textContent = line;
      if (nearBottom(box)) box.scrollTop = box.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      conv.messages.push({
        role: "assistant",
        content: [
          { type: "text", text: "Here is the generated image:" },
          { type: "image_url",
            image_url: { url: "/api/imagine/file/" + encodeURIComponent(end.result) } },
        ],
      });
      saveConversations(conv);
      renderChat();
    } else {
      body.textContent = "Image generation " + end.status +
        " - see the Images page for details.";
    }
  } catch (e) {
    body.textContent = "Image generation failed: " + e.message;
    toast(e.message, true);
  }
}

/** /web <query> - explicit, user-initiated web grounding: search, inject the
 *  results into the conversation, and let the model answer from them. */
async function runWebInChat(query) {
  if (!query) { toast("Usage: /web <query>", true); return; }
  if (chat.abort) { toast("Wait for the current reply to finish", true); return; }
  if (!currentConv()) newConversation();
  const conv = currentConv();
  conv.messages.push({ role: "user", content: "/web " + query });
  if (conv.messages.length === 1) {
    conv.title = query.slice(0, 42) + (query.length > 42 ? "…" : "");
    renderConvList();
  }
  saveConversations(conv);
  renderChat();
  let note;
  try {
    note = await requestWebTool({ name: "web_search", args: { query } });
    note += `\n\nUsing these results, answer: ${query}\nName the sources you used.`;
  } catch (e) {
    toast("Web search failed: " + e.message, true);
    note = `[Web search failed: ${e.message}] Tell the user, and answer ` +
           "from your own knowledge if you can.";
  }
  conv.messages.push({ role: "user", content: note, web: true });
  saveConversations(conv);
  renderChat();
  await runCompletion(conv);
}

/** /music <tags> - generate a default-length instrumental inline; the Music
 *  page has the full form (lyrics, duration, seed…). */
async function runMusicInChat(tags) {
  if (!tags) { toast("Usage: /generate-music <style tags>", true); return; }
  if (!currentConv()) newConversation();
  const conv = currentConv();
  conv.messages.push({ role: "user", content: "/generate-music " + tags });
  saveConversations(conv);
  renderChat();
  const box = $("chat-messages");
  const { body } = addMessageRow(box, "assistant", "");
  body.textContent = "Generating track… (long tracks take a while)";
  box.scrollTop = box.scrollHeight;
  try {
    const r = await fetch("/api/music", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ tags }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      body.textContent = line;
      if (nearBottom(box)) box.scrollTop = box.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      conv.messages.push({
        role: "assistant",
        content: "Here is the generated track:",
        audio: "/api/music/file/" + encodeURIComponent(end.result),
      });
      saveConversations(conv);
      renderChat();
    } else {
      body.textContent = "Music generation " + end.status +
        " - see the Music page for details.";
    }
  } catch (e) {
    body.textContent = "Music generation failed: " + e.message;
    toast(e.message, true);
  }
}

/** /video <prompt> - generate a default-length (~5s) clip inline; the Video
 *  page has the full form (negative, duration, size, start image…). */
async function runVideoInChat(promptText) {
  if (!promptText) { toast("Usage: /generate-video <prompt>", true); return; }
  if (!currentConv()) newConversation();
  const conv = currentConv();
  conv.messages.push({ role: "user", content: "/generate-video " + promptText });
  saveConversations(conv);
  renderChat();
  const box = $("chat-messages");
  const { body } = addMessageRow(box, "assistant", "");
  body.textContent = "Generating clip… (video is slow - expect several minutes)";
  box.scrollTop = box.scrollHeight;
  try {
    const r = await fetch("/api/video", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ prompt: promptText }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const end = await streamJob(data.job_id, (line) => {
      body.textContent = line;
      if (nearBottom(box)) box.scrollTop = box.scrollHeight;
    });
    if (end.status === "done" && end.result) {
      conv.messages.push({
        role: "assistant",
        content: "Here is the generated clip:",
        video: "/api/video/file/" + encodeURIComponent(end.result),
      });
      saveConversations(conv);
      renderChat();
    } else {
      body.textContent = "Video generation " + end.status +
        " - see the Video page for details.";
    }
  } catch (e) {
    body.textContent = "Video generation failed: " + e.message;
    toast(e.message, true);
  }
}

function execChatCommand(cmd, arg) {
  switch (cmd) {
    case "generate-image": case "imagine": runImagineInChat(arg); return true;
    case "generate-music": case "music": runMusicInChat(arg); return true;
    case "generate-video": case "video": runVideoInChat(arg); return true;
    case "web": runWebInChat(arg); return true;
    case "clear": {
      const conv = currentConv();
      if (conv) {
        conv.messages = [];
        conv.branches = [];
        saveConversations(conv);
        renderChat();
      }
      return true;
    }
    case "compact": $("compact-conv").onclick(); return true;
    case "export": exportConversation(); return true;
    case "rename": {
      const conv = currentConv();
      if (conv && arg) { conv.title = arg; saveConversations(conv); renderConvList(); }
      else toast("Usage: /rename <title>", true);
      return true;
    }
    case "remember": rememberFact(arg); return true;
    case "memory": openMemoryModal(); return true;
    case "persona": {
      if (!arg) {
        const names = personaCache.map((p) => p.name);
        toast(names.length ? "Personas: " + names.join(", ")
                           : "No personas saved yet - use the drawer's save…",
              !names.length);
        return true;
      }
      // case-insensitive match for typing convenience
      const hit = personaCache.find(
        (p) => p.name.toLowerCase() === arg.toLowerCase());
      applyPersona(hit ? hit.name : arg);
      return true;
    }
    case "pin": {
      const conv = currentConv();
      if (!conv) return true;
      conv.pinned = !conv.pinned;
      if (!conv.pinned) delete conv.pinned;
      saveConversations(conv);
      renderConvList();
      toast(conv.pinned ? "Pinned" : "Unpinned");
      return true;
    }
    case "folder": {
      const conv = currentConv();
      if (!conv) return true;
      if (arg) conv.folder = arg;
      else delete conv.folder;
      saveConversations(conv);
      renderConvList();
      toast(arg ? `Moved to folder '${arg}'` : "Removed from its folder");
      return true;
    }
    case "system":
      $("params").classList.add("open");
      $("p-system").focus();
      return true;
    case "new": newConversation(); return true;
  }
  return false;
}

function execCoderCommand(cmd) {
  switch (cmd) {
    case "undo": $("coder-undo").onclick(); return true;
    case "files": openFilesModal(); return true;
    case "compact": $("coder-compact").onclick(); return true;
    case "export": exportCoderSession(); return true;
    case "log": $("coder-log").onclick(); return true;
    case "stop": $("coder-stop").onclick(); return true;
    case "end": $("coder-end").onclick(); return true;
    case "help":
      openModal("Coder commands", (body) => {
        for (const c of CODER_COMMANDS) {
          const row = el("div", "log-entry");
          row.appendChild(el("span", "t", "/" + c.cmd));
          row.appendChild(document.createTextNode(c.hint));
          body.appendChild(row);
        }
        body.appendChild(el("div", "sub",
          "Anything not starting with / is sent to the agent as a task."));
      });
      return true;
  }
  return false;
}

/** Attach a slash-command dropdown to a composer textarea. */
function attachSlashMenu(textarea, commands, execute) {
  const menu = el("div", "slash-menu");
  menu.style.display = "none";
  textarea.closest(".composer-wrap").appendChild(menu);
  let selected = 0;
  let visible = [];

  function close() { menu.style.display = "none"; visible = []; }

  function render() {
    const value = textarea.value;
    if (!value.startsWith("/") || value.includes("\n")) { close(); return; }
    // Once a space is typed the command token is complete and the user is
    // entering arguments - close the menu so Enter SENDS the whole line
    // ("/remember some note") instead of the menu's Enter handler calling
    // pick(), which overwrites the input with "/cmd " and discards the args.
    const rest = value.slice(1);
    if (rest.includes(" ")) { close(); return; }
    const typed = rest.toLowerCase();
    visible = commands.filter((c) => c.cmd.startsWith(typed));
    if (!visible.length) { close(); return; }
    selected = Math.min(selected, visible.length - 1);
    menu.replaceChildren();
    visible.forEach((c, i) => {
      const row = el("div", "slash-item" + (i === selected ? " selected" : ""));
      row.appendChild(el("span", "cmd", "/" + c.cmd + (c.args ? " " + c.args : "")));
      row.appendChild(el("span", "hint", c.hint));
      row.onmousedown = (e) => { e.preventDefault(); pick(c); };
      menu.appendChild(row);
    });
    menu.style.display = "block";
  }

  function pick(c) {
    if (!c) { close(); return; }   // empty list in a render/keydown race (LATENT-1)
    if (c.args) {
      textarea.value = "/" + c.cmd + " ";
      textarea.focus();
      close();
    } else {
      textarea.value = "";
      autoGrow(textarea);
      close();
      execute(c.cmd, "");
    }
  }

  textarea.addEventListener("input", () => { selected = 0; render(); });
  textarea.addEventListener("blur", () => setTimeout(close, 150));
  textarea.addEventListener("keydown", (e) => {
    if (menu.style.display === "none") return;
    if (e.key === "ArrowDown") {
      e.preventDefault(); selected = (selected + 1) % visible.length; render();
    } else if (e.key === "ArrowUp") {
      e.preventDefault(); selected = (selected - 1 + visible.length) % visible.length; render();
    } else if (e.key === "Enter" && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
      e.preventDefault(); pick(visible[selected]);
    } else if (e.key === "Escape") {
      close();
    }
  });
}

/** Intercept "/cmd arg" on submit. Returns true when handled (not for the model). */
function handleSlashSubmit(text, execute) {
  if (!text.startsWith("/")) return false;
  const space = text.indexOf(" ");
  const cmd = (space === -1 ? text.slice(1) : text.slice(1, space)).toLowerCase();
  const arg = space === -1 ? "" : text.slice(space + 1).trim();
  const hint = pluginSuggestion(cmd);   // known plugin, just not active yet
  if (hint) { toast(hint, true); return true; }
  if (!execute(cmd, arg)) {
    toast(`Unknown command: /${cmd}`, true);
  }
  return true;   // never send slash input to the model
}

attachSlashMenu($("chat-input"), CHAT_COMMANDS, execChatCommand);
attachSlashMenu($("coder-input"), CODER_COMMANDS, (c) => execCoderCommand(c));

/* ================================================================ */
/*  Init                                                             */
/* ================================================================ */

$("setup-cwd").value = localStorage.getItem("localm.coderCwd") || "";
// API-key gate wiring (shown by showKeyGate on a 401 boot, e.g. a network bind).
if ($("key-gate-submit")) $("key-gate-submit").onclick = submitKeyGate;
if ($("key-gate-scan")) $("key-gate-scan").onclick = startQrScan;
if ($("qr-scan-cancel")) $("qr-scan-cancel").onclick = stopQrScan;
if ($("key-gate-input")) {
  $("key-gate-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); submitKeyGate(); }
  });
}
// Onboarding install gate (mobile): Continue enters the app; Install fires the
// captured beforeinstallprompt (Android/desktop Chrome).
if ($("install-gate-continue")) $("install-gate-continue").onclick = dismissInstallGate;
if ($("install-gate-install")) $("install-gate-install").onclick = () => {
  const d = window.__deferredInstall;
  if (!d) return;
  d.prompt();
  d.userChoice.finally(() => {
    window.__deferredInstall = null;
    applyInstallGateUI({ ios: isIOSSafari(), canPrompt: false });
  });
};

// Hard auth gate (NET-1): when the server REQUIRES a key and this browser has
// none that works, show ONLY the onboarding - do NOT reveal the app shell or
// load any /api data behind an unsatisfied gate. window.__localmLocked lets late
// boot steps (deep-link restore, pages.js) bail too.
function lockUI(message) {
  window.__localmLocked = true;
  const app = $("app");
  if (app) app.style.display = "none";          // nothing of localm behind the gate
  showKeyGate(message || "This LocaLM server requires an API key.");
}
function unlockUI() {
  window.__localmLocked = false;
  const gate = $("key-gate");
  if (gate) gate.style.display = "none";
  hideReconnectOverlay();
  const app = $("app");
  if (app) app.style.display = "";
}

// Recovery (AUTH-1b): when the auth state is WEDGED - the user logged in
// successfully but the page still boots 401 - a stale service-worker shell (or a
// cached navigation that bypassed the loopback cookie re-seed) is the cause, NOT
// the key. Do automatically what the user otherwise has to do by hand (clear
// site data): unregister the SW and drop its caches, then reload once. A
// sessionStorage guard bounds it to a single attempt so it can never loop. We do
// NOT touch SameSite (the cookie IS sent on same-origin fetch; the rejected
// misdiagnosis would only open CSRF).
async function resetServiceWorkerAndCaches() {
  try {
    if ("serviceWorker" in navigator) {
      const regs = await navigator.serviceWorker.getRegistrations();
      await Promise.all(regs.map((r) => r.unregister()));
    }
  } catch (e) { /* best-effort; the reload still fetches a fresh shell */ }
  try {
    if (window.caches) {
      const keys = await caches.keys();
      await Promise.all(keys.map((k) => caches.delete(k)));
    }
  } catch (e) { /* best-effort */ }
}
window.resetServiceWorkerAndCaches = resetServiceWorkerAndCaches;

// Manual escape hatch (offered on the reconnect overlay): wipe EVERY client-side
// artifact that could wedge boot/auth, then reload to a clean state (the key
// gate). Each step is independently guarded so one failure never blocks the rest.
// (The HttpOnly localm_session cookie cannot be cleared from JS, but it never
// wedges the client - a stale one simply yields a 401 -> the key gate.)
async function resetClientState() {
  try {
    document.cookie.split(";").forEach((c) => {
      const n = c.split("=")[0].trim();
      if (n) document.cookie = n + "=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/";
    });
  } catch (e) { /* ignore */ }
  try { localStorage.clear(); } catch (e) { /* ignore */ }
  try { sessionStorage.clear(); } catch (e) { /* ignore */ }
  try { await resetServiceWorkerAndCaches(); } catch (e) { /* ignore */ }
  location.reload();
}
window.resetClientState = resetClientState;

// Server-unreachable lock (AUTH-1b): the server is DOWN (e.g. it crashed - that
// is Lane A's territory), NOT an auth failure. Show a distinct "reconnecting"
// overlay and auto-retry instead of the key gate, so a dead server is not
// mistaken for a bad key and re-entered in a loop. When the server answers
// again, reload for a clean boot (which then handles 200 vs 401 freshly).
let _reconnectTimer = null;
function showReconnectOverlay() {
  let ov = $("reconnect-overlay");
  if (!ov) {
    ov = el("div", "reconnect-overlay");
    ov.id = "reconnect-overlay";
    const panel = el("div", "reconnect-panel");
    panel.appendChild(el("div", "reconnect-spinner"));
    panel.appendChild(el("div", "reconnect-msg",
      "Can't reach the LocaLM server. It may be starting or stopped. Reconnecting..."));
    // Escape hatch: a client must always have a manual way out, so no local
    // artifact (a bad cookie, a wedged shell) can ever trap the user with no
    // recovery. Reset clears all client-side state and reloads to the key gate.
    const reset = el("button", "reconnect-reset", "Reset and re-enter key");
    reset.type = "button";
    reset.onclick = resetClientState;
    panel.appendChild(reset);
    ov.appendChild(panel);
    document.body.appendChild(ov);
  }
  ov.style.display = "flex";
}
function hideReconnectOverlay() {
  const ov = $("reconnect-overlay");
  if (ov) ov.style.display = "none";
}

// R25: first-load progress. Shown immediately at boot so the cold-start wait (the
// first /api/models can block for many seconds while the model loads) shows that
// load is in progress instead of a blank / half-rendered shell. Distinct in copy
// from the reconnect overlay, which means the server is DOWN. Hidden once the
// model list resolves (or fails), so it never stacks over the gate or the app.
function showStartupOverlay() {
  let ov = $("startup-overlay");
  if (!ov) {
    ov = el("div", "reconnect-overlay startup-overlay");
    ov.id = "startup-overlay";
    const panel = el("div", "reconnect-panel");
    panel.appendChild(el("div", "reconnect-spinner"));
    panel.appendChild(el("div", "reconnect-msg",
      "Starting LocaLM... loading the model. The first run can take a moment."));
    ov.appendChild(panel);
    document.body.appendChild(ov);
  }
  ov.style.display = "flex";
}
function hideStartupOverlay() {
  const ov = $("startup-overlay");
  if (ov) ov.style.display = "none";
}
window.showStartupOverlay = showStartupOverlay;
window.hideStartupOverlay = hideStartupOverlay;

// Reachability probe that carries NO auth headers, so it can NEVER fail for a
// client-side reason (a bad cookie/header that makes authHeaders throw). ANY HTTP
// response - even 401 - proves the server is reachable; only a thrown fetch (no
// response at all) means it is genuinely down. This is what makes the
// "server unreachable" verdict actually mean unreachable, never a client problem.
async function serverReachable() {
  try {
    await fetch("/api/models", { cache: "no-store" });
    return true;
  } catch (e) {
    return false;
  }
}
window.serverReachable = serverReachable;

function onServerUnreachable() {
  window.__localmLocked = true;
  const app = $("app");
  if (app) app.style.display = "none";
  const gate = $("key-gate");
  if (gate) gate.style.display = "none";   // a connectivity problem, not a key one
  showReconnectOverlay();
  if (_reconnectTimer) return;
  _reconnectTimer = setInterval(async () => {
    if (!(await serverReachable())) return;   // still down - keep waiting
    clearInterval(_reconnectTimer);
    _reconnectTimer = null;
    location.reload();                         // back up -> clean boot handles 200/401
  }, 3000);
}
window.onServerUnreachable = onServerUnreachable;

// Boot auth probe (AUTH-1b refactor). Returns true when authed (the caller then
// loads the app). status 0 / unreachable -> reconnect overlay (NOT the gate);
// 401 -> key gate, OR a one-shot stale-shell self-heal if a prior login should
// already have authed; 200 -> unlock.
async function bootAuthProbe() {
  let status;
  try {
    const r = await fetch("/api/models", { headers: authHeaders() });
    status = r.status;
  } catch (e) {
    // The authed request could not complete. Before declaring the server
    // unreachable (a dead-end overlay with no recovery), confirm with a header-free
    // probe: if the server answers at all it is UP and the failure was client-side,
    // so treat it as "needs auth" (the recoverable key gate), NOT a dead server.
    status = (await serverReachable()) ? 401 : 0;
  }
  if (status === 0) { onServerUnreachable(); return false; }
  if (status === 401) {
    // A SUCCESSFUL login (marker set by submitKeyGate / the Settings key save)
    // that still boots 401 means the cached shell, not the key, is wedged ->
    // reset it once and reload, so the user never has to clear site data.
    if (sessionStorage.getItem("localm.loginOk") === "1"
        && sessionStorage.getItem("localm.swReset") !== "1") {
      sessionStorage.removeItem("localm.loginOk");
      sessionStorage.setItem("localm.swReset", "1");   // one-shot guard - no loop
      await resetServiceWorkerAndCaches();
      location.reload();
      return false;
    }
    lockUI();
    return false;
  }
  // Authed (or open / loopback mode): clear recovery flags and reveal the app.
  try {
    sessionStorage.removeItem("localm.loginOk");
    sessionStorage.removeItem("localm.swReset");
  } catch (e) { /* sessionStorage may be unavailable in some private modes */ }
  unlockUI();
  return true;
}
window.bootAuthProbe = bootAuthProbe;

(async () => {
  showStartupOverlay();   // R25: immediate first-load feedback (cold start is slow)
  // Probe auth before loading any app data or revealing the shell.
  const authed = await bootAuthProbe();
  if (!authed) { hideStartupOverlay(); return; }   // gate / reconnect overlay takes over
  // On a phone not yet installed, show the one-time install landing first; the
  // app still loads behind it and is revealed by "Continue". Desktop / installed
  // / returning visits fall straight through.
  if (shouldShowInstallGate()) showInstallGate();
  // Authenticated (or open/loopback mode): load the app.
  syncLogoStyleFromConfig();   // reconcile the wordmark with the shared config
  // R25: hide the startup overlay once the model list resolves (the app is usable)
  // or fails - never leave it stuck over the shell.
  refreshModels().then(() => populateSetupModels()).finally(hideStartupOverlay);
  // Server persistence depends on knowing the privacy state first.
  refreshCtxLimit().then(initServerConversations);
})();
// Reveal toggles on the API-key inputs (AUTH-2): the in-page gate and the
// Settings key field, so the user can confirm the key they typed.
addRevealToggle($("key-gate-input"));
addRevealToggle($("gui-api-key"));
refreshKbSelect();
refreshPersonas();
refreshMemory();
refreshVoiceStatus();
// Voice picker + client-side plugins (the tts plugin installs a neural voice).
if (window.speechSynthesis) speechSynthesis.onvoiceschanged = populateVoicePicker;
if ($("p-voice")) $("p-voice").onchange = onVoicePick;
populateVoicePicker();
loadClientPlugins();
refreshPluginCommands();
// Re-sync plugin command hints when the window regains focus, so a plugin
// toggled in another terminal/tab while sitting on the chat view is reflected
// without a reload (the view-switch path in pages.js covers navigation).
window.addEventListener("focus", refreshPluginCommands);
// R50: when a plugin is enabled/disabled in ANOTHER tab, that tab bumps a shared
// localStorage rev; the storage event fires in every OTHER same-origin tab, so we
// re-sync the nav/commands promptly. A tab parked on the now-disabled plugin's
// (static) page is then redirected to chat by reconcileActiveView instead of
// erroring on the plugin's unmounted routes. visibilitychange covers the case
// where the tab becomes visible without a focus event firing.
window.addEventListener("storage", (e) => {
  if (e.key === "localm.pluginsRev") refreshPluginCommands();
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refreshPluginCommands();
});
setInterval(refreshModels, 30000);
startHwStats();   // live CPU/RAM/VRAM/GPU readout in the status bar
setupPerfCard();  // Settings: GPU-layers/context sliders + live VRAM estimate
// The resolved ctx ceiling only exists once a model has loaded - keep the
// compaction threshold in sync as models load or switch.
setInterval(refreshCtxLimit, 30000);
renderConvList();
if (chat.conversations.length) {
  chat.activeId = chat.conversations[0].id;
  renderConvList();
}
renderChat();
reattachSessions();
// Deep links + restore. Deferred a tick so pages.js has installed
// window.onViewShown and the #pull-start handler. Skipped while the hard auth
// gate is locked (nothing of the app should activate behind the onboarding).
{
  const params = new URLSearchParams(location.search);
  const pullSpec = params.get("pull");      // from `localm gui --pull SPEC`
  const viewParam = params.get("view");
  const sharedTo = params.get("shared");    // from the PWA share target (phone)
  if (sharedTo) {
    // A phone shared image(s) into localm: land on chat and ingest them.
    history.replaceState(null, "", location.pathname);
    setTimeout(() => {
      if (window.__localmLocked) return;
      showView("chat");
      ingestSharedFiles();
    }, 0);
  } else if (pullSpec || viewParam) {
    // Strip the query so a reload doesn't restart the download.
    history.replaceState(null, "", location.pathname);
    setTimeout(() => {
      if (window.__localmLocked) return;
      showView(VIEWS.includes(viewParam) ? viewParam : "models");
      if (pullSpec) {
        const specInput = $("pull-spec");
        if (specInput) {
          specInput.value = pullSpec;
          $("pull-start").click();   // kick off the pull with progress
        }
      }
    }, 0);
  } else {
    // Restore the last active page (set in non-privacy mode only).
    const savedView = localStorage.getItem("localm.activeView");
    if (savedView && savedView !== "chat") {
      setTimeout(() => { if (!window.__localmLocked) showView(savedView); }, 0);
    }
  }
}
