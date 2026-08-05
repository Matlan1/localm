// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - models sidebar selector (split from app.js). Classic script: it
   shares the one global lexical environment with the other app/* and
   pages/* scripts, so every cross-section reference resolves by bare
   name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { chat } from "./chat.js";
import { $, GIB, authHeaders, el, fmtDuration, instanceCacheTrusted, openModal, streamJob, toast } from "./helpers.js";
import { refreshPerfEstimate } from "./settings-perf.js";

export const modelSelect = $("model-select");

// ADR-0008 U4: tracks whether the dot is CURRENTLY showing a caller's deliberate
// busy state (e.g. switchModel's "loading X…"), so refreshModels()'s periodic
// "ok" write does not clobber it - the pre-existing race this fix closes.
// Generic on purpose (any future busy-setter is covered), not tied to one
// specific caller.
let _statusBusy = false;

export function setStatus(state, text) {
  _statusBusy = state === "busy";
  $("status-dot").className = "dot " + state;
  $("status-text").textContent = text;
}

// Live hardware monitor in the status bar (CPU/RAM/VRAM/GPU). Renders whatever
// /api/stats reports; any section the box can't measure is simply absent (no
// psutil -> no CPU/RAM; AMD -> no GPU%). VRAM shows used/total when free is
// known, otherwise just total.
export function renderHwStats(data) {
  const el = $("hw-stats");
  if (!el) return;
  const gib = (b) => (b / GIB).toFixed(1);
  // Each metric is its own <span> so the VRAM figure can carry a subtle fullness
  // colour. The colour rides ONLY on a trustworthy used/total (the backend sends
  // `used` only for a fresh, device-global reading - see sysstats._vram); a
  // total-only reading gets no colour, since there is nothing to be "full" of.
  const spans = [];
  const add = (text, cls) => {
    const s = document.createElement("span");
    s.textContent = text;
    if (cls) s.className = cls;
    spans.push(s);
  };
  if (data && data.cpu && typeof data.cpu.percent === "number")
    add(`CPU ${Math.round(data.cpu.percent)}%`);
  if (data && data.ram && typeof data.ram.percent === "number")
    add(`RAM ${Math.round(data.ram.percent)}%`);
  if (data && data.vram && data.vram.total) {
    const v = data.vram;
    if (v.used != null) {
      const frac = v.total ? v.used / v.total : 0;
      const band = frac >= 0.9 ? "vram-full" : frac >= 0.7 ? "vram-busy" : "vram-ok";
      add(`VRAM ${gib(v.used)}/${gib(v.total)} GB`, `vram-usage ${band}`);
    } else {
      add(`VRAM ${gib(v.total)} GB`);
    }
  }
  if (data && data.gpu && typeof data.gpu.percent === "number")
    add(`GPU ${Math.round(data.gpu.percent)}%`);
  el.textContent = "";
  if (spans.length) {
    spans.forEach((s, i) => {
      if (i) el.appendChild(document.createTextNode(" · "));
      el.appendChild(s);
    });
    el.hidden = false;
  } else {
    el.hidden = true;
  }
}

export async function pollHwStats() {
  if (typeof document !== "undefined" && document.hidden) return;
  try {
    const r = await fetch("/api/stats", { headers: authHeaders() });
    if (!r.ok) return;
    renderHwStats(await r.json());
  } catch (e) { /* transient - keep the last reading */ }
}

export let _hwStatsTimer = null;
export function startHwStats(intervalMs = 2500) {
  pollHwStats();
  pollActivity();   // ADR-0008 U4: folded into this existing poll, not a new timer
  if (_hwStatsTimer) clearInterval(_hwStatsTimer);
  _hwStatsTimer = setInterval(() => { pollHwStats(); pollActivity(); }, intervalMs);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) { pollHwStats(); pollActivity(); }   // refresh promptly on tab focus
  });
}

// --------------------------------------------------------------------------- //
//  ADR-0008 U4: cross-session/cross-tab activity. A persistent status-bar     //
//  affordance for background operations this tab did not necessarily start   //
//  (another browser session, or this same tab before a reload) - the         //
//  maintainer's own reported case. Polled on the SAME tick as pollHwStats     //
//  above (no new timer) and reattached at boot, mirroring coder.js's         //
//  reattachSessions()/streamSession(). Deliberately does NOT drive the       //
//  status DOT (setStatus) - only the pill below - to avoid inventing new,    //
//  unrequested semantics about which kinds of background work should read as
//  "the model is busy"; the _statusBusy fix above is what lets a future
//  caller do that safely, not a decision this unit makes for every op kind.
// --------------------------------------------------------------------------- //

// job_id -> {lines: string[]} - lines accumulated from a background streamJob()
// reattach, so the details modal can show "the output it missed" even for an
// operation whose page-specific progress UI was never open in this tab.
const _activityLogs = new Map();
let _activityOps = [];          // the last successfully-read /api/activity list
// The SERVER's clock (epoch seconds) at that same read. Ages are rendered as
// `_activityServerNow - op.created_at`, NEVER against this browser's own
// clock - the /api/activity route's own docstring is explicit that a client
// clock (a phone, a drifted box) disagrees by real amounts, and durations
// are exactly what created_at exists to make renderable.
let _activityServerNow = null;
// True once at least one read has SUCCEEDED. Distinguishes "have not asked
// yet" (boot, before the first poll resolves) from "asked, and there is
// genuinely nothing running" (a real, confirmed answer) - collapsing the two
// into one rendering is the same defect class as loop_lag's fabricated 0.00,
// just in the app shell: an unasked question must never read as a "no".
let _activityKnown = false;

export function isActivityBusy() {
  return _activityOps.some((op) => op.status === "running");
}

function activityLabel(op) {
  return (op && (op.label || op.kind)) || "operation";
}

// A compact "running 12m" / "3m ago" string, or "" when age cannot be
// computed (no server clock yet, or a malformed created_at).
function activityAge(op) {
  if (_activityServerNow == null || typeof op.created_at !== "number") return "";
  const d = fmtDuration(_activityServerNow - op.created_at);
  if (!d) return "";
  return op.status === "running" ? `running ${d}` : `${d} ago`;
}

export function renderActivityPill(ops) {
  const pill = $("activity-pill");
  if (!pill) return;
  const running = ops.filter((op) => op.status === "running");
  if (!running.length) {
    pill.style.display = "none";
    return;
  }
  pill.style.display = "";
  if (running.length === 1) {
    const op = running[0];
    const pct = typeof op.pct === "number" ? ` ${Math.round(op.pct)}%` : "";
    pill.textContent = activityLabel(op) + pct;
  } else {
    pill.textContent = `${running.length} running`;
  }
}

export function showActivityDetails() {
  openModal("Activity", (body) => {
    if (!_activityKnown) {
      // Never render "Nothing running." for a question that was never
      // successfully asked (R1) - this only shows before the very first
      // poll/reattach resolves, or if every attempt so far has failed.
      body.appendChild(el("p", "sub", "Checking…"));
      return;
    }
    if (!_activityOps.length) {
      body.appendChild(el("p", "sub", "Nothing running."));
      return;
    }
    for (const op of _activityOps) {
      const row = el("div", "job-row");
      const head = el("div", "job-head");
      head.appendChild(el("span", "job-name", activityLabel(op)));
      head.appendChild(el("span", "job-state st-" + op.status, op.status));
      const age = activityAge(op);
      if (age) head.appendChild(el("span", "sub", age));
      row.appendChild(head);
      if (typeof op.pct === "number") {
        const dl = el("div", "dl-progress");
        const bar = el("div", "dl-bar");
        const fill = el("div", "dl-fill");
        fill.style.width = Math.max(0, Math.min(100, op.pct)) + "%";
        bar.appendChild(fill);
        dl.appendChild(bar);
        row.appendChild(dl);
      }
      const entry = _activityLogs.get(op.id);
      if (entry && entry.lines.length) {
        const log = el("pre", "job-log");
        log.textContent = entry.lines.join("\n");
        row.appendChild(log);
      }
      body.appendChild(row);
    }
  });
}
if ($("activity-pill")) $("activity-pill").onclick = showActivityDetails;

// Shared read: GET /api/activity, update the module's known state + pill,
// and return the parsed operations list - or null on an unreadable response
// (R1: the caller must keep whatever it last knew, never fabricate "nothing
// running"; _activityKnown/_activityOps/_activityServerNow are simply left
// untouched on failure, matching pollHwStats' own "transient - keep the last
// reading" precedent).
async function _readActivity() {
  try {
    const r = await fetch("/api/activity", { headers: authHeaders() });
    if (!r.ok) return null;
    const data = await r.json();
    const ops = Array.isArray(data.operations) ? data.operations : [];
    _activityOps = ops;
    _activityServerNow = typeof data.now === "number" ? data.now : null;
    _activityKnown = true;
    renderActivityPill(ops);
    return ops;
  } catch (e) {
    return null;
  }
}

export async function pollActivity() {
  if (typeof document !== "undefined" && document.hidden) return;
  await _readActivity();
}

// Boot-time reattach, mirroring coder.js's reattachSessions(): ask the server
// what is running, reattach a live stream to anything found (fire-and-forget -
// an operation can run far longer than boot should wait), and toast once.
// streamJob() already handles reconnect + replay dedup internally (see
// dev-notes/streamJob-reconnect-contract.md) - this needs no dedup logic of
// its own, it only needs to call streamJob() once per running operation.
export async function reattachActivity() {
  try {
    const ops = await _readActivity();
    if (!ops) return;   // server unreachable at boot - same as reattachSessions()
    const running = ops.filter((op) => op.status === "running");
    if (!running.length) return;
    for (const op of running) {
      _activityLogs.set(op.id, { lines: [] });
      streamJob(op.id, (line) => {
        const entry = _activityLogs.get(op.id);
        if (entry) entry.lines.push(line);
      }, (ev) => {
        const idx = _activityOps.findIndex((o) => o.id === op.id);
        if (idx !== -1 && typeof ev.pct === "number") {
          _activityOps[idx] = { ..._activityOps[idx], pct: ev.pct };
          renderActivityPill(_activityOps);
        }
      }).then((end) => {
        // "disconnected" is a CLIENT-only concept (streamJob gave up
        // reconnecting) - it is not a real job outcome, so it must never
        // overwrite the last known SERVER-reported status; the job may well
        // still be running, we have just lost our own update channel to it.
        if (end.status === "disconnected") return;
        const idx = _activityOps.findIndex((o) => o.id === op.id);
        if (idx !== -1) {
          _activityOps[idx] = { ..._activityOps[idx], status: end.status };
          renderActivityPill(_activityOps);
        }
      });
    }
    toast(running.length === 1
      ? `Reattached to a running ${activityLabel(running[0])}`
      : `Reattached to ${running.length} running operations`);
  } catch (e) { /* server unreachable at boot - same as reattachSessions() */ }
}

// In-page API-key gate. Shown when an authed boot returns 401 and this browser
// has no working key - the network/phone case, where the loopback key is never
// auto-seeded. Replaces window.prompt() (suppressed by mobile/PWA browsers, the
// NET-1 white-page cause). Idempotent.
export function showKeyGate(message) {
  const gate = $("key-gate");
  if (!gate) return;
  if (message) { const m = $("key-gate-msg"); if (m) m.textContent = message; }
  gate.style.display = "flex";
  // Show the "Install certificate" step only when the local CA is genuinely NOT
  // trusted yet (see updateKeyGateCertStep), so a returning trusted device is
  // never told to reinstall it each time the gate appears.
  updateKeyGateCertStep();
  // Offer "Scan QR code" wherever the browser can open a camera (secure context).
  // Decoding uses the native BarcodeDetector when present, else the bundled jsQR
  // fallback, so it is not limited to Android Chrome.
  const scan = $("key-gate-scan");
  if (scan) scan.style.display = scanSupported() ? "inline-block" : "none";
  const input = $("key-gate-input");
  if (input) {
    input.value = "";   // HttpOnly key is unreadable; the gate only shows unauthed
    input.focus();
  }
}

// Decide whether the key gate should offer "Install certificate".
export function updateKeyGateCertStep() {
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
    // Firefox keeps its OWN certificate store and ignores the OS one, so a system
    // install still warns here (the common "I installed it but it still warns"
    // case). Surface the Firefox-specific step prominently.
    const ff = $("key-gate-cert-ff");
    if (ff) {
      const isFirefox = /firefox\//i.test(navigator.userAgent || "");
      ff.style.display = isFirefox ? "block" : "none";
    }
  }
}
window.updateKeyGateCertStep = updateKeyGateCertStep;

// POST the entered key to /api/session so the server sets the HttpOnly session
// cookie (the key never lives in JS) and returns the session CSRF token. Stash
// the token so a write issued before the reload already has it (the boot
// re-fetches it via refreshCsrf too), then reload to re-run authenticated.
export async function loginWithKey(key) {
  try {
    const r = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    });
    if (r.ok) {
      try { window.__LOCALM_CSRF__ = (await r.json()).csrf || ""; } catch (e) { /* body optional */ }
    }
    return r.ok;
  } catch (e) { return false; }
}

// Submit the gate: log in with the entered key (trimmed) then reload. An empty
// entry just reloads (still unauthenticated -> the gate shows again).
export function submitKeyGate() {
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
export function addRevealToggle(input) {
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

// Pairing QR scanner (phone): read the key QR shown in the computer's Settings
// (Companion app) with the camera, saving the key without typing. Decoding
// prefers the native BarcodeDetector and falls back to the bundled jsQR, so it
// works on any browser that can open a camera in a secure context (Firefox,
// Brave, Opera, iOS Safari, ...), not just Android Chrome.
export function scanSupported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

// Lazily load the bundled jsQR decoder (only fetched when a scan starts on a
// browser without BarcodeDetector). Sets window.jsQR (UMD). Cached after first
// load; the service worker caches the file for later offline pairing.
export let _jsqrPromise = null;
export function loadJsQR() {
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
export function handleScannedKey(text) {
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

export let _qrStream = null;
export let _qrTimer = null;
export function stopQrScan() {
  if (_qrTimer) { clearInterval(_qrTimer); _qrTimer = null; }
  if (_qrStream) { _qrStream.getTracks().forEach((t) => t.stop()); _qrStream = null; }
  const v = $("qr-video"); if (v) v.srcObject = null;
  const s = $("qr-scanner"); if (s) s.style.display = "none";
}

export async function startQrScan() {
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

// PWA install affordance (P2c): the "Install app" path differs per platform, so
// the Settings card shows the right thing - Android/desktop Chrome fire
// `beforeinstallprompt` and get a real Install button; iOS Safari fires nothing
// (only Share sheet -> Add to Home Screen) so it gets written steps; a standalone
// launch just confirms it. applyInstallUI() is the single decision point,
// unit-tested by branch.

export function pwaDisplayMode() {
  try {
    if (window.matchMedia && window.matchMedia("(display-mode: standalone)").matches) {
      return "standalone";
    }
  } catch (e) { /* matchMedia absent (older/headless) */ }
  // iOS records an installed PWA on navigator.standalone, not display-mode.
  if (window.navigator && window.navigator.standalone === true) return "standalone";
  return "browser";
}

export function isIOSSafari() {
  const nav = window.navigator || {};
  const ua = nav.userAgent || "";
  // iPadOS 13+ reports itself as a Mac, so add the touch check for that case.
  return /iPad|iPhone|iPod/.test(ua)
    || (nav.platform === "MacIntel" && (nav.maxTouchPoints || 0) > 1);
}

// Show exactly one of: installed confirmation, the native Install button, the
// iOS Add-to-Home-Screen steps, or the generic hint. env = {standalone, ios,
// canPrompt}; missing fields are treated as false.
export function applyInstallUI(env) {
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

// Onboarding install gate (mobile, P-mobile): after auth, a not-yet-installed
// phone lands on a one-time install screen (Install on Android / Add-to-Home-
// Screen steps on iOS), then taps "Continue" to enter the app. Desktop, an
// already-installed launch, and a return visit skip it.
export function shouldShowInstallGate() {
  if (pwaDisplayMode() === "standalone") return false;       // already installed
  try {
    // AUD-INSTANCEID (see helpers.js reconcileInstanceId): only trust a cached
    // "already onboarded" flag once this origin confirmed pairing with the
    // connected backend, else a fresh install reusing a prior instance's origin
    // would skip its own onboarding because a DIFFERENT install dismissed it here.
    if (instanceCacheTrusted() && localStorage.getItem("localm.onboarded") === "1") return false;
  } catch (e) { /* storage blocked - treat as not onboarded */ }
  // Phones/tablets only: a touch device with a coarse pointer. A desktop
  // `localm gui` (fine pointer) opens straight into the app.
  return (navigator.maxTouchPoints || 0) > 0
    || !!(window.matchMedia && window.matchMedia("(pointer: coarse)").matches);
}

// Render the gate's install affordance - mirror of applyInstallUI for the gate's
// own elements. env = {ios, canPrompt}; missing fields treated as false.
export function applyInstallGateUI(env) {
  env = env || {};
  const btn = $("install-gate-install");
  const ios = $("install-gate-ios");
  const hint = $("install-gate-hint");
  if (btn) btn.style.display = env.canPrompt ? "" : "none";
  if (ios) ios.style.display = (!env.canPrompt && env.ios) ? "" : "none";
  if (hint) hint.style.display = (!env.canPrompt && !env.ios) ? "" : "none";
}

export function showInstallGate() {
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
export function dismissInstallGate() {
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

export let modelCache = { models: [], active: "" };

// Tracks whether refreshModels() has completed at least once, so the very first
// (boot-time) call can be told apart from a later call that legitimately observes
// NO active model (an external unload, or a poll landing before anything has ever
// loaded) - modelCache.active is "" in both cases, so the value alone can't
// distinguish them. See the truthiness note below.
let _modelsEverRefreshed = false;

export async function refreshModels() {
  try {
    const r = await fetch("/api/models?type=llm", { headers: authHeaders() });
    if (r.status === 401) {
      // No working key (e.g. a network bind, where the loopback key is never
      // auto-seeded). Show the in-page key gate, not window.prompt() - mobile/PWA
      // browsers suppress prompt(), leaving a phone/LAN client on a blank page
      // (NET-1). The gate stores the key and reloads on submit.
      showKeyGate("This LocaLM server requires an API key.");
      return;
    }
    if (!r.ok) {
      // A non-401 error (500, 503, ...) returns a body with no `models` array
      // (e.g. FastAPI's {"detail": ...}); the empty-list fallback below would
      // then silently show an empty dropdown + "ok / no model" status, masking
      // the server error. Surface it instead.
      setStatus("err", `models unavailable (HTTP ${r.status})`);
      return;
    }
    const data = await r.json();
    const previousActive = modelCache.active;
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
    // ADR-0008 U4: do not clobber a deliberate busy state (e.g. a model load
    // still in flight) - this 30s poll has no idea one is running and used to
    // overwrite it unconditionally every time it landed mid-load.
    if (!_statusBusy) setStatus("ok", data.active || "no model");
    renderModelSplitLine(data.active_gpu_split);
    // The active model can change from OUTSIDE this tab too - another tab,
    // another device, the CLI, an MCP client - while Settings is already open. A
    // same-tab switch refreshes the Live Tuning VRAM estimate itself (switchModel's
    // callers each call refreshPerfEstimate directly), but this 30s poll is the
    // only thing that ever learns about an external switch, so it has to carry the
    // estimate refresh too or the panel is left showing the previous model's
    // numbers indefinitely (it otherwise only touches the dropdown/status line).
    // Deliberately NOT `previousActive && modelCache.active && ...`: requiring
    // BOTH sides truthy looks like it merely skips the boot-time call, but "" is
    // also the legitimate value after an external UNLOAD, so a truthy-only guard
    // masks that transition AND poisons the next one - an unload then a reload
    // from elsewhere would both be silently dropped, since neither the outgoing
    // nor the incoming side of one of those two polls is truthy. _modelsEverRefreshed
    // is the only thing that actually means "not the first call"; every value
    // change after that (including through "") must refresh.
    if (_modelsEverRefreshed && modelCache.active !== previousActive) {
      refreshPerfEstimate();
    }
    _modelsEverRefreshed = true;
  } catch (e) {
    setStatus("err", "server unreachable");
  }
}

/** Show the GPU split the ACTIVE model's load actually applied (auto
 *  free-VRAM-proportional, pinned ratios, or the equal fallback) under the
 *  model status - e.g. "Split: GPU 0 33% · GPU 1 67% (by free VRAM)". Hidden
 *  whenever /api/models carries no active_gpu_split (no split, no model, or
 *  a backend that does not record one), and re-hidden when a model switch
 *  drops it - a stale line would claim a distribution the current model does
 *  not have. */
export function renderModelSplitLine(split) {
  const line = document.getElementById("model-split-line");
  if (!line) return;
  const devices = split && Array.isArray(split.devices) ? split.devices : [];
  if (devices.length < 2) { line.hidden = true; line.textContent = ""; return; }
  const parts = devices.map(
    (d) => `GPU ${d.index} ${Math.round((d.share || 0) * 100)}%`);
  const why = split.source === "auto" ? " (by free VRAM)"
    : split.source === "pinned" ? " (pinned)"
    : split.source === "equal" ? " (equal)" : "";
  line.textContent = `Split: ${parts.join(" · ")}${why}`;
  line.hidden = false;
}

// Switch the active model. Returns the server status object
// ({status: "loaded" | "already_active" | "superseded", model, ...}).
// "superseded" means another model was selected while this one was still loading:
// the server aborted this load and the newer selection owns the UI, so we do NOT
// claim success or reset status here (that would flash the abandoned model's
// name). Callers should skip their success toast for it.
export async function switchModel(model) {
  setStatus("busy", "loading " + model + "…");
  const r = await fetch("/api/models/load", {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ model }),
  });
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  const data = await r.json().catch(() => ({ status: "loaded", model }));
  if (data.status === "superseded") return data;
  setStatus("ok", data.model || model);
  // Publish the newly active model NOW. refreshModels() is the only other writer
  // and it polls every 30s, so without this the client believes no model is
  // loaded until the next poll: sendChat's empty-model guard reads
  // modelCache.active and would falsely refuse "No model loaded" for up to ~30s
  // after a sidebar load actually succeeded (REG-471). Use the name the SERVER
  // reports (an alias may resolve to a different one); the next poll reconciles
  // the rest of the cache. Deliberately after the superseded return: that load
  // was abandoned, so claiming it would publish a model that is not loaded.
  modelCache.active = data.model || model;
  return data;
}

// Toast the outcome of a switchModel() call. A model too big to fully fit
// VRAM still loads (the backend deliberately defers to a partial/zero GPU
// offload rather than refusing), so a plain "switched" toast would read as
// success even when the load quietly fell back to (slow) CPU layers -
// res.degraded (from /api/models/load's gpu_layers_offloaded/gpu_layers_total)
// says so; warn instead of a bare success toast (AGENTS.md rule 5).
export function toastLoadResult(res, model) {
  if (res && res.degraded) {
    toast(`Model switched to ${model} (${res.gpu_layers_offloaded}/` +
          `${res.gpu_layers_total} layers on GPU, rest on CPU - slower)`, true);
  } else {
    toast("Model switched to " + model);
  }
}

modelSelect.onchange = async () => {
  const model = modelSelect.value;
  try {
    const res = await switchModel(model);
    // Superseded: a newer selection is now loading - stay quiet and let its own
    // handler report when it lands, instead of toasting a model we abandoned.
    if (!res || res.status !== "superseded") {
      toastLoadResult(res, model);
      // The Settings "Live tuning" VRAM estimate defaults to the active model
      // server-side, but only re-fetches on its own slider input - refresh it
      // here too so a switch made from the model dropdown is reflected without
      // needing to nudge a slider or leave and re-enter Settings.
      refreshPerfEstimate();
    }
  } catch (e) {
    setStatus("err", "load failed");
    toast("Model load failed: " + e.message, true);
    refreshModels();
  }
};

