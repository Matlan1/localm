// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - models sidebar selector (split from app.js). Classic script: it
   shares the one global lexical environment with the other app/* and
   pages/* scripts, so every cross-section reference resolves by bare
   name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { chat } from "./chat.js";
import { $, GIB, authHeaders, el, toast } from "./helpers.js";

/* ================================================================ */
/*  Models (sidebar selector)                                        */
/* ================================================================ */

export const modelSelect = $("model-select");

export function setStatus(state, text) {
  $("status-dot").className = "dot " + state;
  $("status-text").textContent = text;
}

// ---- live hardware monitor in the status bar (CPU/RAM/VRAM/GPU) ----------
// Renders whatever /api/stats reports; any section the box can't measure is
// simply absent (e.g. no psutil -> no CPU/RAM; AMD -> no GPU%). VRAM shows
// used/total when free is known, otherwise just total.
export function renderHwStats(data) {
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
export function showKeyGate(message) {
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

// Decide whether the key gate should offer "Install certificate". 
// The user requested this to be always visible on the API key authentication page.
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
  }
}
window.updateKeyGateCertStep = updateKeyGateCertStep;

// POST the entered key to /api/session so the server sets the HttpOnly session
// cookie (the key never lives in JS) and returns the session's CSRF token, then
// reload so the boot re-runs authenticated. Stash the token so a write issued
// before the reload already has it (the boot re-fetches it via refreshCsrf too).
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

// --- Pairing QR scanner (phone) -------------------------------------------
// Reads the key QR shown in the computer's Settings (Companion app) with the
// camera and saves the key without typing. Decoding prefers the native
// BarcodeDetector and falls back to the bundled jsQR, so it works on browsers
// that lack BarcodeDetector (Firefox, Brave, Opera, iOS Safari, ...) - any
// browser that can open a camera in a secure context.
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

// --- PWA install affordance (P2c) -----------------------------------------
// The browser "Install app" path differs per platform, so the Settings card
// shows the right thing: Android/desktop Chrome fire `beforeinstallprompt` and
// get a real Install button; iOS Safari fires nothing (the only path is the
// Share sheet -> Add to Home Screen), so it gets written steps; an already
// installed launch (standalone display) just confirms it. applyInstallUI() is
// the single decision point and is unit-tested by branch.

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

// --- Onboarding install gate (mobile, P-mobile) ----------------------------
// After auth, a phone that has not installed localm yet lands on a one-time
// install screen first (Install on Android / Add-to-Home-Screen steps on iOS),
// then taps "Continue" to enter the app. Desktop, an already-installed launch,
// and a return visit skip it. This is the "land on a setup page, reach localm
// via Continue" flow - the install affordance was previously buried in Settings.
export function shouldShowInstallGate() {
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

export async function refreshModels() {
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
    if (!r.ok) {
      // A non-401 error (500, 503, ...) returns a JSON body with no `models`
      // array (e.g. FastAPI's {"detail": ...}), so the empty-list fallback below
      // would silently show an empty dropdown + an "ok / no model" status,
      // masking the server error. Surface it instead.
      setStatus("err", `models unavailable (HTTP ${r.status})`);
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

// Switch the active model. Returns the server's status object
// ({status: "loaded" | "already_active" | "superseded", model, ...}).
// A "superseded" result means the user selected another model while this one was
// still loading: the server aborted this load and the newer selection now owns
// the UI, so we do NOT claim success or reset the status here (that would flash
// the abandoned model's name). Callers should skip their success toast for it.
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
  return data;
}

modelSelect.onchange = async () => {
  const model = modelSelect.value;
  try {
    const res = await switchModel(model);
    // Superseded: a newer selection is now loading - stay quiet and let its own
    // handler report when it lands, instead of toasting a model we abandoned.
    if (!res || res.status !== "superseded") toast("Model switched to " + model);
  } catch (e) {
    setStatus("err", "load failed");
    toast("Model load failed: " + e.message, true);
    refreshModels();
  }
};

