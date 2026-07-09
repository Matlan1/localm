// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - init (split from app.js). Classic script: it
   shares the one global lexical environment with the other app/* and
   pages/* scripts, so every cross-section reference resolves by bare
   name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { chat, convUI, ingestSharedFiles, initServerConversations, refreshCtxLimit, renderChat, renderConvList } from "./chat.js";
import { populateSetupModels, reattachSessions } from "./coder.js";
import { $, authHeaders, el, instanceCacheTrusted, refreshCsrf } from "./helpers.js";
import { syncLogoStyleFromConfig } from "./logo.js";
import { addRevealToggle, applyInstallGateUI, dismissInstallGate, isIOSSafari, refreshModels, shouldShowInstallGate, showInstallGate, showKeyGate, startHwStats, startQrScan, stopQrScan, submitKeyGate } from "./models-sidebar.js";
import { loadClientPlugins, onVoicePick, populateVoicePicker, refreshKbSelect, refreshMemory, refreshPersonas, refreshPluginCommands, refreshVoiceStatus, setupPerfCard } from "./settings-perf.js";
import { VIEWS, showView } from "./tabs.js";

// CSRF self-heal: a cookie-authed write can 403 if the in-memory CSRF token went
// stale (server restart rotated the per-process secret; or a POST fired before
// the boot token fetch resolved). On such a 403, refresh from /api/session and
// retry the write ONCE. Safe: a 403 is rejected before the handler runs, so
// nothing is duplicated. Same-origin only, and only when WE sent an X-CSRF-Token
// (our own write). Installed before any fetch so it covers boot too. This, plus
// deriving the token from the session (never a separate cookie), is what makes
// "missing CSRF token" unreachable.
const _rawFetch = window.fetch.bind(window);
window.fetch = async function (input, init) {
  const res = await _rawFetch(input, init);
  try {
    const hdrs = init && init.headers;
    const sent = hdrs instanceof Headers ? hdrs.get("X-CSRF-Token")
               : (hdrs && typeof hdrs === "object") ? hdrs["X-CSRF-Token"] : null;
    if (res.status === 403 && sent) {
      const url = typeof input === "string" ? input : (input && input.url) || "";
      if (url.startsWith("/") || url.startsWith(location.origin)) {
        const fresh = await refreshCsrf();
        if (fresh && fresh !== sent) {
          const h2 = hdrs instanceof Headers ? new Headers(hdrs) : { ...hdrs };
          if (h2 instanceof Headers) h2.set("X-CSRF-Token", fresh);
          else h2["X-CSRF-Token"] = fresh;
          return await _rawFetch(input, { ...init, headers: h2 });
        }
      }
    }
  } catch (e) { /* fall through with the original response */ }
  return res;
};

// AUD-INSTANCEID (see helpers.js reconcileInstanceId): whether this origin
// already confirmed the backend's id on an earlier boot. Only then are the
// instance-scoped localStorage reads below (all run BEFORE this load's /v1/config
// round trip resolves) trusted for the FIRST paint; a brand-new pairing must not
// flash foreign data. refreshCtxLimit()/initServerConversations() then correct
// everything once their round trip lands.
const _instanceTrusted = instanceCacheTrusted();

$("setup-cwd").value = _instanceTrusted ? (localStorage.getItem("localm.coderCwd") || "") : "";
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
export function lockUI(message) {
  window.__localmLocked = true;
  const app = $("app");
  if (app) app.style.display = "none";          // nothing of localm behind the gate
  showKeyGate(message || "This LocaLM server requires an API key.");
}
export function unlockUI() {
  window.__localmLocked = false;
  const gate = $("key-gate");
  if (gate) gate.style.display = "none";
  hideReconnectOverlay();
  const app = $("app");
  if (app) app.style.display = "";
}

// Recovery (AUTH-1b): when auth is WEDGED - logged in successfully but the page
// still boots 401 - the cause is a stale service-worker shell (or a cached
// navigation that bypassed the loopback cookie re-seed), NOT the key. Do
// automatically what the user otherwise does by hand (clear site data):
// unregister the SW, drop its caches, reload once. A sessionStorage guard bounds
// it to one attempt so it can never loop. Do NOT touch SameSite (the cookie IS
// sent same-origin; that misdiagnosis would only open CSRF).
export async function resetServiceWorkerAndCaches() {
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
// (The HttpOnly localm_session cookie cannot be cleared from JS, but a stale one
// never wedges the client - it simply yields a 401 -> the key gate.)
export async function resetClientState() {
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

// Server-unreachable lock (AUTH-1b): the server is DOWN, NOT an auth failure.
// Show a distinct "reconnecting" overlay and auto-retry instead of the key gate,
// so a dead server is not mistaken for a bad key and re-entered in a loop. When
// it answers again, reload for a clean boot (which handles 200 vs 401 freshly).
export let _reconnectTimer = null;
export function showReconnectOverlay() {
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
export function hideReconnectOverlay() {
  const ov = $("reconnect-overlay");
  if (ov) ov.style.display = "none";
}

// R25: first-load progress. Shown immediately at boot so the cold-start wait
// (the first /api/models can block many seconds while the model loads) reads as
// in-progress, not a blank shell. Distinct in copy from the reconnect overlay
// (server DOWN). Hidden once the model list resolves or fails, so it never stacks
// over the gate or the app.
export function showStartupOverlay() {
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
export function hideStartupOverlay() {
  const ov = $("startup-overlay");
  if (ov) ov.style.display = "none";
}
window.showStartupOverlay = showStartupOverlay;
window.hideStartupOverlay = hideStartupOverlay;

// Reachability probe carrying NO auth headers, so it can NEVER fail for a
// client-side reason (a bad cookie/header that makes authHeaders throw). ANY HTTP
// response - even 401 - proves the server is reachable; only a thrown fetch means
// it is genuinely down. Makes the "server unreachable" verdict actually mean it.
export async function serverReachable() {
  try {
    await fetch("/api/models", { cache: "no-store" });
    return true;
  } catch (e) {
    return false;
  }
}
window.serverReachable = serverReachable;

export function onServerUnreachable() {
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
export async function bootAuthProbe() {
  let status;
  try {
    const r = await fetch("/api/models", { headers: authHeaders() });
    status = r.status;
  } catch (e) {
    // The authed request could not complete. Before declaring the server
    // unreachable (a dead-end overlay), confirm with a header-free probe: if the
    // server answers at all it is UP and the failure was client-side, so treat it
    // as "needs auth" (the recoverable key gate), NOT a dead server.
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
  // Stash the session's CSRF token (derived server-side, in lockstep with the
  // session) before any state-changing request can fire, so the first write never
  // has to rely on the 403 self-heal. Awaited so it is ready before the app loads.
  await refreshCsrf();
  // On a phone not yet installed, show the one-time install landing first; the
  // app still loads behind it and is revealed by "Continue". Desktop / installed
  // / returning visits fall straight through.
  if (shouldShowInstallGate()) showInstallGate();
  // Authenticated (or open/loopback mode): load the app.
  syncLogoStyleFromConfig();   // reconcile the wordmark with the shared config
  // R25: hide the startup overlay once the model list resolves (the app is usable)
  // or fails - never leave it stuck over the shell.
  const modelsReady = refreshModels().then(() => populateSetupModels());
  // Server persistence depends on knowing the privacy state first.
  const convReady = refreshCtxLimit().then(initServerConversations);
  // AUD-INSTANCEID (see helpers.js reconcileInstanceId): the startup overlay is
  // position:fixed/inset:0/opaque, fully covering #app and the conversation
  // sidebar. A browser confirmed for a DIFFERENT prior backend (instanceCacheTrusted
  // true but WRONG id) paints that backend's cached conv list synchronously at boot
  // (fast path below this IIFE). So wait on BOTH modelsReady AND convReady (whose
  // refreshCtxLimit runs the instance-id reconciliation) before hiding the overlay,
  // not refreshModels alone - otherwise the foreign list can flash before the
  // mismatch is detected and the DOM corrected.
  Promise.allSettled([modelsReady, convReady]).finally(hideStartupOverlay);
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
// R50: a plugin toggled in ANOTHER tab bumps a shared localStorage rev; the
// storage event fires in every OTHER same-origin tab, so we re-sync nav/commands
// promptly. A tab parked on the now-disabled plugin's page is redirected to chat
// by reconcileActiveView instead of erroring on its unmounted routes.
// visibilitychange covers a tab becoming visible without a focus event.
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
// AUD-INSTANCEID (see helpers.js reconcileInstanceId): never paint the raw,
// unverified localStorage cache before a same-instance confirmation exists for
// this origin (see _instanceTrusted above). A brand-new pairing starts from an
// empty list; the async refreshCtxLimit()/initServerConversations() chain below
// populates it for real once its round trip lands, so the first visible frame
// never shows another install's conversations.
if (!_instanceTrusted) {
  chat.conversations = [];
  chat.activeId = null;
  convUI.collapsed = new Set();
}
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
  const pullToken = params.get("pull_token");  // spec-bound one-time grant, see below
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
    setTimeout(async () => {
      if (window.__localmLocked) return;
      showView(VIEWS.includes(viewParam) ? viewParam : "models");
      if (pullSpec) {
        const specInput = $("pull-spec");
        if (specInput) {
          specInput.value = pullSpec;
          // SEC-PULL-CONFIRM: a `?pull=` query param can be put on ANY link, or
          // in a hidden iframe on any site the user visits while localm is
          // running locally - the page has no way to tell that apart from
          // `localm gui --pull` opening its own tab. Never start a real
          // download from a URL alone.
          //
          // `localm gui --pull` mints a single-use, spec-bound secret
          // server-side (mint_pull_grant, web.py) and passes it as
          // `pull_token` so ITS OWN deep link can redeem it here and
          // auto-start with zero clicks - unattended launches keep working.
          // A forged link cannot know that secret, so the redeem call below
          // 403s and we fall back to an explicit human confirmation instead.
          // window.confirm is a native browser-chrome dialog a page cannot
          // script past, auto-dismiss, or hide, so that fallback holds even
          // from an invisible iframe.
          let authorized = false;
          if (pullToken) {
            try {
              const r = await fetch("/api/models/pull-token/redeem", {
                method: "POST", headers: authHeaders(),
                body: JSON.stringify({ spec: pullSpec, token: pullToken }),
              });
              authorized = r.ok;
            } catch (e) { authorized = false; }
          }
          if (authorized || confirm(
            `Download model "${pullSpec}"?\n\n` +
            "A link or page just asked localm to start this download. Only " +
            "continue if you started this yourself (e.g. via " +
            '"localm gui --pull").')) {
            $("pull-start").click();   // kick off the pull with progress
          }
        }
      }
    }, 0);
  } else {
    // Restore the last active page (set in non-privacy mode only). Gated on a
    // confirmed same-instance cache (AUD-INSTANCEID, see helpers.js): an
    // unverified pairing ignores the stored view and stays on the chat default
    // rather than trust a value that may belong to a different backend.
    const savedView = _instanceTrusted ? localStorage.getItem("localm.activeView") : null;
    if (savedView && savedView !== "chat") {
      setTimeout(() => { if (!window.__localmLocked) showView(savedView); }, 0);
    }
  }
}

