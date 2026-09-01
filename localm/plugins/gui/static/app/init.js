// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - init (split from app.js). Classic script: it
   shares the one global lexical environment with the other app/* and
   pages/* scripts, so every cross-section reference resolves by bare
   name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { chat, convUI, ingestSharedFiles, initServerConversations, refreshCtxLimit, renderChat, renderConvList } from "./chat.js";
import { populateSetupModels, reattachSessions } from "./coder.js";
import { $, authHeaders, el, instanceCacheTrusted, refreshCsrf, sentShellToken } from "./helpers.js";
import { syncLanguageFromConfig } from "./i18n.js";
import { syncLogoStyleFromConfig } from "./logo.js";
import { addRevealToggle, applyInstallGateUI, dismissInstallGate, isIOSSafari, reattachActivity, refreshModels, shouldShowInstallGate, showInstallGate, showKeyGate, startHwStats, startQrScan, stopQrScan, submitKeyGate } from "./models-sidebar.js";
import { capsReady, loadClientPlugins, onVoicePick, populateVoicePicker, refreshKbSelect, refreshMemory, refreshPersonas, refreshPluginCommands, refreshVoiceStatus, setupPerfCard } from "./settings-perf.js";
import { VIEWS, showView } from "./tabs.js";

// CSRF self-heal: a cookie-authed write can 403 if the in-memory CSRF token went
// stale (server restart rotated the per-process secret; or a POST fired before
// the boot token fetch resolved). On such a 403, refresh from /api/session and
// retry the write ONCE. Safe: a 403 is rejected before the handler runs, so
// nothing is duplicated. Same-origin only, and only when WE sent an X-CSRF-Token
// (our own write). Installed before any fetch so it covers boot too. This, plus
// deriving the token from the session (never a separate cookie), is what makes
// "missing CSRF token" unreachable.

// Same-origin paths whose 403 is the ROUTE's own answer rather than a rejected
// credential. The stale-shell recovery below skips them.
// See "a 403 from /api/image-proxy is the route saying the feature is OFF" in
// tests-js/auth-recovery.test.mjs.
// /api/discover/search and /api/discover/files answer 403 the same way
// whenever net_mode=off refuses the lookup (discover.py's _discover_status) -
// the same "route's own answer, not a rejected credential" shape as the
// image proxy, on the same shell-token credential in open mode.
const _ROUTE_OWNED_403 = ["/api/image-proxy", "/api/discover/search", "/api/discover/files"];

function _routeOwns403(url) {
  let path = url;
  try { path = new URL(url, location.origin).pathname; } catch (e) { /* use the raw value */ }
  return _ROUTE_OWNED_403.some((p) => path === p || path.startsWith(p + "/"));
}

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
    // Open-mode counterpart (NEW-RESTART-DEAD-BUTTONS). The self-heal above can
    // NEVER fire in open (keyless loopback) mode: it is gated on `sent`, the
    // X-CSRF-Token WE sent, and open mode has no CSRF token at all -
    // routes/session.py returns csrf="" while no key is configured, so
    // authHeaders() sends the shell-token bearer instead. That left the one
    // credential open mode actually uses with no recovery path of any kind.
    else if (res.status === 403 && sentShellToken(hdrs)) {
      const url = typeof input === "string" ? input : (input && input.url) || "";
      if ((url.startsWith("/") || url.startsWith(location.origin)) && !_routeOwns403(url)) {
        onShellTokenRejected();
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

// --- open-mode shell-token recovery (NEW-RESTART-DEAD-BUTTONS) -------------
// In open (keyless loopback) mode the GUI's ONLY management credential is the
// per-process shell token: http_server.py mints it with secrets.token_urlsafe
// at startup ("per-process so it dies on restart") and web.py bakes THAT
// process's value into the index.html it serves. helpers.js reads it into a
// module-scope const at load, so it can only ever change by loading a new
// document. A page that outlives the process which served it therefore holds a
// dead credential and has no way to refresh it in place.
//
// This is not a write-only problem. The open-mode gate in http_server.py
// requires that bearer for every unsafe method AND every /api|/v1 metadata GET
// (everything except /api/session and /v1/models*), so a stale token 403s
// essentially the whole shell at once - which is what "every button is dead
// after a restart" actually is. Two known ways to end up holding one:
//   - the reconnect poll reloads on the first HTTP answer it gets, and the OLD
//     process keeps answering for the whole unload + wait_for_vram_release
//     window before it re-execs, so that reload can be served by the process
//     that is about to disappear (see onServerUnreachable below, which now
//     waits to observe a down first);
//   - sw.js falls a navigation back to the PRECACHED "/" whenever the network
//     throws, which is exactly the case during the re-exec gap - and that
//     cached copy carries whatever token was live when the worker installed.
//
// The only source of a live token is a fresh document from the CURRENT process,
// so the recovery is: drop the service worker and its caches (otherwise the
// reload can be served the same stale shell straight back out of the cache),
// then reload. A one-shot sessionStorage guard bounds it exactly like the
// AUTH-1b self-heal above, so it can never loop: if a freshly served document
// still has its token rejected then staleness is not the cause, and we SAY so
// rather than reload again or - as before this fix - unlock a shell whose every
// call fails, with nothing on screen to explain it.
export function showShellStaleOverlay() {
  let ov = $("shell-stale-overlay");
  if (!ov) {
    ov = el("div", "reconnect-overlay");
    ov.id = "shell-stale-overlay";
    const panel = el("div", "reconnect-panel");
    panel.appendChild(el("div", "reconnect-msg",
      "This page was served by an earlier run of the LocaLM server, so it can "
      + "no longer make changes. Reloading will reconnect it."));
    const again = el("button", "reconnect-reset", "Reload");
    again.type = "button";
    again.onclick = () => location.reload();
    panel.appendChild(again);
    ov.appendChild(panel);
    document.body.appendChild(ov);
  }
  ov.style.display = "flex";
}
window.showShellStaleOverlay = showShellStaleOverlay;

export let _shellRecoveryStarted = false;
export async function onShellTokenRejected() {
  // A stale token 403s many in-flight calls at once; recover for the first.
  if (_shellRecoveryStarted) return;
  _shellRecoveryStarted = true;
  let alreadyTried = false;
  try { alreadyTried = sessionStorage.getItem("localm.shellReset") === "1"; }
  catch (e) { /* sessionStorage unavailable in some private modes */ }
  if (alreadyTried) { showShellStaleOverlay(); return; }
  // A 403 proves the server ANSWERED, but during a restart the process that
  // answered may be on its way out. Confirm it is still reachable before
  // reloading, so a reload aimed into the re-exec gap lands on our own
  // reconnect overlay (which retries) instead of the browser's error page.
  if (!(await serverReachable())) { onServerUnreachable({ sawDown: true }); return; }
  try { sessionStorage.setItem("localm.shellReset", "1"); } catch (e) { /* ignore */ }
  await resetServiceWorkerAndCaches();
  location.reload();
}
window.onShellTokenRejected = onShellTokenRejected;

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

// Identity probe carrying NO auth headers, same as serverReachable: returns the
// /whoami payload, or null if the server did not answer or the body did not parse.
export async function fetchWhoami() {
  try {
    const r = await fetch("/whoami", { cache: "no-store" });
    if (!r.ok) return null;
    return await r.json();
  } catch (e) {
    return null;
  }
}
window.fetchWhoami = fetchWhoami;

// How many consecutive REACHABLE polls to accept as "the server came back" when
// there is no per-process id to compare against (the sawDown path, or a /whoami
// answer with no instance_id).
const _UP_POLLS_WITHOUT_DOWN = 3;   // ~9s at the 3s poll interval

// *sawDown* records whether the caller has already OBSERVED the server
// unreachable. bootAuthProbe has (a thrown fetch is why it calls this), so it
// passes true and keeps the original behaviour: reload on the first poll that
// answers.
//
// *priorInstanceId*, when given, is the /whoami instance_id captured right
// before a restart was requested. Every poll then compares against it instead
// of guessing from a bounded up-poll count: the OLD process keeps answering
// through its own unload-and-wait sequence before it re-execs, so "still
// reachable" does not mean "the new process is up" - only a DIFFERENT
// instance_id does. Passed only by the Settings "Restart server" button.
export function onServerUnreachable(opts) {
  window.__localmLocked = true;
  const app = $("app");
  if (app) app.style.display = "none";
  const gate = $("key-gate");
  if (gate) gate.style.display = "none";   // a connectivity problem, not a key one
  showReconnectOverlay();
  if (_reconnectTimer) return;
  let downSeen = !!(opts && opts.sawDown);
  const priorInstanceId = opts && opts.priorInstanceId;
  let upPolls = 0;
  _reconnectTimer = setInterval(async () => {
    if (priorInstanceId) {
      const who = await fetchWhoami();
      if (!who) return;                                    // still down - keep waiting
      if (who.instance_id) {
        if (who.instance_id === priorInstanceId) return;    // still the OLD process - keep waiting
        // a different instance_id: this IS the new process
      } else if (++upPolls < _UP_POLLS_WITHOUT_DOWN) {
        return;                                             // no id to compare - bounded fallback
      }
      clearInterval(_reconnectTimer);
      _reconnectTimer = null;
      location.reload();
      return;
    }
    if (!(await serverReachable())) { downSeen = true; return; }   // still down - keep waiting
    if (!downSeen && ++upPolls < _UP_POLLS_WITHOUT_DOWN) return;    // may be the OLD process
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
  if (status === 0) { onServerUnreachable({ sawDown: true }); return false; }
  // A 403 on the boot probe is the open-mode stale-shell-token case
  // (NEW-RESTART-DEAD-BUTTONS): in open mode the gate in http_server.py demands
  // the per-process shell bearer for this very GET, so a page served by an
  // earlier process fails it. This used to fall through to unlockUI() below and
  // reveal a shell whose every subsequent call 403s, with nothing on screen
  // saying why - the dead-buttons report. Recover instead, and never report
  // authed. Scoped to the case we can actually diagnose: only when we HAVE a
  // shell token, i.e. we really are the open-mode client this describes. A 403
  // without one is some other condition and keeps its existing handling rather
  // than being swept into this recovery.
  if (status === 403 && sentShellToken(authHeaders())) {
    await onShellTokenRejected();
    return false;
  }
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
    // Same for the shell-token guard: this boot's credential works, so a LATER
    // restart in the same tab must get its own recovery attempt. Leaving it set
    // would make the second restart of a session skip straight to the "reload
    // by hand" overlay having never retried.
    sessionStorage.removeItem("localm.shellReset");
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
  syncLanguageFromConfig();    // reconcile the interface language with the shared config
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

  // P1a: everything below talks to an authenticated /api endpoint (or starts a
  // timer that will), so none of it may run until bootAuthProbe() above has
  // actually confirmed this client is authed (or in open/loopback mode). These
  // used to sit at the top level, past the end of this IIFE - but an async IIFE
  // that is never awaited does not block the module body that follows it: the
  // module's synchronous execution reached these calls and fired them before
  // bootAuthProbe()'s own `await fetch(...)` had a chance to resolve, so a
  // keyless client (a 401 boot) fired every one of them before the key gate
  // ever showed. Moving them here - behind the `authed` check above - is a
  // SEQUENCING fix, not a flag check: a `window.__localmLocked` guard on each
  // function would have been a structural no-op for this exact race, since the
  // flag itself is not set until lockUI()/unlockUI() run, which is further
  // down this same async chain.
  refreshKbSelect();
  refreshPersonas();
  refreshMemory();
  refreshVoiceStatus();
  // Kept (not fired and forgotten) because the `?view=` deep link below has to
  // wait for the plugin tabs' view sections to exist before it can land on one.
  const clientPluginsReady = loadClientPlugins();
  refreshPluginCommands();
  setInterval(refreshModels, 30000);
  startHwStats();   // live CPU/RAM/VRAM/GPU readout in the status bar
  setupPerfCard();  // Settings: GPU-layers/context sliders + live VRAM estimate
  // The resolved ctx ceiling only exists once a model has loaded - keep the
  // compaction threshold in sync as models load or switch.
  setInterval(refreshCtxLimit, 30000);
  reattachSessions();
  reattachActivity();   // ADR-0008 U4: cross-session/cross-tab background operations

  // Deep links + restore (P1a, second half). Every branch below ends in
  // showView(), and tabs.js's showView calls window.onViewShown(name) ->
  // pages/dispatch.js, which fires authenticated /api and /v1 reads for
  // whichever page it lands on (models -> refreshModelsPage; settings ->
  // refreshSettingsPage + refreshUploadsList + refreshPerfEstimate; images/
  // music/video -> warmComfyStatus + history). The `?shared=` branch fetches
  // /api/share/pending and the `?pull=` branch POSTs the redeem endpoint
  // directly.
  //
  // This block used to sit at the TOP LEVEL, below this IIFE, each branch
  // wrapped in `setTimeout(..., 0)` and guarded on `window.__localmLocked`.
  // Both halves of that were ineffective, in the same way the first half of
  // P1a was: a 0 ms timer is a MACROTASK queued during module evaluation, so
  // it runs BEFORE bootAuthProbe()'s `await fetch(...)` resolves - at which
  // point the flag is still `undefined`, because it is only ever assigned by
  // lockUI()/unlockUI()/onServerUnreachable() further down this same async
  // chain. `undefined` fails OPEN in BOTH polarities that were used
  // (`if (window.__localmLocked) return;` did not return; `if
  // (!window.__localmLocked) showView(...)` proceeded), so a keyless (401)
  // client fired the whole page's authenticated reads before the key gate
  // ever showed. No prior state was needed: `/?view=models` alone reached it,
  // from any link or hidden iframe pointing at the local server - the same
  // threat model SEC-PULL-CONFIRM names below.
  //
  // Running here instead is a SEQUENCING fix: `authed` above is a resolved
  // network answer, not a flag that may not be set yet. A bare `=== false`
  // flip would NOT have worked - the timer still precedes unlockUI(), so it
  // would have dropped the restore on every healthy authed boot instead.
  // The guards are gone rather than kept as defence in depth: reaching this
  // line already proves bootAuthProbe() returned true and unlockUI() ran, so
  // a re-check here would be provably dead code implying a protection this
  // block no longer needs. The `setTimeout(..., 0)` wrappers are gone with
  // them - they existed only to let pages/*.js install window.onViewShown and
  // the #pull-start handler first, and awaiting the boot probe's round trip
  // already orders this after the entire module graph has evaluated.
  const params = new URLSearchParams(location.search);
  const pullSpec = params.get("pull");      // from `localm gui --pull SPEC`
  const pullToken = params.get("pull_token");  // spec-bound one-time grant, see below
  const viewParam = params.get("view");
  const sharedTo = params.get("shared");    // from the PWA share target (phone)
  if (sharedTo) {
    // A phone shared image(s) into localm: land on chat and ingest them.
    history.replaceState(null, "", location.pathname);
    showView("chat");
    ingestSharedFiles();
  } else if (pullSpec || viewParam) {
    // Strip the query so a reload doesn't restart the download.
    history.replaceState(null, "", location.pathname);
    // A plugin tab (coder/images/music/video/knowledge/jobs) is NOT in VIEWS at
    // module-eval time: tabs.js seeds VIEWS with CORE_VIEWS alone, and
    // rebuildViews() appends the plugin tabs only once /api/capabilities has
    // been read. Deciding before that lands makes a perfectly valid tab fail the
    // includes() test below and silently become "models" - and the line above
    // has already stripped the query, so nothing can retry it. `localm gui`
    // prints /?view=models, a CORE view, which is why this never showed up in
    // normal use.
    //
    // So wait for the ANSWER instead of deciding from an absence, exactly as
    // pages/settings.js waits on capsReady before reading caps.fsAccess. BOTH
    // signals are needed and they are not the same one: capsReady covers VIEWS
    // (refreshPluginCommands -> renderNav -> rebuildViews, all inside the try
    // whose `finally` resolves it), while #view-jobs is built by the jobs client
    // entry itself and is reached only through loadClientPlugins()'s dynamic
    // imports - without that second wait showView() finds no #view-jobs and
    // falls back to chat. Neither promise can reject, and capsReady resolves in
    // a `finally` on every exit path, so an unreachable server degrades to the
    // "models" fallback below rather than hanging here.
    //
    // convReady is the THIRD signal and it is about precedence, not readiness.
    // Every browser that has never paired with this backend reconciles as
    // "mismatched" (helpers.js reconcileInstanceId - a brand-new origin counts,
    // not just a genuinely foreign one), and chat.js answers that by calling
    // showView("chat") to drop a saved view restored from another install. That
    // correction lands whenever /v1/config resolves, so without waiting for it
    // the deep link and the correction race: MEASURED here as a real
    // intermittent, ?view=knowledge landing on view-chat in one run of three
    // while passing in the other two. Waiting makes the explicit url win
    // deterministically instead of usually. It does not weaken the correction -
    // every instance-scoped wipe it performs still happens, and a `?view=` in
    // THIS url is not the foreign-install leftover that branch exists to clear.
    //
    // Gated on viewParam so a bare `?pull=` keeps its current timing.
    if (viewParam) {
      await Promise.allSettled([capsReady, clientPluginsReady, convReady]);
    }
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
  } else {
    // Restore the last active page (set in non-privacy mode only). Gated on
    // _instanceTrusted (AUD-INSTANCEID, see helpers.js instanceCacheTrusted) -
    // which is a PRESENCE check only (some instance id was confirmed for this
    // origin at some point), not yet a confirmed match with the backend THIS
    // load is talking to; that confirmation is an async round trip
    // (refreshCtxLimit -> reconcileInstanceId, chat.js) that may not have
    // resolved yet. So this restore is optimistic: correct for the common
    // case (the same backend as last time), and self-corrects via chat.js's
    // mismatch branch (showView("chat")) once the round trip confirms
    // otherwise - never trust this read as a real per-backend confirmation on
    // its own.
    //
    // NOT deferred the way the `?view=` branch above is, and that asymmetry is
    // deliberate. chat.js's confirmed-mismatch correction (showView("chat"),
    // AUD-INSTANCEID residual 1) is written on the premise that nothing
    // re-asserts the view after it runs. Awaiting the plugin set here would let
    // this restore land AFTER that correction and put a foreign install's
    // last-open page back on screen - reintroducing the exact leftover that
    // branch exists to clear. A `?view=` deep link carries no such residue: it
    // is an explicit request in THIS url, not cached state from another backend.
    const savedView = _instanceTrusted ? localStorage.getItem("localm.activeView") : null;
    if (savedView && savedView !== "chat") showView(savedView);
  }
})();
// Reveal toggles on the API-key inputs (AUTH-2): the in-page gate and the
// Settings key field, so the user can confirm the key they typed. Pure DOM
// wiring (no fetch) - stays outside the authed gate above so the key gate's
// own input is usable while still locked.
addRevealToggle($("key-gate-input"));
addRevealToggle($("gui-api-key"));
// Voice picker + client-side plugins (the tts plugin installs a neural voice).
// populateVoicePicker() itself only reads local browser APIs (speechSynthesis /
// the not-yet-installed ttsProvider) - no fetch, so it is safe before auth is
// confirmed too.
if (window.speechSynthesis) speechSynthesis.onvoiceschanged = populateVoicePicker;
if ($("p-voice")) $("p-voice").onchange = onVoicePick;
populateVoicePicker();
// Re-sync plugin command hints when the window regains focus, so a plugin
// toggled in another terminal/tab while sitting on the chat view is reflected
// without a reload (the view-switch path in pages.js covers navigation).
// Registering the listener is harmless while locked; P1a is about the fetch
// each callback fires. `=== false` requires bootAuthProbe to have EXPLICITLY
// unlocked (unlockUI sets it false) - undefined (never resolved yet) and true
// (locked) both correctly fall through and skip the call. A real focus/
// storage/visibility event needs a user already looking at the page, well
// after bootAuthProbe's one network round trip has settled in practice, but
// the guard costs nothing and closes the same class of gap this fix exists for.
window.addEventListener("focus", () => {
  if (window.__localmLocked === false) refreshPluginCommands();
});
// R50: a plugin toggled in ANOTHER tab bumps a shared localStorage rev; the
// storage event fires in every OTHER same-origin tab, so we re-sync nav/commands
// promptly. A tab parked on the now-disabled plugin's page is redirected to chat
// by reconcileActiveView instead of erroring on its unmounted routes.
// visibilitychange covers a tab becoming visible without a focus event.
window.addEventListener("storage", (e) => {
  if (e.key === "localm.pluginsRev" && window.__localmLocked === false) refreshPluginCommands();
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && window.__localmLocked === false) refreshPluginCommands();
});
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
// reattachSessions()/reattachActivity() and the deep-link/restore block both
// moved into the authed IIFE above (P1a) - they fetch authenticated state and
// used to run here, unguarded in practice, before bootAuthProbe() resolved.

