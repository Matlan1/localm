// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - init. */
"use strict";

import { chat, convUI, ingestSharedFiles, initServerConversations, refreshCtxLimit, renderChat, renderConvList } from "./chat.js";
import { populateSetupModels, reattachSessions } from "./coder.js";
import { $, authHeaders, el, instanceCacheTrusted, refreshCsrf, sentShellToken } from "./helpers.js";
import { syncLogoStyleFromConfig } from "./logo.js";
import { addRevealToggle, applyInstallGateUI, dismissInstallGate, isIOSSafari, reattachActivity, refreshModels, shouldShowInstallGate, showInstallGate, showKeyGate, startHwStats, startQrScan, stopQrScan, submitKeyGate } from "./models-sidebar.js";
import { capsReady, loadClientPlugins, onVoicePick, populateVoicePicker, refreshKbSelect, refreshMemory, refreshPersonas, refreshPluginCommands, refreshVoiceStatus, setupPerfCard } from "./settings-perf.js";
import { VIEWS, showView } from "./tabs.js";

// CSRF self-heal: on a 403 for a same-origin request that carried our own
// X-CSRF-Token, refresh the token from /api/session and retry the write once.
// Installed before any fetch, so it covers boot too.
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
    // Open-mode counterpart: open (keyless loopback) mode has no CSRF token, so
    // the self-heal above never fires there and a rejected shell-token bearer
    // routes to its own recovery instead.
    else if (res.status === 403 && sentShellToken(hdrs)) {
      const url = typeof input === "string" ? input : (input && input.url) || "";
      if (url.startsWith("/") || url.startsWith(location.origin)) onShellTokenRejected();
    }
  } catch (e) { /* fall through with the original response */ }
  return res;
};

// Whether this origin already confirmed the backend's id on an earlier boot.
// Only then are the instance-scoped localStorage reads below - all of which run
// before this load's /v1/config round trip resolves - used for the first paint.
// refreshCtxLimit()/initServerConversations() correct everything once their
// round trip lands.
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

// Hard auth gate: hide the app shell and show only the onboarding, loading no
// /api data. window.__localmLocked lets late boot steps (deep-link restore,
// pages.js) bail too.
export function lockUI(message) {
  window.__localmLocked = true;
  const app = $("app");
  if (app) app.style.display = "none";
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

// Unregister the service worker and drop its caches. Callers pair this with a
// reload and a sessionStorage guard that bounds it to one attempt.
export async function resetServiceWorkerAndCaches() {
  try {
    if ("serviceWorker" in navigator) {
      const regs = await navigator.serviceWorker.getRegistrations();
      await Promise.all(regs.map((r) => r.unregister()));
    }
  } catch (e) { /* best-effort */ }
  try {
    if (window.caches) {
      const keys = await caches.keys();
      await Promise.all(keys.map((k) => caches.delete(k)));
    }
  } catch (e) { /* best-effort */ }
}
window.resetServiceWorkerAndCaches = resetServiceWorkerAndCaches;

// Manual escape hatch, offered on the reconnect overlay: wipe the client-side
// cookies, storage, service worker and caches, then reload to the key gate. Each
// step is independently guarded so one failure does not block the rest. (The
// HttpOnly localm_session cookie cannot be cleared from JS; a stale one yields a
// 401 and the key gate.)
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

// --- open-mode shell-token recovery ---------------------------------------
// In open (keyless loopback) mode the GUI's only management credential is the
// per-process shell token, baked into the index.html the serving process
// returns and read into a module-scope const in helpers.js, so it changes only
// by loading a new document. A page that outlives the process that served it
// holds a dead token, and the open-mode gate demands that bearer for every
// unsafe method and every /api|/v1 metadata GET.
//
// Recovery: drop the service worker and its caches so the reload cannot be
// served the same stale shell, then reload. A one-shot sessionStorage guard
// bounds it to a single attempt; a second rejection shows the overlay below
// instead of reloading again.
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
  // Confirm the server is still reachable before reloading, so a reload aimed
  // into a restart gap lands on the reconnect overlay rather than a browser
  // error page.
  if (!(await serverReachable())) { onServerUnreachable({ sawDown: true }); return; }
  try { sessionStorage.setItem("localm.shellReset", "1"); } catch (e) { /* ignore */ }
  await resetServiceWorkerAndCaches();
  location.reload();
}
window.onShellTokenRejected = onShellTokenRejected;

// Server-unreachable lock: a distinct "reconnecting" overlay with an auto-retry,
// shown instead of the key gate. When the server answers again, the page reloads
// for a clean boot.
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
    // Manual escape hatch: clears all client-side state and reloads to the key
    // gate.
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

// First-load progress overlay, shown immediately at boot and hidden once the
// model list resolves or fails.
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

// Reachability probe carrying no auth headers, so it cannot fail for a
// client-side reason. Any HTTP response, 401 included, counts as reachable; only
// a thrown fetch counts as down.
export async function serverReachable() {
  try {
    await fetch("/api/models", { cache: "no-store" });
    return true;
  } catch (e) {
    return false;
  }
}
window.serverReachable = serverReachable;

// How many consecutive reachable polls count as "the server came back" when it
// was never observed down.
const _UP_POLLS_WITHOUT_DOWN = 3;   // ~9s at the 3s poll interval

// *opts.sawDown* records whether the caller has already observed the server
// unreachable. bootAuthProbe passes true and reloads on the first poll that
// answers. The Settings "Restart server" button does not: it calls this while
// the old process is still answering, so this path waits until the server is
// seen down, or until _UP_POLLS_WITHOUT_DOWN answers have arrived, before
// reloading.
export function onServerUnreachable(opts) {
  window.__localmLocked = true;
  const app = $("app");
  if (app) app.style.display = "none";
  const gate = $("key-gate");
  if (gate) gate.style.display = "none";   // connectivity, not the key
  showReconnectOverlay();
  if (_reconnectTimer) return;
  let downSeen = !!(opts && opts.sawDown);
  let upPolls = 0;
  _reconnectTimer = setInterval(async () => {
    if (!(await serverReachable())) { downSeen = true; return; }   // still down - keep waiting
    if (!downSeen && ++upPolls < _UP_POLLS_WITHOUT_DOWN) return;    // may be the old process
    clearInterval(_reconnectTimer);
    _reconnectTimer = null;
    location.reload();                         // back up: a clean boot handles 200/401
  }, 3000);
}
window.onServerUnreachable = onServerUnreachable;

// Boot auth probe. Returns true when authed, and the caller then loads the app.
// Status 0 / unreachable -> reconnect overlay, not the gate; 401 -> key gate, or
// a one-shot stale-shell self-heal if a prior login should already have authed;
// 200 -> unlock.
export async function bootAuthProbe() {
  let status;
  try {
    const r = await fetch("/api/models", { headers: authHeaders() });
    status = r.status;
  } catch (e) {
    // The authed request could not complete. A header-free probe separates a
    // client-side failure (treated as 401, the key gate) from a dead server.
    status = (await serverReachable()) ? 401 : 0;
  }
  if (status === 0) { onServerUnreachable({ sawDown: true }); return false; }
  // A 403 on the boot probe while we hold a shell token is the open-mode
  // stale-shell-token case: run the recovery and never report authed. A 403
  // without a shell token falls through to the handling below.
  if (status === 403 && sentShellToken(authHeaders())) {
    await onShellTokenRejected();
    return false;
  }
  if (status === 401) {
    // A successful login (marked by submitKeyGate or the Settings key save)
    // that still boots 401 means the cached shell is wedged: reset it once and
    // reload.
    if (sessionStorage.getItem("localm.loginOk") === "1"
        && sessionStorage.getItem("localm.swReset") !== "1") {
      sessionStorage.removeItem("localm.loginOk");
      sessionStorage.setItem("localm.swReset", "1");   // one-shot guard
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
    // Clearing the shell-token guard re-arms the recovery for a later restart
    // in the same tab.
    sessionStorage.removeItem("localm.shellReset");
  } catch (e) { /* sessionStorage may be unavailable in some private modes */ }
  unlockUI();
  return true;
}
window.bootAuthProbe = bootAuthProbe;

(async () => {
  showStartupOverlay();   // immediate first-load feedback
  // Probe auth before loading any app data or revealing the shell.
  const authed = await bootAuthProbe();
  if (!authed) { hideStartupOverlay(); return; }   // gate / reconnect overlay takes over
  // Stash the session's CSRF token before any state-changing request can fire.
  // Awaited so it is ready before the app loads.
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

