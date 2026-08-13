// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages } from "./harness.mjs";

// NET-1: over a network bind the SPA boots without a key (the loopback key is
// never auto-seeded to remote clients). The boot must show an IN-PAGE key gate
// so a phone/LAN client can enter a key - NOT window.prompt(), which mobile/PWA
// browsers suppress (leaving a blank page). These tests pin that behaviour.

const tick = () => new Promise((r) => setTimeout(r, 100));

const keyless401 = () => Promise.resolve({
  ok: false, status: 401,
  json: async () => ({ detail: "Missing or invalid API key" }),
  text: async () => "Missing or invalid API key",
});
const allOk = () => Promise.resolve({
  ok: true, status: 200,
  json: async () => ({ models: [], active: "" }),
  text: async () => "",
});

test("NET-1: a keyless (401) boot shows the in-page key gate, not window.prompt", async () => {
  const { window } = loadApp({ fetchImpl: keyless401 });
  let promptCalls = 0;
  window.prompt = () => { promptCalls++; return null; };
  await tick();
  const gate = window.document.getElementById("key-gate");
  assert.ok(gate, "#key-gate element exists in the shell");
  assert.notEqual(gate.style.display, "none", "key gate is visible after a 401 boot");
  assert.equal(promptCalls, 0, "window.prompt() is NOT used (mobile/PWA suppress it)");
});

test("NET-1: a working backend leaves the gate hidden (loopback case unaffected)", async () => {
  const { window } = loadApp({ fetchImpl: allOk });
  await tick();
  const gate = window.document.getElementById("key-gate");
  assert.ok(gate, "#key-gate element exists");
  assert.equal(gate.style.display, "none", "key gate stays hidden when fetches succeed");
});

test("S2: submitting the gate POSTs the trimmed key to /api/session (no localStorage)", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    calls.push({ url: String(url), opts });
    if (String(url).includes("/api/session")) {
      return { ok: true, status: 200, json: async () => ({ authed: true }), text: async () => "" };
    }
    return keyless401();
  };
  const { window } = loadApp({ fetchImpl });
  try { Object.defineProperty(window.location, "reload", { configurable: true, value: () => {} }); } catch { /* jsdom no-op nav */ }
  await tick();
  window.document.getElementById("key-gate-input").value = "  my-secret-key  ";
  window.document.getElementById("key-gate-submit").click();
  await tick();
  const login = calls.find((c) => c.url.includes("/api/session")
    && (c.opts.method || "").toUpperCase() === "POST");
  assert.ok(login, "submitting POSTs the key to /api/session");
  assert.equal(JSON.parse(login.opts.body).key, "my-secret-key", "the trimmed key is sent");
  assert.equal(window.localStorage.getItem("localm.apiKey"), null,
    "the key is NOT written to localStorage (it lives only in the HttpOnly cookie)");
});

test("NET-1: over HTTPS an UNTRUSTED CA (SW failed to register) offers the Install-certificate link", async () => {
  const { window } = loadApp({ fetchImpl: keyless401, url: "https://192.168.0.5:8651/" });
  await tick();
  const cert = window.document.getElementById("key-gate-cert");
  const link = window.document.getElementById("key-gate-cert-link");
  assert.ok(cert, "#key-gate-cert exists in the shell");
  // Trust is unknown until SW registration resolves; we now OFFER the cert in that
  // state too. On a phone (Firefox, a clicked-through cert) the heuristic often
  // stays unknown, and hiding it there stranded the user on a self-signed cert
  // with no way to download it. Only a CONFIRMED-trusted CA hides the step.
  assert.notEqual(cert.style.display, "none",
    "unknown trust over HTTPS still offers the certificate (mobile must not be stranded)");
  // SW failed to register => the local CA is not trusted yet => offer the install.
  window.__swFailed = true;
  window.updateKeyGateCertStep();
  assert.notEqual(cert.style.display, "none", "cert link shown when the CA is untrusted over HTTPS");
  assert.equal(link.getAttribute("href"), "https://192.168.0.5:8651/localm-ca.crt",
    "links to the CA download over an absolute https URL, never the http port (J2)");
});

test("SEAMLESS: over HTTPS a TRUSTED CA (SW registered) HIDES the cert step - no reinstall nag", async () => {
  const { window } = loadApp({ fetchImpl: keyless401, url: "https://192.168.0.5:8651/" });
  await tick();
  // The service worker registered, which only happens behind a trusted cert.
  window.__swFailed = false;
  window.updateKeyGateCertStep();
  const cert = window.document.getElementById("key-gate-cert");
  assert.equal(cert.style.display, "none",
    "a returning trusted device is NOT told to reinstall the certificate every time");
});

test("the cert step surfaces a Firefox-specific note ONLY on Firefox", async () => {
  // Firefox uses its own certificate store and ignores the Windows/OS one, so a
  // system install leaves the warning here - the exact case the user hit. The step
  // must call this out, but only for Firefox (Chrome/Edge use the OS store).
  const { window } = loadApp({ fetchImpl: keyless401, url: "https://192.168.0.5:8651/" });
  await tick();
  window.__swFailed = true;   // untrusted CA over HTTPS -> the cert step is shown
  const ff = window.document.getElementById("key-gate-cert-ff");
  assert.ok(ff, "#key-gate-cert-ff exists in the shell");

  const setUA = (ua) => { try { Object.defineProperty(window.navigator, "userAgent",
    { value: ua, configurable: true }); } catch (e) { /* jsdom */ } };

  setUA("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36");
  window.updateKeyGateCertStep();
  assert.equal(ff.style.display, "none", "hidden for Chrome/Edge (they use the OS store)");

  setUA("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0");
  window.updateKeyGateCertStep();
  assert.notEqual(ff.style.display, "none", "shown for Firefox (its own cert store)");
});

test("NET-1: over plain HTTP the cert link stays hidden (loopback dev gate)", async () => {
  const { window } = loadApp({ fetchImpl: keyless401, url: "http://localhost:8642/" });
  await tick();
  // Even if SW registration "failed", there is no CA to install over plain HTTP.
  window.__swFailed = true;
  window.updateKeyGateCertStep();
  const cert = window.document.getElementById("key-gate-cert");
  assert.ok(cert, "#key-gate-cert exists");
  assert.equal(cert.style.display, "none", "no CA to install over plain HTTP, so hidden");
});

test("NET-1 hard gate: a keyless (401) boot HIDES the app shell - nothing of localm loads behind the gate", async () => {
  const { window } = loadApp({ fetchImpl: keyless401 });
  await tick();
  const app = window.document.getElementById("app");
  assert.equal(app.style.display, "none", "#app is hidden while locked (no shell behind the onboarding)");
  assert.equal(window.__localmLocked, true, "the lock flag is set so late boot steps bail too");
  const gate = window.document.getElementById("key-gate");
  assert.notEqual(gate.style.display, "none", "the onboarding/key gate is shown");
});

test("P2b/S2: scanning a localm-key QR logs in via /api/session (prefix stripped, no localStorage)", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    calls.push({ url: String(url), opts });
    if (String(url).includes("/api/session")) {
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    return keyless401();
  };
  const { window } = loadApp({ fetchImpl });
  try { Object.defineProperty(window.location, "reload", { configurable: true, value: () => {} }); } catch { /* jsdom no-op nav */ }
  await tick();
  assert.equal(window.handleScannedKey("localm-key:secret-123"), true);
  await tick();
  const login = calls.find((c) => c.url.includes("/api/session")
    && (c.opts.method || "").toUpperCase() === "POST");
  assert.ok(login, "a valid pairing QR logs in via /api/session");
  assert.equal(JSON.parse(login.opts.body).key, "secret-123");
  assert.equal(window.localStorage.getItem("localm.apiKey"), null);
});

test("P2b: a non-localm QR is ignored (no login attempt)", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => { calls.push({ url: String(url) }); return keyless401(); };
  const { window } = loadApp({ fetchImpl });
  await tick();
  assert.equal(window.handleScannedKey("https://example.com"), false);
  await tick();
  assert.ok(!calls.some((c) => c.url.includes("/api/session")),
    "a non-localm QR triggers no /api/session login");
});

const _safeFetch = async () => ({ ok: true, status: 200,
  json: async () => ({ models: [], conversations: [], plugins: [] }), text: async () => "" });

test("scanSupported: offered wherever a camera can open, not only with BarcodeDetector", async () => {
  const { window } = loadApp({ fetchImpl: _safeFetch });
  await tick();
  // No camera API -> hidden (e.g. an insecure context).
  assert.equal(window.scanSupported(), false);
  // A camera + getUserMedia is enough; the jsQR fallback covers decoding even
  // with no BarcodeDetector (Firefox / Brave / Opera / iOS Safari).
  Object.defineProperty(window.navigator, "mediaDevices",
    { value: { getUserMedia() {} }, configurable: true });
  assert.equal("BarcodeDetector" in window, false, "no native detector in this env");
  assert.equal(window.scanSupported(), true, "still offered via the jsQR fallback");
});

test("loadJsQR injects the bundled decoder script once", async () => {
  const { window } = loadApp({ fetchImpl: _safeFetch });
  await tick();
  const p1 = window.loadJsQR();
  const p2 = window.loadJsQR();
  assert.equal(p1, p2, "concurrent calls share one load promise (no double-inject)");
  const scripts = [...window.document.querySelectorAll('script[src="/vendor/jsQR.js"]')];
  assert.equal(scripts.length, 1, "exactly one jsQR script tag added");
});

test("NET-1 onboarding: the guided cert-install step + per-platform help render over HTTPS", async () => {
  const { window } = loadApp({ fetchImpl: keyless401, url: "https://192.168.0.5:8651/" });
  await tick();
  const help = window.document.querySelector("#key-gate-cert .onboard-help");
  assert.ok(help, "per-platform cert-install help is present");
  assert.match(help.textContent, /iOS|iPadOS/, "includes the iOS trust steps");
  const titles = [...window.document.querySelectorAll(".onboard-step-title")]
    .map((e) => e.textContent);
  assert.ok(titles.some((t) => /Trust this device/.test(t)), "cert step titled");
  assert.ok(titles.some((t) => /Enter your API key/.test(t)), "key-entry step titled");
});

test("NET-1 hard gate: a working (200) boot reveals the app and hides the gate", async () => {
  const { window } = loadApp({ fetchImpl: allOk });
  await tick();
  const app = window.document.getElementById("app");
  assert.notEqual(app.style.display, "none", "#app is visible when authed / open mode");
  assert.equal(window.__localmLocked, false, "unlocked");
  const gate = window.document.getElementById("key-gate");
  assert.equal(gate.style.display, "none", "gate hidden when authed / open mode");
});

test("S2: authHeaders sends the in-memory CSRF token and no localStorage bearer", async () => {
  const { window } = loadApp({ fetchImpl: allOk });
  await tick();
  // The CSRF token is DERIVED from the session server-side and stashed in memory
  // (fetched from GET /api/session), NOT read from a cookie that could desync.
  window.__LOCALM_CSRF__ = "tok123";
  const h = window.authHeaders();
  assert.equal(h["X-CSRF-Token"], "tok123", "CSRF token echoed from the in-memory session token");
  assert.ok(!("Authorization" in h), "no bearer header (session auth; no localStorage key)");
});

test("S3: once a session exists, authHeaders drops the shell-token bearer", async () => {
  // Open mode (loopback shell token, no session): the shell token is the
  // management credential and rides the Authorization header.
  const { window } = loadApp({ fetchImpl: allOk, shellToken: "SHELLXYZ" });
  await tick();
  window.__LOCALM_CSRF__ = "";     // boot's refreshCsrf found no session token
  const open = window.authHeaders();
  assert.equal(open["Authorization"], "Bearer SHELLXYZ", "open mode sends the shell bearer");
  assert.ok(!("X-CSRF-Token" in open), "no CSRF token before a session exists");
  // Session established (e.g. right after minting the first key): the header MUST
  // NOT carry the now-dead shell token, which would win server-side and 401.
  window.__LOCALM_CSRF__ = "tok999";
  const authed = window.authHeaders();
  assert.equal(authed["X-CSRF-Token"], "tok999", "session mode echoes the CSRF token");
  assert.ok(!("Authorization" in authed), "shell-token bearer suppressed once a session exists");
});

test("AUTH-2: both API-key inputs get a show/hide reveal toggle", async () => {
  const { window } = loadApp({ fetchImpl: allOk });
  await tick();
  for (const id of ["key-gate-input", "gui-api-key"]) {
    const input = window.document.getElementById(id);
    assert.ok(input, `#${id} exists`);
    assert.equal(input.type, "password", `${id} starts masked`);
    const wrap = input.closest(".input-reveal");
    assert.ok(wrap, `${id} is wrapped for the reveal toggle`);
    const btn = wrap.querySelector(".reveal-btn");
    assert.ok(btn, `${id} has a reveal button`);
    assert.equal(btn.type, "button", "the toggle never submits a form");
    assert.equal(btn.textContent, "show");
    btn.click();
    assert.equal(input.type, "text", `${id} reveals on click`);
    assert.equal(btn.textContent, "hide");
    btn.click();
    assert.equal(input.type, "password", `${id} re-masks on second click`);
    assert.equal(btn.textContent, "show");
  }
});

// P1a (COORDINATOR-DISPATCH-2026-08-11): several boot-time functions
// (refreshKbSelect/refreshPersonas/refreshMemory/refreshVoiceStatus/
// loadClientPlugins/refreshPluginCommands/setupPerfCard/startHwStats/
// reattachSessions/reattachActivity, plus the two setInterval polls) used to
// sit at the top level of init.js, past the end of the async boot IIFE - an
// async IIFE that is never awaited does not block the module body that
// follows it, so a keyless (401) client fired every one of them, with a
// bearer/session header, before the key gate ever showed. They now run
// sequenced behind bootAuthProbe() inside that same IIFE.
test("P1a: a keyless (401) boot fires ONLY the one auth probe - zero other " +
     "/api requests before the key gate is shown", async () => {
  const calls = [];
  const fetchImpl = async (url) => {
    calls.push(String(url));
    return {
      ok: false, status: 401,
      json: async () => ({ detail: "Missing or invalid API key" }),
      text: async () => "Missing or invalid API key",
    };
  };
  const { window } = loadApp({ fetchImpl });
  await tick(); await tick();

  assert.deepEqual(calls, ["/api/models"],
    "a keyless boot must fire exactly the one auth probe (bootAuthProbe's own " +
    "/api/models check) and nothing else - every other boot-time /api call must " +
    "wait behind a confirmed auth state, not race it");
  const gate = window.document.getElementById("key-gate");
  assert.notEqual(gate.style.display, "none", "sanity check: the key gate is showing");
});

test("P1a: an authenticated (200) boot still fires the normal boot-time /api " +
     "calls - the sequencing fix must not silently drop them for the common case", async () => {
  const calls = [];
  const fetchImpl = async (url) => {
    calls.push(String(url));
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl });
  await tick(); await tick();

  for (const path of ["/api/rag/collections", "/api/prompts", "/api/memory",
                       "/api/voice/status", "/api/capabilities", "/api/coder/sessions",
                       "/api/activity", "/api/stats"]) {
    assert.ok(calls.some((u) => u.startsWith(path)),
      `authed boot still fires ${path} (not gated away entirely)`);
  }
  assert.equal(window.__localmLocked, false, "sanity check: this boot is unlocked");
});

// P1a, SECOND HALF. The three boot-restore branches (`?shared=`, `?pull=`/
// `?view=`, and the saved-view fallback) stayed at the top level of init.js
// after the first half shipped, each wrapped in `setTimeout(..., 0)` and
// guarded on `window.__localmLocked`. A 0 ms timer is a MACROTASK queued during
// module evaluation, so it runs BEFORE bootAuthProbe's `await fetch(...)`
// resolves - the flag is still `undefined` there, and `undefined` fails OPEN in
// both polarities that were used. So a keyless client reached showView(), which
// calls window.onViewShown -> pages/dispatch.js -> the whole page's
// authenticated reads, before the key gate ever appeared.
//
// These use loadAppWithPages, not loadApp, ON PURPOSE. The existing P1a test
// above cannot fail on any of this in TWO independent ways: loadApp injects
// only app/*, so window.onViewShown is undefined and a firing showView() issues
// zero fetches; and it seeds no query string and no localStorage, so none of
// the three branches is even reachable. A fixture whose value space cannot
// intersect the defect's trigger space reads as covered while being unable to
// go red.
// THE FETCH MUST BE SLOW, AND THAT IS THE WHOLE EXPERIMENT - do not "simplify"
// it back to an immediately-resolved value. Every other mock in this file
// returns an already-resolved promise, which settles on the MICROTASK queue. A
// `setTimeout(..., 0)` is a MACROTASK, and microtasks all run first - so with an
// instant mock, bootAuthProbe resolves and lockUI() sets window.__localmLocked
// BEFORE the deep-link timer ever fires, and the old `if (window.__localmLocked)
// return;` guard reads `true` and correctly bails. The fixture would then pass
// on the UNFIXED code (measured: it did - all four cases green against pre-fix
// init.js) while the browser does the opposite, because a real network round
// trip resolves LONG after a 0 ms timer.
//
// Delaying the mock past one macrotask turn restores the real ordering: the
// timer fires first, reads `undefined`, and fails open. This is a third,
// independent way this fixture could not express the defect, on top of the two
// named above (no trigger seeded, no onViewShown loaded) - and the only one that
// survives fixing those, so it is the one that decides whether these tests mean
// anything.
const NET_MS = 25;   // > the 0 ms deep-link timer, like any real round trip
const keylessCounting = (calls) => async (url) => {
  calls.push(String(url));
  await new Promise((r) => setTimeout(r, NET_MS));
  return {
    ok: false, status: 401,
    json: async () => ({ detail: "Missing or invalid API key" }),
    text: async () => "Missing or invalid API key",
  };
};

for (const [label, opts] of [
  ["?view=models (no prior state at all - any link or hidden iframe reaches this)",
   { url: "http://localhost:8642/?view=models" }],
  ["?view=settings (fires the config schema, uploads list and VRAM estimate)",
   { url: "http://localhost:8642/?view=settings" }],
  ["?shared=1 (the PWA share target - ingestSharedFiles reads /api/share/pending)",
   { url: "http://localhost:8642/?shared=1" }],
  ["a saved view restored for a returning browser with a trusted instance id",
   { seedLocalStorage: { "localm.instanceId": "inst-abc", "localm.activeView": "settings" } }],
]) {
  test(`P1a: a keyless (401) boot with ${label} still fires ONLY the auth probe`, async () => {
    const calls = [];
    const { window } = loadAppWithPages({ fetchImpl: keylessCounting(calls), ...opts });
    await tick(); await tick();

    assert.deepEqual(calls, ["/api/models"],
      "the boot deep-link/restore path must not reach showView() -> onViewShown " +
      "-> the page's authenticated /api and /v1 reads until bootAuthProbe has " +
      "confirmed this client is authed; exactly the one probe may fire");
    assert.equal(window.__localmLocked, true, "sanity check: this boot is locked");
    const gate = window.document.getElementById("key-gate");
    assert.notEqual(gate.style.display, "none", "sanity check: the key gate is showing");
  });
}

test("P1a: an AUTHED boot still restores ?view=settings - the fix must sequence " +
     "the restore behind the probe, not drop it", async () => {
  // The trap this pins: flipping the guards to `window.__localmLocked === false`
  // would pass every locked case above and silently break the healthy one, since
  // the 0 ms timer still precedes unlockUI() and the flag is `undefined` then
  // too. Only moving the block behind the awaited `authed` satisfies both.
  const calls = [];
  const fetchImpl = async (url) => {
    calls.push(String(url));
    await new Promise((r) => setTimeout(r, NET_MS));   // see the note above
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", items: [], conversations: [] }) };
  };
  const { window } = loadAppWithPages({ fetchImpl, url: "http://localhost:8642/?view=settings" });
  await tick(); await tick();

  // Assert on a URL ONLY the settings page fetches. Matching "/api/models"
  // would prove nothing here: bootAuthProbe fetches it, and so does the 30 s
  // poll, so it is present whether or not the restore ran.
  assert.ok(calls.some((u) => u.startsWith("/api/uploads")),
    "the settings view was actually entered (refreshUploadsList ran)");
  assert.equal(window.__localmLocked, false, "sanity check: this boot is unlocked");
});
