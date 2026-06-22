// SPDX-License-Identifier: AGPL-3.0-or-later
// AUTH-1(b): recover a WEDGED auth state without a manual "clear site data".
//   - A successful login that still boots 401 (a stale service-worker shell)
//     self-heals: unregister the SW + drop caches + reload ONCE (guarded).
//   - A 401 with no prior successful login just shows the key gate (no reset).
//   - An UNREACHABLE server shows a "reconnecting" overlay, NOT the key gate, so
//     a dead server is not mistaken for a bad key and re-entered in a loop.
// We deliberately do NOT touch SameSite (the rejected misdiagnosis).

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 80));
const keyless401 = () => Promise.resolve({
  ok: false, status: 401, json: async () => ({}), text: async () => "",
});
const allOk = () => Promise.resolve({
  ok: true, status: 200, json: async () => ({ models: [], active: "" }), text: async () => "",
});
const down = () => Promise.reject(new Error("ECONNREFUSED"));

// jsdom's location.reload cannot always be redefined (and navigation is a no-op
// that only logs), so this is best-effort - the existing keygate tests do the
// same. The self-heal is proven by its observable side effects (SW unregister,
// cache delete, the one-shot guard), not by counting reloads.
function stubReload(window) {
  try { Object.defineProperty(window.location, "reload", { configurable: true, value: () => {} }); }
  catch (e) { /* jsdom no-op nav */ }
}
function stubSWAndCaches(window) {
  const unregistered = [];
  const deleted = [];
  Object.defineProperty(window.navigator, "serviceWorker", {
    configurable: true,
    value: { getRegistrations: async () => [{ unregister: async () => { unregistered.push(1); return true; } }] },
  });
  Object.defineProperty(window, "caches", {
    configurable: true,
    value: { keys: async () => ["localm-shell-v10"], delete: async (k) => { deleted.push(k); return true; } },
  });
  return { unregistered, deleted };
}

test("AUTH-1b: a successful login that still boots 401 self-heals (SW reset + reload, once)", async () => {
  const { window } = loadApp({ fetchImpl: keyless401 });
  await tick();   // load-time probe: 401, no marker -> gate shown, NO reset
  stubReload(window);
  const { unregistered, deleted } = stubSWAndCaches(window);
  window.sessionStorage.setItem("localm.loginOk", "1");   // a prior login DID succeed

  const ok1 = await window.bootAuthProbe();
  assert.equal(ok1, false, "still locked (it reloads to recover)");
  assert.equal(unregistered.length, 1, "the stale service worker was unregistered");
  assert.deepEqual(deleted, ["localm-shell-v10"], "its caches were dropped");
  assert.equal(window.sessionStorage.getItem("localm.swReset"), "1", "the one-shot guard is set");
  assert.equal(window.sessionStorage.getItem("localm.loginOk"), null, "the login marker was consumed");

  // It must NOT loop: a second wedged probe (guard already set) just shows the gate.
  const ok2 = await window.bootAuthProbe();
  assert.equal(ok2, false);
  assert.equal(unregistered.length, 1, "no second reset (guarded - cannot loop)");
  assert.notEqual(window.document.getElementById("key-gate").style.display, "none", "the gate is shown");
});

test("AUTH-1b: a 401 with NO prior login just shows the gate (no SW nuke)", async () => {
  const { window } = loadApp({ fetchImpl: keyless401 });
  await tick();
  stubReload(window);
  const { unregistered, deleted } = stubSWAndCaches(window);
  // no loginOk marker
  const ok = await window.bootAuthProbe();
  assert.equal(ok, false);
  assert.equal(unregistered.length, 0, "an ordinary keyless 401 never resets the SW");
  assert.equal(deleted.length, 0);
  assert.notEqual(window.document.getElementById("key-gate").style.display, "none", "the key gate is shown");
});

test("AUTH-1b: an UNREACHABLE server shows the reconnect overlay, not the key gate", async () => {
  const { window } = loadApp({ fetchImpl: down });
  await tick();
  const ov = window.document.getElementById("reconnect-overlay");
  assert.ok(ov, "a reconnect overlay is created");
  assert.notEqual(ov.style.display, "none", "and shown");
  assert.equal(window.document.getElementById("key-gate").style.display, "none",
    "the key gate is NOT shown for a server-down state (so the key is not re-entered in a loop)");
  assert.equal(window.document.getElementById("app").style.display, "none", "no app behind it");
  assert.equal(window.__localmLocked, true, "locked");
});

test("AUTH-1b: submitting the gate with a good key marks the login (so the reload can self-heal)", async () => {
  const fetchImpl = async (url) => (String(url).includes("/api/session") ? allOk() : keyless401());
  const { window } = loadApp({ fetchImpl });
  stubReload(window);
  await tick();
  window.document.getElementById("key-gate-input").value = "good-key";
  window.document.getElementById("key-gate-submit").click();
  await tick();
  assert.equal(window.sessionStorage.getItem("localm.loginOk"), "1",
    "a successful login is marked so a still-401 reload self-heals instead of looping");
});

test("AUTH-1b: a 200 boot clears the recovery markers and reveals the app", async () => {
  const { window } = loadApp({ fetchImpl: allOk });
  await tick();
  window.sessionStorage.setItem("localm.loginOk", "1");
  window.sessionStorage.setItem("localm.swReset", "1");
  const ok = await window.bootAuthProbe();
  assert.equal(ok, true, "authed");
  assert.equal(window.sessionStorage.getItem("localm.loginOk"), null, "markers cleared on success");
  assert.equal(window.sessionStorage.getItem("localm.swReset"), null);
  assert.equal(window.document.getElementById("app").style.display, "", "app revealed");
  assert.equal(window.document.getElementById("key-gate").style.display, "none", "gate hidden");
});
