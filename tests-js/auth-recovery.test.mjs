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
import { loadApp, runScript } from "./harness.mjs";

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

// --- NEW-RESTART-DEAD-BUTTONS: the open-mode shell token ------------------
// In open (keyless loopback) mode the ONLY management credential is the
// per-process shell token baked into the served HTML. It rotates on every
// server start, helpers.js reads it into a const at load, and the open-mode
// gate demands it for every /api|/v1 metadata GET as well as every write - so a
// page that outlives the process which served it 403s on essentially
// everything. Before this fix nothing detected that: the #399 self-heal is
// gated on an X-CSRF-Token we never send in open mode, and a 403 fell through
// bootAuthProbe to unlockUI(), revealing a shell whose every call failed with
// nothing on screen to say why. That is the reported "every button is dead
// after a restart".
const SHELL = "SHELL-TOKEN-FROM-THE-OLD-PROCESS";
const forbidden = () => Promise.resolve({
  ok: false, status: 403,
  json: async () => ({ detail: "Open-mode management requires the localm GUI shell" }),
  text: async () => "",
});

test("RESTART: an open-mode boot whose shell token is STALE (403) never unlocks " +
     "the shell, and attempts the recovery", async () => {
  const { window } = loadApp({ fetchImpl: forbidden, shellToken: SHELL });
  await tick();
  // The load-time probe already ran. Assert on what it left behind: the shell
  // must NOT have been revealed as working, and the recovery must have been
  // attempted (its one-shot guard is the durable trace of that).
  assert.notEqual(window.__localmLocked, false,
    "a 403 must never reach unlockUI() - that is what presented a dead shell as a live one");
  assert.equal(window.sessionStorage.getItem("localm.shellReset"), "1",
    "the stale-shell recovery ran (drop the SW + caches, reload for a fresh token)");
});

test("RESTART: a 403 that SURVIVES the one-shot recovery says so instead of " +
     "reloading again or unlocking", async () => {
  const { window } = loadApp({ fetchImpl: forbidden, shellToken: SHELL });
  await tick();
  // Simulate "the reload already happened and the fresh document still 403s":
  // the guard is set, and we re-arm the in-page latch so the probe can run again.
  runScript(window, "_shellRecoveryStarted = false;");
  const { unregistered } = stubSWAndCaches(window);
  const ok = await window.bootAuthProbe();
  assert.equal(ok, false, "still not authed");
  assert.equal(unregistered.length, 0, "no second SW reset - the recovery cannot loop");
  const ov = window.document.getElementById("shell-stale-overlay");
  assert.ok(ov, "an explanatory overlay is created");
  assert.notEqual(ov.style.display, "none",
    "and shown - the failure is stated, not swallowed into a silently dead shell");
});

test("RESTART: a shell-token 403 on an ORDINARY call (the real restart case: the " +
     "page booted fine, then the server re-execed) triggers the recovery", async () => {
  // Boot healthy in open mode, exactly as a user would before clicking Restart.
  let status = 200;
  const fetchImpl = async () => (status === 200
    ? { ok: true, status: 200, json: async () => ({ models: [], active: "" }), text: async () => "" }
    : { ok: false, status: 403, json: async () => ({}), text: async () => "" });
  const { window } = loadApp({ fetchImpl, shellToken: SHELL });
  await tick();
  assert.equal(window.__localmLocked, false, "premise: this boot unlocked normally");

  stubReload(window);
  const { unregistered, deleted } = stubSWAndCaches(window);
  status = 403;                       // the server re-execed; our token is dead
  await window.fetch("/api/stats", { headers: window.authHeaders() });
  await tick();

  assert.equal(unregistered.length, 1,
    "the fetch wrapper recognised a rejected shell token and recovered - without " +
    "this the app just keeps 403ing with no signal");
  assert.deepEqual(deleted, ["localm-shell-v10"],
    "the SW caches go too: sw.js falls a navigation back to its PRECACHED index.html, " +
    "which carries whatever token was live when the worker installed");
});

test("RESTART: a 403 from /api/image-proxy is the route saying the feature is OFF, " +
     "so it must NOT trigger the shell recovery", async () => {
  // The test above is this one's control: the SAME 403, the SAME shell token, on
  // an ordinary route, DOES recover. Only the path differs.
  //
  // /api/image-proxy answers 403 whenever "Show remote images in replies" is off,
  // which is the shipped default, and helpers.js sends it with authHeaders(). So
  // in open mode a model reply carrying `![](https://host/x.png)` produced a 403
  // on a shell-token request, the wrapper read that as a rejected credential, and
  // the page unregistered its service worker, dropped its caches and reloaded -
  // mid-reply. Measured in a real browser: proxy OFF reloaded the page every
  // time, proxy ON never did, same page and same image URL.
  let status = 200;
  const fetchImpl = async () => (status === 200
    ? { ok: true, status: 200, json: async () => ({ models: [], active: "" }), text: async () => "" }
    : { ok: false, status: 403, json: async () => ({}), text: async () => "" });
  const { window } = loadApp({ fetchImpl, shellToken: SHELL });
  await tick();
  assert.equal(window.__localmLocked, false, "premise: this boot unlocked normally");

  stubReload(window);
  const { unregistered, deleted } = stubSWAndCaches(window);
  status = 403;
  await window.fetch(
    "/api/image-proxy?url=" + encodeURIComponent("https://example.invalid/a.png"),
    { headers: window.authHeaders() });
  await tick();

  assert.equal(unregistered.length, 0,
    "the service worker must survive: the credential was fine, the feature is off");
  assert.deepEqual(deleted, [], "and its caches with it");
  assert.equal(window.sessionStorage.getItem("localm.shellReset"), null,
    "the one-shot recovery guard was never armed, so a later REAL stale token " +
    "still gets its single recovery");
});

test("RESTART: a 403 with NO shell token is NOT swept into the shell recovery", async () => {
  // Negative control. The recovery keys on the shell token's VALUE, so it fires
  // only for the open-mode case it can actually diagnose; every other 403 keeps
  // its existing handling.
  const { window } = loadApp({ fetchImpl: forbidden });   // no shellToken
  await tick();
  assert.equal(window.sessionStorage.getItem("localm.shellReset"), null,
    "no shell token means this is some other 403, not a stale open-mode shell");
});

test("RESTART: sentShellToken tells the two credential modes apart - it is what " +
     "keeps this recovery off every other kind of 403", async () => {
  // The predicate, asserted directly, because that is the only thing here that
  // CAN fail. An end-to-end "a session-mode 403 takes the CSRF self-heal
  // instead" test was written first and then dropped: the CSRF branch of the
  // fetch wrapper RETURNS its retried response, the shell branch is an `else`,
  // AND authHeaders never emits both credentials - so the outcome is guaranteed
  // three times over and no single realistic mutation could turn that test red.
  // It would have read as coverage of the interaction while being structurally
  // incapable of detecting a regression in it. (The invariant it was really
  // leaning on - a session drops the shell bearer - is already pinned by "S3:
  // once a session exists, authHeaders drops the shell-token bearer" in
  // keygate.test.mjs.)
  const { window } = loadApp({ fetchImpl: allOk, shellToken: SHELL });
  await tick();

  window.__LOCALM_CSRF__ = "";                       // open mode
  assert.equal(window.sentShellToken(window.authHeaders()), true,
    "an open-mode request carries the shell bearer, so a 403 on it IS the stale-token case");

  window.__LOCALM_CSRF__ = "a-session-token";         // session mode
  assert.equal(window.sentShellToken(window.authHeaders()), false,
    "a session request sends X-CSRF-Token and no bearer - never route it here");

  assert.equal(window.sentShellToken({ Authorization: "Bearer somebody-elses-token" }), false,
    "matching the VALUE, not just the presence of a bearer, keeps this off a " +
    "hand-built Authorization header that has nothing to do with the shell token");
  assert.equal(window.sentShellToken(null), false, "no headers at all is not a shell request");
});

// Driving the reconnect poll. Two mechanics, both forced on us by the
// environment rather than chosen:
//
//  - jsdom's location.reload is NON-CONFIGURABLE, so it cannot be replaced and
//    reloads cannot be counted (the note at the top of this file says the same,
//    and stubReload above is best-effort for exactly this reason). The poll
//    clears its own interval on the line immediately before it reloads, and on
//    no other path, so a clearInterval call is a faithful 1:1 marker of "this
//    poll decided to reload" - which is the branch under test.
//  - setInterval is replaced so the 3s poll can be stepped by hand; waiting on
//    wall clock would make these tests take ~15s for no extra signal.
//
// Reachability is driven through the REAL serverReachable() by making fetch
// throw, rather than by stubbing serverReachable itself - the poll's whole job
// is to interpret that function's answer, so replacing it would test the stub.
function armPoll(window) {
  const state = { reloadDecisions: 0, poll: null, down: false };
  window.setInterval = (fn) => { state.poll = fn; return 42; };
  window.clearInterval = () => { state.reloadDecisions += 1; };
  return state;
}

test("RESTART: the reconnect poll does not reload until it has seen the server " +
     "actually go DOWN, when the caller never observed a down", async () => {
  // (d): _do_restart unloads the engines and waits on wait_for_vram_release
  // BEFORE os.execv, so the OLD process keeps answering for seconds after the
  // restart POST. Reloading on its first answer hands the browser a document
  // from the process that is about to die - a brand new page holding an already
  // doomed shell token, which is one of the two ways the dead-buttons state is
  // reached.
  const state = { down: false };
  const fetchImpl = async () => {
    if (state.down) throw new Error("ECONNREFUSED");
    return { ok: true, status: 200, json: async () => ({ models: [] }), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl, shellToken: SHELL });
  await tick();
  const p = armPoll(window);

  window.onServerUnreachable();          // the restart path: no down observed yet
  assert.ok(p.poll, "a reconnect poll was armed");
  await p.poll();
  assert.equal(p.reloadDecisions, 0,
    "the first answer may still be the OLD process mid-unload - do not reload into it");

  state.down = true;                     // the re-exec: it really goes away
  await p.poll();
  assert.equal(p.reloadDecisions, 0, "still down - keep waiting");

  state.down = false;                    // the NEW process answers
  await p.poll();
  assert.equal(p.reloadDecisions, 1,
    "a down-then-up transition proves this is a new process - now reload for a fresh token");
});

test("RESTART: the wait for a down is BOUNDED - a re-exec faster than one poll " +
     "must not strand the user on the overlay forever", async () => {
  // The gap between os.execv and the new process serving can be shorter than
  // the 3s poll, so the down transition can be missed entirely. Waiting forever
  // for a transition that already happened would be strictly worse than the bug
  // being fixed, so the wait gives up and reloads anyway.
  const { window } = loadApp({ fetchImpl: allOk, shellToken: SHELL });
  await tick();
  const p = armPoll(window);
  window.onServerUnreachable();
  await p.poll();
  await p.poll();
  assert.equal(p.reloadDecisions, 0, "still holding out for a down");
  await p.poll();
  assert.equal(p.reloadDecisions, 1,
    "bounded: after 3 straight answers it reloads regardless, which is the " +
    "pre-existing behaviour - and if that lands on the doomed process, the " +
    "shell-token recovery picks it up");
});

test("RESTART: a caller that DID observe the server down still reloads on the " +
     "first answer (no added latency for a plain outage)", async () => {
  const { window } = loadApp({ fetchImpl: allOk, shellToken: SHELL });
  await tick();
  const p = armPoll(window);

  window.onServerUnreachable({ sawDown: true });   // bootAuthProbe's path
  await p.poll();
  assert.equal(p.reloadDecisions, 1,
    "a confirmed outage reloads as soon as the server answers - the extra wait " +
    "applies only to the restart path, which has not seen a down yet");
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
