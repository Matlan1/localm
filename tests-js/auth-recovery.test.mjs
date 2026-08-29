// SPDX-License-Identifier: AGPL-3.0-or-later

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

// Best-effort: jsdom's location.reload cannot always be redefined.
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
  window.sessionStorage.setItem("localm.loginOk", "1");   // a prior login succeeded

  const ok1 = await window.bootAuthProbe();
  assert.equal(ok1, false, "still locked (it reloads to recover)");
  assert.equal(unregistered.length, 1, "the stale service worker was unregistered");
  assert.deepEqual(deleted, ["localm-shell-v10"], "its caches were dropped");
  assert.equal(window.sessionStorage.getItem("localm.swReset"), "1", "the one-shot guard is set");
  assert.equal(window.sessionStorage.getItem("localm.loginOk"), null, "the login marker was consumed");

  // A second wedged probe, guard already set, just shows the gate.
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

// --- the open-mode shell token --------------------------------------------
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
  // The load-time probe already ran; assert on what it left behind.
  assert.notEqual(window.__localmLocked, false,
    "a 403 must never reach unlockUI() - that is what presented a dead shell as a live one");
  assert.equal(window.sessionStorage.getItem("localm.shellReset"), "1",
    "the stale-shell recovery ran (drop the SW + caches, reload for a fresh token)");
});

test("RESTART: a 403 that SURVIVES the one-shot recovery says so instead of " +
     "reloading again or unlocking", async () => {
  const { window } = loadApp({ fetchImpl: forbidden, shellToken: SHELL });
  await tick();
  // The guard is already set; re-arm the in-page latch so the probe runs again.
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
  // Boot healthy in open mode.
  let status = 200;
  const fetchImpl = async () => (status === 200
    ? { ok: true, status: 200, json: async () => ({ models: [], active: "" }), text: async () => "" }
    : { ok: false, status: 403, json: async () => ({}), text: async () => "" });
  const { window } = loadApp({ fetchImpl, shellToken: SHELL });
  await tick();
  assert.equal(window.__localmLocked, false, "premise: this boot unlocked normally");

  stubReload(window);
  const { unregistered, deleted } = stubSWAndCaches(window);
  status = 403;                       // the server re-execed; the token is stale
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

test("RESTART: a 403 from /api/discover/search is the route's own net_mode=off " +
     "answer, so it must NOT trigger the shell recovery either", async () => {
  // Same shape as the image-proxy case above: net_mode=off makes
  // discover.py's _ensure_online refuse, and _discover_status maps that
  // specific refusal to 403 - a real, expected outcome of a model search, not
  // a rejected credential. Reproduces a live browser bug: searching while
  // net_mode=off reloaded the whole app to the Chat tab mid-search, discarding
  // the query and any results already on screen.
  let status = 200;
  const fetchImpl = async () => (status === 200
    ? { ok: true, status: 200, json: async () => ({ models: [], active: "" }), text: async () => "" }
    : { ok: false, status: 403, json: async () => ({ detail: "Network access is off. Turn it on, or allow downloads only, in Settings → Network." }), text: async () => "" });
  const { window } = loadApp({ fetchImpl, shellToken: SHELL });
  await tick();
  assert.equal(window.__localmLocked, false, "premise: this boot unlocked normally");

  stubReload(window);
  const { unregistered, deleted } = stubSWAndCaches(window);
  status = 403;
  await window.fetch(
    "/api/discover/search?q=smollm&formats=gguf&types=llm",
    { headers: window.authHeaders() });
  await tick();

  assert.equal(unregistered.length, 0,
    "the service worker must survive: the credential was fine, net_mode said no");
  assert.deepEqual(deleted, [], "and its caches with it");
  assert.equal(window.sessionStorage.getItem("localm.shellReset"), null,
    "the one-shot recovery guard was never armed, so a later REAL stale token " +
    "still gets its single recovery");
});

test("RESTART: a 403 with NO shell token is NOT swept into the shell recovery", async () => {
  const { window } = loadApp({ fetchImpl: forbidden });   // no shellToken
  await tick();
  assert.equal(window.sessionStorage.getItem("localm.shellReset"), null,
    "no shell token means this is some other 403, not a stale open-mode shell");
});

test("RESTART: sentShellToken tells the two credential modes apart - it is what " +
     "keeps this recovery off every other kind of 403", async () => {
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

// Replaces setInterval so the reconnect poll can be stepped by hand, and counts
// clearInterval calls as reload decisions: the poll clears its own interval only
// on the line immediately before it reloads.
function armPoll(window) {
  const state = { reloadDecisions: 0, poll: null, down: false };
  window.setInterval = (fn) => { state.poll = fn; return 42; };
  window.clearInterval = () => { state.reloadDecisions += 1; };
  return state;
}

test("RESTART: the reconnect poll does not reload until it has seen the server " +
     "actually go DOWN, when the caller never observed a down", async () => {
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

  state.down = true;                     // the re-exec: the server goes away
  await p.poll();
  assert.equal(p.reloadDecisions, 0, "still down - keep waiting");

  state.down = false;                    // the NEW process answers
  await p.poll();
  assert.equal(p.reloadDecisions, 1,
    "a down-then-up transition proves this is a new process - now reload for a fresh token");
});

test("RESTART: the wait for a down is BOUNDED - a re-exec faster than one poll " +
     "must not strand the user on the overlay forever", async () => {
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

test("RESTART: with a priorInstanceId, the poll waits for a DIFFERENT " +
     "instance_id rather than reloading after a bounded count of \"still " +
     "reachable\" answers - the old process keeps answering /whoami with its " +
     "own id throughout its unload-and-wait sequence, well past that bound", async () => {
  const state = { instanceId: "old-proc" };
  const fetchImpl = async (url) => {
    if (String(url) === "/whoami") {
      return { ok: true, status: 200, json: async () => ({ instance_id: state.instanceId }),
        text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({ models: [] }), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl, shellToken: SHELL });
  await tick();
  const p = armPoll(window);

  window.onServerUnreachable({ priorInstanceId: "old-proc" });
  assert.ok(p.poll, "a reconnect poll was armed");

  for (let i = 0; i < 6; i++) {
    await p.poll();
    assert.equal(p.reloadDecisions, 0,
      `poll ${i + 1}: still the OLD process's instance_id - must not reload into it`);
  }

  state.instanceId = "new-proc";   // the re-exec'd process is up
  await p.poll();
  assert.equal(p.reloadDecisions, 1,
    "a DIFFERENT instance_id proves this is the new process - reload now");
});

test("RESTART: with a priorInstanceId, a /whoami answer with no instance_id " +
     "field falls back to the bounded up-poll count", async () => {
  const fetchImpl = async (url) => {
    if (String(url) === "/whoami") {
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({ models: [] }), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl, shellToken: SHELL });
  await tick();
  const p = armPoll(window);

  window.onServerUnreachable({ priorInstanceId: "old-proc" });
  await p.poll();
  await p.poll();
  assert.equal(p.reloadDecisions, 0, "still within the fallback bound");
  await p.poll();
  assert.equal(p.reloadDecisions, 1,
    "no instance_id to compare against - falls back to the bounded-count heuristic");
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
