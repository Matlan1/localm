// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the opt-in inline browser mirror in the coder session
// transcript (NEW-CAP-BROWSER item 1, browser_inline_live_view).
//
// jsdom's classic-script harness cannot perform a real dynamic import() (it
// rejects with "A dynamic import callback was not specified" - verified
// directly, not assumed), so loadBrowserPluginModule is a reassignable
// top-level let a test can substitute a fake module into.
//
// coder (and loadBrowserPluginModule) are top-level let/const, which - per
// this harness's own contract - live in the shared lexical environment, NOT
// as window properties, so `window.coder = ...` is a silent no-op and must
// go through runScript() instead. A fake object is stashed on window first
// (plain property assignment always works) and wired in from a runScript
// snippet that runs in the shared realm.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const settle = async (n = 20) => { for (let i = 0; i < n; i++) await Promise.resolve(); };

function fetchLog(routes) {
  const calls = [];
  const impl = async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    calls.push({ url: u, method, body: opts.body, signal: opts.signal });
    for (const [match, handler] of routes) {
      if (typeof match === "string" ? u.endsWith(match) : match(u, method)) {
        return handler(u, method, opts);
      }
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };
  return { calls, impl };
}

/** Builds a fake session, stashes it on window.__s and registers it in the
 *  real coder.sessions map via runScript, so both the test (window.__s) and
 *  the app code (coder.sessions.get(id)) see the SAME object. */
function makeSession(window, id, { active = false, withPanel = false } = {}) {
  // No top-level const/let in this snippet: classic scripts injected via
  // runScript() share ONE global lexical environment, so a second call
  // declaring the same const name throws "already been declared" - which
  // jsdom reports without halting the process, silently leaving window.__s
  // pointing at the PREVIOUS session. Caught live: a two-session test had
  // its second session collapse onto the first this way.
  window.__sid = id;
  runScript(window, `
    window.__s = { info: { id: window.__sid, cwd: "Z:/proj" },
      feedEl: document.createElement("div"), busy: false,
      lastEventAt: null, liveBody: null, liveText: "", liveReasoning: "",
      pendingCards: [], confirmCards: new Map(), closed: false,
      inlineBrowserAttempted: false, inlineBrowser: null };
    coder.sessions.set(window.__sid, window.__s);
  `);
  const s = window.__s;
  if (active) runScript(window, `coder.activeId = window.__sid;`);
  if (withPanel) {
    s.inlineBrowser = {
      mod: null, img: window.document.createElement("img"),
      status: window.document.createElement("div"), abort: null, jobId: null,
    };
  }
  return s;
}

/** Installs a fake browser.js module as the loader's result, via runScript
 *  so it overrides the real top-level `let loadBrowserPluginModule`. */
function useFakeModule(window, mod) {
  window.__fakeMod = mod;
  runScript(window, `loadBrowserPluginModule = () => Promise.resolve(window.__fakeMod);`);
}

function fakeBrowserModule() {
  const watchCalls = [];
  return {
    mod: {
      frameSrc: (data) => (typeof data === "string" && data ? "data:image/jpeg;base64," + data : null),
      watchFrames: (jobId, opts) => { watchCalls.push({ jobId, opts }); return new Promise(() => {}); },
    },
    watchCalls,
  };
}

test("the setting off: no panel is built, but the gate itself was checked", async () => {
  const { calls, impl } = fetchLog([
    ["/api/browser/state", async () => ({ ok: true, status: 200,
      json: async () => ({ open: false, enabled: true, inlineLiveView: false }) })],
  ]);
  const { window } = loadApp({ fetchImpl: impl });
  // A WORKING fake module, installed even though the panel must not use it:
  // without this, a real dynamic import always rejects under jsdom (verified
  // live), so s.inlineBrowser stays null whether the gate refused or the
  // gate itself is broken - the two are indistinguishable unless the module
  // load can actually succeed for the assertion to mean anything.
  const { mod } = fakeBrowserModule();
  useFakeModule(window, mod);
  const s = makeSession(window, "s1");

  await window.maybeAttachInlineBrowser(s);
  await settle();

  assert.equal(s.inlineBrowser, null, "a panel was built despite the setting being off");
  assert.ok(calls.some((c) => c.url.endsWith("/api/browser/state")),
    "the gating check itself never ran");
});

test("the setting on: the plugin module is loaded and a panel is attached", async () => {
  const { impl } = fetchLog([
    ["/api/browser/state", async () => ({ ok: true, status: 200,
      json: async () => ({ open: false, enabled: true, inlineLiveView: true }) })],
  ]);
  const { window } = loadApp({ fetchImpl: impl });
  const { mod } = fakeBrowserModule();
  useFakeModule(window, mod);
  const s = makeSession(window, "s1", { active: true });

  await window.maybeAttachInlineBrowser(s);
  await settle();

  assert.ok(s.inlineBrowser, "no panel was attached with the setting on");
  assert.equal(s.inlineBrowser.mod, mod);
  assert.equal(s.feedEl.querySelectorAll(".inline-browser-panel").length, 1,
    "the panel was not appended to the feed");
});

test("a failed module load (no browser plugin installed) is a silent no-op", async () => {
  const { impl } = fetchLog([
    ["/api/browser/state", async () => ({ ok: true, status: 200,
      json: async () => ({ open: false, enabled: true, inlineLiveView: true }) })],
  ]);
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, `loadBrowserPluginModule = () => Promise.reject(new Error("404"));`);
  const s = makeSession(window, "s1");

  await window.maybeAttachInlineBrowser(s);
  await settle();

  assert.equal(s.inlineBrowser, null);
});

test("attaching twice for the same session builds only one panel", async () => {
  const { impl } = fetchLog([
    ["/api/browser/state", async () => ({ ok: true, status: 200,
      json: async () => ({ open: false, enabled: true, inlineLiveView: true }) })],
  ]);
  const { window } = loadApp({ fetchImpl: impl });
  const { mod } = fakeBrowserModule();
  useFakeModule(window, mod);
  const s = makeSession(window, "s1", { active: true });

  await window.maybeAttachInlineBrowser(s);
  await settle();
  await window.maybeAttachInlineBrowser(s);
  await settle();

  assert.equal(s.feedEl.querySelectorAll(".inline-browser-panel").length, 1,
    "a second attach attempt built a second panel");
});

test("a browser_* tool_call attaches once; a later browser call does not re-attach", async () => {
  const { calls, impl } = fetchLog([
    ["/api/browser/state", async () => ({ ok: true, status: 200,
      json: async () => ({ open: false, enabled: true, inlineLiveView: true }) })],
    ["/api/browser/agent", async () => ({ ok: true, status: 200,
      json: async () => ({ job_id: "j1", session_id: "s1" }) })],
  ]);
  const { window } = loadApp({ fetchImpl: impl });
  const { mod } = fakeBrowserModule();
  useFakeModule(window, mod);
  const s = makeSession(window, "s1", { active: true });

  window.handleCoderEvent(s, { type: "tool_call", tool: "browser_navigate", args: {} });
  await settle();
  window.handleCoderEvent(s, { type: "tool_call", tool: "browser_click", args: {} });
  await settle();

  assert.equal(s.feedEl.querySelectorAll(".inline-browser-panel").length, 1,
    "more than one attach happened across several browser tool calls");
  assert.equal(
    calls.filter((c) => c.url.endsWith("/api/browser/state")).length, 1,
    "the gating check ran more than once for one session");
});

test("a non-browser tool_call never triggers an attach attempt", async () => {
  const { calls, impl } = fetchLog([
    ["/api/browser/state", async () => ({ ok: true, status: 200,
      json: async () => ({ open: false, enabled: true, inlineLiveView: true }) })],
  ]);
  const { window } = loadApp({ fetchImpl: impl });
  const s = makeSession(window, "s1");
  await settle(40);   // let loadApp's own async boot sequence finish first

  window.handleCoderEvent(s, { type: "tool_call", tool: "write_file", args: {} });
  await settle();

  assert.deepEqual(calls.filter((c) => c.url.includes("/api/browser/")), [],
    "a non-browser tool call reached a browser route");
  assert.equal(s.inlineBrowser, null);
});

test("startInlineBrowserStream targets this session's own coder_session_id and streams frames", async () => {
  const { calls, impl } = fetchLog([
    ["/api/browser/agent", async () => ({ ok: true, status: 200,
      json: async () => ({ job_id: "j-42", session_id: "s1" }) })],
  ]);
  const { window } = loadApp({ fetchImpl: impl });
  const { mod, watchCalls } = fakeBrowserModule();
  const s = makeSession(window, "s1");
  s.inlineBrowser = { mod, img: window.document.createElement("img"),
                      status: window.document.createElement("div"), abort: null, jobId: null };

  window.startInlineBrowserStream(s);
  await settle();

  const post = calls.find((c) => c.url.endsWith("/api/browser/agent") && c.method === "POST");
  assert.ok(post, "no request was sent to watch the agent browser");
  assert.deepEqual(JSON.parse(post.body), { coder_session_id: "s1" },
    "the request did not target this session's own coder_session_id");
  assert.equal(watchCalls.length, 1, "watchFrames was not called after the job was opened");
  assert.equal(watchCalls[0].jobId, "j-42");
  assert.equal(s.inlineBrowser.jobId, "j-42");

  watchCalls[0].opts.onFrame("AAA");
  assert.equal(s.inlineBrowser.img.src, "data:image/jpeg;base64,AAA");
  assert.equal(s.inlineBrowser.img.hidden, false);

  watchCalls[0].opts.onLine("watching the agent browser");
  assert.equal(s.inlineBrowser.status.textContent, "watching the agent browser");
});

test("starting twice while already streaming does not open a second job", async () => {
  const { calls, impl } = fetchLog([
    ["/api/browser/agent", async () => ({ ok: true, status: 200,
      json: async () => ({ job_id: "j-1", session_id: "s1" }) })],
  ]);
  const { window } = loadApp({ fetchImpl: impl });
  const { mod, watchCalls } = fakeBrowserModule();
  const s = makeSession(window, "s1");
  s.inlineBrowser = { mod, img: window.document.createElement("img"),
                      status: window.document.createElement("div"), abort: null, jobId: null };

  window.startInlineBrowserStream(s);
  await settle();
  window.startInlineBrowserStream(s);
  await settle();

  assert.equal(calls.filter((c) => c.url.endsWith("/api/browser/agent")).length, 1,
    "a second start opened a second viewer job while one was already running");
  assert.equal(watchCalls.length, 1);
});

test("stopInlineBrowserStream aborts the stream and cancels the job server-side", async () => {
  const { calls, impl } = fetchLog([
    ["/api/browser/agent", async () => ({ ok: true, status: 200,
      json: async () => ({ job_id: "j-9", session_id: "s1" }) })],
  ]);
  const { window } = loadApp({ fetchImpl: impl });
  const { mod, watchCalls } = fakeBrowserModule();
  const s = makeSession(window, "s1");
  s.inlineBrowser = { mod, img: window.document.createElement("img"),
                      status: window.document.createElement("div"), abort: null, jobId: null };

  window.startInlineBrowserStream(s);
  await settle();
  assert.equal(watchCalls.length, 1);
  const signal = watchCalls[0].opts.signal;
  assert.equal(signal.aborted, false);

  window.stopInlineBrowserStream(s);

  assert.equal(signal.aborted, true, "the watch's own AbortSignal was not aborted");
  assert.equal(s.inlineBrowser.abort, null);
  assert.ok(calls.some((c) => c.method === "POST" && c.url.endsWith("/api/jobs/j-9/cancel")),
    "the server-side viewer job was never cancelled, so it keeps streaming to nobody");
});

test("stopping when nothing is streaming is a no-op", async () => {
  const { window } = loadApp({});
  const s = makeSession(window, "s1", { withPanel: true });
  window.stopInlineBrowserStream(s);   // must not throw
  assert.equal(s.inlineBrowser.abort, null);
});

test("switching the active session stops the old stream and starts the new one", async () => {
  const { calls, impl } = fetchLog([
    ["/api/browser/agent", async (u, m, opts) => {
      const body = JSON.parse(opts.body);
      return { ok: true, status: 200,
               json: async () => ({ job_id: "job-" + body.coder_session_id,
                                    session_id: body.coder_session_id }) };
    }],
  ]);
  const { window } = loadApp({ fetchImpl: impl });
  const { mod: modA, watchCalls: watchA } = fakeBrowserModule();
  const { mod: modB, watchCalls: watchB } = fakeBrowserModule();
  const a = makeSession(window, "a");
  a.inlineBrowser = { mod: modA, img: window.document.createElement("img"),
                      status: window.document.createElement("div"), abort: null, jobId: null };
  const b = makeSession(window, "b");
  b.inlineBrowser = { mod: modB, img: window.document.createElement("img"),
                      status: window.document.createElement("div"), abort: null, jobId: null };

  window.activateSession("a");
  await settle();
  assert.equal(watchA.length, 1, "activating a did not start a's stream");
  const signalA = watchA[0].opts.signal;

  window.activateSession("b");
  await settle();
  assert.equal(signalA.aborted, true, "leaving a for b did not stop a's stream");
  assert.equal(watchB.length, 1, "activating b did not start b's stream");

  window.activateSession("a");
  await settle();
  assert.equal(watchA.length, 2, "returning to a did not resume a's stream");
});

test("closing a session stops its inline stream", async () => {
  const { impl } = fetchLog([
    ["/api/browser/agent", async () => ({ ok: true, status: 200,
      json: async () => ({ job_id: "j-close", session_id: "s1" }) })],
    ["/api/coder/sessions/s1", async () => ({ ok: true, status: 200, json: async () => ({}) })],
  ]);
  const { window } = loadApp({ fetchImpl: impl });
  const { mod, watchCalls } = fakeBrowserModule();
  const s = makeSession(window, "s1", { active: true });
  s.inlineBrowser = { mod, img: window.document.createElement("img"),
                      status: window.document.createElement("div"), abort: null, jobId: null };

  window.startInlineBrowserStream(s);
  await settle();
  const signal = watchCalls[0].opts.signal;

  await window.closeCoderSession(s);
  await settle();

  assert.equal(signal.aborted, true, "closing the session left its inline stream running");
});

test("the server closing the session (a 404 on its own event stream) stops the inline view too",
  async () => {
    const { impl } = fetchLog([
      ["/api/browser/agent", async () => ({ ok: true, status: 200,
        json: async () => ({ job_id: "j-404", session_id: "s1" }) })],
      [(u) => u.includes("/api/coder/sessions/s1/events"),
       async () => ({ ok: false, status: 404, json: async () => ({}) })],
    ]);
    const { window } = loadApp({ fetchImpl: impl });
    const { mod, watchCalls } = fakeBrowserModule();
    const s = makeSession(window, "s1", { active: true });
    s.inlineBrowser = { mod, img: window.document.createElement("img"),
                        status: window.document.createElement("div"), abort: null, jobId: null };

    window.startInlineBrowserStream(s);
    await settle();
    const signal = watchCalls[0].opts.signal;

    await window.streamSession(s, false);

    assert.equal(signal.aborted, true,
      "a 404 on the session's own event stream left the inline view streaming");
    assert.equal(s.closed, true);
  });
