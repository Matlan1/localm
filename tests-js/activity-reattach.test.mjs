// SPDX-License-Identifier: AGPL-3.0-or-later
// ADR-0008 U4: a persistent status-bar activity affordance for background
// operations this tab did not necessarily start (another browser session, or
// this same tab before a reload) - the maintainer's own reported case
// ("another localm session for that same server has no idea", "closing and
// reopening the browser tab also loses the info"). Polled on the same tick
// as pollHwStats (models-sidebar.js's startHwStats), reattached at boot
// mirroring coder.js's reattachSessions().
//
// streamJob() is stubbed directly in most tests here (its own reconnect/
// dedup contract is covered by streamjob-reconnect.test.mjs) so these tests
// stay focused on the WIRING: does reattachActivity() call it correctly,
// handle its result correctly, and never fabricate a state it does not have.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));

function makeActivityFetch(responses) {
  // responses: array of either an operations array, or "fail" for a bad read.
  // The LAST entry repeats for any call beyond its length.
  let call = 0;
  const calls = [];
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u === "/api/activity") {
      calls.push(u);
      const r = responses[Math.min(call, responses.length - 1)];
      call++;
      if (r === "fail") return { ok: false, status: 500, json: async () => ({}) };
      return { ok: true, status: 200, json: async () => ({ operations: r }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  return { fetchImpl, calls };
}

// --------------------------------------------------------------------------- //
//  pollActivity() / renderActivityPill()                                     //
// --------------------------------------------------------------------------- //

test("pollActivity: a single running operation shows its label and pct in the pill", async () => {
  const { fetchImpl } = makeActivityFetch([
    [{ id: "j1", kind: "pull", label: "Model pull qwen2.5", status: "running", pct: 42 }],
  ]);
  const { window: win } = loadApp({ fetchImpl });
  await win.pollActivity();

  const pill = win.document.getElementById("activity-pill");
  assert.notEqual(pill.style.display, "none");
  assert.match(pill.textContent, /Model pull qwen2\.5/);
  assert.match(pill.textContent, /42%/);
});

test("pollActivity: multiple running operations collapse to 'N running'", async () => {
  const { fetchImpl } = makeActivityFetch([
    [{ id: "j1", kind: "pull", status: "running" },
     { id: "j2", kind: "imagine", status: "running" }],
  ]);
  const { window: win } = loadApp({ fetchImpl });
  await win.pollActivity();

  const pill = win.document.getElementById("activity-pill");
  assert.match(pill.textContent, /2 running/);
});

test("pollActivity: no running operations hides the pill (done/failed/cancelled do not count)", async () => {
  const { fetchImpl } = makeActivityFetch([
    [{ id: "j1", kind: "pull", status: "done" }, { id: "j2", kind: "imagine", status: "failed" }],
  ]);
  const { window: win } = loadApp({ fetchImpl });
  await win.pollActivity();

  const pill = win.document.getElementById("activity-pill");
  assert.equal(pill.style.display, "none");
});

test("pollActivity: an unreadable state (R1) keeps the pill's last known rendering, never fabricates 'nothing running'", async () => {
  // Response[0] is consumed by init.js's OWN automatic startHwStats() ->
  // pollActivity() call at boot (loadApp() runs the real boot sequence) -
  // this deliberately tests that real path rather than fighting it. The
  // explicit call below is then the "next poll tick", which fails.
  const { fetchImpl } = makeActivityFetch([
    [{ id: "j1", kind: "pull", label: "Model pull", status: "running", pct: 10 }],
    "fail",
  ]);
  const { window: win } = loadApp({ fetchImpl });
  await tick(); await tick(); await tick();
  const pill = win.document.getElementById("activity-pill");
  assert.notEqual(pill.style.display, "none", "precondition: pill visible after boot's own read");
  const textBefore = pill.textContent;

  await win.pollActivity();   // simulates the next 2500ms tick, which fails
  assert.notEqual(pill.style.display, "none",
    "a failed read must not hide the pill - that would falsely claim nothing is running");
  assert.equal(pill.textContent, textBefore, "the stale-but-known reading is kept verbatim");
});

// --------------------------------------------------------------------------- //
//  refreshModels() must not clobber a deliberate busy state (the status dot) //
// --------------------------------------------------------------------------- //

function makeModelsFetch() {
  return async (url) => {
    if (String(url) === "/api/models?type=llm") {
      return { ok: true, status: 200, json: async () => ({ models: [], active: "some-model" }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

test("refreshModels: does not clobber a deliberately-set busy status dot", async () => {
  const { window: win } = loadApp({ fetchImpl: makeModelsFetch() });
  win.setStatus("busy", "loading something…");
  await win.refreshModels();

  assert.equal(win.document.getElementById("status-dot").className, "dot busy",
    "the 30s poll must not overwrite an in-flight busy state");
  assert.equal(win.document.getElementById("status-text").textContent, "loading something…");
});

test("refreshModels: still sets 'ok' normally when nothing is busy (unchanged behavior)", async () => {
  const { window: win } = loadApp({ fetchImpl: makeModelsFetch() });
  await win.refreshModels();

  assert.equal(win.document.getElementById("status-dot").className, "dot ok");
  assert.equal(win.document.getElementById("status-text").textContent, "some-model");
});

// --------------------------------------------------------------------------- //
//  reattachActivity() - boot-time, mirrors reattachSessions()                 //
// --------------------------------------------------------------------------- //

// NOTE: none of these tests call reattachActivity() explicitly - init.js's own
// boot sequence (which loadAppWithPages() runs for real) already calls it
// once, and streamJob is stubbed via runScript BEFORE that pending call
// reaches its own await point, so the stub is what boot's automatic call
// actually uses. Calling it a second time here would double-count.

test("reattachActivity: a running operation is reattached via streamJob and toasted once", async () => {
  const { fetchImpl } = makeActivityFetch([
    [{ id: "j1", kind: "pull", label: "Model pull qwen2.5", status: "running" }],
  ]);
  const { window: win } = loadAppWithPages({ fetchImpl });
  runScript(win, `
    window.__streamJobCalls = [];
    streamJob = async (id, onLine, onProgress) => {
      window.__streamJobCalls.push(id);
      return new Promise(() => {});   // stay "running" - do not resolve in this test
    };
  `);
  await tick(); await tick(); await tick();

  assert.deepEqual(Array.from(win.__streamJobCalls), ["j1"], "streamJob was called once, for the running job's id");
  const toastEl = win.document.getElementById("toast");
  assert.match(toastEl.textContent, /Reattached to a running Model pull qwen2\.5/);
});

test("reattachActivity: no running operations reattaches nothing and does not toast", async () => {
  const { fetchImpl } = makeActivityFetch([
    [{ id: "j1", kind: "pull", status: "done" }],
  ]);
  const { window: win } = loadAppWithPages({ fetchImpl });
  runScript(win, `
    window.__streamJobCalls = [];
    streamJob = async (id) => { window.__streamJobCalls.push(id); return { status: "done" }; };
  `);
  await tick(); await tick(); await tick();

  assert.deepEqual(Array.from(win.__streamJobCalls), []);
  const toastEl = win.document.getElementById("toast");
  assert.equal(toastEl.textContent, "", "no toast when nothing needed reattaching");
});

test("reattachActivity: multiple running operations reattach each and toast a summary count", async () => {
  const { fetchImpl } = makeActivityFetch([
    [{ id: "j1", kind: "pull", status: "running" }, { id: "j2", kind: "imagine", status: "running" }],
  ]);
  const { window: win } = loadAppWithPages({ fetchImpl });
  runScript(win, `
    window.__streamJobCalls = [];
    streamJob = async (id) => { window.__streamJobCalls.push(id); return new Promise(() => {}); };
  `);
  await tick(); await tick(); await tick();

  assert.deepEqual(Array.from(win.__streamJobCalls).sort(), ["j1", "j2"]);
  const toastEl = win.document.getElementById("toast");
  assert.match(toastEl.textContent, /Reattached to 2 running operations/);
});

test("reattachActivity: a streamJob line is collected into the details modal's log", async () => {
  const { fetchImpl } = makeActivityFetch([
    [{ id: "j1", kind: "pull", label: "Model pull", status: "running" }],
  ]);
  const { window: win } = loadAppWithPages({ fetchImpl });
  runScript(win, `
    streamJob = async (id, onLine) => {
      onLine("downloading 10%");
      onLine("downloading 50%");
      return new Promise(() => {});
    };
  `);
  await tick(); await tick(); await tick();

  win.showActivityDetails();
  const modalBody = win.document.getElementById("modal-body");
  assert.match(modalBody.textContent, /downloading 10%/);
  assert.match(modalBody.textContent, /downloading 50%/);
});

test("reattachActivity: a reattach that ends 'disconnected' does not overwrite the last known status", async () => {
  const { fetchImpl } = makeActivityFetch([
    [{ id: "j1", kind: "pull", label: "Model pull", status: "running" }],
  ]);
  const { window: win } = loadAppWithPages({ fetchImpl });
  runScript(win, `
    streamJob = async () => ({ status: "disconnected" });
  `);
  await tick(); await tick(); await tick();

  win.showActivityDetails();
  const modalBody = win.document.getElementById("modal-body");
  assert.match(modalBody.textContent, /running/,
    "a client-only 'disconnected' outcome must not overwrite the server's last known 'running' status");
  assert.doesNotMatch(modalBody.textContent, /disconnected/);
});
