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
  // responses: array of "fail", a bare operations array (now omitted from the
  // reply, matching an old/malformed server), or {ops, now} for control over
  // the server clock field. The LAST entry repeats for any call beyond its
  // length.
  let call = 0;
  const calls = [];
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u === "/api/activity") {
      calls.push(u);
      const r = responses[Math.min(call, responses.length - 1)];
      call++;
      if (r === "fail") return { ok: false, status: 500, json: async () => ({}) };
      const body = Array.isArray(r) ? { operations: r } : { operations: r.ops, now: r.now };
      return { ok: true, status: 200, json: async () => body };
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

// --------------------------------------------------------------------------- //
//  Real /api/activity contract: the "now" field and the tri-state read       //
//  (never asked / confirmed empty / read failed) - #1075's merged route.     //
// --------------------------------------------------------------------------- //

test("showActivityDetails: before any read has succeeded, shows 'Checking…' - never 'Nothing running.' (R1)", async () => {
  // Every read fails - the boot-time automatic reattachActivity() call never
  // gets a successful response, so _activityKnown must stay false.
  const { fetchImpl } = makeActivityFetch(["fail"]);
  const { window: win } = loadAppWithPages({ fetchImpl });
  await tick(); await tick(); await tick();

  win.showActivityDetails();
  const modalBody = win.document.getElementById("modal-body");
  assert.match(modalBody.textContent, /Checking/);
  assert.doesNotMatch(modalBody.textContent, /Nothing running/,
    "a question that was never successfully asked must not be rendered as a confirmed 'no'");
});

test("showActivityDetails: a confirmed empty read shows 'Nothing running.' (a real, not a fabricated, answer)", async () => {
  const { fetchImpl } = makeActivityFetch([{ ops: [], now: 1000 }]);
  const { window: win } = loadAppWithPages({ fetchImpl });
  await tick(); await tick(); await tick();

  win.showActivityDetails();
  const modalBody = win.document.getElementById("modal-body");
  assert.match(modalBody.textContent, /Nothing running/);
});

test("showActivityDetails: renders a running operation's age from the SERVER's clock (now - created_at), not the browser's own", async () => {
  // now=1000, created_at=880 -> 120s ago -> fmtDuration(120) == "2m 00s".
  // If this were computed against Date.now() instead, the result would be
  // wildly different (real epoch seconds vs this fixture's tiny numbers) -
  // the assertion below can only pass if the SERVER's now is what got used.
  //
  // No "running " prefix on the age itself - verified live against a real
  // server that a prefix there reads as "runningrunning 21s" next to the
  // adjacent .job-state pill, which already says "running" on its own.
  const { fetchImpl } = makeActivityFetch([
    { ops: [{ id: "j1", kind: "pull", label: "Model pull", status: "running", created_at: 880 }], now: 1000 },
  ]);
  const { window: win } = loadAppWithPages({ fetchImpl });
  await tick(); await tick(); await tick();

  win.showActivityDetails();
  const modalBody = win.document.getElementById("modal-body");
  assert.match(modalBody.textContent, /2m 00s/);
  assert.doesNotMatch(modalBody.textContent, /runningrunning/,
    "the age must not repeat the word the adjacent status pill already shows");
});

test("showActivityDetails: a finished operation's age reads '<duration> ago', not 'running <duration>'", async () => {
  const { fetchImpl } = makeActivityFetch([
    { ops: [{ id: "j1", kind: "pull", label: "Model pull", status: "done", created_at: 940 }], now: 1000 },
  ]);
  const { window: win } = loadAppWithPages({ fetchImpl });
  await tick(); await tick(); await tick();

  win.showActivityDetails();
  const modalBody = win.document.getElementById("modal-body");
  assert.match(modalBody.textContent, /1m 00s ago/);
  assert.doesNotMatch(modalBody.textContent, /running 1m 00s/);
});

test("showActivityDetails: a finished operation's age is computed from finished_at, not created_at (#1078 post-merge review)", async () => {
  // created_at=940 (started 60s before "now"), finished_at=990 (actually
  // finished only 10s before "now") - a two-hour pull that finished 10s ago
  // must read "10s ago", not "1m 00s ago". The fixture in the test above
  // this one OMITTED finished_at entirely, which is why the created_at bug
  // was not caught the first time - see the fallback test right below this
  // one for that omitted-field case, now covered deliberately.
  const { fetchImpl } = makeActivityFetch([
    { ops: [{ id: "j1", kind: "pull", label: "Model pull", status: "done",
              created_at: 940, finished_at: 990 }], now: 1000 },
  ]);
  const { window: win } = loadAppWithPages({ fetchImpl });
  await tick(); await tick(); await tick();

  win.showActivityDetails();
  const modalBody = win.document.getElementById("modal-body");
  assert.match(modalBody.textContent, /10s ago/);
  assert.doesNotMatch(modalBody.textContent, /1m 00s ago/,
    "must use finished_at's age, not created_at's, once finished_at is present");
});

test("showActivityDetails: a finished operation with no finished_at falls back to created_at's age", async () => {
  // Legacy/partial data shape (an older server, or a job snapshot taken
  // before mark_finished() ran) - still renders something instead of
  // nothing, using the pre-fix behaviour as the degrade path.
  const { fetchImpl } = makeActivityFetch([
    { ops: [{ id: "j1", kind: "pull", label: "Model pull", status: "done", created_at: 940 }], now: 1000 },
  ]);
  const { window: win } = loadAppWithPages({ fetchImpl });
  await tick(); await tick(); await tick();

  win.showActivityDetails();
  const modalBody = win.document.getElementById("modal-body");
  assert.match(modalBody.textContent, /1m 00s ago/);
});

// --------------------------------------------------------------------------- //
//  #1078 post-merge review: the details modal must not be a frozen snapshot   //
// --------------------------------------------------------------------------- //

test("showActivityDetails: an open modal re-renders on the next poll, not a frozen snapshot", async () => {
  // A count/index-driven fixture (like makeActivityFetch's array form) is
  // fragile here: boot fires TWO independent /api/activity reads of its own
  // (init.js's startHwStats() -> pollActivity() AND reattachActivity()'s own
  // read), so exactly how many array slots boot eats before any test code
  // runs is an implementation detail this test should not have to track.
  // Instead: every call succeeds: pct=10 until `advanced` flips to true.
  let advanced = false;
  const fetchImpl = async (url) => {
    if (String(url) !== "/api/activity")
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    return {
      ok: true, status: 200,
      json: async () => ({ operations: [
        { id: "j1", kind: "pull", label: "Model pull", status: "running", pct: advanced ? 55 : 10 },
      ] }),
    };
  };
  const { window: win } = loadApp({ fetchImpl });
  await tick(); await tick(); await tick();   // let boot's own automatic reads resolve
  win.showActivityDetails();
  const modalBody = win.document.getElementById("modal-body");
  // pct renders as a bar WIDTH in the modal, never as "N%" text (that literal
  // text only appears in the pill) - matching the wrong representation here
  // would silently pass regardless of which poll's data was actually shown.
  const barWidth = () => modalBody.querySelector(".dl-fill").style.width;
  assert.equal(barWidth(), "10%", "precondition: modal shows the current (pre-advance) read");

  advanced = true;
  await win.pollActivity();   // simulates the next 2500ms tick while the modal stays open
  assert.equal(barWidth(), "55%",
    "the open modal must reflect the latest poll, not what was open at open time");
});

test("showActivityDetails: a DIFFERENT open modal is never overwritten by an activity poll", async () => {
  const { fetchImpl } = makeActivityFetch([
    [{ id: "j1", kind: "pull", label: "Model pull", status: "running", pct: 10 }],
    [{ id: "j1", kind: "pull", label: "Model pull", status: "running", pct: 90 }],
  ]);
  const { window: win } = loadApp({ fetchImpl });
  await win.pollActivity();
  win.openModal("Something else", (body) => { body.textContent = "unrelated content"; });

  await win.pollActivity();
  const modalBody = win.document.getElementById("modal-body");
  assert.equal(modalBody.textContent, "unrelated content",
    "a modal the activity feature does not own must never be rewritten out from under it");
});

test("showActivityDetails: a CLOSED activity modal is not silently reopened by a later poll", async () => {
  const { fetchImpl } = makeActivityFetch([
    [{ id: "j1", kind: "pull", label: "Model pull", status: "running", pct: 10 }],
    [{ id: "j1", kind: "pull", label: "Model pull", status: "running", pct: 90 }],
  ]);
  const { window: win } = loadApp({ fetchImpl });
  await win.pollActivity();
  win.showActivityDetails();
  win.document.getElementById("modal").style.display = "none";   // user closed it

  await win.pollActivity();
  assert.equal(win.document.getElementById("modal").style.display, "none",
    "a background poll must never reopen a modal the user closed");
});

// --------------------------------------------------------------------------- //
//  #1078 post-merge review (AC4): repeated failed reads must surface, not     //
//  silently keep (or silently drop) a claim nobody has confirmed lately.      //
// --------------------------------------------------------------------------- //

test("renderActivityPill: a single failed poll does not flap the pill (routine at a 2.5s tick)", async () => {
  const { fetchImpl } = makeActivityFetch([
    [{ id: "j1", kind: "pull", label: "Model pull", status: "running", pct: 10 }],
    "fail",
  ]);
  const { window: win } = loadApp({ fetchImpl });
  await tick(); await tick(); await tick();   // boot's own auto pollActivity() consumes index0
  await win.pollActivity();   // index1 (the array's last entry repeats): one failure

  const pill = win.document.getElementById("activity-pill");
  assert.match(pill.textContent, /Model pull/);
  assert.doesNotMatch(pill.textContent, /unknown/i, "one dropped poll must not yet claim the state is unknown");
});

test("renderActivityPill: reads failing 3 times in a row marks a running op's pill as unknown", async () => {
  const { fetchImpl } = makeActivityFetch([
    [{ id: "j1", kind: "pull", label: "Model pull", status: "running", pct: 10 }],
    "fail", "fail", "fail",
  ]);
  const { window: win } = loadApp({ fetchImpl });
  await tick(); await tick(); await tick();   // boot's own auto pollActivity() consumes index0
  await win.pollActivity();   // index1 (fail 1)
  await win.pollActivity();   // index2 (fail 2)
  await win.pollActivity();   // index3 (fail 3 -> stale)

  const pill = win.document.getElementById("activity-pill");
  assert.notEqual(pill.style.display, "none", "must not hide - that would falsely claim nothing running");
  assert.match(pill.textContent, /status unknown/i);
  assert.match(pill.className, /st-unknown/);
});

test("renderActivityPill: reads failing 3 times in a row surfaces 'status unknown' even with nothing previously running", async () => {
  const { fetchImpl } = makeActivityFetch([
    { ops: [], now: 1000 },
    "fail", "fail", "fail",
  ]);
  const { window: win } = loadApp({ fetchImpl });
  await tick(); await tick(); await tick();   // boot's own auto pollActivity() consumes index0
  const pill = win.document.getElementById("activity-pill");
  assert.equal(pill.style.display, "none", "precondition: a confirmed-empty read hides the pill as usual");

  await win.pollActivity();   // index1 (fail 1)
  await win.pollActivity();   // index2 (fail 2)
  await win.pollActivity();   // index3 (fail 3 -> stale)
  assert.notEqual(pill.style.display, "none",
    "once a known 'nothing running' has gone stale, hiding it silently repeats the fabrication R1 forbids");
  assert.match(pill.textContent, /status unknown/i);
});

test("renderActivityPill: a successful read after failures immediately clears the unknown state", async () => {
  // Same reasoning as the modal test above: a fixed-index fixture cannot
  // account for boot's own two independent /api/activity reads without
  // hand-counting them, so this drives the mock from state instead of a
  // position - only the very first call ever succeeds (establishing known
  // state), every call after that fails until `recovered` flips.
  let firstCallDone = false;
  let recovered = false;
  const fetchImpl = async (url) => {
    if (String(url) !== "/api/activity")
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    if (!firstCallDone || recovered) {
      firstCallDone = true;
      return {
        ok: true, status: 200,
        json: async () => ({ operations: [
          { id: "j1", kind: "pull", label: "Model pull", status: "running", pct: recovered ? 77 : 10 },
        ] }),
      };
    }
    return { ok: false, status: 500, json: async () => ({}) };
  };
  const { window: win } = loadApp({ fetchImpl });
  await tick(); await tick(); await tick();   // boot's first read succeeds, its second (and on) fail
  const pill = win.document.getElementById("activity-pill");

  await win.pollActivity();
  await win.pollActivity();
  await win.pollActivity();   // several more failed polls - well past the 3-in-a-row threshold
  assert.match(pill.className, /st-unknown/, "marked unknown after enough consecutive failures");

  recovered = true;
  await win.pollActivity();
  assert.doesNotMatch(pill.className, /st-unknown/);
  assert.match(pill.textContent, /77%/);
  assert.doesNotMatch(pill.textContent, /unknown/i);
});

test("showActivityDetails: the modal notes staleness once reads have failed 3 times in a row", async () => {
  const { fetchImpl } = makeActivityFetch([
    [{ id: "j1", kind: "pull", label: "Model pull", status: "running", pct: 10 }],
    "fail", "fail", "fail",
  ]);
  const { window: win } = loadApp({ fetchImpl });
  await tick(); await tick(); await tick();   // boot's own auto pollActivity() consumes index0
  win.showActivityDetails();
  await win.pollActivity();   // index1 (fail 1)
  await win.pollActivity();   // index2 (fail 2)
  await win.pollActivity();   // index3 (fail 3 -> stale)

  const modalBody = win.document.getElementById("modal-body");
  assert.match(modalBody.textContent, /[Cc]ould not reach the server/);
  assert.match(modalBody.textContent, /Model pull/, "the last known op list is still shown below the notice");
});
