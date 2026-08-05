// SPDX-License-Identifier: AGPL-3.0-or-later
// streamJob (app/helpers.js) used to report a job as FAILED whenever its SSE
// connection ended without an "end" frame - a transport-level disconnect
// rendered as an application-level outcome. Verified live 2026-08-05:
// aborting a real GET /api/jobs/{id}/events connection ~5ms after opening it
// (a real model-pull job, a real server) produced exactly this - the reader
// either threw (AbortError) or resolved {done:true} with no "end" event -
// while the job went on to finish successfully several seconds later.
// models.js's pull handler then painted a red "failed" bar and toasted
// "Pull failed", though the pull was running fine.
//
// The fix reconnects (matching coder.js's streamSession: 1500ms backoff,
// retry until the operation is confirmed over) and only ever returns
// {status:"disconnected"} - never "failed" - when it genuinely gives up or
// the job is confirmed gone (404). "failed" stays reserved for the job's own
// end event saying so.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

// Builds a fetchImpl that, on each successive call to GET
// /api/jobs/<id>/events, plays back the next scenario in `attempts`:
//   { events: [...] }              - deliver these frames, then a clean EOF
//                                     (done:true) with no "end" frame - the
//                                     "resolved without an exception" drop shape.
//   { events: [...], thenThrow: true } - deliver these frames, then throw -
//                                     the "AbortError / network TypeError" drop
//                                     shape (the one reproduced live).
//   { notFound: true }             - a 404: the job is confirmed gone.
// The LAST scenario in the array repeats for any call beyond its length.
function makeReconnectableFetch(attempts) {
  let call = 0;
  const calls = [];
  const fetchImpl = async (url) => {
    const u = String(url);
    if (!/\/api\/jobs\/[^/]+\/events$/.test(u)) {
      return { ok: true, status: 200, json: async () => ({}) };
    }
    calls.push(u);   // only job-events calls - the harness's own app-init
                      // fetches (models, config, coder sessions, ...) must
                      // not count toward "did streamJob reconnect".
    const scenario = attempts[Math.min(call, attempts.length - 1)];
    call++;
    if (scenario.notFound) {
      return { ok: false, status: 404, statusText: "Not Found" };
    }
    const frames = (scenario.events || []).map((ev) => `data: ${JSON.stringify(ev)}\n\n`);
    let idx = 0;
    const enc = new TextEncoder();
    return {
      ok: true,
      status: 200,
      body: {
        getReader() {
          return {
            async read() {
              if (idx < frames.length) {
                const chunk = enc.encode(frames[idx]);
                idx++;
                return { done: false, value: chunk };
              }
              if (scenario.thenThrow) {
                const e = new Error("simulated network drop");
                e.name = "TypeError";
                throw e;
              }
              return { done: true, value: undefined };
            },
            async cancel() {},
          };
        },
      },
    };
  };
  return { fetchImpl, calls };
}

// Speed up the reconnect loop for tests: real production tuning (1500ms x 20
// attempts) would make every test here take tens of seconds for no reason.
function speedUpReconnect(win, { delayMs = 2, maxAttempts } = {}) {
  runScript(win, `JOB_RECONNECT_DELAY_MS = ${delayMs};`);
  if (maxAttempts != null) runScript(win, `JOB_RECONNECT_MAX_ATTEMPTS = ${maxAttempts};`);
}

test("a stream that drops (thrown exception) before 'end' reconnects and returns the job's real outcome, not a false failure", async () => {
  const { fetchImpl, calls } = makeReconnectableFetch([
    { events: [{ type: "line", text: "starting" }], thenThrow: true },
    { events: [{ type: "line", text: "starting" },
               { type: "line", text: "still going" },
               { type: "end", status: "done", returncode: 0 }] },
  ]);
  const { window: win } = loadApp({ fetchImpl, shellToken: "tok" });
  speedUpReconnect(win);

  const lines = [];
  runScript(win, "window.__result = null;");
  win.streamJob("job1", (line) => lines.push(line)).then((r) => { win.__result = r; });

  const deadline = Date.now() + 3000;
  while (win.__result === null && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 5));
  }

  assert.ok(win.__result, "streamJob must eventually resolve");
  assert.equal(win.__result.status, "done", "resolves with the job's REAL end status, not a false failure");
  assert.notEqual(win.__result.status, "failed", "a transient disconnect must never be reported as a job failure");
  assert.ok(calls.length >= 2, "a reconnect attempt must have happened (fetch called more than once)");
  assert.deepEqual(lines, ["starting", "still going"],
    "each real line is delivered exactly once - the replayed 'starting' line on reconnect must not be double-printed");
});

test("a stream that drops (clean EOF, no exception) before 'end' also reconnects rather than reporting {status:'failed'}", async () => {
  // The OTHER shape of the same defect: readSSE's loop just breaks when
  // reader.read() resolves {done:true} with nothing pending - the exact
  // helpers.js:461 fallback this unit fixes.
  const { fetchImpl, calls } = makeReconnectableFetch([
    { events: [{ type: "progress", phase: "download", downloaded: 0, total: 100, pct: 0 }] },
    { events: [{ type: "progress", phase: "download", downloaded: 0, total: 100, pct: 0 },
               { type: "end", status: "done", returncode: 0 }] },
  ]);
  const { window: win } = loadApp({ fetchImpl, shellToken: "tok" });
  speedUpReconnect(win);

  runScript(win, "window.__result = null;");
  win.streamJob("job2", null, null).then((r) => { win.__result = r; });

  const deadline = Date.now() + 3000;
  while (win.__result === null && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 5));
  }

  assert.ok(win.__result);
  assert.equal(win.__result.status, "done");
  assert.ok(calls.length >= 2, "a reconnect attempt must have happened");
});

test("a genuinely failed job still reports status 'failed' - the fix must not mask a real failure", async () => {
  const { fetchImpl } = makeReconnectableFetch([
    { events: [{ type: "line", text: "trying" },
               { type: "end", status: "failed", returncode: 1 }] },
  ]);
  const { window: win } = loadApp({ fetchImpl, shellToken: "tok" });
  speedUpReconnect(win);

  runScript(win, "window.__result = null;");
  win.streamJob("job3", null, null).then((r) => { win.__result = r; });

  const deadline = Date.now() + 2000;
  while (win.__result === null && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 5));
  }

  assert.ok(win.__result);
  assert.equal(win.__result.status, "failed", "a real job failure must still surface as failed");
  assert.equal(win.__result.returncode, 1);
});

test("a job confirmed gone (404) returns 'disconnected' immediately, without exhausting the retry budget", async () => {
  const { fetchImpl, calls } = makeReconnectableFetch([{ notFound: true }]);
  const { window: win } = loadApp({ fetchImpl, shellToken: "tok" });
  // Deliberately do NOT speed up the delay/attempts here - a 404 must return
  // on the FIRST call, not after burning through the retry budget.

  runScript(win, "window.__result = null;");
  win.streamJob("job4", null, null).then((r) => { win.__result = r; });

  const deadline = Date.now() + 2000;
  while (win.__result === null && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 5));
  }

  assert.ok(win.__result, "a 404 must resolve promptly, not hang through the full retry budget");
  assert.equal(win.__result.status, "disconnected");
  assert.equal(calls.length, 1, "no retry after a confirmed 404 - reconnecting cannot help");
});

test("a permanently dropped connection eventually gives up and reports 'disconnected', never 'failed'", async () => {
  // Every attempt drops before any "end" frame - simulates a server that is
  // genuinely, persistently unreachable (not just a transient blip).
  const { fetchImpl, calls } = makeReconnectableFetch([{ events: [], thenThrow: true }]);
  const { window: win } = loadApp({ fetchImpl, shellToken: "tok" });
  speedUpReconnect(win, { delayMs: 2, maxAttempts: 3 });

  runScript(win, "window.__result = null;");
  win.streamJob("job5", null, null).then((r) => { win.__result = r; });

  const deadline = Date.now() + 2000;
  while (win.__result === null && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 5));
  }

  assert.ok(win.__result, "must eventually give up and resolve, not hang forever");
  assert.equal(win.__result.status, "disconnected");
  assert.notEqual(win.__result.status, "failed",
    "giving up after exhausting retries is still not the same fact as the job failing");
  assert.equal(calls.length, 3, "retried exactly JOB_RECONNECT_MAX_ATTEMPTS times before giving up");
});
