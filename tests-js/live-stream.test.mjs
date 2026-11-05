// SPDX-License-Identifier: AGPL-3.0-or-later
// The status bar follows GET /api/live (pushed hardware stats and activity) and
// polls /api/stats and /api/activity only while that stream is not open.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const GIB = 1024 ** 3;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitFor(cond, timeout = 2000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    if (cond()) return true;
    await sleep(5);
  }
  return !!cond();
}

const statsEvent = (usedGib, cpu = 10) => ({
  type: "stats",
  data: { cpu: { percent: cpu }, vram: { used: usedGib * GIB, total: 16 * GIB } },
});

// One open /api/live connection the test drives by hand.
function liveStream(signal) {
  const enc = new TextEncoder();
  const queue = [];
  let waiter = null;
  let ended = false;
  const wake = () => { if (waiter) { const w = waiter; waiter = null; w(); } };
  const stream = {
    signal,
    send(text) { queue.push(enc.encode(text)); wake(); },
    push(ev) { stream.send("data: " + JSON.stringify(ev) + "\n\n"); },
    comment() { stream.send(": keepalive\n\n"); },
    end() { ended = true; wake(); },
    response: {
      ok: true, status: 200,
      body: {
        getReader() {
          return {
            read: () => new Promise((resolve, reject) => {
              const next = () => {
                if (signal && signal.aborted) { reject(new DOMException("aborted", "AbortError")); return true; }
                if (queue.length) { resolve({ done: false, value: queue.shift() }); return true; }
                if (ended) { resolve({ done: true }); return true; }
                return false;
              };
              if (next()) return;
              waiter = () => { next(); };
              if (signal) signal.addEventListener("abort", () => { next(); }, { once: true });
            }),
          };
        },
      },
    },
  };
  return stream;
}

// A page whose server answers /api/live with whatever `liveHandler(n, opts,
// streams)` returns for the nth attempt, and polls with fixed bodies.
function makePage({ liveHandler, statsUsedGib = 10, statsDelayMs = 0, operations = [] }) {
  const calls = [];
  const streams = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ u, at: Date.now() });
    if (u === "/api/live") {
      const n = calls.filter((c) => c.u === "/api/live").length;
      return liveHandler(n, opts, streams);
    }
    if (u === "/api/stats") {
      if (statsDelayMs) await sleep(statsDelayMs);
      const body = statsEvent(statsUsedGib).data;
      return { ok: true, status: 200, json: async () => body };
    }
    if (u === "/api/activity") {
      return { ok: true, status: 200, json: async () => ({ now: 1000, operations, stalls: [] }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window: win } = loadApp({ fetchImpl });
  const warnings = [];
  win.console.warn = (...a) => warnings.push(a.join(" "));
  const count = (u) => calls.filter((c) => c.u === u).length;
  return { win, calls, streams, warnings, count };
}

const acceptStream = (n, opts, streams) => {
  const s = liveStream(opts.signal);
  streams.push(s);
  return s.response;
};

const hw = (win) => win.document.getElementById("hw-stats").textContent;

test("pushed stats render, and polling stops once the stream is delivering", async () => {
  const p = makePage({ liveHandler: acceptStream });
  runScript(p.win, "LIVE_BACKOFF_MIN_MS = 5;");
  p.win.startHwStats(40);
  await waitFor(() => p.streams.length === 1);
  p.streams[0].push(statsEvent(2));
  assert.ok(await waitFor(() => /VRAM 2\.0\/16\.0 GB/.test(hw(p.win))), hw(p.win));
  const stats = p.count("/api/stats"), activity = p.count("/api/activity");
  await sleep(250);
  assert.equal(p.count("/api/stats"), stats, "still polling /api/stats beside an open stream");
  assert.equal(p.count("/api/activity"), activity, "still polling /api/activity beside an open stream");
  p.streams[0].push(statsEvent(9));
  assert.ok(await waitFor(() => /VRAM 9\.0\/16\.0 GB/.test(hw(p.win))), "a second push did not render");
});

test("a pushed activity list drives the pill", async () => {
  const p = makePage({ liveHandler: acceptStream });
  p.win.startHwStats(40);
  await waitFor(() => p.streams.length === 1);
  p.streams[0].push(statsEvent(2));
  p.streams[0].push({
    type: "activity", now: 1000, stalls: [],
    operations: [{ id: "j1", kind: "pull", label: "Model pull a/b", status: "running", pct: 42 }],
  });
  const pill = p.win.document.getElementById("activity-pill");
  assert.ok(await waitFor(() => /Model pull a\/b/.test(pill.textContent) && /42%/.test(pill.textContent)),
    pill.textContent);
  assert.notEqual(pill.style.display, "none");
});

test("a poll answered after a push does not put the older reading back", async () => {
  const p = makePage({ liveHandler: acceptStream, statsUsedGib: 10, statsDelayMs: 120 });
  p.win.startHwStats(1000);
  await waitFor(() => p.streams.length === 1);
  p.streams[0].push(statsEvent(2));
  assert.ok(await waitFor(() => /VRAM 2\.0\/16\.0 GB/.test(hw(p.win))));
  await sleep(250);   // the slow boot poll answers with 10 GB meanwhile
  assert.match(hw(p.win), /VRAM 2\.0\/16\.0 GB/, "the stale poll overwrote the pushed reading");
});

test("a stream that cannot be opened falls back to polling, logs once, and is retried with a growing delay", async () => {
  const p = makePage({ liveHandler: async () => ({ ok: false, status: 404, statusText: "Not Found" }) });
  runScript(p.win, "LIVE_BACKOFF_MIN_MS = 20; LIVE_BACKOFF_MAX_MS = 1000;");
  p.win.startHwStats(30);
  assert.ok(await waitFor(() => p.count("/api/live") >= 4, 3000), `only ${p.count("/api/live")} attempts`);
  assert.ok(p.count("/api/stats") >= 3, "polling did not carry the status bar");
  assert.match(hw(p.win), /VRAM 10\.0\/16\.0 GB/);
  const tries = p.calls.filter((c) => c.u === "/api/live").map((c) => c.at);
  const gaps = tries.slice(1).map((t, i) => t - tries[i]);
  assert.ok(gaps[1] >= gaps[0] && gaps[2] >= gaps[1] && gaps[2] > gaps[0],
    `retry delays did not grow: ${gaps.join(", ")}`);
  assert.equal(p.warnings.filter((w) => /live stats stream/.test(w)).length, 1,
    `expected one fallback line, got: ${JSON.stringify(p.warnings)}`);
  assert.match(p.warnings[0], /HTTP 404/);
});

test("the retry delay stops growing at its ceiling", async () => {
  const p = makePage({ liveHandler: async () => { throw new Error("offline"); } });
  runScript(p.win, "LIVE_BACKOFF_MIN_MS = 10; LIVE_BACKOFF_MAX_MS = 30;");
  p.win.startHwStats(1000);
  assert.ok(await waitFor(() => p.count("/api/live") >= 7, 3000));
  const tries = p.calls.filter((c) => c.u === "/api/live").map((c) => c.at);
  const gaps = tries.slice(1).map((t, i) => t - tries[i]);
  assert.ok(Math.max(...gaps.slice(2)) < 150, `delay kept growing: ${gaps.join(", ")}`);
});

test("when a retry succeeds the stream takes over from polling", async () => {
  const p = makePage({
    liveHandler: async (n, opts, streams) =>
      n === 1 ? { ok: false, status: 503 } : acceptStream(n, opts, streams),
  });
  runScript(p.win, "LIVE_BACKOFF_MIN_MS = 10;");
  p.win.startHwStats(30);
  assert.ok(await waitFor(() => p.streams.length === 1), "never retried");
  p.streams[0].push(statsEvent(3));
  assert.ok(await waitFor(() => /VRAM 3\.0\/16\.0 GB/.test(hw(p.win))));
  const polled = p.count("/api/stats");
  await sleep(200);
  assert.equal(p.count("/api/stats"), polled, "polling went on after the stream recovered");
});

test("a stream the server closes falls back to polling and reconnects", async () => {
  const p = makePage({ liveHandler: acceptStream });
  runScript(p.win, "LIVE_BACKOFF_MIN_MS = 10;");
  p.win.startHwStats(30);
  await waitFor(() => p.streams.length === 1);
  p.streams[0].push(statsEvent(2));
  await waitFor(() => /VRAM 2\.0/.test(hw(p.win)));
  p.streams[0].end();
  assert.ok(await waitFor(() => p.streams.length === 2), "did not reconnect");
  assert.ok(p.warnings.some((w) => /live stats stream/.test(w)), "the fallback was not logged");
});

test("a connection that goes silent is dropped, and a keepalive comment keeps it", async () => {
  const p = makePage({ liveHandler: acceptStream });
  runScript(p.win, "LIVE_STALL_MS = 120; LIVE_BACKOFF_MIN_MS = 10;");
  p.win.startHwStats(1000);
  await waitFor(() => p.streams.length === 1);
  p.streams[0].push(statsEvent(2));
  for (let i = 0; i < 6; i++) { await sleep(50); p.streams[0].comment(); }
  assert.equal(p.streams[0].signal.aborted, false, "a connection sending keepalives was cut");
  assert.ok(await waitFor(() => p.streams[0].signal.aborted, 1500), "a silent connection was never dropped");
  assert.ok(await waitFor(() => p.streams.length === 2), "no reconnect after the watchdog");
  assert.ok(p.warnings.some((w) => /no data for/.test(w)), JSON.stringify(p.warnings));
});

test("a hidden tab closes the stream and does not reconnect; showing it again reopens", async () => {
  const p = makePage({ liveHandler: acceptStream });
  runScript(p.win, "LIVE_BACKOFF_MIN_MS = 10;");
  let hidden = false;
  Object.defineProperty(p.win.document, "hidden", { configurable: true, get: () => hidden });
  p.win.startHwStats(1000);
  await waitFor(() => p.streams.length === 1);
  p.streams[0].push(statsEvent(2));
  await waitFor(() => /VRAM 2\.0/.test(hw(p.win)));

  hidden = true;
  p.win.document.dispatchEvent(new p.win.Event("visibilitychange"));
  assert.equal(p.streams[0].signal.aborted, true, "the stream stayed open in a hidden tab");
  await sleep(150);
  assert.equal(p.streams.length, 1, "reconnected while hidden");
  assert.equal(p.warnings.length, 0, "closing for a hidden tab was reported as a failure");

  hidden = false;
  p.win.document.dispatchEvent(new p.win.Event("visibilitychange"));
  assert.ok(await waitFor(() => p.streams.length === 2), "did not reopen when shown");
  p.streams[1].push(statsEvent(7));
  assert.ok(await waitFor(() => /VRAM 7\.0\/16\.0 GB/.test(hw(p.win))));
});

test("a frame that is not JSON is skipped and the next one still renders", async () => {
  const p = makePage({ liveHandler: acceptStream });
  p.win.startHwStats(1000);
  await waitFor(() => p.streams.length === 1);
  p.streams[0].send("data: {not json\n\n");
  p.streams[0].push(statsEvent(4));
  assert.ok(await waitFor(() => /VRAM 4\.0\/16\.0 GB/.test(hw(p.win))));
  assert.equal(p.streams[0].signal.aborted, false);
});

test("a frame that cannot be rendered is logged and the stream carries on", async () => {
  const p = makePage({ liveHandler: acceptStream });
  const errors = [];
  p.win.console.error = (...a) => errors.push(a.map(String).join(" "));
  p.win.startHwStats(1000);
  await waitFor(() => p.streams.length === 1);
  p.streams[0].push({ type: "stats", data: { cpu: { percent: 5 } } });
  await waitFor(() => /CPU 5%/.test(hw(p.win)));
  p.streams[0].push({ type: "activity", now: 1, operations: [], stalls: [null] });
  assert.ok(await waitFor(() => errors.some((e) => /could not render a live frame/.test(e))),
    JSON.stringify(errors));
  p.streams[0].push(statsEvent(6));
  assert.ok(await waitFor(() => /VRAM 6\.0\/16\.0 GB/.test(hw(p.win))), "the stream stopped rendering");
  assert.equal(p.streams[0].signal.aborted, false);
  assert.equal(p.streams.length, 1, "a render error caused a reconnect");
});

test("an activity frame alone does not turn polling off: the readout still needs its first stats frame", async () => {
  const p = makePage({ liveHandler: acceptStream });
  p.win.startHwStats(30);
  await waitFor(() => p.streams.length === 1);
  p.streams[0].push({ type: "activity", now: 1, operations: [], stalls: [] });
  await sleep(120);
  assert.match(hw(p.win), /VRAM 10\.0\/16\.0 GB/, "the readout went blank before the stream's first stats frame");
  const polled = p.count("/api/stats");
  await sleep(120);
  assert.ok(p.count("/api/stats") > polled, "polling stopped on an activity frame");
  p.streams[0].push(statsEvent(2));
  assert.ok(await waitFor(() => /VRAM 2\.0\/16\.0 GB/.test(hw(p.win))));
  const after = p.count("/api/stats");
  await sleep(150);
  assert.equal(p.count("/api/stats"), after, "polling went on after the first stats frame");
});
