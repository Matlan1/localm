// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the coder busy-pill activity signal: readSSE's onAnyFrame
// callback, streamSession wiring it to a per-session lastEventAt timestamp,
// and tickCoderBusyIndicator rendering it into the #coder-state pill.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function okFetch() {
  return async () => ({ ok: true, status: 200, json: async () => ({}) });
}

// Delivers `before` frames immediately, then blocks until the test itself
// calls the returned `releaseGate()` before delivering `after` - a
// deterministic gap (no reliance on real-time delays racing jsdom's own
// overhead) so a test can observe state between the two. Once `after` is
// exhausted the reader stalls forever (never resolves `done: true`), so
// streamSession's loop does not reconnect mid-test.
function makeGatedSseFetch({ before, after }) {
  let idx = 0;
  const enc = new TextEncoder();
  let releaseGate;
  const gate = new Promise((resolve) => { releaseGate = resolve; });
  const fetchImpl = async (url) => {
    if (!/\/api\/coder\/sessions\/[^/]+\/events/.test(String(url))) {
      return { ok: true, status: 200, json: async () => ({}) };
    }
    return {
      ok: true,
      status: 200,
      body: {
        getReader() {
          return {
            async read() {
              if (idx < before.length) {
                const chunk = enc.encode(before[idx]);
                idx++;
                return { done: false, value: chunk };
              }
              if (idx === before.length) await gate;
              const j = idx - before.length;
              if (j < after.length) {
                const chunk = enc.encode(after[j]);
                idx++;
                return { done: false, value: chunk };
              }
              return new Promise(() => {});   // stall - the connection stays open
            },
            async cancel() {},
          };
        },
      },
    };
  };
  return { fetchImpl, releaseGate: () => releaseGate() };
}

test("readSSE invokes onAnyFrame once per parsed frame, including a bare comment, while onData still only sees data: payloads", async () => {
  const raw = ": keepalive\n\n" + `data: ${JSON.stringify({ a: 1 })}\n\n` + ": keepalive\n\n";
  const enc = new TextEncoder();
  let delivered = false;
  const response = {
    body: {
      getReader() {
        return {
          async read() {
            if (!delivered) { delivered = true; return { done: false, value: enc.encode(raw) }; }
            return { done: true, value: undefined };
          },
          async cancel() {},
        };
      },
    },
  };
  const { window } = loadApp({ fetchImpl: okFetch() });
  const payloads = [];
  let frames = 0;
  await window.readSSE(response, (p) => payloads.push(p), () => { frames++; });
  assert.equal(frames, 3, "onAnyFrame must fire for every parsed frame, keepalive comments included");
  assert.deepEqual(payloads, [JSON.stringify({ a: 1 })], "onData must still fire only for data: lines");
});

test("a keepalive-only SSE frame does not touch the session's activity clocks", async () => {
  const { fetchImpl, releaseGate } = makeGatedSseFetch({
    before: [": keepalive\n\n", ": keepalive\n\n"],
    after: [`data: ${JSON.stringify({ type: "turn", turn: 1, total_tokens: 0 })}\n\n`],
  });
  const { window } = loadApp({ fetchImpl, shellToken: "tok" });

  runScript(window, `
    const feedEl = document.createElement("div");
    window.__s = { info: { id: "sid1", cwd: "Z:/proj" }, feedEl, busy: false,
      lastEventAt: null, phaseStartedAt: null, liveBody: null, liveText: "", liveReasoning: "",
      pendingCards: [], confirmCards: new Map(), closed: false };
    coder.sessions.set("sid1", window.__s);
    coder.activeId = "sid1";
    streamSession(window.__s, false);
  `);

  await new Promise((r) => setTimeout(r, 100));
  assert.equal(window.__s.lastEventAt, null, "a keepalive comment is not a real event");
  assert.equal(window.__s.phaseStartedAt, null, "a keepalive comment must not start or reset the phase clock");
  assert.equal(window.__s.busy, false);

  releaseGate();
  const doneDeadline = Date.now() + 2000;
  while (!window.__s.busy && Date.now() < doneDeadline) {
    await new Promise((r) => setTimeout(r, 5));
  }
  assert.equal(window.__s.busy, true, "the turn event must still arrive normally after the keepalive-only period");
  assert.equal(typeof window.__s.phaseStartedAt, "number", "the turn event starts the phase clock");
});

test("tickCoderBusyIndicator shows elapsed time since the phase began while the active session is busy", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  runScript(window, `
    coder.activeId = "sid1";
    coder.sessions.set("sid1", { info: { id: "sid1" }, busy: true, phaseStartedAt: Date.now() - 12000, lastEventAt: Date.now() });
    const node = document.getElementById("coder-state");
    node.textContent = "working…";
    node.className = "job-state st-running";
  `);
  window.tickCoderBusyIndicator();
  const text = window.document.getElementById("coder-state").textContent;
  assert.match(text, /^working… \d+s$/, "the pill text now carries a live elapsed-seconds readout");
  const secs = Number(text.match(/(\d+)s$/)[1]);
  assert.ok(secs >= 10, `expected roughly 12s elapsed, got "${text}"`);
});

test("tickCoderBusyIndicator leaves the pill alone when the active session is not busy", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  runScript(window, `
    coder.activeId = "sid1";
    coder.sessions.set("sid1", { info: { id: "sid1" }, busy: false, phaseStartedAt: Date.now() - 12000, lastEventAt: Date.now() });
    const node = document.getElementById("coder-state");
    node.textContent = "idle";
    node.className = "job-state st-pending";
  `);
  window.tickCoderBusyIndicator();
  assert.equal(window.document.getElementById("coder-state").textContent, "idle");
});

test("tickCoderBusyIndicator leaves the pill alone when no phase has started yet for the busy session", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  runScript(window, `
    coder.activeId = "sid1";
    coder.sessions.set("sid1", { info: { id: "sid1" }, busy: true, phaseStartedAt: null, lastEventAt: null });
    const node = document.getElementById("coder-state");
    node.textContent = "working…";
    node.className = "job-state st-running";
  `);
  window.tickCoderBusyIndicator();
  assert.equal(window.document.getElementById("coder-state").textContent, "working…",
    "with no phaseStartedAt yet there is nothing to count from, so the base label is left untouched");
});

test("a status event shows in the busy pill until the next event", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  runScript(window, `
    coder.activeId = "sid1";
    window.__s = { info: { id: "sid1" }, busy: true, phaseStartedAt: Date.now(), lastEventAt: Date.now() };
    coder.sessions.set("sid1", window.__s);
    const node = document.getElementById("coder-state");
    node.textContent = "working…";
    node.className = "job-state st-running";
  `);
  const pill = () => window.document.getElementById("coder-state").textContent;
  window.handleCoderEvent(window.__s, { type: "status", text: "Loading model...", code: "loading_model" });
  assert.match(pill(), /^Loading model… \d+s$/);
  window.handleCoderEvent(window.__s, { type: "status", text: "Something new...", code: null });
  assert.match(pill(), /^Something new\.\.\. \d+s$/);
  window.handleCoderEvent(window.__s, { type: "some_later_event" });
  window.tickCoderBusyIndicator();
  assert.match(pill(), /^working… \d+s$/, "the status clears once the model answers");
  assert.ok(!(window.__s.eventLog || []).some((e) => e.type === "status"),
    "statuses are not kept in the exported event log");
});

test("a parameterised status is localized from its args and takes the warning style", async () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  runScript(window, `
    coder.activeId = "sid1";
    window.__s = { info: { id: "sid1" }, busy: true, phaseStartedAt: Date.now(), lastEventAt: Date.now() };
    coder.sessions.set("sid1", window.__s);
    const node = document.getElementById("coder-state");
    node.className = "job-state st-running";
  `);
  const node = window.document.getElementById("coder-state");
  window.handleCoderEvent(window.__s, { type: "status", text: "english text", code: "loading_beside",
                                        args: { model: "big", resident: "small" } });
  assert.match(node.textContent, /big/);
  assert.doesNotMatch(node.textContent, /english text/, "the catalog text is used, filled from the args");
  assert.ok(node.classList.contains("st-warn"));
  window.handleCoderEvent(window.__s, { type: "some_later_event" });
  window.tickCoderBusyIndicator();
  assert.ok(!node.classList.contains("st-warn"), "the warning ends with the status");
});

function busyPill(window, fields) {
  runScript(window, `
    coder.activeId = "sid1";
    window.__s = Object.assign({ info: { id: "sid1" }, busy: true }, ${JSON.stringify(fields)});
    coder.sessions.set("sid1", window.__s);
    const node = document.getElementById("coder-state");
    node.textContent = "working…";
    node.className = "job-state st-running";
  `);
  return () => window.document.getElementById("coder-state").textContent;
}

test("the counter keeps climbing past 15 s while only keepalive frames arrive", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  const now = Date.now();
  const pill = busyPill(window, { phaseStartedAt: now - 20000, lastEventAt: now - 20000 });
  window.tickCoderBusyIndicator();
  assert.match(pill(), /^working… (19|20|21)s$/);
  runScript(window, `window.__s.phaseStartedAt -= 25000; window.__s.lastEventAt -= 25000;`);
  window.tickCoderBusyIndicator();
  assert.match(pill(), /^working… 45s \(no output for 45s\)$|^working… 4[5-6]s \(no output for 4[5-6]s\)$/);
});

test("no stall note while real events are recent; it appears at 30 s of silence", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  const now = Date.now();
  const pill = busyPill(window, { phaseStartedAt: now - 90000, lastEventAt: now - 5000 });
  window.tickCoderBusyIndicator();
  assert.match(pill(), /^working… 1m 30s$/, "long durations read as minutes and seconds, no stall note");
  runScript(window, `window.__s.lastEventAt = Date.now() - 40000;`);
  window.tickCoderBusyIndicator();
  assert.match(pill(), /^working… 1m 30s \(no output for 4\ds\)$/);
});

test("formatCoderElapsed formats seconds, minutes and hours", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  assert.equal(window.formatCoderElapsed(0), "0s");
  assert.equal(window.formatCoderElapsed(59.9), "59s");
  assert.equal(window.formatCoderElapsed(65), "1m 05s");
  assert.equal(window.formatCoderElapsed(3725), "1h 02m");
});

test("a status change restarts the phase clock; same-phase events and repeated statuses do not", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  const old = Date.now() - 50000;
  busyPill(window, { phaseStartedAt: old, lastEventAt: old, phaseKey: "" });
  window.handleCoderEvent(window.__s, { type: "some_later_event" });
  assert.equal(window.__s.phaseStartedAt, old, "an event in the same phase leaves the clock alone");
  window.handleCoderEvent(window.__s, { type: "status", text: "Loading model...", code: "loading_model" });
  const loadStart = window.__s.phaseStartedAt;
  assert.ok(loadStart > old + 40000, "a new status starts a new phase");
  window.__s.phaseStartedAt = old;
  window.handleCoderEvent(window.__s, { type: "status", text: "Loading model...", code: "loading_model" });
  assert.equal(window.__s.phaseStartedAt, old, "the same status again is the same phase");
  window.handleCoderEvent(window.__s, { type: "some_later_event" });
  assert.ok(window.__s.phaseStartedAt > old + 40000, "the status clearing starts a new phase");
});
