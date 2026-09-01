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

test("a keepalive-only SSE frame marks the session's last-activity timestamp before any data event arrives", async () => {
  const { fetchImpl, releaseGate } = makeGatedSseFetch({
    before: [": keepalive\n\n", ": keepalive\n\n"],
    after: [`data: ${JSON.stringify({ type: "turn", turn: 1, total_tokens: 0 })}\n\n`],
  });
  const { window } = loadApp({ fetchImpl, shellToken: "tok" });

  runScript(window, `
    const feedEl = document.createElement("div");
    window.__s = { info: { id: "sid1", cwd: "Z:/proj" }, feedEl, busy: false,
      lastEventAt: null, liveBody: null, liveText: "", liveReasoning: "",
      pendingCards: [], confirmCards: new Map(), closed: false };
    coder.sessions.set("sid1", window.__s);
    coder.activeId = "sid1";
    streamSession(window.__s, false);
  `);

  const midDeadline = Date.now() + 2000;
  while (window.__s.lastEventAt == null && Date.now() < midDeadline) {
    await new Promise((r) => setTimeout(r, 5));
  }
  assert.ok(window.__s.lastEventAt, "a keepalive comment must register as received activity");
  // The data frame is still held behind the gate at this point - it cannot
  // have reached streamSession yet, so this is not a timing race.
  assert.equal(window.__s.busy, false,
    "no data event has arrived yet, so a keepalive alone must not mark the session busy");

  releaseGate();
  const doneDeadline = Date.now() + 2000;
  while (!window.__s.busy && Date.now() < doneDeadline) {
    await new Promise((r) => setTimeout(r, 5));
  }
  assert.equal(window.__s.busy, true, "the turn event must still arrive normally after the keepalive-only period");
});

test("tickCoderBusyIndicator shows elapsed seconds since the last frame while the active session is busy", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  runScript(window, `
    coder.activeId = "sid1";
    coder.sessions.set("sid1", { info: { id: "sid1" }, busy: true, lastEventAt: Date.now() - 12000 });
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
    coder.sessions.set("sid1", { info: { id: "sid1" }, busy: false, lastEventAt: Date.now() - 12000 });
    const node = document.getElementById("coder-state");
    node.textContent = "idle";
    node.className = "job-state st-pending";
  `);
  window.tickCoderBusyIndicator();
  assert.equal(window.document.getElementById("coder-state").textContent, "idle");
});

test("tickCoderBusyIndicator leaves the pill alone when no frame has been received yet for the busy session", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  runScript(window, `
    coder.activeId = "sid1";
    coder.sessions.set("sid1", { info: { id: "sid1" }, busy: true, lastEventAt: null });
    const node = document.getElementById("coder-state");
    node.textContent = "working…";
    node.className = "job-state st-running";
  `);
  window.tickCoderBusyIndicator();
  assert.equal(window.document.getElementById("coder-state").textContent, "working…",
    "with no lastEventAt yet there is nothing to count from, so the base label is left untouched");
});
