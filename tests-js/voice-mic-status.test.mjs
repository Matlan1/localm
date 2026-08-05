// SPDX-License-Identifier: AGPL-3.0-or-later
// #1078 post-merge review bug 1: refreshVoiceStatus() only ever wrote
// chat-mic's tooltip in its two UNAVAILABLE branches (404, or available:false),
// so once voice became available the tooltip kept claiming "needs the [voice]
// extra" forever - nothing on the success path ever reset it.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));

function fetchFor(responses) {
  let call = 0;
  return async (url) => {
    const u = String(url);
    if (u === "/api/voice/status") {
      const r = responses[Math.min(call, responses.length - 1)];
      call++;
      return r;
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

// init.js calls refreshVoiceStatus() once automatically at boot (loadApp()
// runs the real boot sequence, same as pollActivity() - see
// activity-reattach.test.mjs), which consumes the FIRST fetch response before
// any test code runs. Waiting a few ticks lets that resolve deterministically
// so the test's own explicit call is known to consume the array's next entry.
test("refreshVoiceStatus: an unavailable response sets the actionable tooltip", async () => {
  const fetchImpl = fetchFor([
    { ok: true, status: 200, json: async () => ({ available: false, reason: "needs the extra" }) },
  ]);
  const { window: win } = loadApp({ fetchImpl });
  await tick(); await tick(); await tick();   // boot's own auto refreshVoiceStatus() consumes index0

  const mic = win.document.getElementById("chat-mic");
  assert.equal(mic.title, "needs the extra");
  assert.ok(mic.classList.contains("unavailable"));
});

test("refreshVoiceStatus: becoming available afterwards RESETS the tooltip, not leaves the stale unavailable text", async () => {
  const fetchImpl = fetchFor([
    { ok: true, status: 200, json: async () => ({ available: false, reason: "needs the [voice] extra" }) },
    { ok: true, status: 200, json: async () => ({ available: true, reason: "" }) },
  ]);
  const { window: win } = loadApp({ fetchImpl });
  await tick(); await tick(); await tick();   // boot's own auto call consumes index0 (unavailable)
  const mic = win.document.getElementById("chat-mic");
  assert.match(mic.title, /needs the \[voice\] extra/, "precondition: starts unavailable");

  await win.refreshVoiceStatus();   // index1 (now available)
  assert.doesNotMatch(mic.title, /needs the \[voice\] extra/,
    "a mic that is now available must not keep claiming it needs the extra");
  assert.doesNotMatch(mic.title, /needs/i);
  assert.ok(!mic.classList.contains("unavailable"));
});

test("refreshVoiceStatus: a 404 (plugin not installed) sets the install hint, unaffected by this fix", async () => {
  const fetchImpl = fetchFor([{ ok: false, status: 404, json: async () => ({}) }]);
  const { window: win } = loadApp({ fetchImpl });
  await tick(); await tick(); await tick();

  const mic = win.document.getElementById("chat-mic");
  assert.match(mic.title, /not installed/);
  assert.ok(mic.classList.contains("unavailable"));
});
