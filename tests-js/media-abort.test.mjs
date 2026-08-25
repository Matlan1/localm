// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

// Media generation (image / music / video) runs as a background job. The Stop
// button reveals while it runs and POSTs /api/jobs/<id>/cancel. showStop /
// hideStop (pages.js) + cancelJob (app.js) implement it.

function setup() {
  const calls = [];
  const fetchImpl = async (url, opts) => {
    calls.push({ url: String(url), opts: opts || {} });
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  return { window, calls };
}

test("showStop reveals the Stop button and wires it to cancel that job", async () => {
  const { window, calls } = setup();
  const btn = window.document.getElementById("img-stop");
  assert.ok(btn, "img-stop button exists in the markup");

  window.showStop("img-stop", "job-abc");
  assert.notEqual(btn.style.display, "none", "Stop button is shown");
  assert.equal(btn.disabled, false);

  btn.onclick();                                   // user presses Stop
  // cancelJob calls fetch() before its first await, so the call is already
  // recorded. Flush microtasks only: a macrotask would run the rAF-deferred
  // page init.
  await Promise.resolve();

  const cancels = calls.filter((c) => c.url.includes("/api/jobs/job-abc/cancel"));
  assert.equal(cancels.length, 1, "exactly one cancel POST for this job");
  assert.equal(cancels[0].opts.method, "POST");
  assert.equal(btn.disabled, true, "button disables itself once pressed");
});

test("hideStop hides the Stop button and clears its handler", async () => {
  const { window } = setup();
  const btn = window.document.getElementById("music-stop");
  window.showStop("music-stop", "j1");
  window.hideStop("music-stop");
  assert.equal(btn.style.display, "none");
  assert.equal(btn.onclick, null);
  assert.equal(btn.textContent, "Stop", "label is reset for the next run");
});

test("every media page has a Stop button", () => {
  const { window } = setup();
  for (const id of ["img-stop", "music-stop", "video-stop"]) {
    assert.ok(window.document.getElementById(id), `${id} exists`);
  }
});
