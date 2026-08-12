// SPDX-License-Identifier: AGPL-3.0-or-later
// The Settings "Runtime update" card: check surfaces an available llama.cpp
// build and reveals "Update runtime"; apply streams a job and reports honestly.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function makeFetch(routes) {
  return async (url, opts = {}) => {
    const u = String(url);
    for (const [frag, resp] of routes) {
      if (u.includes(frag)) {
        const body = typeof resp === "function" ? resp(opts) : resp;
        return { ok: true, status: 200, text: async () => "", json: async () => body };
      }
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function flush() {
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
}

test("Runtime update: check surfaces a different build and reveals 'Update runtime'", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", { installed: true, backend: "vulkan", current: "b10300",
      target: "b10361", newer: true, pinned: null }],
  ]) });
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  assert.match(doc.getElementById("runtime-update-status").textContent, /b10361/);
  assert.equal(doc.getElementById("runtime-update-apply").hidden, false);
});

test("Runtime update: up to date hides 'Update runtime'", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", { installed: true, backend: "vulkan", current: "b10361",
      target: "b10361", newer: false, pinned: null }],
  ]) });
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  assert.match(doc.getElementById("runtime-update-status").textContent, /Up to date/);
  assert.equal(doc.getElementById("runtime-update-apply").hidden, true);
});

test("Runtime update: nothing installed says so and hides the button", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", { installed: false, backend: null, current: null,
      target: null, newer: false, pinned: null }],
  ]) });
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();
  assert.match(doc.getElementById("runtime-update-status").textContent, /run setup first/);
  assert.equal(doc.getElementById("runtime-update-apply").hidden, true);
});

test("Runtime update: 'Update runtime' POSTs, streams the job, and re-checks on success", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", { installed: true, backend: "vulkan", current: "b10300",
      target: "b10361", newer: true, pinned: null }],
    ["/api/runtime/update", (opts) => { calls.push(opts.method); return { job_id: "j1" }; }],
  ]) });
  runScript(window, `streamJob = (id, onLine) => {
    onLine("Fetching release b10361 ...");
    onLine("OK - the fetched build loads on this machine.");
    return Promise.resolve({ status: "done" });
  };`);
  const doc = window.document;
  // A real user only ever reaches "Update runtime" after a check reveals it.
  doc.getElementById("runtime-update-check").click();
  await flush();
  assert.equal(doc.getElementById("runtime-update-apply").hidden, false);

  doc.getElementById("runtime-update-apply").click();
  await flush();
  assert.equal(calls[0], "POST");
  assert.match(doc.getElementById("runtime-update-log").textContent, /loads on this machine/);
  // Success re-runs the check so the card reflects the real post-apply state
  // (same pattern as the managed-ComfyUI panel) - this fixture's check is
  // still wired to report "newer", so the button stays visible; the
  // up-to-date case is covered by the "hides 'Update runtime'" test above.
  assert.match(doc.getElementById("runtime-update-status").textContent, /b10361/);
});

test("Runtime update: a failed apply shows the job's own reason, never a fake success", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/check", { installed: true, backend: "vulkan", current: "b10300",
      target: "b10361", newer: true, pinned: null }],
    ["/api/runtime/update", { job_id: "j2" }],
  ]) });
  runScript(window, `streamJob = (id, onLine) => {
    onLine("Cannot provision the runtime right now: another setup-llama run is busy.");
    return Promise.resolve({ status: "failed" });
  };`);
  const doc = window.document;
  doc.getElementById("runtime-update-check").click();
  await flush();

  doc.getElementById("runtime-update-apply").click();
  await flush();
  assert.match(doc.getElementById("runtime-update-status").textContent, /another setup-llama run is busy/);
  // The apply button must NOT hide on a failure (no re-check runs) - the
  // user needs to be able to retry.
  assert.equal(doc.getElementById("runtime-update-apply").hidden, false);
});

test("Runtime update: a 409 from the route (already running) is reported without a job stream", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/runtime/update", { ok: false, status: 409, detail: "A runtime update is already running." }],
  ]) });
  // Force the ok:false shape through our makeFetch (which always returns ok:true) -
  // override just this route directly.
  window.fetch = async (url, opts = {}) => {
    if (String(url).includes("/api/runtime/update")) {
      return { ok: false, status: 409, statusText: "Conflict",
        json: async () => ({ detail: "A runtime update is already running." }) };
    }
    return makeFetch([])(url, opts);
  };
  const doc = window.document;
  doc.getElementById("runtime-update-apply").click();
  await flush();
  assert.match(doc.getElementById("runtime-update-status").textContent,
    /A runtime update is already running/);
});
