// SPDX-License-Identifier: AGPL-3.0-or-later
// scan-reason (audit finding E): the Models page "Re-scan ComfyUI folder" button
// must surface WHY an empty scan found nothing instead of a bare "Added 0".
//
// The scanner returns a ScanResult whose `method` carries the reason on an empty
// scan ("none (comfy_workdir not configured)", "none (models folder not found
// under {path})"). A real scan is job-based now (see gui_scan_models /
// scan_comfy_models's progress_cb): POST /api/models/scan returns {job_id}
// immediately and the final progress event (phase "done") before "end" carries
// the same added/skipped/method fields the old synchronous response body used
// to - this drives the real button + fetch + SSE + toast path and asserts the
// humanized reason still shows, and that the internal folder-walk/hybrid method
// jargon never leaks into the toast.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

// Flush enough microtask/macrotask turns for the async click handler's
// fetch -> job -> SSE -> toast -> refreshModelsPage chain to settle.
const flush = async () => {
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
};

// A single-shot SSE playback, shaped exactly like streamjob-reconnect.test.mjs's
// mock and models-import-comfy.test.mjs's sseResponse: one "data: <json>\n\n"
// frame per read() call, then a clean EOF.
//
// delayMs>0 resolves each frame on a REAL timer instead of an already-resolved
// microtask - plain `await`-chained microtasks all drain within a single event
// loop turn, so reading the DOM after just one `setTimeout(...,0)` would
// otherwise observe the FULLY SETTLED end state, not a mid-stream one, no
// matter how many progress events preceded it. Use delayMs + waitFor (below)
// to actually catch an in-flight moment.
function sseResponse(events, { delayMs = 0 } = {}) {
  const frames = events.map((ev) => `data: ${JSON.stringify(ev)}\n\n`);
  let idx = 0;
  const enc = new TextEncoder();
  return {
    ok: true, status: 200,
    body: {
      getReader() {
        return {
          read() {
            const emit = () => {
              if (idx < frames.length) {
                const chunk = enc.encode(frames[idx]);
                idx++;
                return { done: false, value: chunk };
              }
              return { done: true, value: undefined };
            };
            if (!delayMs) return Promise.resolve(emit());
            return new Promise((resolve) => setTimeout(() => resolve(emit()), delayMs));
          },
          async cancel() {},
        };
      },
    },
  };
}

// Poll *cond* until it returns true or *timeout* elapses.
async function waitFor(cond, { timeout = 2000, interval = 5 } = {}) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (cond()) return true;
    await new Promise((r) => setTimeout(r, interval));
  }
  return false;
}

function makeScanFetch(scanResult) {
  return async (url) => {
    const u = String(url);
    if (/\/api\/jobs\/[^/]+\/events$/.test(u)) {
      const added = scanResult.added || 0;
      const skipped = scanResult.skipped || 0;
      return sseResponse([
        { type: "progress", phase: "done", done: added + skipped, total: added + skipped,
          added, skipped, method: scanResult.method },
        { type: "end", status: "done", returncode: 0 },
      ]);
    }
    if (u.startsWith("/api/models/scan")) {
      return { ok: true, status: 200, json: async () => ({ job_id: "scan-btn-job" }), text: async () => "" };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models: [], active: null }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

async function clickScan(scanResult) {
  const { window } = loadAppWithPages({ fetchImpl: makeScanFetch(scanResult) });
  const btn = window.document.getElementById("models-scan-btn");
  assert.ok(btn, "the Re-scan ComfyUI folder button exists");
  btn.click();
  await flush();
  return window.document.getElementById("toast").textContent;
}

test("scan-reason: an unconfigured ComfyUI workdir surfaces the config hint, not 'Added 0'", async () => {
  const msg = await clickScan({ added: 0, skipped: 0, method: "none (comfy_workdir not configured)" });
  // Negative case: the old bare "Added 0" text must be gone.
  assert.ok(!/Added 0/.test(msg), `must not show a bare "Added 0" (got: ${msg})`);
  // The humanized reason must point the user at where to fix it.
  assert.match(msg, /workdir/i, `must name the missing workdir (got: ${msg})`);
  assert.match(msg, /Settings/i, `must point the user to Settings (got: ${msg})`);
});

test("scan-reason: a missing models folder names the path it looked under", async () => {
  const path = "D:\\path\\to\\comfy";
  const msg = await clickScan({
    added: 0, skipped: 0,
    method: `none (models folder not found under ${path})`,
  });
  assert.ok(!/Added 0/.test(msg), `must not show a bare "Added 0" (got: ${msg})`);
  assert.match(msg, /models folder not found/i, `must say the folder was not found (got: ${msg})`);
  assert.ok(msg.includes(path), `must include the path it looked under (got: ${msg})`);
});

test("scan-reason: a successful scan keeps the clean 'Added X, skipped Y' and leaks no method jargon", async () => {
  const msg = await clickScan({ added: 3, skipped: 2, method: "hybrid (folder-walk + /object_info)" });
  assert.match(msg, /Added 3 models/, `success toast keeps the count (got: ${msg})`);
  assert.match(msg, /skipped 2 existing/, `success toast keeps the skipped count (got: ${msg})`);
  // The internal method jargon must never surface in the user-facing toast.
  assert.ok(!/folder-walk/i.test(msg), `must not leak "folder-walk" jargon (got: ${msg})`);
  assert.ok(!/hybrid/i.test(msg), `must not leak "hybrid" jargon (got: ${msg})`);
  assert.ok(!/object_info/i.test(msg), `must not leak "object_info" jargon (got: ${msg})`);
});

test("scan-reason: an empty-but-configured scan reads better than 'Added 0'", async () => {
  // Folders exist and were walked, but held no models: not a misconfig, but a
  // bare "Added 0 models, skipped 0 existing" is still poor. It should read as
  // "no models found", and must not leak the method jargon.
  const msg = await clickScan({ added: 0, skipped: 0, method: "folder-walk" });
  assert.ok(!/folder-walk/i.test(msg), `must not leak "folder-walk" jargon (got: ${msg})`);
  assert.match(msg, /no .*models found/i, `should read as no models found (got: ${msg})`);
});

test("scan-reason: a real scan shows live 'registering model N of M' progress before the toast", async () => {
  // The concrete gap this unit closes for the plain button too: it used to
  // toast a single "Scanning..." line and then show NOTHING until the whole
  // batch finished.
  const { window } = loadAppWithPages({
    fetchImpl: async (url) => {
      const u = String(url);
      if (/\/api\/jobs\/[^/]+\/events$/.test(u)) {
        // Real per-frame delay (see sseResponse's doc comment) so the test
        // below can genuinely catch an in-flight state via polling.
        return sseResponse([
          { type: "progress", phase: "registering", done: 1, total: 2, name: "a.safetensors" },
          { type: "progress", phase: "registering", done: 2, total: 2, name: "b.safetensors" },
          { type: "progress", phase: "done", done: 2, total: 2, added: 2, skipped: 0, method: "folder-walk" },
          { type: "end", status: "done", returncode: 0 },
        ], { delayMs: 30 });
      }
      if (u.startsWith("/api/models/scan")) {
        return { ok: true, status: 200, json: async () => ({ job_id: "scan-btn-job" }) };
      }
      return { ok: true, status: 200, json: async () => ({ models: [], active: null }), text: async () => "" };
    },
  });
  const btn = window.document.getElementById("models-scan-btn");
  btn.click();
  const pct = window.document.getElementById("scan-pct");
  // Catch the FIRST tick specifically - proves the count genuinely advances
  // per event rather than jumping straight to the end.
  const sawFirst = await waitFor(() => /registering model 1 of 2/i.test(pct.textContent));
  assert.ok(sawFirst, `never observed "1 of 2" while the job was in flight (last seen: ${pct.textContent})`);

  // The remaining frames (2/2, the final "done" progress event, "end") are
  // still resolving on their own 30ms real timers - wait for the button to
  // re-enable (the click handler's own `finally`, which only runs once
  // streamJob has fully drained the stream) rather than a fixed-tick flush(),
  // which would return long before those timers fire and leave dangling work
  // behind for a LATER test to trip over.
  const settled = await waitFor(() => btn.disabled === false, { timeout: 3000 });
  assert.ok(settled, "the scan button never re-enabled - the click handler did not finish");
  assert.match(window.document.getElementById("toast").textContent, /added 2 models/i);
});

test("scan-reason: scanResultMessage maps method reasons directly (unit)", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeScanFetch({}) });
  const fn = window.scanResultMessage;
  assert.equal(typeof fn, "function", "scanResultMessage is exported for reuse/testing");

  // The notfound and success mappings are already exercised end to end via the
  // click-driven tests above; this unit layer keeps one representative case.
  const unconfigured = fn({ added: 0, skipped: 0, method: "none (comfy_workdir not configured)" });
  assert.ok(!/Added 0/.test(unconfigured) && /workdir/i.test(unconfigured) && /Settings/i.test(unconfigured));
});
