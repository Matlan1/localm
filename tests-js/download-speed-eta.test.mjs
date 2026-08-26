// SPDX-License-Identifier: AGPL-3.0-or-later
// The model download bar's smoothed (rolling-window) speed and ETA: the pure
// downloadRate() helper and fmtDuration() formatting.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

function call(win, expr) {
  runScript(win, `globalThis.__out = ${expr};`);
  return win.__out;
}

// Default fetch shape for app.js's lazy refreshes on a bare window.
function stubFetch() {
  return async () => ({
    ok: true, status: 200, text: async () => "",
    json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
  });
}

// Cross-realm objects built inside jsdom fail deepStrictEqual on prototype
// identity; assert the two fields directly.
function assertNull(r) {
  assert.equal(r.bytesPerSec, null);
  assert.equal(r.etaSec, null);
}

// --------------------------------------------------------------------------- //
//  downloadRate(samples, total) - smoothed bytes/sec + ETA                     //
// --------------------------------------------------------------------------- //

test("downloadRate averages over the sample window", () => {
  const { window: win } = loadApp({ fetchImpl: stubFetch() });
  // 10s window, 10 MB transferred -> 1 MB/s.
  const r = call(win,
    `downloadRate([{t:0,downloaded:0},{t:3000,downloaded:5000000},` +
    `{t:10000,downloaded:10000000}], 20000000)`);
  assert.equal(r.bytesPerSec, 1000000, "averaged from first..last sample");
  // ETA over the remaining 10 MB at 1 MB/s = 10s.
  assert.equal(r.etaSec, 10);
});

test("downloadRate needs two samples and forward progress", () => {
  const { window: win } = loadApp({ fetchImpl: stubFetch() });
  assertNull(call(win, "downloadRate([], 100)"));
  assertNull(call(win, "downloadRate([{t:0,downloaded:5}], 100)"));
  // zero elapsed time -> cannot divide
  assertNull(call(win, "downloadRate([{t:5,downloaded:0},{t:5,downloaded:9}], 100)"));
  // no forward progress -> null
  assertNull(call(win, "downloadRate([{t:0,downloaded:9},{t:1000,downloaded:9}], 100)"));
});

test("downloadRate: speed without a known total still works; eta null", () => {
  const { window: win } = loadApp({ fetchImpl: stubFetch() });
  const r = call(win,
    "downloadRate([{t:0,downloaded:0},{t:2000,downloaded:2000}], null)");
  assert.equal(r.bytesPerSec, 1000);
  assert.equal(r.etaSec, null);
});

test("downloadRate: at the finish line eta is 0", () => {
  const { window: win } = loadApp({ fetchImpl: stubFetch() });
  const r = call(win,
    "downloadRate([{t:0,downloaded:0},{t:1000,downloaded:1000}], 1000)");
  assert.equal(r.etaSec, 0);
});

// --------------------------------------------------------------------------- //
//  fmtDuration(sec)                                                            //
// --------------------------------------------------------------------------- //

test("fmtDuration formats seconds / minutes / hours", () => {
  const { window: win } = loadApp({ fetchImpl: stubFetch() });
  assert.equal(call(win, "fmtDuration(0)"), "0s");
  assert.equal(call(win, "fmtDuration(45)"), "45s");
  assert.equal(call(win, "fmtDuration(90)"), "1m 30s");
  assert.equal(call(win, "fmtDuration(3661)"), "1h 01m");
  assert.equal(call(win, "fmtDuration(null)"), "");
  assert.equal(call(win, "fmtDuration(-5)"), "");
});

// --------------------------------------------------------------------------- //
//  Wiring: the pull handler renders speed + ETA in the bar text               //
// --------------------------------------------------------------------------- //

function makeFetch() {
  return async (url, opts = {}) => {
    if (url === "/api/models/pull") {
      return { ok: true, status: 200, json: async () => ({ job_id: "j1" }), text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

test("pull progress text includes the smoothed speed and ETA", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  // Fixed rate plus a streamJob that fires one progress event and captures the
  // bar text before the handler overwrites it with "done".
  runScript(win, `
    downloadRate = () => ({ bytesPerSec: 5 * 1024 * 1024, etaSec: 90 });
    streamJob = async (jobId, onLine, onProgress) => {
      onProgress({ pct: 50, total: 1000000, downloaded: 500000 });
      globalThis.__pct = document.getElementById("pull-pct").textContent;
      return { status: "done" };
    };
  `);

  win.document.getElementById("pull-spec").value = "owner/repo:m.gguf";
  win.document.getElementById("pull-start").click();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  const text = win.__pct;
  assert.match(text, /50%/, "still shows percent");
  assert.match(text, /5\.0 MB\/s/, "shows the smoothed speed");
  assert.match(text, /ETA 1m 30s/, "shows the ETA");
});

// --------------------------------------------------------------------------- //
//  a multi-file (split GGUF) download shows "file N of M: <name>"               //
// --------------------------------------------------------------------------- //

test("pull progress shows the current file for a multi-file download", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  runScript(win, `
    streamJob = async (jobId, onLine, onProgress) => {
      onProgress({ pct: 40, total: 1000, downloaded: 400,
                   name: "model-00002-of-00003.gguf", index: 2, count: 3 });
      const el = document.getElementById("pull-file");
      globalThis.__file = { hidden: el.hidden, text: el.textContent };
      return { status: "done" };
    };
  `);
  win.document.getElementById("pull-spec").value = "owner/repo:m.gguf";
  win.document.getElementById("pull-start").click();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(win.__file.hidden, false, "the file line is shown for a multi-file pull");
  assert.match(win.__file.text, /file 2 of 3: model-00002-of-00003\.gguf/);
});

test("pull progress hides the file line for a single-file download", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  runScript(win, `
    streamJob = async (jobId, onLine, onProgress) => {
      onProgress({ pct: 70, total: 1000, downloaded: 700 });   // no count/name
      globalThis.__hidden = document.getElementById("pull-file").hidden;
      return { status: "done" };
    };
  `);
  win.document.getElementById("pull-spec").value = "owner/repo:m.gguf";
  win.document.getElementById("pull-start").click();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(win.__hidden, true, "single-file pull keeps the file line hidden");
});

// --------------------------------------------------------------------------- //
// A lost SSE connection (streamJob giving up after exhausting its reconnect   //
// budget) renders differently from a genuine pull failure.                    //
// --------------------------------------------------------------------------- //

test("a disconnected pull (streamJob gave up reconnecting) is NOT painted as failed", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  runScript(win, `
    streamJob = async (jobId, onLine, onProgress) => ({ status: "disconnected" });
  `);
  win.document.getElementById("pull-spec").value = "owner/repo:m.gguf";
  win.document.getElementById("pull-start").click();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  const bar = win.document.getElementById("pull-bar");
  const pct = win.document.getElementById("pull-pct").textContent;
  assert.ok(!bar.classList.contains("failed"),
    "a lost connection must not be styled as a failed pull");
  assert.doesNotMatch(pct, /failed/i, "the status text must not claim the pull failed");
  assert.match(pct, /connection|lost/i, "the status text names the real fact - a lost connection");
});

test("a genuinely failed pull is still painted as failed - the fix must not mask a real one", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  runScript(win, `
    streamJob = async (jobId, onLine, onProgress) => ({ status: "failed", returncode: 1 });
  `);
  win.document.getElementById("pull-spec").value = "owner/repo:m.gguf";
  win.document.getElementById("pull-start").click();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  const bar = win.document.getElementById("pull-bar");
  const pct = win.document.getElementById("pull-pct").textContent;
  assert.ok(bar.classList.contains("failed"), "a real failure is still styled as failed");
  assert.match(pct, /failed/i);
});
