// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings > Diagnostics card (pages/settings.js). The five ACTIVE self-checks
// from `localm doctor`, run from the app.
//
// The probes themselves are Python and are covered there. What can only go wrong
// HERE is the rendering, and the risk is specific: this card has to state a
// conclusion where a terminal would show a transcript, so the failure mode is a
// card that OVERCLAIMS - "all good" on a run that never happened, a stale phase
// that reads as still-running, or a failure whose reassuring half is the line the
// user sees. Those are what these assert.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const COVERS = [
  { key: "llama_lib", label: "llama.cpp library" },
  { key: "native_abi", label: "Native ABI" },
  { key: "worker_spawn", label: "Worker process spawn" },
  { key: "venv", label: "Nested venv creation" },
  { key: "hf_backend", label: "HF (transformers) backend" },
];

function body(over = {}) {
  return {
    running: false, job_id: null, started_at: null, finished_at: null,
    progress: null, report: null, covers: COVERS, ...over,
  };
}

function check(key, status, summary, findings) {
  return {
    key, label: COVERS.find((c) => c.key === key).label, status, summary,
    findings: findings || [{ status, text: summary }],
  };
}

function load() {
  const { window: win } = loadAppWithPages({
    fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
  });
  return win;
}

const statusText = (win) => win.document.getElementById("doctor-status").textContent;
const rows = (win) =>
  [...win.document.getElementById("doctor-checks").querySelectorAll(".doctor-check")];
const pillOf = (row) => row.querySelector(".job-state");


// --------------------------------------------------------------------------- //
//  Before a run                                                                //
// --------------------------------------------------------------------------- //

test("before any run the card names the five checks instead of showing a bare button", () => {
  const win = load();
  win.renderDoctorReport(body());
  assert.match(statusText(win), /Not run yet/);
  assert.deepEqual(rows(win).map((r) => r.querySelector(".job-name").textContent),
                   COVERS.map((c) => c.label));
});

test("an unrun check gets the NEUTRAL pill, never a warning colour", () => {
  // Every ordinary box has an optional backend it does not have installed. If
  // "not run" painted yellow, every ordinary box would show warnings and the
  // colour would stop carrying information.
  const win = load();
  win.renderDoctorReport(body());
  for (const row of rows(win)) {
    assert.equal(pillOf(row).className, "job-state ");
    assert.equal(pillOf(row).textContent, "not run");
  }
});


// --------------------------------------------------------------------------- //
//  A completed run                                                             //
// --------------------------------------------------------------------------- //

test("an all-clear verdict says what it checked and does not claim more", () => {
  const win = load();
  win.renderDoctorReport(body({
    finished_at: 2, report: { verdict: "ok", checks: [
      check("llama_lib", "ok", "llama.dll found in /x"),
      check("native_abi", "ok", "native ABI: struct layout matches this build"),
      check("worker_spawn", "ok", "background worker spawn: OK"),
      check("venv", "ok", "venv creation: OK"),
      check("hf_backend", "skipped", "not installed - the HF backend is optional"),
    ] },
  }));
  const text = statusText(win);
  // 4 ran, 1 skipped: the count must be what actually ran.
  assert.match(text, /All 4 active checks passed/);
  assert.match(text, /not everything about your system/);
});

test("a failing check is counted and the row carries the failure pill", () => {
  const win = load();
  win.renderDoctorReport(body({
    report: { verdict: "fail", checks: [
      check("llama_lib", "fail", "llama.dll in /x is empty (0 bytes) - corrupt"),
      check("native_abi", "skipped", "not checked - no healthy library"),
      check("worker_spawn", "ok", "background worker spawn: OK"),
      check("venv", "warn", "venv creation was inconclusive"),
      check("hf_backend", "skipped", "not installed"),
    ] },
  }));
  assert.match(statusText(win), /2 of 3 active checks need attention/);
  assert.equal(pillOf(rows(win)[0]).textContent, "failed");
  assert.ok(pillOf(rows(win)[0]).className.includes("st-error"));
  assert.equal(pillOf(rows(win)[3]).textContent, "warning");
  assert.ok(pillOf(rows(win)[3]).className.includes("st-warn"));
});

test("a run that could NOT happen never renders as a clean result", () => {
  // The sharpest way this card could lie: an unrunnable diagnostic looking like
  // a pass. The verdict is "error", so nothing may say "passed".
  const win = load();
  win.renderDoctorReport(body({
    report: { verdict: "error", checks: [],
              error: "the diagnostics run did not finish within 360s" },
  }));
  const text = statusText(win);
  assert.match(text, /could not be run/);
  assert.match(text, /did not finish within 360s/);
  assert.doesNotMatch(text, /passed/);
});

test("the reassuring half of a failure is not what the row leads with", () => {
  // The library check reports a GREEN "found it" line and then the BLAS kernel
  // failure underneath. The headline must be the failure; the green line is
  // context below it.
  const win = load();
  win.renderDoctorReport(body({
    report: { verdict: "fail", checks: [
      check("llama_lib", "fail",
            "rocblas is installed but its rocblas/ kernel directory is empty",
            [{ status: "ok", text: "llama.dll found in /x" },
             { status: "fail",
               text: "rocblas is installed but its rocblas/ kernel directory is empty" }]),
    ] },
  }));
  const row = rows(win)[0];
  assert.match(row.querySelector(".sub").textContent, /^rocblas is installed/);
  const hints = [...row.querySelectorAll(".doctor-check-hint")].map((n) => n.textContent);
  assert.deepEqual(hints, ["llama.dll found in /x"]);
});

test("a finding's hint lines are shown, so an ABI mismatch names the fields", () => {
  const win = load();
  win.renderDoctorReport(body({
    report: { verdict: "fail", checks: [
      check("native_abi", "fail", "native ABI MISMATCH",
            [{ status: "fail", text: "native ABI MISMATCH",
               hints: ["n_gpu_layers at offset 12, expected 16", "size 96 != 88"] }]),
    ] },
  }));
  const hints = [...rows(win)[0].querySelectorAll(".doctor-check-hint")]
    .map((n) => n.textContent);
  assert.deepEqual(hints,
                   ["n_gpu_layers at offset 12, expected 16", "size 96 != 88"]);
});


// --------------------------------------------------------------------------- //
//  While a run is in flight                                                    //
// --------------------------------------------------------------------------- //

test("a running card names the check in flight and counts only what finished", () => {
  const win = load();
  win.renderDoctorReport(body({
    running: true,
    progress: { phase: "Nested venv creation", done: 3, total: 5 },
  }));
  assert.match(statusText(win), /Running: Nested venv creation \(3 of 5 done\)/);
  assert.equal(win.document.getElementById("doctor-run").disabled, true);
});

test("a run with no progress yet does not invent a number", () => {
  // ADR-0008 R1: an operation that has not established where it is is at an
  // UNKNOWN point, never at 0 of 5.
  const win = load();
  win.renderDoctorReport(body({ running: true, progress: { phase: "", done: 0, total: 5 } }));
  assert.equal(statusText(win), "Running the checks...");
});

test("the button is re-enabled once the run finishes", () => {
  const win = load();
  win.renderDoctorReport(body({ running: true, progress: { phase: "x", done: 1, total: 5 } }));
  assert.equal(win.document.getElementById("doctor-run").disabled, true);
  win.renderDoctorReport(body({ report: { verdict: "ok", checks: [check("venv", "ok", "ok")] } }));
  assert.equal(win.document.getElementById("doctor-run").disabled, false);
});


// --------------------------------------------------------------------------- //
//  Reattaching                                                                 //
// --------------------------------------------------------------------------- //

test("refreshDiagnosticsCard paints a run that is already in flight elsewhere", async () => {
  // ADR-0008: a run started in another tab, or in this one before a reload, must
  // not be undiscoverable just because this page did not start it.
  const { window: win } = loadAppWithPages({
    fetchImpl: async (url) => {
      if (String(url) === "/api/doctor") {
        return { ok: true, status: 200, json: async () => body({
          running: true, job_id: "abc",
          progress: { phase: "Worker process spawn", done: 2, total: 5 },
        }) };
      }
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    },
  });
  await win.refreshDiagnosticsCard();
  assert.match(statusText(win), /Running: Worker process spawn \(2 of 5 done\)/);
});

test("a refresh that cannot reach the server leaves the card as it was", async () => {
  const { window: win } = loadAppWithPages({
    fetchImpl: async () => { throw new Error("offline"); },
  });
  win.renderDoctorReport(body({ report: { verdict: "ok", checks: [check("venv", "ok", "venv creation: OK")] } }));
  const before = statusText(win);
  await win.refreshDiagnosticsCard();
  assert.equal(statusText(win), before);
});

test("mid-run, each row says where IT is rather than all claiming to be waiting", () => {
  // The card said "4 of 5 done" on the line above while every row said
  // "waiting..." - not false (the browser has no per-check result until the run
  // ends) but self-contradictory. A finished row says it was checked and that
  // the result is coming; it does NOT guess a verdict.
  const win = load();
  win.renderDoctorReport(body({
    running: true,
    progress: { phase: "Nested venv creation", done: 3, total: 5 },
  }));
  const said = rows(win).map((r) => r.querySelector(".sub").textContent);
  assert.deepEqual(said, [
    "checked - result when the run finishes",
    "checked - result when the run finishes",
    "checked - result when the run finishes",
    "checking now...",
    "waiting...",
  ]);
  // And still no verdict pill on any of them.
  for (const row of rows(win)) assert.equal(pillOf(row).textContent, "not run");
});
