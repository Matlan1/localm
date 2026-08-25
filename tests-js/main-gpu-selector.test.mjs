// SPDX-License-Identifier: AGPL-3.0-or-later
// The Settings "Main GPU" selector (setupMainGpuSelector /
// refreshMainGpuSelector in app/settings-perf.js): populated from GET
// /api/gpus, hidden on a single-GPU box, and PATCHes main_gpu_index on change.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const GIB = 1024 ** 3;
const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, timeout = 800) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await settle(15); }
  return false;
}

// Records calls; serves /api/gpus (configurable gpu list) + the bootstrap
// endpoints every loadApp() init pass hits.
function makeFetch(calls, { gpus = [], mainGpuIndex = null } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    calls.push({ u, method, body: opts.body });
    if (u.includes("/api/gpus"))
      return { ok: true, status: 200, json: async () => ({ gpus, main_gpu_index: mainGpuIndex }) };
    if (u.endsWith("/v1/config") && method === "GET")
      return { ok: true, status: 200, json: async () => ({ n_ctx: 8192, n_gpu_layers: 99 }) };
    if (u.endsWith("/v1/config") && method === "PATCH")
      return { ok: true, status: 200, json: async () => ({ ok: true }) };
    if (u.includes("/api/vram-estimate"))
      return { ok: true, status: 200, json: async () => ({
        model: "m", model_bytes: 4 * GIB, weights: 4 * GIB, kv_cache: 0.6 * GIB,
        overhead: 1.5 * GIB, needed: 6.1 * GIB, free: 11 * GIB, total: 16 * GIB,
        fits: true, approximate: true }) };
    if (u.includes("/api/models"))
      return { ok: true, status: 200, json: async () => ({ models: [], active: "" }) };
    if (u.includes("/api/plugins"))
      return { ok: true, status: 200, json: async () => ({ plugins: [] }) };
    return { ok: true, status: 200, json: async () => ({}) };
  };
}

test("two detected GPUs: selector shows, populated with name + VRAM, preselects the configured index", async () => {
  const calls = [];
  const gpus = [
    { index: 0, name: "NVIDIA RTX 4090", total: 24 * GIB, free: 20 * GIB },
    { index: 1, name: "NVIDIA RTX 3060", total: 12 * GIB, free: 10 * GIB },
  ];
  const { window } = loadApp({ fetchImpl: makeFetch(calls, { gpus, mainGpuIndex: 1 }) });
  const row = window.document.getElementById("perf-gpu-select-row");
  const sel = window.document.getElementById("perf-main-gpu");
  assert.ok(await waitFor(() => row.hidden === false), "row becomes visible for 2+ GPUs");
  assert.equal(sel.options.length, 2, "one option per detected GPU");
  assert.match(sel.options[0].textContent, /RTX 4090/);
  assert.match(sel.options[0].textContent, /24\.0 GB/);
  assert.match(sel.options[1].textContent, /RTX 3060/);
  assert.equal(sel.value, "1", "preselected on the configured main_gpu_index");
});

test("a single detected GPU keeps the selector hidden", async () => {
  const calls = [];
  const gpus = [{ index: 0, name: "Solo GPU", total: 16 * GIB, free: 12 * GIB }];
  const { window } = loadApp({ fetchImpl: makeFetch(calls, { gpus, mainGpuIndex: null }) });
  const row = window.document.getElementById("perf-gpu-select-row");
  const sel = window.document.getElementById("perf-main-gpu");
  // Let the async populate run before asserting the row stayed hidden.
  await waitFor(() => calls.some((c) => c.u.includes("/api/gpus")));
  await settle(30);
  assert.equal(row.hidden, true, "no useful choice on a single-GPU box");
  // The row starts `hidden` in the markup, so also check the populate ran and
  // returned early.
  assert.equal(sel.options.length, 0, "selector left unpopulated for a single GPU");
});

test("no GPUs detected (endpoint reachable but empty) keeps the selector hidden", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: makeFetch(calls, { gpus: [], mainGpuIndex: null }) });
  const row = window.document.getElementById("perf-gpu-select-row");
  const sel = window.document.getElementById("perf-main-gpu");
  await waitFor(() => calls.some((c) => c.u.includes("/api/gpus")));
  await settle(30);
  assert.equal(row.hidden, true);
  assert.equal(sel.options.length, 0);
});

test("changing the selection PATCHes main_gpu_index to /v1/config", async () => {
  const calls = [];
  const gpus = [
    { index: 0, name: "GPU A", total: 24 * GIB, free: 20 * GIB },
    { index: 1, name: "GPU B", total: 12 * GIB, free: 10 * GIB },
  ];
  const { window } = loadApp({ fetchImpl: makeFetch(calls, { gpus, mainGpuIndex: 0 }) });
  const sel = window.document.getElementById("perf-main-gpu");
  assert.ok(await waitFor(() => sel.options.length === 2), "selector populated");
  sel.value = "1";
  sel.dispatchEvent(new window.Event("change", { bubbles: true }));
  assert.ok(await waitFor(() => calls.some(
    (c) => c.u.endsWith("/v1/config") && c.method === "PATCH")), "change issues a PATCH");
  const patch = calls.find((c) => c.u.endsWith("/v1/config") && c.method === "PATCH");
  const body = JSON.parse(patch.body);
  assert.equal(body.main_gpu_index, 1, "PATCH carries the newly selected index");
});
