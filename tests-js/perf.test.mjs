// jsdom tests for the Settings "Performance" card (setupPerfCard /
// refreshPerfEstimate in app.js): the GPU-layers + context sliders seed from
// /v1/config, the live VRAM readout reflects /api/vram-estimate, and Apply
// PATCHes the two engine keys.

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

// Records calls; serves config + estimate + the bootstrap endpoints.
function makeFetch(calls, { fits = true } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    calls.push({ u, method, body: opts.body });
    if (u.includes("/api/vram-estimate"))
      return { ok: true, status: 200, json: async () => ({
        model: "m", model_bytes: 4 * GIB, weights: 4 * GIB, kv_cache: 0.6 * GIB,
        overhead: 1.5 * GIB, needed: 6.1 * GIB, free: 11 * GIB, total: 16 * GIB,
        fits, approximate: true }) };
    if (u.endsWith("/v1/config") && method === "GET")
      return { ok: true, status: 200, json: async () => ({ n_ctx: 8192, n_gpu_layers: 99 }) };
    if (u.endsWith("/v1/config") && method === "PATCH")
      return { ok: true, status: 200, json: async () => ({ ok: true }) };
    if (u.includes("/api/models"))
      return { ok: true, status: 200, json: async () => ({ models: [], active: "" }) };
    if (u.includes("/api/plugins"))
      return { ok: true, status: 200, json: async () => ({ plugins: [] }) };
    return { ok: true, status: 200, json: async () => ({}) };
  };
}

test("perf card seeds sliders from config and renders the estimate", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: makeFetch(calls) });
  const ctx = window.document.getElementById("perf-ctx");
  const est = window.document.getElementById("perf-estimate");
  assert.ok(await waitFor(() => ctx.value === "8192"), "ctx slider seeded from config (8192)");
  assert.equal(window.document.getElementById("perf-gl-val").textContent, "all",
    "gpu-layers 99 shows as 'all'");
  assert.ok(await waitFor(() => /needed/.test(est.textContent)), "VRAM estimate rendered");
  assert.match(est.textContent, /fits/);
});

test("moving the context slider refetches the estimate", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: makeFetch(calls) });
  await waitFor(() => /needed/.test(window.document.getElementById("perf-estimate").textContent));
  const before = calls.filter((c) => c.u.includes("/api/vram-estimate")).length;
  const ctx = window.document.getElementById("perf-ctx");
  ctx.value = "16384";
  ctx.dispatchEvent(new window.Event("input", { bubbles: true }));
  assert.equal(window.document.getElementById("perf-ctx-val").textContent, "16384",
    "label updates immediately");
  assert.ok(await waitFor(
    () => calls.filter((c) => c.u.includes("/api/vram-estimate")).length > before),
    "a new estimate is fetched after the slider moves");
});

test("Apply PATCHes n_ctx and n_gpu_layers to /v1/config", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: makeFetch(calls) });
  await waitFor(() => window.document.getElementById("perf-ctx").value === "8192");
  window.document.getElementById("perf-apply").click();
  assert.ok(await waitFor(() => calls.some(
    (c) => c.u.endsWith("/v1/config") && c.method === "PATCH")), "Apply issues a PATCH");
  const patch = calls.find((c) => c.u.endsWith("/v1/config") && c.method === "PATCH");
  const body = JSON.parse(patch.body);
  assert.ok("n_ctx" in body && "n_gpu_layers" in body, "PATCH carries both engine keys");
});

test("estimate flags a model that may not fit (negative)", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: makeFetch(calls, { fits: false }) });
  const est = window.document.getElementById("perf-estimate");
  assert.ok(await waitFor(() => /may not fit/.test(est.textContent)),
    "a non-fitting model is flagged");
  assert.ok(est.classList.contains("perf-warn"), "warn styling applied");
});
