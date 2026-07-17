// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the Settings "Performance" card (setupPerfCard /
// refreshPerfEstimate in app.js): the GPU-layers + context sliders seed from
// /v1/config, the live VRAM readout reflects /api/vram-estimate, and Apply
// PATCHes the two engine keys. Also covers the "Split across GPUs" checkbox
// row (refreshGpuSplitCheckboxes / onGpuSplitChange / setupGpuSplitCheckboxes
// in app/settings-perf.js), which sits right below the Main GPU selector and
// shares its GET /api/gpus data source and single-GPU-hides-row gate.

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
function makeFetch(calls, { fits = true, gpus = [], gpuSplitIndices = null,
                            free = 11 * GIB } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    calls.push({ u, method, body: opts.body });
    if (u.includes("/api/gpus"))
      return { ok: true, status: 200,
        json: async () => ({ gpus, main_gpu_index: null, gpu_split_indices: gpuSplitIndices }) };
    if (u.includes("/api/vram-estimate"))
      return { ok: true, status: 200, json: async () => ({
        model: "m", model_bytes: 4 * GIB, weights: 4 * GIB, kv_cache: 0.6 * GIB,
        overhead: 1.5 * GIB, needed: 6.1 * GIB, free, total: 16 * GIB,
        fits, approximate: true }) };
    if (u.includes("/api/models/load") && method === "POST") {
      const body = JSON.parse(opts.body || "{}");
      return { ok: true, status: 200, json: async () => ({ status: "loaded", model: body.model }) };
    }
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
  // GPU layers is now an editable number input (id perf-gpu-layers), not a span
  // showing "all" - it seeds its value from config (n_gpu_layers 99 -> "99").
  assert.ok(await waitFor(() => window.document.getElementById("perf-gpu-layers").value === "99"),
    "gpu-layers input seeded from config (99)");
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

test("switching the model via the dropdown refreshes the VRAM estimate", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: makeFetch(calls) });
  await waitFor(() => /needed/.test(window.document.getElementById("perf-estimate").textContent));
  const before = calls.filter((c) => c.u.includes("/api/vram-estimate")).length;
  const sel = window.document.getElementById("model-select");
  // The select is normally populated by refreshModels() from /api/models; add
  // the option this test switches to directly so .value actually takes (jsdom,
  // like a real browser, ignores setting .value to a value with no matching
  // <option>).
  const opt = window.document.createElement("option");
  opt.value = "other-model";
  sel.appendChild(opt);
  sel.value = "other-model";
  sel.dispatchEvent(new window.Event("change", { bubbles: true }));
  assert.ok(await waitFor(
    () => calls.some((c) => c.u.includes("/api/models/load") && c.method === "POST")),
    "the dropdown change issues a model-load request");
  assert.ok(await waitFor(
    () => calls.filter((c) => c.u.includes("/api/vram-estimate")).length > before),
    "a fresh estimate is fetched after the model switch");
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

test("estimate shows 'free VRAM unknown' and no fit verdict when free is withheld", async () => {
  // free:null + fits:null is what the backend returns when the VRAM reading is
  // stale or process-blind: an untrustworthy free must not drive a fit verdict.
  const calls = [];
  const { window } = loadApp({ fetchImpl: makeFetch(calls, { free: null, fits: null }) });
  const est = window.document.getElementById("perf-estimate");
  assert.ok(await waitFor(() => /free VRAM unknown/.test(est.textContent)),
    "a withheld free renders as 'free VRAM unknown', not a number");
  assert.doesNotMatch(est.textContent, /fits|may not fit/,
    "no fit verdict is claimed without a trustworthy free");
  assert.equal(est.classList.contains("perf-warn"), false,
    "no warn styling when fit is genuinely unknown");
});

test("two detected GPUs with a configured split: the checkbox row shows, both boxes pre-checked", async () => {
  const calls = [];
  const gpus = [
    { index: 0, name: "GPU A", total: 24 * GIB, free: 20 * GIB },
    { index: 1, name: "GPU B", total: 12 * GIB, free: 10 * GIB },
  ];
  const { window } = loadApp({ fetchImpl: makeFetch(calls, { gpus, gpuSplitIndices: [0, 1] }) });
  const row = window.document.getElementById("perf-gpu-split-row");
  const list = window.document.getElementById("perf-gpu-split-list");
  assert.ok(await waitFor(() => row.hidden === false), "split row becomes visible for 2+ GPUs");
  const boxes = [...list.querySelectorAll("input[type=checkbox]")];
  assert.equal(boxes.length, 2, "one checkbox per detected GPU");
  assert.ok(boxes.every((cb) => cb.checked), "both checkboxes pre-checked from gpu_split_indices");
});

test("a single detected GPU keeps the split checkbox row hidden", async () => {
  const calls = [];
  const gpus = [{ index: 0, name: "Solo GPU", total: 16 * GIB, free: 12 * GIB }];
  const { window } = loadApp({ fetchImpl: makeFetch(calls, { gpus, gpuSplitIndices: null }) });
  const row = window.document.getElementById("perf-gpu-split-row");
  const list = window.document.getElementById("perf-gpu-split-list");
  await waitFor(() => calls.some((c) => c.u.includes("/api/gpus")));
  await settle(30);
  assert.equal(row.hidden, true, "no useful split on a single-GPU box");
  assert.equal(list.querySelectorAll("input[type=checkbox]").length, 0,
    "checkbox list left unpopulated for a single GPU");
});

test("unchecking down to 1 checked PATCHes gpu_split_indices to null", async () => {
  const calls = [];
  const gpus = [
    { index: 0, name: "GPU A", total: 24 * GIB, free: 20 * GIB },
    { index: 1, name: "GPU B", total: 12 * GIB, free: 10 * GIB },
  ];
  const { window } = loadApp({ fetchImpl: makeFetch(calls, { gpus, gpuSplitIndices: [0, 1] }) });
  const list = window.document.getElementById("perf-gpu-split-list");
  assert.ok(await waitFor(() => list.querySelectorAll("input[type=checkbox]").length === 2),
    "checkbox list populated");
  const boxes = [...list.querySelectorAll("input[type=checkbox]")];
  boxes[0].checked = false;
  boxes[0].dispatchEvent(new window.Event("change", { bubbles: true }));
  assert.ok(await waitFor(() => calls.some(
    (c) => c.u.endsWith("/v1/config") && c.method === "PATCH")), "unchecking issues a PATCH");
  const patch = calls.filter((c) => c.u.endsWith("/v1/config") && c.method === "PATCH").at(-1);
  const body = JSON.parse(patch.body);
  assert.equal(body.gpu_split_indices, null,
    "fewer than 2 checked clears the split rather than silently keeping it on");
});

test("2 checked boxes PATCH gpu_split_indices with both selected indices", async () => {
  const calls = [];
  const gpus = [
    { index: 0, name: "GPU A", total: 24 * GIB, free: 20 * GIB },
    { index: 1, name: "GPU B", total: 12 * GIB, free: 10 * GIB },
  ];
  const { window } = loadApp({ fetchImpl: makeFetch(calls, { gpus, gpuSplitIndices: null }) });
  const list = window.document.getElementById("perf-gpu-split-list");
  assert.ok(await waitFor(() => list.querySelectorAll("input[type=checkbox]").length === 2),
    "checkbox list populated");
  const boxes = [...list.querySelectorAll("input[type=checkbox]")];
  assert.ok(boxes.every((cb) => !cb.checked), "neither box starts checked (no configured split)");
  boxes[0].checked = true;
  boxes[0].dispatchEvent(new window.Event("change", { bubbles: true }));
  boxes[1].checked = true;
  boxes[1].dispatchEvent(new window.Event("change", { bubbles: true }));
  assert.ok(await waitFor(() => calls.some(
    (c) => c.u.endsWith("/v1/config") && c.method === "PATCH")), "checking issues a PATCH");
  const patch = calls.filter((c) => c.u.endsWith("/v1/config") && c.method === "PATCH").at(-1);
  const body = JSON.parse(patch.body);
  assert.deepEqual(body.gpu_split_indices, [0, 1],
    "PATCH carries both checked GPU indices once 2 are checked");
});
