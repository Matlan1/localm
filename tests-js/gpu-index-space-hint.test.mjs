// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the native index-space hint on the Settings GPU selectors
// (settings-perf.js): when GET /api/gpus says index_space "native" (the
// vulkan build serving ggml's own device registry - GPU-SPLIT-VKINDEX), both
// the "Main GPU" selector row and the "Split across GPUs" row must carry a
// visible hint that the numbering is the native backend's load-time order,
// and must carry NO such hint otherwise.

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

function makeFetch(calls, { gpus = [], indexSpace = null } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    calls.push({ u, method, body: opts.body });
    if (u.includes("/api/gpus")) {
      const payload = { gpus, main_gpu_index: null, gpu_split_indices: null,
                        probe_status: "ok" };
      if (indexSpace) payload.index_space = indexSpace;
      return { ok: true, status: 200, json: async () => payload };
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

const TWO_NATIVE = [
  { index: 0, name: "AMD Radeon RX 6900 XT (RADV NAVI21)", total: 16 * GIB, free: 15 * GIB },
  { index: 1, name: "llvmpipe (LLVM 19.1.7, 256 bits)", total: 8 * GIB, free: 7 * GIB },
];

test("index_space native: both GPU rows show the native-numbering hint once", async () => {
  const calls = [];
  const { window } = loadApp({
    fetchImpl: makeFetch(calls, { gpus: TWO_NATIVE, indexSpace: "native" }),
  });
  const doc = window.document;
  const selRow = doc.getElementById("perf-gpu-select-row");
  const splitRow = doc.getElementById("perf-gpu-split-row");
  assert.ok(await waitFor(() => selRow.hidden === false), "selector row visible");
  assert.ok(await waitFor(() => splitRow.hidden === false), "split row visible");
  const selHints = selRow.querySelectorAll(".perf-index-space-hint");
  const splitHints = splitRow.querySelectorAll(".perf-index-space-hint");
  assert.equal(selHints.length, 1, "exactly one hint under the Main GPU row");
  assert.equal(splitHints.length, 1, "exactly one hint under the split row");
  assert.match(selHints[0].textContent, /Vulkan backend/);
  assert.match(splitHints[0].textContent, /Vulkan backend/);
});

test("no index_space: no native-numbering hint appears", async () => {
  const calls = [];
  const gpus = [
    { index: 0, name: "NVIDIA RTX 4090", total: 24 * GIB, free: 20 * GIB },
    { index: 1, name: "NVIDIA RTX 3060", total: 12 * GIB, free: 10 * GIB },
  ];
  const { window } = loadApp({ fetchImpl: makeFetch(calls, { gpus }) });
  const doc = window.document;
  const selRow = doc.getElementById("perf-gpu-select-row");
  assert.ok(await waitFor(() => selRow.hidden === false), "selector row visible");
  await settle(30);
  assert.equal(doc.querySelectorAll(".perf-index-space-hint").length, 0,
               "no hint on a torch/nvidia-smi-sourced device list");
});

test("a refresh after switching away from native removes the stale hint", async () => {
  // First load: native. Then the fetch double flips to a plain list (as after
  // a runtime re-provision) and a manual refresh must not leave a stale hint.
  const calls = [];
  const state = { gpus: TWO_NATIVE, indexSpace: "native" };
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u.includes("/api/gpus")) {
      const payload = { gpus: state.gpus, main_gpu_index: null,
                        gpu_split_indices: null, probe_status: "ok" };
      if (state.indexSpace) payload.index_space = state.indexSpace;
      return { ok: true, status: 200, json: async () => payload };
    }
    return makeFetch(calls)(url, opts);
  };
  const { window } = loadApp({ fetchImpl });
  const doc = window.document;
  const selRow = doc.getElementById("perf-gpu-select-row");
  assert.ok(await waitFor(() => selRow.querySelectorAll(".perf-index-space-hint").length === 1),
            "hint present after the native load");
  state.indexSpace = null;
  // Classic-script injection lands the top-level functions on the window
  // (see harness.mjs) - call the refreshers directly for the second pass.
  await window.refreshMainGpuSelector();
  await window.refreshGpuSplitCheckboxes();
  await settle(30);
  assert.equal(doc.querySelectorAll(".perf-index-space-hint").length, 0,
               "stale hint removed once the source is no longer native");
});
