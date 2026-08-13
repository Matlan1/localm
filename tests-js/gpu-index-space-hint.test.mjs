// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the native index-space hint on the Settings GPU selectors
// (settings-perf.js): when GET /api/gpus says index_space "native" (the
// vulkan build serving ggml's own device registry - GPU-SPLIT-VKINDEX), the
// numbering shown is the native backend's load-time order and the page must
// say so - once - and must say nothing otherwise.
//
// CONTRACT CHANGED 2026-08-13 (finding M1 of the settings conformance sweep).
// It used to be appended PER ROW, so a multi-GPU Vulkan box rendered the
// identical sentence twice about 130px apart. It is now ONE element that both
// refreshers drive, so these tests assert on its VISIBILITY rather than on how
// many copies exist, and additionally assert that a second copy never appears.
// The property being guarded is unchanged: the note is present exactly when the
// device list is native-sourced and a GPU row is actually on screen.

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

/** The one shared note. Visible = present in the DOM and not `hidden`. */
function hint(doc) { return doc.getElementById("perf-gpu-index-space-hint"); }
function hintVisible(doc) {
  const h = hint(doc);
  return !!h && h.hidden === false;
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

test("index_space native: the native-numbering hint is shown ONCE for both rows", async () => {
  const calls = [];
  const { window } = loadApp({
    fetchImpl: makeFetch(calls, { gpus: TWO_NATIVE, indexSpace: "native" }),
  });
  const doc = window.document;
  const selRow = doc.getElementById("perf-gpu-select-row");
  const splitRow = doc.getElementById("perf-gpu-split-row");
  assert.ok(await waitFor(() => selRow.hidden === false), "selector row visible");
  assert.ok(await waitFor(() => splitRow.hidden === false), "split row visible");
  assert.ok(await waitFor(() => hintVisible(doc)), "the shared hint is visible");

  // M1 is exactly this assertion: BOTH rows are on screen, and the sentence
  // still appears once. Before the fix this was 2.
  const all = doc.querySelectorAll(".perf-index-space-hint");
  assert.equal(all.length, 1,
               `the note must exist once, found ${all.length} copies`);
  assert.match(hint(doc).textContent, /Vulkan backend/);

  // ...and it is NOT nested inside either row, which is what made it duplicate.
  assert.equal(selRow.querySelectorAll(".perf-index-space-hint").length, 0,
               "the note must not live inside the Main GPU row");
  assert.equal(splitRow.querySelectorAll(".perf-index-space-hint").length, 0,
               "the note must not live inside the split row");
});

test("no index_space: no native-numbering hint is shown", async () => {
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
  assert.equal(hintVisible(doc), false,
               "no hint on a torch/nvidia-smi-sourced device list");
});

test("a single-GPU box hides the rows, so the shared hint stays hidden too", async () => {
  // The note is shared, so it must not be left stranded under two hidden rows.
  // A single-GPU box is the common case and the one that would show it.
  const calls = [];
  const one = [{ index: 0, name: "AMD Radeon RX 6900 XT", total: 16 * GIB, free: 15 * GIB }];
  const { window } = loadApp({
    fetchImpl: makeFetch(calls, { gpus: one, indexSpace: "native" }),
  });
  const doc = window.document;
  const selRow = doc.getElementById("perf-gpu-select-row");
  assert.ok(await waitFor(() => selRow.hidden === true), "selector row hidden");
  await settle(30);
  assert.equal(hintVisible(doc), false,
               "the shared note must not show when no GPU row is on screen");
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
  assert.ok(await waitFor(() => hintVisible(doc)), "hint present after the native load");
  state.indexSpace = null;
  // Classic-script injection lands the top-level functions on the window
  // (see harness.mjs) - call the refreshers directly for the second pass.
  await window.refreshMainGpuSelector();
  await window.refreshGpuSplitCheckboxes();
  await settle(30);
  assert.equal(hintVisible(doc), false,
               "stale hint hidden once the source is no longer native");
});
