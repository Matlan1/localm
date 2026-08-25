// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the Settings "Backend" row + dismissable NVIDIA+vulkan hint
// (refreshBackendInfo / shouldShowBackendHint / setupBackendHintDismiss in
// app/settings-perf.js): reads GET /api/backend, shows the hint only for the
// NVIDIA+vulkan combination, and a dismissal persists across a reload except
// in privacy mode, which writes no localStorage.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const GIB = 1024 ** 3;
const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, timeout = 800) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await settle(15); }
  return false;
}

// Records calls; serves /api/backend (configurable payload) plus the bootstrap
// endpoints every loadApp() init pass hits.
function makeFetch(calls, { installed = null, vendor = null, recommended = null } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    calls.push({ u, method, body: opts.body });
    if (u.includes("/api/backend"))
      return { ok: true, status: 200, json: async () => ({ installed, vendor, recommended }) };
    if (u.includes("/api/gpus"))
      return { ok: true, status: 200, json: async () => ({ gpus: [], main_gpu_index: null }) };
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

test("shouldShowBackendHint: true only for NVIDIA + vulkan + not-dismissed", () => {
  const { window } = loadApp({ fetchImpl: makeFetch([]) });
  const cases = [
    [{ vendor: "nvidia", installed: "vulkan" }, false, true],
    [{ vendor: "nvidia", installed: "cuda" }, false, false],    // already on cuda
    [{ vendor: "amd", installed: "vulkan" }, false, false],     // not nvidia
    [{ vendor: "intel", installed: "vulkan" }, false, false],   // not nvidia
    [{ vendor: null, installed: "vulkan" }, false, false],      // no vendor detected
    [{ vendor: "nvidia", installed: null }, false, false],      // nothing installed yet
    [{ vendor: "nvidia", installed: "vulkan" }, true, false],   // dismissed
  ];
  for (const [data, dismissed, expected] of cases) {
    runScript(window, `window.__r = shouldShowBackendHint(${JSON.stringify(data)}, ${dismissed});`);
    assert.equal(window.__r, expected, `data=${JSON.stringify(data)} dismissed=${dismissed}`);
  }
});

test("backend row shows the installed value, and the hint shows for NVIDIA+vulkan", async () => {
  const calls = [];
  const { window } = loadApp({
    fetchImpl: makeFetch(calls, { installed: "vulkan", vendor: "nvidia", recommended: "cuda" }),
  });
  const row = window.document.getElementById("perf-backend-row");
  const value = window.document.getElementById("perf-backend-value");
  const hint = window.document.getElementById("perf-backend-hint");
  assert.ok(await waitFor(() => row.hidden === false), "backend row becomes visible");
  assert.equal(value.textContent, "vulkan");
  assert.ok(await waitFor(() => hint.hidden === false), "hint shows for NVIDIA+vulkan");
});

test("hint stays hidden for AMD+vulkan - a real combination, just not the hinted one", async () => {
  const calls = [];
  const { window } = loadApp({
    fetchImpl: makeFetch(calls, { installed: "vulkan", vendor: "amd", recommended: "vulkan" }),
  });
  const row = window.document.getElementById("perf-backend-row");
  const hint = window.document.getElementById("perf-backend-hint");
  assert.ok(await waitFor(() => row.hidden === false), "backend row still shows");
  await settle(30);
  assert.equal(hint.hidden, true, "no hint outside the NVIDIA+vulkan combination");
});

test("hint stays hidden once an NVIDIA box is already on cuda", async () => {
  const calls = [];
  const { window } = loadApp({
    fetchImpl: makeFetch(calls, { installed: "cuda", vendor: "nvidia", recommended: "cuda" }),
  });
  const row = window.document.getElementById("perf-backend-row");
  const hint = window.document.getElementById("perf-backend-hint");
  assert.ok(await waitFor(() => row.hidden === false));
  await settle(30);
  assert.equal(hint.hidden, true, "already on the recommended backend - nothing to hint");
});

test("dismissing the hint hides it immediately and persists across a reload", async () => {
  const calls = [];
  const { window } = loadApp({
    fetchImpl: makeFetch(calls, { installed: "vulkan", vendor: "nvidia", recommended: "cuda" }),
  });
  const hint = window.document.getElementById("perf-backend-hint");
  const dismiss = window.document.getElementById("perf-backend-hint-dismiss");
  assert.ok(await waitFor(() => hint.hidden === false), "hint shows before dismissal");

  dismiss.click();
  assert.equal(hint.hidden, true, "dismiss hides it immediately");
  assert.equal(window.localStorage.getItem("localm.backendHintDismissed"), "1",
    "dismissal is remembered");

  // A later page load: a fresh window seeded with the same localStorage.
  const { window: reload } = loadApp({
    fetchImpl: makeFetch([], { installed: "vulkan", vendor: "nvidia", recommended: "cuda" }),
    seedLocalStorage: { "localm.backendHintDismissed": "1" },
  });
  const row2 = reload.document.getElementById("perf-backend-row");
  const hint2 = reload.document.getElementById("perf-backend-hint");
  assert.ok(await waitFor(() => row2.hidden === false), "backend row still shows on reload");
  await settle(30);
  assert.equal(hint2.hidden, true, "a dismissal from a prior load suppresses the hint");
});

test("dismissing in privacy mode hides it now but leaves no localStorage trace", async () => {
  const calls = [];
  const { window } = loadApp({
    fetchImpl: makeFetch(calls, { installed: "vulkan", vendor: "nvidia", recommended: "cuda" }),
  });
  const hint = window.document.getElementById("perf-backend-hint");
  const dismiss = window.document.getElementById("perf-backend-hint-dismiss");
  assert.ok(await waitFor(() => hint.hidden === false));

  runScript(window, "chat.privacy = true;");
  dismiss.click();
  assert.equal(hint.hidden, true, "still hides for the rest of this session");
  assert.equal(window.localStorage.getItem("localm.backendHintDismissed"), null,
    "privacy mode leaves no localStorage trace, mirroring dismissInstallGate");
});
