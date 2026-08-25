// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the Settings "Max resident models" and "Pinned models"
// controls (setupResidencyControls in app/settings-perf.js): both seed from
// GET /v1/config and PATCH their own key back on change, independently of the
// GPU-layers/context sliders and the Apply button.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, timeout = 800) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await settle(15); }
  return false;
}

// Records calls, and serves /v1/config with configurable residency fields plus
// the bootstrap endpoints every loadApp() init pass hits.
function makeFetch(calls, { maxResident = null, pinnedModels = null } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    calls.push({ u, method, body: opts.body });
    if (u.endsWith("/v1/config") && method === "GET")
      return { ok: true, status: 200, json: async () => ({
        n_ctx: 8192, n_gpu_layers: 99,
        max_resident_models: maxResident, pinned_models: pinnedModels }) };
    if (u.endsWith("/v1/config") && method === "PATCH")
      return { ok: true, status: 200, json: async () => ({ ok: true }) };
    if (u.includes("/api/gpus"))
      return { ok: true, status: 200, json: async () => ({ gpus: [], main_gpu_index: null }) };
    if (u.includes("/api/vram-estimate"))
      return { ok: true, status: 200, json: async () => ({
        model: "m", model_bytes: 0, weights: 0, kv_cache: 0, overhead: 0,
        needed: 0, free: 0, total: 0, fits: true, approximate: true }) };
    if (u.includes("/api/models"))
      return { ok: true, status: 200, json: async () => ({ models: [], active: "" }) };
    if (u.includes("/api/plugins"))
      return { ok: true, status: 200, json: async () => ({ plugins: [] }) };
    return { ok: true, status: 200, json: async () => ({}) };
  };
}

test("both fields seed from GET /v1/config", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: makeFetch(calls,
    { maxResident: 2, pinnedModels: ["chat-model", "coder-model"] }) });
  const cap = window.document.getElementById("perf-max-resident");
  const pinned = window.document.getElementById("perf-pinned-models");
  assert.ok(await waitFor(() => cap.value === "2"), "cap seeded from max_resident_models");
  assert.equal(pinned.value, "chat-model, coder-model",
    "pinned field seeded as a comma-joined list");
});

test("no cap / no pins configured leaves both fields blank", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: makeFetch(calls) });
  await waitFor(() => calls.some((c) => c.u.endsWith("/v1/config") && c.method === "GET"));
  await settle(30);
  assert.equal(window.document.getElementById("perf-max-resident").value, "");
  assert.equal(window.document.getElementById("perf-pinned-models").value, "");
});

test("setting a cap PATCHes max_resident_models as a number", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: makeFetch(calls) });
  const cap = window.document.getElementById("perf-max-resident");
  await waitFor(() => calls.some((c) => c.u.endsWith("/v1/config") && c.method === "GET"));
  cap.value = "2";
  cap.dispatchEvent(new window.Event("change", { bubbles: true }));
  assert.ok(await waitFor(() => calls.some(
    (c) => c.u.endsWith("/v1/config") && c.method === "PATCH")), "change issues a PATCH");
  const patch = calls.find((c) => c.u.endsWith("/v1/config") && c.method === "PATCH");
  const body = JSON.parse(patch.body);
  assert.equal(body.max_resident_models, 2, "PATCH carries the cap as a real number");
});

test("clearing the cap field PATCHes max_resident_models to null", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: makeFetch(calls, { maxResident: 2 }) });
  const cap = window.document.getElementById("perf-max-resident");
  assert.ok(await waitFor(() => cap.value === "2"), "cap seeded");
  cap.value = "";
  cap.dispatchEvent(new window.Event("change", { bubbles: true }));
  assert.ok(await waitFor(() => calls.some(
    (c) => c.u.endsWith("/v1/config") && c.method === "PATCH")), "clearing issues a PATCH");
  const patch = calls.filter((c) => c.u.endsWith("/v1/config") && c.method === "PATCH").at(-1);
  assert.equal(JSON.parse(patch.body).max_resident_models, null,
    "a blank field clears the cap rather than leaving it out or coercing to 0");
  // The PATCH is recorded before the handler reaches its toast() call, so the
  // toast text needs its own wait.
  const toastEl = window.document.getElementById("toast");
  assert.ok(await waitFor(() => /Cap cleared/.test(toastEl.textContent)),
    "clearing the cap gets its own confirmation, not the generic 'Saved' text "
    + "shared with setting a real value");
});

test("a cap below 1 is rejected client-side: no PATCH, a toast explains why", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: makeFetch(calls) });
  const cap = window.document.getElementById("perf-max-resident");
  await waitFor(() => calls.some((c) => c.u.endsWith("/v1/config") && c.method === "GET"));
  const before = calls.length;
  cap.value = "0";
  cap.dispatchEvent(new window.Event("change", { bubbles: true }));
  await settle(50);
  assert.equal(calls.length, before, "no PATCH is sent for an out-of-range cap");
  const toastEl = window.document.getElementById("toast");
  assert.match(toastEl.textContent, /whole number of 1 or more/);
});

test("typing pinned model names PATCHes pinned_models as a trimmed array", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: makeFetch(calls) });
  const pinned = window.document.getElementById("perf-pinned-models");
  await waitFor(() => calls.some((c) => c.u.endsWith("/v1/config") && c.method === "GET"));
  pinned.value = " chat-model ,coder-model,, ";
  pinned.dispatchEvent(new window.Event("change", { bubbles: true }));
  assert.ok(await waitFor(() => calls.some(
    (c) => c.u.endsWith("/v1/config") && c.method === "PATCH")), "change issues a PATCH");
  const patch = calls.find((c) => c.u.endsWith("/v1/config") && c.method === "PATCH");
  const body = JSON.parse(patch.body);
  assert.deepEqual(body.pinned_models, ["chat-model", "coder-model"],
    "entries are trimmed and empty ones (blank/trailing comma) dropped");
});

test("clearing the pinned-models field PATCHes pinned_models to null", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: makeFetch(calls, { pinnedModels: ["m1"] }) });
  const pinned = window.document.getElementById("perf-pinned-models");
  assert.ok(await waitFor(() => pinned.value === "m1"), "pinned field seeded");
  pinned.value = "";
  pinned.dispatchEvent(new window.Event("change", { bubbles: true }));
  assert.ok(await waitFor(() => calls.some(
    (c) => c.u.endsWith("/v1/config") && c.method === "PATCH")), "clearing issues a PATCH");
  const patch = calls.filter((c) => c.u.endsWith("/v1/config") && c.method === "PATCH").at(-1);
  assert.equal(JSON.parse(patch.body).pinned_models, null,
    "a blank field clears every pin rather than sending an empty-string list");
  // The PATCH is recorded synchronously inside the fetch mock, before the
  // handler's own `await fetch(...)` resolves and reaches its toast() call, so
  // the toast text needs its own wait.
  const toastEl = window.document.getElementById("toast");
  assert.ok(await waitFor(() => /Pins cleared/.test(toastEl.textContent)),
    "the user is told the pins were cleared");
});
