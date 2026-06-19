// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the status-bar hardware monitor (renderHwStats / pollHwStats
// in localm/plugins/gui/static/app.js). renderHwStats renders whatever
// /api/stats reports; absent sections must simply not appear, and an all-empty
// payload must hide the readout entirely.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const GIB = 1024 ** 3;
const STATS = {
  cpu: { percent: 12.4 },
  ram: { used: 4 * GIB, total: 8 * GIB, percent: 50.0 },
  vram: { used: 2 * GIB, total: 16 * GIB, percent: 12.5 },
  gpu: { percent: 30.0 },
};

const settle = () => new Promise((r) => setTimeout(r, 0));

// Well-formed responses for the endpoints app.js's bootstrap touches on load, so
// the unrelated init code (refreshModels -> populateSetupModels, plugins) never
// throws while we exercise the hardware monitor. `calls` records URLs hit.
function goodFetch(calls = []) {
  return async (url) => {
    const u = String(url);
    calls.push(u);
    if (u.endsWith("/api/stats"))
      return { ok: true, status: 200, json: async () => STATS };
    if (u.includes("/api/models"))
      return { ok: true, status: 200, json: async () => ({ models: [], active: "" }) };
    if (u.includes("/api/plugins"))
      return { ok: true, status: 200, json: async () => ({ plugins: [] }) };
    return { ok: true, status: 200, json: async () => ({}) };
  };
}

test("renderHwStats writes a compact CPU/RAM/VRAM/GPU line and unhides", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  assert.equal(typeof window.renderHwStats, "function", "renderHwStats on window");
  const el = window.document.getElementById("hw-stats");
  assert.ok(el, "#hw-stats element exists in index.html");
  window.renderHwStats(STATS);
  assert.equal(el.hidden, false, "shown when there is data");
  assert.match(el.textContent, /CPU 12%/);
  assert.match(el.textContent, /RAM 50%/);
  assert.match(el.textContent, /VRAM 2\.0\/16\.0 GB/);
  assert.match(el.textContent, /GPU 30%/);
});

test("renderHwStats hides the readout when nothing is measurable", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const el = window.document.getElementById("hw-stats");
  window.renderHwStats(STATS);
  assert.equal(el.hidden, false);
  window.renderHwStats({});              // negative: empty payload
  assert.equal(el.hidden, true, "empty stats hide the readout");
});

test("renderHwStats shows VRAM total-only when free is unknown", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const el = window.document.getElementById("hw-stats");
  window.renderHwStats({ vram: { total: 8 * GIB } });
  assert.match(el.textContent, /VRAM 8\.0 GB/);
  assert.doesNotMatch(el.textContent, /\//, "no used/total slash when free unknown");
});

test("pollHwStats fetches /api/stats and renders the result", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: goodFetch(calls) });
  await window.pollHwStats();
  await settle();
  assert.ok(calls.some((u) => u.endsWith("/api/stats")), "polls GET /api/stats");
  assert.match(window.document.getElementById("hw-stats").textContent, /CPU 12%/);
});
