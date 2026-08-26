// SPDX-License-Identifier: AGPL-3.0-or-later
// A model too big to fully fit VRAM still loads with a partial or zero GPU
// offload, and /api/models/load reports gpu_layers_offloaded /
// gpu_layers_total / degraded. These tests drive the real
// modelSelect.onchange -> switchModel -> toastLoadResult path and check the
// toast wording and class for each payload shape.
import { test } from "node:test";
import assert from "node:assert/strict";

import { loadApp, runScript } from "./harness.mjs";

function stubLoad(window, body) {
  window.fetch = (url) => {
    const u = String(url);
    if (u.includes("/api/models/load")) {
      return Promise.resolve({
        ok: true, status: 200, json: async () => body, text: async () => "",
      });
    }
    return Promise.resolve({
      ok: true, status: 200, body: null,
      json: async () => ({}), text: async () => "",
    });
  };
}

async function selectAndLoad(window, model) {
  const select = window.document.getElementById("model-select");
  const opt = window.document.createElement("option");
  opt.value = model;
  select.appendChild(opt);
  select.value = model;
  runScript(window, "window.__onchange = modelSelect.onchange;");
  await window.__onchange();
}

test("a fully-offloaded load toasts plain success, no GPU-layer wording", async () => {
  const { window } = loadApp();
  stubLoad(window, {
    status: "loaded", model: "small-model",
    gpu_layers_offloaded: 32, gpu_layers_total: 32, degraded: false,
  });

  await selectAndLoad(window, "small-model");

  const toastEl = window.document.getElementById("toast");
  assert.match(toastEl.textContent, /Model switched to small-model/);
  assert.ok(!/layers on GPU/.test(toastEl.textContent),
    "a full offload does not need to mention layer counts");
  assert.ok(!toastEl.className.includes("error"),
    "a full offload is a plain success, not a warning");
});

test("a partial-offload (silent CPU fallback) load warns with the real layer counts", async () => {
  const { window } = loadApp();
  stubLoad(window, {
    status: "loaded", model: "big-model",
    gpu_layers_offloaded: 12, gpu_layers_total: 32, degraded: true,
  });

  await selectAndLoad(window, "big-model");

  const toastEl = window.document.getElementById("toast");
  assert.match(toastEl.textContent, /big-model/);
  assert.match(toastEl.textContent, /12\/32 layers on GPU/,
    "must name the real placement, not just say something degraded");
  assert.match(toastEl.textContent, /CPU/i);
  assert.ok(toastEl.className.includes("error"),
    "a silent CPU fallback must not read as a plain success toast (AGENTS.md rule 5)");
});

test("a zero-offload (fully CPU) load still warns, not just a full offload", async () => {
  const { window } = loadApp();
  stubLoad(window, {
    status: "loaded", model: "huge-model",
    gpu_layers_offloaded: 0, gpu_layers_total: 32, degraded: true,
  });

  await selectAndLoad(window, "huge-model");

  const toastEl = window.document.getElementById("toast");
  assert.match(toastEl.textContent, /0\/32 layers on GPU/);
  assert.ok(toastEl.className.includes("error"));
});

test("a backend that cannot report placement (no gpu_layers_* fields) still toasts plain success", async () => {
  // The gpu_layers_* fields are absent entirely, as from the HF backend.
  const { window } = loadApp();
  stubLoad(window, { status: "loaded", model: "hf-model" });

  await selectAndLoad(window, "hf-model");

  const toastEl = window.document.getElementById("toast");
  assert.match(toastEl.textContent, /Model switched to hf-model/);
  assert.ok(!toastEl.className.includes("error"));
});

test("a superseded load skips the toast entirely, degraded or not", async () => {
  const { window } = loadApp();
  stubLoad(window, {
    status: "superseded", model: "abandoned", by: "newer-model",
    gpu_layers_offloaded: 4, gpu_layers_total: 32, degraded: true,
  });

  await selectAndLoad(window, "abandoned");

  const toastEl = window.document.getElementById("toast");
  assert.equal(toastEl.textContent, "",
    "an abandoned load must not toast anything, even a degraded one");
});
