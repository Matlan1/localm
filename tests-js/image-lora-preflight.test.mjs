// SPDX-License-Identifier: AGPL-3.0-or-later
// Gap-1 regression: images.js's img-generate handler must forward the
// selected LoRA into the preflight overrides, mirroring how it already
// forwards model_overrides. Without this, checkModelsBeforeGenerate never
// sees lora_name and can never detect/report a missing LoRA before the real
// generate call fails (see helpers.js's checkModelsBeforeGenerate and the
// /api/media/image/preflight route in routes/models.py).

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));

function setup() {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    calls.push({ url: String(url), opts });
    if (String(url) === "/api/media/image/preflight") {
      return { ok: true, status: 200, json: async () => ({ missing: [] }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  return { window, calls };
}

function selectLora(window, value) {
  const sel = window.document.getElementById("img-lora");
  const opt = window.document.createElement("option");
  opt.value = value;
  opt.textContent = value;
  sel.appendChild(opt);
  sel.value = value;
}

test("a selected LoRA is forwarded into the preflight overrides", async () => {
  const { window, calls } = setup();
  window.document.getElementById("img-prompt").value = "a fox in snow";
  selectLora(window, "my_style.safetensors");

  window.document.getElementById("img-generate").onclick();
  await tick(); await tick();

  const preflight = calls.find((c) => c.url === "/api/media/image/preflight");
  assert.ok(preflight, "preflight was called");
  const body = JSON.parse(preflight.opts.body);
  assert.equal(body.lora_name, "my_style.safetensors");
});

test("no LoRA selected: lora_name is omitted from the preflight overrides", async () => {
  const { window, calls } = setup();
  window.document.getElementById("img-prompt").value = "a fox in snow";
  // img-lora stays at its default "" (None) selection - no option added.

  window.document.getElementById("img-generate").onclick();
  await tick(); await tick();

  const preflight = calls.find((c) => c.url === "/api/media/image/preflight");
  assert.ok(preflight, "preflight was called");
  const body = JSON.parse(preflight.opts.body);
  assert.equal(body.lora_name, undefined, "no LoRA selected -> lora_name is not sent at all");
});
