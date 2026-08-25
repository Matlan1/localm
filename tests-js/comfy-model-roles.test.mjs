// SPDX-License-Identifier: AGPL-3.0-or-later
// The Workflow panel's model picker once it consumes what a plugin declares
// through host.register_model_role plus localm's own registry slice
// (localm/plugins/gui/static/pages/workflow.js).
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function ok(j) {
  return { ok: true, status: 200, json: async () => j, text: async () => "" };
}

function makeFetch(payload) {
  return async (url, opts = {}) => {
    const method = opts.method || "GET";
    if (url === "/api/image/workflows" && method === "GET") {
      return ok({ workflows: [], selected: null });
    }
    if (url === "/api/imagine/comfy-models" && method === "GET") return ok(payload);
    return ok({ models: [], active: "", conversations: [], plugins: [] });
  };
}

async function drain() {
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
}

async function render(payload) {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(payload) });
  runScript(win, `refreshWorkflowPanel("image");`);
  await drain();
  return win.document.querySelector(`[data-media="image"]`);
}

const UNET_SLOT = {
  node_id: "1", class_type: "UnetLoaderGGUFAdvanced", input_name: "unet_name",
  current: "flux1-dev-Q8_0.gguf", options: ["flux1-dev-Q8_0.gguf"],
  model_type: "diffusion-unet", role_id: "image-unet",
  role_label: "Diffusion model (UNet)", installed: true,
};
const VAE_SLOT = {
  node_id: "3", class_type: "VAELoader", input_name: "vae_name",
  current: "ae.safetensors", options: ["ae.safetensors"],
  model_type: "vae", role_id: "image-vae", role_label: "VAE", installed: true,
};

function role(id, label, extra = {}) {
  return {
    role_id: id, label, model_type: "vae", required: false, description: "",
    slot: null, current: null, in_workflow: true, installed: true,
    registry_models: [], registry_only: [], ...extra,
  };
}

test("a slot dropdown is labelled with its declared role AND its raw field name", async () => {
  const box = await render({
    reachable: true, api_url: "http://127.0.0.1:8188", slots: [UNET_SLOT],
    loras: [], roles: [role("image-unet", "Diffusion model (UNet)")],
    registry_models: {},
  });
  const label = box.querySelector(".comfy-model-label");
  assert.match(label.textContent, /Diffusion model \(UNet\)/,
    "the role label is shown");
  assert.match(label.textContent, /unet_name/,
    "the raw input_name stays visible - the role pairing is positional and can "
    + "be off by one on a hand-exported graph, so it must never REPLACE the field");
});

test("a slot with no declared role falls back to the raw field name", async () => {
  const bare = { ...VAE_SLOT, role_id: null, role_label: null };
  const box = await render({
    reachable: true, api_url: "u", slots: [bare], loras: [], roles: [],
    registry_models: {},
  });
  assert.equal(box.querySelector(".comfy-model-label").textContent, "vae_name");
});

test("a registered model ComfyUI is not offering is surfaced with what to do", async () => {
  const unsatisfied = { ...VAE_SLOT, current: "nowhere.safetensors", installed: false };
  const box = await render({
    reachable: true, api_url: "u", slots: [unsatisfied], loras: [],
    roles: [role("image-vae", "VAE", {
      installed: false,
      registry_only: [{ name: "my-vae", filename: "my_own_vae.safetensors" }],
    })],
    registry_models: {},
  });
  const hint = box.querySelector(".comfy-model-registry-hint");
  assert.ok(hint, "the registry-only hint rendered");
  assert.match(hint.textContent, /my_own_vae\.safetensors/);
  assert.match(hint.textContent, /models folder/, "it says what to do about it");
});

test("the hint LEADS with the exact file when the workflow's own file is registered", async () => {
  const unsatisfied = { ...VAE_SLOT, current: "ae.safetensors", installed: false, options: [] };
  const box = await render({
    reachable: true, api_url: "u", slots: [unsatisfied], loras: [],
    roles: [role("image-vae", "VAE", {
      installed: false,
      registry_only: [
        { name: "other", filename: "my_own_vae.safetensors" },
        { name: "the-ae", filename: "ae.safetensors" },
      ],
    })],
    registry_models: {},
  });
  const hint = box.querySelector(".comfy-model-registry-hint");
  assert.match(hint.textContent, /ae\.safetensors IS registered/);
  assert.match(hint.textContent, /the-ae/, "names the registry entry to look for");
  assert.ok(!/my_own_vae/.test(hint.textContent),
    "the exact match is the message; the also-rans would only dilute it");
});

test("the registry-only hint also renders on a slot ComfyUI has NOTHING for", async () => {
  // A slot with no options skips the <select> entirely.
  const empty = { ...VAE_SLOT, current: "nowhere.safetensors", options: [], installed: false };
  const box = await render({
    reachable: true, api_url: "u", slots: [empty], loras: [],
    roles: [role("image-vae", "VAE", {
      installed: false,
      registry_only: [{ name: "my-vae", filename: "my_own_vae.safetensors" }],
    })],
    registry_models: {},
  });
  assert.ok(box.querySelector(".comfy-model-missing-value"), "still shown as missing");
  assert.ok(box.querySelector(".comfy-model-registry-hint"),
    "the hint is not lost on the not-installed row");
});

test("a REQUIRED role the active workflow has no slot for is called out", async () => {
  const box = await render({
    reachable: true, api_url: "u", slots: [UNET_SLOT], loras: [],
    roles: [
      role("image-unet", "Diffusion model (UNet)", { required: true }),
      role("image-vae", "VAE", { required: true, in_workflow: false, installed: null }),
    ],
    registry_models: {},
  });
  const notes = [...box.querySelectorAll(".comfy-model-missing")]
    .map((n) => n.textContent).join(" ");
  assert.match(notes, /no slot for: VAE/);
});

test("an OPTIONAL role with no slot is not reported as a problem", async () => {
  const box = await render({
    reachable: true, api_url: "u", slots: [UNET_SLOT], loras: [],
    roles: [
      role("image-unet", "Diffusion model (UNet)", { required: true }),
      role("image-lora", "LoRA", { in_workflow: false, installed: null }),
    ],
    registry_models: {},
  });
  const notes = [...box.querySelectorAll(".comfy-model-missing")]
    .map((n) => n.textContent).join(" ");
  assert.ok(!/LoRA/.test(notes),
    "an optional component the workflow simply does not use is not a warning");
});

test("an unreachable ComfyUI still lists the roles and what is registered", async () => {
  const box = await render({
    reachable: false, api_url: "u", slots: [], loras: [],
    message: "ComfyUI is not running - launch it to see available models.",
    roles: [
      role("image-unet", "Diffusion model (UNet)", {
        in_workflow: null, installed: null,
        registry_models: [{ name: "flux-q8", filename: "flux1-dev-Q8_0.gguf" }],
      }),
      role("image-vae", "VAE", { in_workflow: null, installed: null }),
    ],
    registry_models: {},
  });
  const text = box.textContent;
  assert.match(text, /not running/, "the honest unreachable message stays");
  assert.match(text, /Diffusion model \(UNet\)/, "the workflow's needs are named");
  assert.match(text, /flux-q8/, "a registered model that could fill it is shown");
  assert.match(text, /none registered in localm/,
    "and a role with nothing registered says so, rather than showing blank");
  assert.ok(!box.querySelector(".comfy-model-select"),
    "still no dropdowns - nothing may be picked without ComfyUI");
});

test("an unreachable ComfyUI with no declared roles is unchanged from before", async () => {
  const box = await render({
    reachable: false, api_url: "u", slots: [], loras: [], roles: [],
    message: "ComfyUI is not running - launch it to see available models.",
    registry_models: {},
  });
  assert.match(box.textContent, /not running/);
  assert.ok(!box.querySelector(".comfy-model-picker-head"),
    "no empty 'Models this needs' heading when there is nothing to list");
});

test("a dropdown never displays a file other than the one the workflow will use", async () => {
  // The workflow's current file is not among the options ComfyUI offers.
  const mismatched = {
    ...VAE_SLOT, current: "wan2.2_vae.safetensors",
    options: ["ae.safetensors"], installed: false,
  };
  const box = await render({
    reachable: true, api_url: "u", slots: [mismatched], loras: [],
    roles: [role("image-vae", "VAE", { installed: false })],
    registry_models: {},
  });
  const sel = box.querySelector(".comfy-model-select");
  assert.equal(sel.value, "wan2.2_vae.safetensors",
    "the select reports the value the workflow will actually use");
  const shown = sel.options[sel.selectedIndex];
  assert.match(shown.textContent, /wan2\.2_vae\.safetensors \(not installed\)/);
  assert.equal(shown.disabled, true, "it cannot be re-picked, only replaced");
  assert.ok([...sel.options].some((o) => o.value === "ae.safetensors" && !o.disabled),
    "the live options are still offered");
});

test("a dropdown whose current value IS installed gains no extra option", async () => {
  const box = await render({
    reachable: true, api_url: "u", slots: [VAE_SLOT], loras: [],
    roles: [role("image-vae", "VAE")], registry_models: {},
  });
  const sel = box.querySelector(".comfy-model-select");
  assert.equal(sel.options.length, 1, "no phantom row on the healthy path");
  assert.equal(sel.value, "ae.safetensors");
});
