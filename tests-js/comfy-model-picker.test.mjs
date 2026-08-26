// SPDX-License-Identifier: AGPL-3.0-or-later
// The Workflow panel's "Launch ComfyUI" button + per-slot model dropdowns
// (localm/plugins/gui/static/pages/workflow.js), and their wiring into the
// Images page's Generate request body (images.js).
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function ok(j) {
  return { ok: true, status: 200, json: async () => j, text: async () => "" };
}
function bad(status = 500) {
  return { ok: false, status, json: async () => ({}), text: async () => "" };
}

// A stateful stub covering the image plugin's workflow-management routes
// (/api/image/*) and generation/ComfyUI routes (/api/imagine/*).
function makeFetch(calls, { reachable = false, slots = [], workflows = [], selected = null } = {}) {
  return async (url, opts = {}) => {
    const method = opts.method || "GET";
    if (url === "/api/image/workflows" && method === "GET") {
      return ok({ workflows, selected });
    }
    if (url === "/api/image/workflows/select" && method === "POST") {
      const b = JSON.parse(opts.body);
      calls.push(["select", b]);
      selected = b.name;
      workflows = workflows.map((w) => ({ ...w, is_active: w.name === selected }));
      return ok({ selected, workflows });
    }
    if (url === "/api/imagine/comfy-models" && method === "GET") {
      calls.push(["comfy-models"]);
      return ok(reachable
        ? { reachable: true, api_url: "http://127.0.0.1:8188", slots }
        : { reachable: false, api_url: "http://127.0.0.1:8188", slots: [],
            message: "ComfyUI is not running - launch it to see available models." });
    }
    if (url === "/api/imagine/comfy-launch" && method === "POST") {
      calls.push(["comfy-launch"]);
      return ok({ ok: true, message: "ComfyUI is running.", api_url: "http://127.0.0.1:8188" });
    }
    if (url === "/api/imagine" && method === "POST") {
      calls.push(["generate", JSON.parse(opts.body)]);
      return ok({ job_id: "j1" });
    }
    if (url === "/api/jobs/j1/events") {
      // Streaming is not exercised here; fail fast instead of modelling SSE.
      return bad(500);
    }
    return ok({ models: [], active: "", conversations: [], plugins: [] });
  };
}

async function drain() {
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
}

async function render(win, media = "image") {
  runScript(win, `refreshWorkflowPanel(${JSON.stringify(media)});`);
  await drain();
  return win.document.querySelector(`[data-media="${media}"]`);
}

test("Launch ComfyUI button POSTs comfy-launch and opens the returned api_url", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  win.open = (url) => calls.push(["open", url]);
  const box = await render(win);
  const btn = [...box.querySelectorAll("button")].find((b) => b.textContent === "Launch ComfyUI");
  assert.ok(btn, "Launch ComfyUI button rendered");
  assert.equal(btn.type, "button");
  btn.onclick();
  await drain();
  assert.ok(calls.find((c) => c[0] === "comfy-launch"), "comfy-launch POSTed");
  assert.ok(calls.find((c) => c[0] === "open" && c[1] === "http://127.0.0.1:8188"),
    "the running ComfyUI is opened in a new tab");
});

test("Launch ComfyUI surfaces a failure message instead of a silent no-op", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    if (url === "/api/imagine/comfy-launch") {
      return ok({ ok: false, message: "ComfyUI failed to start.", api_url: "http://127.0.0.1:8188" });
    }
    return makeFetch(calls)(url, opts);
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  win.open = () => calls.push(["open"]);
  const box = await render(win);
  const btn = [...box.querySelectorAll("button")].find((b) => b.textContent === "Launch ComfyUI");
  btn.onclick();
  await drain();
  assert.ok(!calls.some((c) => c[0] === "open"), "a failed launch does not open a tab");
});

test("model picker shows an honest message when ComfyUI is unreachable", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([], { reachable: false }) });
  const box = await render(win);
  assert.ok(!box.querySelector(".comfy-model-select"), "no dropdowns when unreachable");
  assert.ok([...box.querySelectorAll(".sub")].some((d) => /not running/.test(d.textContent)),
    "unreachability message shown, not a silently-empty picker");
});

test("model picker renders one dropdown per slot, defaulting to the workflow's current value", async () => {
  const slots = [
    { node_id: "1", class_type: "UnetLoaderGGUFAdvanced", input_name: "unet_name",
      current: "a.gguf", options: ["a.gguf", "b.gguf"] },
    { node_id: "3", class_type: "VAELoader", input_name: "vae_name",
      current: "ae.safetensors", options: ["ae.safetensors"] },
  ];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([], { reachable: true, slots }) });
  const box = await render(win);
  const selects = box.querySelectorAll(".comfy-model-select");
  assert.equal(selects.length, 2, "one dropdown per slot");
  assert.equal(selects[0].value, "a.gguf");
  assert.equal(selects[0].querySelectorAll("option").length, 2);
  assert.equal(selects[1].value, "ae.safetensors");
});

test("model picker surfaces a slot ComfyUI has zero live options for, instead of hiding it", async () => {
  // ComfyUI is reachable, and one slot has a real `current` filename but an
  // empty live `options` array.
  const slots = [
    { node_id: "1", class_type: "UnetLoaderGGUFAdvanced", input_name: "unet_name",
      current: "flux1-dev-Q8_0.gguf", options: [] },
    { node_id: "3", class_type: "VAELoader", input_name: "vae_name",
      current: "ae.safetensors", options: ["ae.safetensors"] },
  ];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([], { reachable: true, slots }) });
  const box = await render(win);
  const selects = box.querySelectorAll(".comfy-model-select");
  assert.equal(selects.length, 1, "only the slot WITH live options gets a dropdown");
  const missingValue = box.querySelector(".comfy-model-missing-value");
  assert.ok(missingValue, "the missing slot's current value is shown instead");
  assert.match(missingValue.textContent, /flux1-dev-Q8_0\.gguf/);
  assert.match(missingValue.textContent, /not installed/);
  const summary = [...box.querySelectorAll(".comfy-model-missing")]
    .find((d) => /not found in ComfyUI/.test(d.textContent));
  assert.ok(summary, "a summary line calls out the missing model file(s)");
  assert.match(summary.textContent, /^1 required model file /);
});

test("changing a dropdown records the override and it survives a plain panel refresh", async () => {
  const slots = [
    { node_id: "1", class_type: "UnetLoaderGGUFAdvanced", input_name: "unet_name",
      current: "a.gguf", options: ["a.gguf", "b.gguf"] },
  ];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([], { reachable: true, slots }) });
  let box = await render(win);
  let sel = box.querySelector(".comfy-model-select");
  sel.value = "b.gguf";
  sel.onchange();

  // modelOverrides lives in jsdom's own realm, so compare via JSON: strict
  // deepEqual rejects cross-realm objects on constructor identity.
  runScript(win, "window.__mo = modelOverrides;");
  assert.equal(JSON.stringify(win.__mo.image), JSON.stringify({ "1": { unet_name: "b.gguf" } }));

  // Re-entering the tab re-renders the panel.
  box = await render(win);
  sel = box.querySelector(".comfy-model-select");
  assert.equal(sel.value, "b.gguf", "the override survives a plain refresh");
});

test("selecting a different workflow clears the stored overrides", async () => {
  const slots = [
    { node_id: "1", class_type: "UnetLoaderGGUFAdvanced", input_name: "unet_name",
      current: "a.gguf", options: ["a.gguf", "b.gguf"] },
  ];
  const workflows = [{ name: "other.json", is_active: false, size_bytes: 5 }];
  const { window: win } = loadAppWithPages(
    { fetchImpl: makeFetch([], { reachable: true, slots, workflows }) });
  let box = await render(win);
  const sel = box.querySelector(".comfy-model-select");
  sel.value = "b.gguf";
  sel.onchange();
  runScript(win, "window.__mo1 = modelOverrides.image;");
  assert.equal(JSON.stringify(win.__mo1), JSON.stringify({ "1": { unet_name: "b.gguf" } }));

  const pick = [...box.querySelectorAll(".workflow-pick")]
    .find((p) => /other\.json/.test(p.textContent));
  assert.ok(pick, "the uploaded workflow row is selectable");
  pick.onclick();
  await drain();

  runScript(win, "window.__mo2 = modelOverrides.image;");
  assert.equal(JSON.stringify(win.__mo2), "{}",
    "switching the active workflow drops stale node-id overrides");
});

test("Images Generate includes model_overrides in the POST body when a pick was made", async () => {
  const calls = [];
  const slots = [
    { node_id: "1", class_type: "UnetLoaderGGUFAdvanced", input_name: "unet_name",
      current: "a.gguf", options: ["a.gguf", "b.gguf"] },
  ];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { reachable: true, slots }) });
  const box = await render(win);
  const sel = box.querySelector(".comfy-model-select");
  sel.value = "b.gguf";
  sel.onchange();

  win.document.querySelector("#img-prompt").value = "a fox in snow";
  await win.document.querySelector("#img-generate").onclick();
  await drain();

  const gen = calls.find((c) => c[0] === "generate");
  assert.ok(gen, "the generate request was sent");
  assert.deepEqual(gen[1].model_overrides, { "1": { unet_name: "b.gguf" } });
});

test("Images Generate omits model_overrides when nothing was picked", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  await render(win);
  win.document.querySelector("#img-prompt").value = "a fox in snow";
  await win.document.querySelector("#img-generate").onclick();
  await drain();
  const gen = calls.find((c) => c[0] === "generate");
  assert.ok(gen, "the generate request was sent");
  assert.ok(!("model_overrides" in gen[1]), "no override key sent when the picker made no changes");
});
