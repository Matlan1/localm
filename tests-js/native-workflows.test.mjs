// SPDX-License-Identifier: AGPL-3.0-or-later
// The native workflow card (app/native-workflows.js) on the Images, Video and
// Music pages: listing, apply (picker + fields + download offer for missing
// models), save current, rename/delete guards, export and import.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const LIST = [
  { id: "builtin:sd-turbo", name: "SD-Turbo (fast, 512 px)", description: "Quick.", builtin: true },
  { id: "builtin:flux1-dev-q4", name: "FLUX.1 dev (Q4_K_S)", description: "Dev.", builtin: true },
  { id: "mine", name: "Mine", description: "", builtin: false },
];

const RESOLVED = {
  id: "builtin:sd-turbo", name: "SD-Turbo (fast, 512 px)",
  native_models: { model: "sd-turbo" }, loras: [{ name: "aidma-flux", strength: 0.6 }],
  params: { steps: 2, cfg: 1, width: 512, height: 512, sample_method: "euler_a",
            negative_prompt: "blurry" },
  missing: [], notes: [],
};

const MISSING = {
  slot: "model", name: "sd-turbo", filename: "sd_turbo-f16-q8_0.gguf", native: true,
  searchable: false,
  source: { repo: "Green-Sky/SD-Turbo-GGUF", file: "sd_turbo-f16-q8_0.gguf",
            spec: "Green-Sky/SD-Turbo-GGUF:sd_turbo-f16-q8_0.gguf", sha256: "d".repeat(64),
            size_bytes: 2023745376, model_type: "diffusion-unet", name: "sd-turbo", origin: "curated" },
};

const NM_OPTIONS = {
  models: [{ name: "sd-turbo", file: "sd_turbo-f16-q8_0.gguf", kind: "checkpoint" }],
  slots: { clip_l: [], clip_g: [], t5xxl: [], llm: [], vae: [] },
  loras: [{ name: "aidma-flux", file: "aidma.safetensors", kind: "lora" }],
  current: {}, preview: null,
  sampling: { sample_methods: ["euler", "euler_a"], schedulers: ["discrete", "karras"], hires_upscalers: [] },
};

function makeFetch(calls, opts) {
  const resolves = [...(opts.resolves || [RESOLVED])];
  return async (url, init = {}) => {
    const u = String(url);
    const method = init.method || "GET";
    const body = init.body ? JSON.parse(init.body) : undefined;
    calls.push({ url: u, method, body });
    const ok = (data) => ({ ok: true, status: 200, json: async () => data, text: async () => "" });
    if (u.includes("/native-workflows")) {
      if (u.endsWith("/resolve")) return ok(resolves.length > 1 ? resolves.shift() : resolves[0]);
      if (u.endsWith("/import")) {
        return ok({ id: "imported", workflows: [...LIST, { id: "imported", name: "graph", description: "",
          builtin: false }], report: opts.report || { kind: "native", mapped: [], implicit: [],
          unsupported: [], notes: [] } });
      }
      if (u.endsWith("/rename")) return ok({ id: "mine", workflows: LIST });
      if (method === "DELETE") return ok({ workflows: LIST.slice(0, 2) });
      if (method === "POST") return ok({ id: "saved", workflows: [...LIST, { id: "saved", name: body.workflow.name,
        description: "", builtin: false }] });
      if (/native-workflows\/[^/]+$/.test(u)) {
        return ok({ id: "builtin:sd-turbo", builtin: true,
          workflow: { format: "localm-native-workflow", version: 1, name: "SD-Turbo" } });
      }
      return ok({ workflows: LIST });
    }
    if (u.includes("/preflight")) return ok({ status: "verified", missing: opts.preflightMissing || [], warning: null });
    if (u.includes("/native-models")) {
      if (u.includes("/api/music/")) {
        return ok({ components: { dit: [{ name: "acestep-dit", file: "dit.gguf" }], text_encoder: [], vae: [], lm: [] },
          current: {}, unsupported: [] });
      }
      return ok(NM_OPTIONS);
    }
    if (u.includes("/backend")) {
      return ok({ choice: "auto", active: opts.active || "native",
        native: { model: null, runtime: "vulkan", recommended: { name: "sd-turbo", width: 832, height: 480 },
                  models: {}, missing: [] } });
    }
    if (u.endsWith("/api/video") && method === "POST") {
      return { ok: false, status: 400, json: async () => ({ detail: "stop here" }) };
    }
    return ok({ models: [], active: "", conversations: [], plugins: [] });
  };
}

async function settle(win, code) {
  runScript(win, `globalThis.__p = (async () => { ${code} })();`);
  await win.__p;
}

function app(opts = {}) {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, opts) });
  return { win, calls, $: (id) => win.document.getElementById(id) };
}

const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms));

function button(win, mountId, action) {
  return win.document.querySelector(`#${mountId} button[data-action="${action}"]`);
}

test("native image backend shows the native workflow card and hides the ComfyUI one", async () => {
  const { win, $ } = app();
  await settle(win, "await refreshImageBackend(); await imageWorkflowCard.refresh();");
  assert.equal($("img-native-workflow-card").hidden, false);
  assert.equal($("img-workflow-card").hidden, true);
  const groups = [...win.document.querySelectorAll("#img-native-workflows optgroup")].map((g) => g.label);
  assert.deepEqual(groups, ["Built in", "Saved"]);
  assert.equal(win.document.querySelector("#img-native-workflows select").value, "builtin:sd-turbo");
  assert.equal(button(win, "img-native-workflows", "delete").disabled, true);
  assert.equal(button(win, "img-native-workflows", "rename").disabled, true);
  assert.equal(win.document.querySelector("#img-native-workflows .native-workflow-desc").textContent, "Quick.");
});

test("ComfyUI backend keeps the native card hidden", async () => {
  const { win, $ } = app({ active: "comfy" });
  await settle(win, "await refreshImageBackend();");
  assert.equal($("img-native-workflow-card").hidden, true);
  assert.equal($("img-workflow-card").hidden, false);
});

test("apply fills the model picker and the page fields", async () => {
  const { win, $, calls } = app();
  await settle(win, "await refreshImageBackend(); await imageWorkflowCard.refresh();");
  $("img-guidance").value = "7";
  await settle(win, "await imageWorkflowCard.apply('builtin:sd-turbo');");
  assert.equal(win.eval("imageModelPicker.state.models.model"), "sd-turbo");
  assert.deepEqual(JSON.parse(win.eval("JSON.stringify(imageModelPicker.state.loras)")),
    [{ name: "aidma-flux", strength: 0.6 }]);
  assert.equal($("img-steps").value, "2");
  assert.equal($("img-cfg").value, "1");
  assert.equal($("img-guidance").value, "", "a setting the workflow does not set is cleared");
  assert.equal($("img-size").value, "512x512");
  assert.equal($("img-negative").value, "blurry");
  assert.equal($("img-sampler").value, "euler_a");
  assert.ok(!calls.some((c) => c.url.includes("/preflight")), "nothing missing: no download offer");
  assert.match(win.document.querySelector("#img-native-workflows .native-workflow-status").textContent,
    /Applied: SD-Turbo/);
});

test("apply offers missing models through the preflight, and again before generating", async () => {
  const stillMissing = { ...RESOLVED, native_models: {}, missing: [MISSING] };
  const { win, $, calls } = app({
    resolves: [stillMissing, stillMissing, stillMissing, RESOLVED],
    preflightMissing: [MISSING],
  });
  await settle(win, "await refreshImageBackend(); await imageWorkflowCard.refresh();");
  runScript(win, "globalThis.__a = imageWorkflowCard.apply('builtin:sd-turbo');");
  for (let i = 0; i < 50 && $("modal").style.display !== "flex"; i++) await tick(5);
  assert.equal($("modal").style.display, "flex", "the download offer opens");
  assert.match($("modal-body").textContent, /Green-Sky\/SD-Turbo-GGUF/);
  [...$("modal-body").querySelectorAll("button")].find((b) => b.textContent === "Not now").click();
  await win.__a;
  const pre = calls.filter((c) => c.url.includes("/api/media/image/preflight"));
  assert.deepEqual(pre[0].body, { native_workflow: "builtin:sd-turbo" });
  assert.match(win.document.querySelector("#img-native-workflows .native-workflow-missing").textContent,
    /sd_turbo-f16-q8_0\.gguf/);
  assert.ok(win.eval("imageWorkflowCard.pending"));

  runScript(win, "globalThis.__b = imageWorkflowCard.prepare();");
  for (let i = 0; i < 50 && $("modal").style.display !== "flex"; i++) await tick(5);
  [...$("modal-body").querySelectorAll("button")].find((b) => b.textContent === "Not now").click();
  assert.equal(await win.__b, true);
  assert.equal(win.eval("imageModelPicker.state.models.model"), "sd-turbo",
    "a model registered by now goes into its slot");
  assert.equal(win.eval("imageWorkflowCard.pending"), null);
});

test("save current sends the picker's models and the page settings", async () => {
  const { win, $, calls } = app();
  await settle(win, "await refreshImageBackend(); await imageWorkflowCard.refresh();");
  await settle(win, "await imageModelPicker.setModels({ model: 'sd-turbo' });");
  $("img-steps").value = "6";
  $("img-size").value = "640x384";
  button(win, "img-native-workflows", "save").click();
  for (let i = 0; i < 50 && !$("modal-body").querySelector("input"); i++) await tick(5);
  const input = $("modal-body").querySelector("input");
  input.value = "My preset";
  input.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Enter" }));
  for (let i = 0; i < 50 && !calls.some((c) => c.method === "POST" && c.url.endsWith("/native-workflows")); i++) {
    await tick(5);
  }
  const save = calls.find((c) => c.method === "POST" && c.url.endsWith("/api/image/native-workflows"));
  assert.equal(save.body.workflow.format, "localm-native-workflow");
  assert.equal(save.body.workflow.name, "My preset");
  assert.deepEqual(save.body.workflow.models, { model: { name: "sd-turbo" } });
  assert.equal(save.body.workflow.params.steps, 6);
  assert.equal(save.body.workflow.params.width, 640);
  assert.equal(save.body.workflow.params.height, 384);
});

test("import sends the parsed file and shows the ComfyUI report", async () => {
  const report = { kind: "comfy-api", mapped: [{ node: "3", class_type: "KSampler", to: "steps" }],
    implicit: [], unsupported: [{ node: "40", class_type: "UpscaleModelLoader", title: "" }],
    notes: ["The seed was not kept."] };
  const { win, $, calls } = app({ report });
  await settle(win, "await refreshImageBackend(); await imageWorkflowCard.refresh();");
  const input = win.document.querySelector("#img-native-workflows .native-workflow-file");
  const graph = { 3: { class_type: "KSampler", inputs: { steps: 9 } } };
  Object.defineProperty(input, "files", { configurable: true,
    value: [{ name: "fox graph.json", text: async () => JSON.stringify(graph) }] });
  input.dispatchEvent(new win.Event("change"));
  for (let i = 0; i < 50 && !win.document.querySelector("#img-native-workflows .native-workflow-report"); i++) {
    await tick(5);
  }
  const imp = calls.find((c) => c.url.endsWith("/api/image/native-workflows/import"));
  assert.deepEqual(imp.body, { data: graph, name: "fox graph" });
  const rep = win.document.querySelector("#img-native-workflows .native-workflow-report");
  assert.equal(rep.open, true);
  assert.match(rep.querySelector(".native-workflow-unsupported").textContent, /UpscaleModelLoader #40/);
  assert.match(rep.querySelector(".native-workflow-notes").textContent, /seed/);
  assert.equal(win.document.querySelector("#img-native-workflows select").value, "imported");
});

test("import of a file that is not JSON is refused before any request", async () => {
  const { win, calls } = app();
  await settle(win, "await refreshImageBackend(); await imageWorkflowCard.refresh();");
  const input = win.document.querySelector("#img-native-workflows .native-workflow-file");
  Object.defineProperty(input, "files", { configurable: true,
    value: [{ name: "x.json", text: async () => "{nope" }] });
  input.dispatchEvent(new win.Event("change"));
  await tick(20);
  assert.ok(!calls.some((c) => c.url.endsWith("/import")));
});

test("export downloads the workflow JSON", async () => {
  const { win } = app();
  await settle(win, "await refreshImageBackend(); await imageWorkflowCard.refresh();");
  const saved = [];
  win.URL.createObjectURL = () => "blob:x";
  win.URL.revokeObjectURL = () => {};
  win.HTMLAnchorElement.prototype.click = function () { saved.push(this.download); };
  button(win, "img-native-workflows", "export").click();
  for (let i = 0; i < 50 && !saved.length; i++) await tick(5);
  assert.deepEqual(saved, ["SD-Turbo-fast-512-px.image.json"]);
});

test("a saved workflow can be renamed and deleted", async () => {
  const { win, $, calls } = app();
  await settle(win, "await refreshImageBackend(); await imageWorkflowCard.refresh();");
  const sel = win.document.querySelector("#img-native-workflows select");
  sel.value = "mine";
  sel.dispatchEvent(new win.Event("change"));
  assert.equal(button(win, "img-native-workflows", "delete").disabled, false);
  button(win, "img-native-workflows", "delete").click();
  for (let i = 0; i < 50 && $("modal").style.display !== "flex"; i++) await tick(5);
  [...$("modal-body").querySelectorAll("button")].find((b) => b.textContent === "Delete").click();
  for (let i = 0; i < 50 && !calls.some((c) => c.method === "DELETE"); i++) await tick(5);
  assert.ok(calls.some((c) => c.method === "DELETE" && c.url.endsWith("/api/image/native-workflows/mine")));
});

test("video: the native card replaces the ComfyUI card and the flow shift is sent", async () => {
  const { win, $, calls } = app();
  await settle(win, "await refreshVideoBackend();");
  assert.equal($("video-native-workflow-card").hidden, false);
  assert.equal($("video-workflow-card").hidden, true);
  assert.equal($("video-flow-shift").closest(".video-native-only").hidden, false);
  await settle(win, "await videoWorkflowCard.apply('builtin:sd-turbo');");
  $("video-prompt").value = "a cat";
  $("video-flow-shift").value = "5";
  $("video-generate").click();
  for (let i = 0; i < 100 && !calls.some((c) => c.url.endsWith("/api/video")); i++) await tick(5);
  const gen = calls.find((c) => c.url.endsWith("/api/video"));
  assert.equal(gen.body.flow_shift, 5);
  assert.equal(gen.body.native_models.model, "sd-turbo");
});

test("music: apply sets the ACE-Step parts and the shift field", async () => {
  const resolved = { ...RESOLVED, native_models: { dit: "acestep-dit" }, loras: [],
    params: { steps: 8, shift: 3, duration_seconds: 60, tags: "lofi" } };
  const { win, $ } = app({ resolves: [resolved] });
  await settle(win, "await refreshMusicBackend();");
  assert.equal($("music-native-workflow-card").hidden, false);
  assert.equal($("music-shift").closest(".music-native-only").hidden, false);
  await settle(win, "await musicWorkflowCard.apply('builtin:ace');");
  assert.equal(win.eval("musicModelPicker.state.dit"), "acestep-dit");
  assert.equal($("music-steps").value, "8");
  assert.equal($("music-shift").value, "3");
  assert.equal($("music-duration").value, "60");
  assert.equal($("music-tags").value, "lofi");
  assert.deepEqual(JSON.parse(win.eval("JSON.stringify(musicModelPicker.choice())")),
    { native_models: { dit: "acestep-dit" } });
});
