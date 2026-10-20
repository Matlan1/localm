// SPDX-License-Identifier: AGPL-3.0-or-later
// The native image extras on the Images page (app/image-inputs.js): the mask,
// reference image, ControlNet and embedding controls, what each sends in the
// /api/imagine body, what stops a job before it is sent, and "reuse settings".
// The canvas mask editor itself needs a real browser and is not driven here.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));
const PNG = "data:image/png;base64,iVBORw0KGgo=";

function options(capabilities) {
  return {
    models: [{ name: "sd15", file: "sd15.safetensors", kind: "checkpoint", family: "sd1",
               family_label: "Stable Diffusion 1.x" }],
    slots: { clip_l: [], clip_g: [], t5xxl: [], llm: [], vae: [] },
    loras: [],
    controlnets: [
      { name: "canny15", file: "control_v11p_sd15_canny.safetensors", kind: "controlnet",
        family: "sd1", fits: true },
      { name: "canny-xl", file: "canny_xl.safetensors", kind: "controlnet", family: "sdxl",
        fits: false }],
    embeddings: [
      { name: "EasyNeg", file: "EasyNegative.safetensors", kind: "textual_inversion",
        family: "sd1", trigger: "EasyNegative", fits: true },
      { name: "xl-emb", file: "xl_emb.safetensors", kind: "textual_inversion",
        family: "sdxl", trigger: "xl_emb", fits: false }],
    current: { model: "sd15", clip_l: "", clip_g: "", t5xxl: "", llm: "", vae: "" },
    sampling: { sample_methods: [], schedulers: [], hires_upscalers: [] },
    upscalers: [],
    preview: { model: "sd15", family: "sd1", family_label: "Stable Diffusion 1.x",
               kind: "checkpoint", needs: [], components: {}, missing: [], notes: [],
               defaults: null, error: null, capabilities },
  };
}

const SD1_CAPS = { references: false, controlnet: true, embeddings: true, inpaint: false,
                   edit: false };
const FLUX_CAPS = { references: true, controlnet: false, embeddings: false, inpaint: false,
                    edit: false };

function setup(capabilities = SD1_CAPS) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    if (u.startsWith("/api/imagine/native-models")) {
      return { ok: true, status: 200, json: async () => options(capabilities) };
    }
    if (u === "/api/imagine/backend") {
      return { ok: true, status: 200, json: async () => ({
        choice: "auto", active: "native", native: { model: "sd15", runtime: "rocm", recommended: {} } }) };
    }
    if (u === "/api/imagine") return { ok: true, status: 200, json: async () => ({ job_id: "j1" }) };
    if (u === "/api/jobs/j1/events") {
      return { ok: true, status: 200, body: { getReader() { return { async read() {
        return { done: true, value: undefined }; } }; } } };
    }
    return { ok: true, status: 200, json: async () => ({ models: [], images: [] }), text: async () => "" };
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  return { win, calls, $: (id) => win.document.getElementById(id) };
}

async function ready(capabilities) {
  const s = setup(capabilities);
  runScript(s.win, "globalThis.__p = refreshImageBackend();");
  await s.win.__p;
  for (let i = 0; i < 6; i++) await tick();
  return s;
}

async function generate({ win, $ }) {
  $("img-prompt").value = $("img-prompt").value || "a fox";
  $("img-generate").onclick();
  for (let i = 0; i < 5; i++) await tick();
}

const imagineBody = (calls) => {
  const call = calls.find((c) => c.url === "/api/imagine");
  return call ? JSON.parse(call.opts.body) : null;
};

const buttons = (root, text) => [...root.querySelectorAll("button")].filter((b) => b.textContent === text);

test("the extras show for the native backend with ControlNets and embeddings listed", async () => {
  const { $ } = await ready();
  const mount = $("img-native-inputs");
  assert.equal(mount.hidden, false);
  const sel = $("img-control-model");
  assert.deepEqual([...sel.options].map((o) => o.value), ["", "canny15", "canny-xl"]);
  assert.match(sel.options[2].textContent, /other family/);
  const chips = [...mount.querySelectorAll(".iin-chip")].map((c) => c.textContent);
  assert.ok(chips[0].startsWith("EasyNegative"));
  assert.match(chips[1], /other family/);
  assert.equal(buttons(mount, "+ prompt").length, 1, "an embedding of another family has no add buttons");
});

test("untouched extras send nothing", async () => {
  const s = await ready();
  await generate(s);
  const body = imagineBody(s.calls);
  for (const key of ["mask", "input_image_data", "ref_images", "control", "embeddings"]) {
    assert.ok(!(key in body), `${key} must not be sent when untouched`);
  }
});

test("an embedding is sent when its trigger word is in the prompt or negative prompt", async () => {
  const s = await ready();
  const add = buttons(s.$("img-native-inputs"), "+ negative")[0];
  s.$("img-negative").value = "blurry";
  add.onclick();
  add.onclick();
  assert.equal(s.$("img-negative").value, "blurry, EasyNegative", "the trigger is added once");
  await generate(s);
  assert.deepEqual(imagineBody(s.calls).embeddings, ["EasyNeg"]);
});

test("an embedding name inside a longer word is not a use of it", async () => {
  const s = await ready();
  s.$("img-prompt").value = "EasyNegativeV2 style, xl_emb";
  await generate(s);
  assert.ok(!("embeddings" in imagineBody(s.calls)),
    "neither a longer word nor another family's trigger sends an embedding");
});

test("a ControlNet without a control image stops the job with a reason", async () => {
  const s = await ready();
  const sel = s.$("img-control-model");
  sel.value = "canny15";
  sel.onchange();
  await generate(s);
  assert.equal(imagineBody(s.calls), null, "nothing is sent");
  assert.match(s.win.document.body.textContent, /Choose a control image/);
});

test("a ControlNet with its image, strength and canny is sent", async () => {
  const s = await ready();
  const sel = s.$("img-control-model");
  sel.value = "canny15";
  sel.onchange();
  runScript(s.win, `imageInputs.state.control.image = "${PNG}"; imageInputs.render();`);
  s.$("img-control-strength").value = "0.6";
  s.$("img-control-strength").oninput();
  s.$("img-control-preprocess").value = "canny";
  s.$("img-control-preprocess").onchange();
  await generate(s);
  assert.deepEqual(imagineBody(s.calls).control,
    { model: "canny15", image: PNG, preprocess: "canny", strength: 0.6 });
});

test("a painted mask sends its base image as data instead of the input path", async () => {
  const s = await ready();
  s.$("img-input").value = "20260101_000000_abc.png";
  runScript(s.win, `imageInputs.state.base = "${PNG}"; imageInputs.state.mask = "${PNG}";
                    imageInputs.state.size = { w: 64, h: 48 }; imageInputs.render();`);
  assert.match(s.$("img-native-inputs").textContent, /Mask set on a 64x48 image/);
  await generate(s);
  const body = imagineBody(s.calls);
  assert.equal(body.input_image_data, PNG);
  assert.equal(body.mask, PNG);
  assert.equal(body.input_image, null, "the server refuses a path and data together");
  buttons(s.$("img-native-inputs"), "Remove mask")[0].onclick();
  assert.equal(s.win.eval("imageInputs.state.mask"), null);
});

test("reference images are sent to an edit model and stop a job for one without them", async () => {
  const flux = await ready(FLUX_CAPS);
  runScript(flux.win, `imageInputs.state.refs.push("${PNG}", "${PNG}"); imageInputs.render();`);
  assert.match(flux.$("img-native-inputs").textContent, /FLUX.1 Kontext/);
  await generate(flux);
  assert.deepEqual(imagineBody(flux.calls).ref_images, [PNG, PNG]);

  const sd = await ready(SD1_CAPS);
  runScript(sd.win, `imageInputs.state.refs.push("${PNG}"); imageInputs.render();`);
  assert.match(sd.$("img-native-inputs").textContent, /does not take reference images/);
  await generate(sd);
  assert.equal(imagineBody(sd.calls), null);
});

test("a model without ControlNet support says so under the ControlNet picker", async () => {
  const { $ } = await ready(FLUX_CAPS);
  assert.match($("img-native-inputs").textContent, /does not take a ControlNet/);
});

test("reuse settings restores the ControlNet model, strength and preprocess", async () => {
  const s = await ready();
  runScript(s.win, `reuseImageSettings({ meta: { backend: "native", prompt: "p", model: "sd15",
    control: { model: "canny15", strength: 0.5, preprocess: "canny" } } });`);
  for (let i = 0; i < 6; i++) await tick();
  assert.equal(s.$("img-control-model").value, "canny15");
  assert.equal(s.$("img-control-strength").value, "0.5");
  assert.equal(s.$("img-control-preprocess").value, "canny");
});

test("iinHasWord matches whole words only, ignoring case and regex characters", async () => {
  const { win } = await ready();
  const has = (text, word) => win.eval(`iinHasWord(${JSON.stringify(text)}, ${JSON.stringify(word)})`);
  assert.equal(has("a fox, easynegative", "EasyNegative"), true);
  assert.equal(has("EasyNegativeV2", "EasyNegative"), false);
  assert.equal(has("my(emb)", "(emb)"), false);
  assert.equal(has("x (emb) y", "(emb)"), true);
  assert.equal(has("", "a"), false);
});
