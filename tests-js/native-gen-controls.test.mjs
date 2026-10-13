// SPDX-License-Identifier: AGPL-3.0-or-later
// The native generation controls on the Images and Video pages (app/native-models.js
// createNativeGenControls): batch count, sampler, scheduler and hires fix, how they
// reach the request body, how "reuse settings" restores them, the gallery
// "Upscale" action, and the per-image result stream.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));

const SAMPLING = {
  sample_methods: ["euler", "euler_a", "dpm++2m"],
  schedulers: ["discrete", "karras"],
  hires_upscalers: [
    { id: "latent", label: "Latent" }, { id: "latent-bicubic", label: "Latent (bicubic)" },
    { id: "lanczos", label: "Lanczos" }, { id: "model", label: "Model" }],
};
const OPTIONS = {
  models: [{ name: "sd-turbo", file: "sd_turbo.gguf", kind: "checkpoint", family: "sd2",
             family_label: "Stable Diffusion 2.x" }],
  slots: { clip_l: [], clip_g: [], t5xxl: [], llm: [], vae: [] },
  loras: [{ name: "style", file: "style.safetensors", kind: "lora", family: "sd2" }],
  current: { model: "", clip_l: "", clip_g: "", t5xxl: "", llm: "", vae: "" },
  sampling: SAMPLING,
  upscalers: [{ name: "real-esrgan", file: "RealESRGAN_x4plus.safetensors" }],
};

function sseResponse(events) {
  const body = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("");
  const bytes = new TextEncoder().encode(body);
  let sent = false;
  return {
    ok: true, status: 200,
    body: { getReader() { return { async read() {
      if (sent) return { done: true, value: undefined };
      sent = true;
      return { done: false, value: bytes };
    } }; } },
  };
}

function setup({ active = "native", jobEvents = [{ type: "end", status: "done", result: "a.png" }],
                 options = OPTIONS } = {}) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    if (u.startsWith("/api/imagine/native-models")) {
      return { ok: true, status: 200, json: async () => ({ ...options, preview: null }) };
    }
    if (u.startsWith("/api/video/native-models")) {
      return { ok: true, status: 200, json: async () => ({ ...options, preview: null }) };
    }
    if (u === "/api/imagine/backend") {
      return { ok: true, status: 200, json: async () => ({
        choice: "auto", active, native: { model: "sd-turbo", runtime: "vulkan", recommended: {} } }) };
    }
    if (u === "/api/video/backend") {
      return { ok: true, status: 200, json: async () => ({
        choice: "auto", active, native: { model: "wan", runtime: "vulkan", recommended: {} } }) };
    }
    if (u === "/api/imagine" || u === "/api/video" || u.endsWith("/upscale")) {
      return { ok: true, status: 200, json: async () => ({ job_id: "j1" }) };
    }
    if (u === "/api/jobs/j1/events") return sseResponse(jobEvents);
    if (u.startsWith("/api/imagine/file/")) {
      return { ok: true, status: 200, blob: async () => ({}), json: async () => ({}),
               text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({ models: [], images: [] }), text: async () => "" };
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  return { win, calls, $: (id) => win.document.getElementById(id) };
}

async function settle(win, code) {
  runScript(win, `globalThis.__p = (async () => { ${code} })();`);
  await win.__p;
}

const imagineBody = (calls) => JSON.parse(calls.find((c) => c.url === "/api/imagine").opts.body);

test("the native controls show for the native backend and stay hidden for ComfyUI", async () => {
  const native = setup();
  await settle(native.win, "await refreshImageBackend();");
  const nodes = [...native.win.document.querySelectorAll("#view-images .img-native-only")];
  assert.ok(nodes.length >= 4);
  assert.ok(nodes.every((n) => n.hidden === false));
  const comfy = setup({ active: "comfy" });
  await settle(comfy.win, "await refreshImageBackend();");
  assert.ok([...comfy.win.document.querySelectorAll("#view-images .img-native-only")]
    .every((n) => n.hidden === true));
});

test("the dropdowns are filled from the server's names; hires offers the non-latent upscalers", async () => {
  const { win, $ } = setup();
  await settle(win, "await refreshImageBackend();");
  await tick(); await tick();
  const opts = (id) => [...$(id).options].map((o) => o.value);
  assert.deepEqual(opts("img-sampler"), ["", "euler", "euler_a", "dpm++2m"]);
  assert.deepEqual(opts("img-scheduler"), ["", "discrete", "karras"]);
  assert.deepEqual(opts("img-hires-upscaler"), ["", "latent-bicubic", "lanczos", "model"]);
  assert.deepEqual(opts("img-hires-model"), ["", "real-esrgan"]);
  assert.equal($("img-sampler").options[0].textContent, "Model default");
});

test("untouched controls send nothing extra", async () => {
  const { win, $, calls } = setup();
  await settle(win, "await refreshImageBackend();");
  await tick(); await tick();
  $("img-prompt").value = "a fox";
  $("img-generate").onclick();
  await tick(); await tick(); await tick();
  const body = imagineBody(calls);
  for (const key of ["batch", "sample_method", "scheduler", "hires"]) {
    assert.ok(!(key in body), `${key} must not be sent when untouched`);
  }
});

test("batch, sampler, scheduler and hires are sent when set", async () => {
  const { win, $, calls } = setup();
  await settle(win, "await refreshImageBackend();");
  await tick(); await tick();
  $("img-prompt").value = "a fox";
  $("img-batch").value = "4";
  $("img-sampler").value = "euler_a";
  $("img-scheduler").value = "karras";
  $("img-hires").checked = true;
  $("img-hires-upscaler").value = "model";
  $("img-hires-upscaler").dispatchEvent(new win.Event("change"));
  assert.equal($("img-hires-model-row").hidden, false, "the model row shows for the model upscaler");
  $("img-hires-model").value = "real-esrgan";
  $("img-hires-scale").value = "1.5";
  $("img-hires-steps").value = "8";
  $("img-hires-denoise").value = "0.4";
  $("img-generate").onclick();
  await tick(); await tick(); await tick();
  const body = imagineBody(calls);
  assert.equal(body.batch, 4);
  assert.equal(body.sample_method, "euler_a");
  assert.equal(body.scheduler, "karras");
  assert.deepEqual(body.hires, { upscaler: "model", scale: 1.5, steps: 8, denoise: 0.4,
                                 model: "real-esrgan" });
});

test("a hires fix with only the checkbox ticked uses the latent defaults", async () => {
  const { win, $, calls } = setup();
  await settle(win, "await refreshImageBackend();");
  await tick(); await tick();
  $("img-prompt").value = "a fox";
  $("img-hires").checked = true;
  $("img-generate").onclick();
  await tick(); await tick(); await tick();
  assert.deepEqual(imagineBody(calls).hires, { upscaler: "latent", scale: 2, steps: 0, denoise: 0.7 });
});

test("the controls are not sent while the backend is ComfyUI", async () => {
  const { win, $, calls } = setup({ active: "comfy" });
  await settle(win, "await refreshImageBackend();");
  $("img-prompt").value = "a fox";
  $("img-batch").value = "3";
  $("img-hires").checked = true;
  $("img-generate").onclick();
  await tick(); await tick(); await tick();
  const body = imagineBody(calls);
  assert.ok(!("batch" in body) && !("hires" in body));
});

test("reuse settings restores sampler, scheduler, hires, size, model and LoRAs", async () => {
  const { win, $ } = setup();
  await settle(win, "await refreshImageBackend();");
  await tick(); await tick();
  const meta = {
    prompt: "p", seed: 5, backend: "native", model: "sd-turbo", size: "512x768",
    sample_method: "dpm++2m", scheduler: "karras",
    hires: { upscaler: "model", model: "real-esrgan", scale: 1.5, steps: 6, denoise: 0.4 },
    loras: [{ name: "style", strength: 0.6 }], components: {},
  };
  runScript(win, `reuseImageSettings({ meta: ${JSON.stringify(meta)} });`);
  await tick(); await tick();
  assert.equal($("img-prompt").value, "p");
  assert.equal($("img-size").value, "512x768");
  assert.equal($("img-sampler").value, "dpm++2m");
  assert.equal($("img-scheduler").value, "karras");
  assert.equal($("img-hires").checked, true);
  assert.equal($("img-hires-upscaler").value, "model");
  assert.equal($("img-hires-model").value, "real-esrgan");
  assert.equal($("img-hires-scale").value, "1.5");
  assert.equal($("img-hires-steps").value, "6");
  assert.equal($("img-hires-denoise").value, "0.4");
  assert.equal($("img-hires-model-row").hidden, false);
  runScript(win, "globalThis.__c = JSON.stringify(imageModelPicker.choice());");
  assert.deepEqual(JSON.parse(win.__c), {
    native_models: { model: "sd-turbo" }, loras: [{ name: "style", strength: 0.6 }] });
  assert.equal($("img-advanced").open, true, "the fold opens so the restored values are visible");
});

test("reusing an image made without hires clears the hires fields", async () => {
  const { win, $ } = setup();
  await settle(win, "await refreshImageBackend();");
  await tick(); await tick();
  $("img-hires").checked = true;
  $("img-hires-scale").value = "3";
  $("img-sampler").value = "euler";
  runScript(win, `reuseImageSettings({ meta: { prompt: "p", backend: "native", model: "sd-turbo" } });`);
  assert.equal($("img-hires").checked, false);
  assert.equal($("img-hires-scale").value, "");
  assert.equal($("img-sampler").value, "");
});

test("a restored sampler the server no longer lists is kept, not dropped", async () => {
  const { win, $ } = setup();
  await settle(win, "await refreshImageBackend();");
  await tick(); await tick();
  runScript(win, `reuseImageSettings({ meta: { backend: "native", sample_method: "retired_sampler" } });`);
  assert.equal($("img-sampler").value, "retired_sampler");
});

test("each saved image of a batch appears as its event arrives", async () => {
  const { win, $, calls } = setup({ jobEvents: [
    { type: "line", text: "Generating image 1/2" },
    { type: "image", name: "a.png" },
    { type: "image", name: "a_2.png" },
    { type: "end", status: "done", result: "a.png" }] });
  await settle(win, "await refreshImageBackend();");
  await tick(); await tick();
  $("img-prompt").value = "a fox";
  $("img-batch").value = "2";
  $("img-generate").onclick();
  for (let i = 0; i < 8; i++) await tick();
  const fetched = calls.map((c) => c.url).filter((u) => u.startsWith("/api/imagine/file/"));
  assert.deepEqual([...new Set(fetched)].sort(), ["/api/imagine/file/a.png", "/api/imagine/file/a_2.png"]);
  assert.equal($("img-result").querySelectorAll("img").length, 2,
    "the end event's result repeats the first image and must not add a third");
  assert.match($("img-log").textContent, /Generating image 1\/2/);
});

test("a cancelled batch still shows the images it saved", async () => {
  const { win, $ } = setup({ jobEvents: [
    { type: "image", name: "a.png" },
    { type: "line", text: "Cancelled. 1 of 3 images were saved." },
    { type: "end", status: "cancelled", result: "a.png" }] });
  await settle(win, "await refreshImageBackend();");
  await tick(); await tick();
  $("img-prompt").value = "a fox";
  $("img-generate").onclick();
  for (let i = 0; i < 8; i++) await tick();
  assert.equal($("img-result").querySelectorAll("img").length, 1);
});

test("the gallery Upscale action posts the chosen upscaler for that image", async () => {
  const { win, $, calls } = setup({ jobEvents: [
    { type: "image", name: "up.png" }, { type: "end", status: "done", result: "up.png" }] });
  await settle(win, "await refreshImageBackend();");
  await tick(); await tick();
  runScript(win, `globalThis.__row = buildUpscaleAction({ name: "src one.png" }, { closeModal() {} });`);
  const row = win.__row;
  for (let i = 0; i < 4; i++) await tick();
  const sel = row.querySelector("select");
  const btn = row.querySelector("button");
  assert.equal(btn.disabled, false);
  assert.deepEqual([...sel.options].map((o) => o.value), ["real-esrgan"]);
  btn.onclick();
  for (let i = 0; i < 8; i++) await tick();
  const post = calls.find((c) => c.url.endsWith("/upscale"));
  assert.ok(post, "the upscale route was called");
  assert.equal(post.url, "/api/imagine/file/src%20one.png/upscale");
  assert.deepEqual(JSON.parse(post.opts.body), { upscaler: "real-esrgan" });
  assert.equal($("img-result").querySelectorAll("img").length, 1);
});

test("with no registered upscaler the Upscale button stays disabled and says why", async () => {
  const { win } = setup({ options: { ...OPTIONS, upscalers: [] } });
  await settle(win, "await refreshImageBackend();");
  await tick(); await tick();
  runScript(win, `globalThis.__row = buildUpscaleAction({ name: "a.png" }, { closeModal() {} });`);
  const row = win.__row;
  for (let i = 0; i < 4; i++) await tick();
  const btn = row.querySelector("button");
  assert.equal(btn.disabled, true);
  assert.match(btn.title, /No upscaler model is registered/);
});

test("the video page offers a sampler and scheduler for native and sends them", async () => {
  const { win, $, calls } = setup();
  await settle(win, "await refreshVideoBackend();");
  await tick(); await tick();
  assert.deepEqual([...$("video-sampler").options].map((o) => o.value),
    ["", "euler", "euler_a", "dpm++2m"]);
  assert.equal($("video-sampler").closest(".video-native-only").hidden, false);
  $("video-prompt").value = "waves";
  $("video-sampler").value = "euler";
  $("video-scheduler").value = "karras";
  $("video-generate").onclick();
  await tick(); await tick(); await tick();
  const body = JSON.parse(calls.find((c) => c.url === "/api/video").opts.body);
  assert.equal(body.sample_method, "euler");
  assert.equal(body.scheduler, "karras");
});

test("the video page sends no sampler when ComfyUI generates", async () => {
  const { win, $, calls } = setup({ active: "comfy" });
  await settle(win, "await refreshVideoBackend();");
  $("video-prompt").value = "waves";
  $("video-generate").onclick();
  await tick(); await tick(); await tick();
  const body = JSON.parse(calls.find((c) => c.url === "/api/video").opts.body);
  assert.ok(!("sample_method" in body) && !("scheduler" in body));
});

test("a job that only reports its result (ComfyUI) still shows that image", async () => {
  const { win, $ } = setup({ active: "comfy", jobEvents: [
    { type: "line", text: "Submitting workflow" },
    { type: "end", status: "done", result: "comfy.png" }] });
  await settle(win, "await refreshImageBackend();");
  $("img-prompt").value = "a fox";
  $("img-generate").onclick();
  for (let i = 0; i < 8; i++) await tick();
  assert.equal($("img-result").querySelectorAll("img").length, 1);
});
