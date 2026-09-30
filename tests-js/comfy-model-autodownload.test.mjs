// SPDX-License-Identifier: AGPL-3.0-or-later
// checkModelsBeforeGenerate(): the pre-generate model-existence check. A missing
// model with a curated source shows a confirm modal (repo/file/size plus a
// Download button); one without a curated source, or nothing missing at all,
// falls through with no modal.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));

function sseResponse(events) {
  const body = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("");
  const bytes = new TextEncoder().encode(body);
  let sent = false;
  return {
    ok: true, status: 200,
    body: {
      getReader() {
        return {
          async read() {
            if (sent) return { done: true, value: undefined };
            sent = true;
            return { done: false, value: bytes };
          },
        };
      },
    },
  };
}

function makeFetch({ missing, pulls, status = "verified", lookups = [], lookup = null }) {
  return async (url, opts = {}) => {
    const method = opts.method || "GET";
    if (url === "/api/media/image/preflight" && method === "POST") {
      return { ok: true, status: 200, json: async () => ({ status, missing, warning: "" }) };
    }
    if (url === "/api/models/comfy-source/lookup" && method === "POST") {
      lookups.push(JSON.parse(opts.body));
      return { ok: true, status: 200, json: async () => lookup };
    }
    if (url === "/api/models/pull-comfy-source" && method === "POST") {
      pulls.push(JSON.parse(opts.body));
      return { ok: true, status: 200, json: async () => ({ job_id: "job-1" }) };
    }
    if (url === "/api/jobs/job-1/events") {
      return sseResponse([{ type: "line", text: "downloading..." },
                          { type: "end", status: "done" }]);
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

test("nothing missing: resolves true, no modal, no pull POST", async () => {
  const pulls = [];
  const { window: win } = loadApp({ fetchImpl: makeFetch({ missing: [], pulls }) });
  await tick();
  const proceed = await win.checkModelsBeforeGenerate("image", null);
  assert.equal(proceed, true);
  assert.notEqual(win.document.querySelector("#modal").style.display, "flex");
  assert.deepEqual(pulls, []);
});

test("preflight unavailable: non-blocking warning shown, still resolves true, no modal, no pull POST", async () => {
  const pulls = [];
  const { window: win } = loadApp({
    fetchImpl: makeFetch({ missing: [], pulls, status: "unavailable" }),
  });
  await tick();
  const log = win.document.createElement("div");
  const proceed = await win.checkModelsBeforeGenerate("image", log);
  assert.equal(proceed, true, "unavailable never blocks generation");
  assert.notEqual(win.document.querySelector("#modal").style.display, "flex");
  assert.deepEqual(pulls, []);

  const toastEl = win.document.getElementById("toast");
  assert.ok(toastEl.textContent.toLowerCase().includes("could not check"),
    "toast says the pre-check itself could not run, not that nothing is missing");
  assert.equal(toastEl.className, "show error", "surfaced with error-level visual weight");
  assert.ok(log.textContent.toLowerCase().includes("could not check"),
    "the persistent log also gets the line");
});

test("missing WITHOUT a curated source: no modal, but an honest toast+log message", async () => {
  const pulls = [];
  const missing = [{ class_type: "CheckpointLoaderSimple", input_name: "ckpt_name",
                     filename: "custom.safetensors", source: null, dest_dir: null }];
  const { window: win } = loadApp({ fetchImpl: makeFetch({ missing, pulls }) });
  await tick();
  const log = win.document.createElement("div");
  const proceed = await win.checkModelsBeforeGenerate("image", log);
  assert.equal(proceed, true);
  assert.notEqual(win.document.querySelector("#modal").style.display, "flex");
  assert.deepEqual(pulls, []);

  const toastEl = win.document.getElementById("toast");
  assert.ok(toastEl.textContent.includes("custom.safetensors"), "toast names the missing file");
  assert.ok(toastEl.textContent.includes("CheckpointLoaderSimple.ckpt_name"),
    "toast names the class_type/input_name generically, not LoRA-specific wording");
  assert.ok(toastEl.textContent.toLowerCase().includes("no automatic download"),
    "toast is honest that it cannot auto-fetch this file");
  assert.equal(toastEl.className, "show error", "surfaced with error-level visual weight");
  assert.ok(log.textContent.includes("custom.safetensors"), "the persistent log also gets the line");
});

test("missing WITHOUT a curated source: a LoRA miss gets the same honest message", async () => {
  const pulls = [];
  const missing = [{ class_type: "LoraLoader", input_name: "lora_name",
                     filename: "my_style.safetensors", source: null, dest_dir: null }];
  const { window: win } = loadApp({ fetchImpl: makeFetch({ missing, pulls }) });
  await tick();
  const proceed = await win.checkModelsBeforeGenerate("image", null, { lora_name: "my_style.safetensors" });
  assert.equal(proceed, true);
  assert.notEqual(win.document.querySelector("#modal").style.display, "flex");
  assert.deepEqual(pulls, []);
  const toastEl = win.document.getElementById("toast");
  assert.ok(toastEl.textContent.includes("my_style.safetensors"));
  assert.ok(toastEl.textContent.includes("LoraLoader.lora_name"));
});

test("missing WITH a curated source: shows repo/file/size, offers Download", async () => {
  const pulls = [];
  const missing = [{
    class_type: "UnetLoaderGGUF", input_name: "unet_name",
    filename: "flux1-dev-Q8_0.gguf",
    source: { repo: "city96/FLUX.1-dev-gguf", file: "flux1-dev-Q8_0.gguf",
             size_bytes: 12708281504, model_type: "diffusion-unet" },
    dest_dir: "D:\\comfy\\models\\unet",
  }];
  const { window: win } = loadApp({ fetchImpl: makeFetch({ missing, pulls }) });
  await tick();
  const proceedPromise = win.checkModelsBeforeGenerate("image", null);
  await tick();
  const modal = win.document.querySelector("#modal");
  assert.equal(modal.style.display, "flex", "modal is shown for a curated missing model");
  const text = win.document.querySelector("#modal-body").textContent;
  assert.ok(text.includes("flux1-dev-Q8_0.gguf"), "shows the filename");
  assert.ok(text.includes("city96/FLUX.1-dev-gguf"), "shows the repo");
  assert.ok(text.includes("GB"), "shows a human-readable size");
  const buttons = [...win.document.querySelectorAll("#modal-body button")]
    .map((b) => b.textContent);
  assert.ok(buttons.includes("Download"), "a real Download button, never silent auto-pull");
  assert.ok(buttons.includes("Not now"));

  // Clicking Download POSTs the filename and the plugin whose ComfyUI folder the
  // destination resolves against, and nothing else; the server re-resolves
  // repo/path itself.
  [...win.document.querySelectorAll("#modal-body button")]
    .find((b) => b.textContent === "Download").click();
  await tick(); await tick(); await tick();
  const proceed = await proceedPromise;
  assert.equal(proceed, true);
  assert.deepEqual(pulls, [{ filename: "flux1-dev-Q8_0.gguf", plugin: "image",
                             class_type: "UnetLoaderGGUF", input_name: "unet_name" }]);
});

test("Not now skips the download without any pull POST", async () => {
  const pulls = [];
  const missing = [{
    class_type: "VAELoader", input_name: "vae_name", filename: "ae.safetensors",
    source: { repo: "black-forest-labs/FLUX.1-schnell", file: "ae.safetensors",
             size_bytes: 335304388, model_type: "vae" },
    dest_dir: "D:\\comfy\\models\\vae",
  }];
  const { window: win } = loadApp({ fetchImpl: makeFetch({ missing, pulls }) });
  await tick();
  const proceedPromise = win.checkModelsBeforeGenerate("image", null);
  await tick();
  [...win.document.querySelectorAll("#modal-body button")]
    .find((b) => b.textContent === "Not now").click();
  const proceed = await proceedPromise;
  assert.equal(proceed, true);
  assert.deepEqual(pulls, [], "skipping must never trigger a download");
});

const SEARCHABLE = [{
  class_type: "CheckpointLoaderSimple", input_name: "ckpt_name",
  filename: "wan2.1_t2v_1.3B_fp16.safetensors", source: null, dest_dir: null,
  searchable: true, reason: "", detail: "",
}];

function buttons(win) {
  return [...win.document.querySelectorAll("#modal-body button")];
}

function click(win, label) {
  const b = buttons(win).find((x) => x.textContent === label);
  assert.ok(b, `a "${label}" button is shown`);
  b.click();
}

test("searchable miss: nothing is sent until Search is clicked, then a found file is offered with its repository", async () => {
  const pulls = [];
  const lookups = [];
  const lookup = {
    status: "found", reason: "", detail: "", dest_dir: "D:\\comfy\\models\\unet",
    source: { repo: "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
              file: "split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
              size_bytes: 2838303560, model_type: "diffusion-unet", origin: "huggingface" },
  };
  const { window: win } = loadApp({
    fetchImpl: makeFetch({ missing: SEARCHABLE, pulls, lookups, lookup }) });
  await tick();
  const proceedPromise = win.checkModelsBeforeGenerate("image", null);
  await tick();
  assert.equal(win.document.querySelector("#modal").style.display, "flex");
  assert.deepEqual(lookups, [], "no search request before the user asks for one");
  assert.ok(buttons(win).map((b) => b.textContent).includes("Not now"));

  click(win, "Search Hugging Face");
  await tick(); await tick(); await tick();
  assert.deepEqual(lookups, [{
    filename: "wan2.1_t2v_1.3B_fp16.safetensors", class_type: "CheckpointLoaderSimple",
    input_name: "ckpt_name", plugin: "image" }]);
  const text = win.document.querySelector("#modal-body").textContent;
  assert.equal(win.document.querySelector("#modal").style.display, "flex",
    "the found file is offered in a second dialog");
  assert.ok(text.includes("Comfy-Org/Wan_2.1_ComfyUI_repackaged"), "shows the repository");
  assert.ok(text.includes("split_files/diffusion_models/"), "shows the path in the repository");
  assert.ok(text.includes("not from localm's own catalog"), "says it is a search result");
  assert.deepEqual(pulls, [], "finding a file does not download it");

  click(win, "Download");
  await tick(); await tick(); await tick();
  assert.equal(await proceedPromise, true);
  assert.deepEqual(pulls, [{ filename: "wan2.1_t2v_1.3B_fp16.safetensors", plugin: "image",
                             class_type: "CheckpointLoaderSimple", input_name: "ckpt_name" }]);
});

test("searchable miss: Not now sends neither a search nor a download", async () => {
  const pulls = [];
  const lookups = [];
  const { window: win } = loadApp({
    fetchImpl: makeFetch({ missing: SEARCHABLE, pulls, lookups }) });
  await tick();
  const proceedPromise = win.checkModelsBeforeGenerate("image", null);
  await tick();
  click(win, "Not now");
  assert.equal(await proceedPromise, true);
  assert.deepEqual(lookups, []);
  assert.deepEqual(pulls, []);
});

for (const [status, phrase] of [
  ["not_found", "no public hugging face repository"],
  ["offline", "network access is off"],
  ["failed", "could not search hugging face"],
]) {
  test(`searchable miss, lookup ${status}: reported, no download`, async () => {
    const pulls = [];
    const lookups = [];
    const lookup = { status, source: null, dest_dir: null, reason: "", detail: "x" };
    const { window: win } = loadApp({
      fetchImpl: makeFetch({ missing: SEARCHABLE, pulls, lookups, lookup }) });
    await tick();
    const log = win.document.createElement("div");
    const proceedPromise = win.checkModelsBeforeGenerate("image", log);
    await tick();
    click(win, "Search Hugging Face");
    await tick(); await tick(); await tick();
    assert.equal(await proceedPromise, true);
    assert.equal(lookups.length, 1);
    assert.deepEqual(pulls, []);
    assert.notEqual(win.document.querySelector("#modal").style.display, "flex");
    const toastEl = win.document.getElementById("toast");
    assert.ok(toastEl.textContent.toLowerCase().includes(phrase), toastEl.textContent);
    assert.ok(toastEl.textContent.includes("wan2.1_t2v_1.3B_fp16.safetensors"));
    assert.ok(log.textContent.toLowerCase().includes(phrase));
  });
}

for (const [reason, phrase] of [
  ["format", ".safetensors and .gguf"],
  ["folder", "which comfyui models folder"],
  ["name", "too generic"],
]) {
  test(`unsearchable miss (${reason}): explained, no dialog, no request`, async () => {
    const pulls = [];
    const lookups = [];
    const missing = [{ class_type: "UpscaleModelLoader", input_name: "model_name",
                       filename: "RealESRGAN_x4plus.pth", source: null, dest_dir: null,
                       searchable: false, reason, detail: "" }];
    const { window: win } = loadApp({
      fetchImpl: makeFetch({ missing, pulls, lookups }) });
    await tick();
    assert.equal(await win.checkModelsBeforeGenerate("image", null), true);
    assert.notEqual(win.document.querySelector("#modal").style.display, "flex");
    assert.deepEqual(lookups, []);
    assert.deepEqual(pulls, []);
    const toastEl = win.document.getElementById("toast");
    assert.ok(toastEl.textContent.toLowerCase().includes(phrase), toastEl.textContent);
  });
}

test("a source with no known size says so instead of printing a blank size", async () => {
  const pulls = [];
  const missing = [{
    class_type: "VAELoader", input_name: "vae_name", filename: "qwen_image_vae.safetensors",
    source: { repo: "Comfy-Org/Qwen-Image_ComfyUI", file: "split_files/vae/qwen_image_vae.safetensors",
              size_bytes: null, model_type: "vae", origin: "huggingface" },
    dest_dir: "D:\\comfy\\models\\vae",
  }];
  const { window: win } = loadApp({ fetchImpl: makeFetch({ missing, pulls }) });
  await tick();
  const proceedPromise = win.checkModelsBeforeGenerate("image", null);
  await tick();
  const text = win.document.querySelector("#modal-body").textContent;
  assert.ok(text.includes("This workflow needs 'qwen_image_vae.safetensors', which isn't installed."),
    text);
  assert.ok(!text.includes("()"), "no empty size in parentheses");
  click(win, "Not now");
  assert.equal(await proceedPromise, true);
});
