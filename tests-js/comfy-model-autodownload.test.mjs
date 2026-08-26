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

function makeFetch({ missing, pulls }) {
  return async (url, opts = {}) => {
    const method = opts.method || "GET";
    if (url === "/api/media/image/preflight" && method === "POST") {
      return { ok: true, status: 200, json: async () => ({ missing }) };
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
  assert.deepEqual(pulls, [{ filename: "flux1-dev-Q8_0.gguf", plugin: "image" }]);
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
