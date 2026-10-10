// SPDX-License-Identifier: AGPL-3.0-or-later
// checkModelsBeforeGenerate (app/helpers.js) offers each missing model in a
// download dialog. An entry from a native backend (native: true) is a model
// file, not a ComfyUI workflow input, so its dialog never mentions a workflow.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function fetchWith(missing) {
  return async (url) => ({
    ok: true, status: 200, text: async () => "",
    json: async () => (String(url).includes("/preflight")
      ? { status: "verified", warning: null, missing }
      : { models: [], active: "", conversations: [], plugins: [] }),
  });
}

async function dialogText(entry) {
  const { window: win } = loadApp({ fetchImpl: fetchWith([entry]) });
  runScript(win, `globalThis.__done = checkModelsBeforeGenerate("music", null);`);
  for (let i = 0; i < 50 && win.document.getElementById("modal").style.display === "none"; i++) {
    await new Promise((r) => setTimeout(r, 10));
  }
  return win.document.getElementById("modal").textContent;
}

const NATIVE = {
  filename: "vae-BF16.gguf", native: true, searchable: false,
  source: { repo: "owner/repo", file: "vae-BF16.gguf", spec: "owner/repo:vae-BF16.gguf",
            name: "vae-BF16", size_bytes: 337420928, model_type: "vae", origin: "curated" },
};

test("a native missing model is offered without mentioning a workflow", async () => {
  const text = await dialogText(NATIVE);
  assert.match(text, /Generating needs 'vae-BF16\.gguf' \(.+\), which isn't downloaded yet\./);
  assert.doesNotMatch(text, /workflow/i);
});

test("a native missing model without a size is offered without mentioning a workflow", async () => {
  const text = await dialogText({ ...NATIVE, source: { ...NATIVE.source, size_bytes: null } });
  assert.match(text, /Generating needs 'vae-BF16\.gguf', which isn't downloaded yet\./);
  assert.doesNotMatch(text, /workflow/i);
});

test("a ComfyUI missing model still names the workflow", async () => {
  const text = await dialogText({
    filename: "ace.safetensors", class_type: "CheckpointLoaderSimple", input_name: "ckpt_name",
    source: { repo: "owner/repo", file: "ace.safetensors", size_bytes: 1024, origin: "curated" },
  });
  assert.match(text, /This workflow needs 'ace\.safetensors'/);
});
