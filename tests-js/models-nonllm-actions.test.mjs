// SPDX-License-Identifier: AGPL-3.0-or-later
// Every Models-page row offers alias, rename and remove whatever its type;
// Use, Reload and Packages stay on chat models. The buttons call the existing
// routes unchanged.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const TYPES = ["llm", "embedding", "diffusion-unet", "text-encoder", "vae", "lora",
  "upscaler", "controlnet", "unknown", "mmproj", "tts"];

function makeFetch(models, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models, active: null }) };
    }
    if (/^\/api\/models\/(alias|rename|remove)$/.test(u)) {
      calls.push({ url: u, body: JSON.parse(opts.body) });
      const body = u.endsWith("remove") ? { job_id: "j1" }
        : u.endsWith("alias") ? { status: "aliased", alias: "a-1" }
          : { status: "renamed", new_name: "n-1", notes: [] };
      return { ok: true, status: 200, json: async () => body, text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

async function tick(n = 1) {
  for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0));
}

function rowFor(window, name) {
  return [...window.document.querySelectorAll("#models-table tbody tr")]
    .find((tr) => tr.querySelector(".name")?.textContent === name);
}

async function rowButtons(window, name) {
  await window.refreshModelsPage();
  await tick();
  let row = rowFor(window, name);
  if (!row) {
    for (const tab of window.document.querySelectorAll("#models-tab-nav .tab-btn")) {
      tab.click();
      await tick();
      row = rowFor(window, name);
      if (row) break;
    }
  }
  assert.ok(row, `no row rendered for ${name}`);
  return [...row.querySelectorAll("button")];
}

for (const type of TYPES) {
  test(`models actions: a ${type} row has alias, rename and remove`, async () => {
    const model = { name: "m-" + type, model_type: type, active: false, loaded: false, size_bytes: 10 };
    const { window } = loadAppWithPages({ fetchImpl: makeFetch([model], []) });
    const labels = (await rowButtons(window, model.name)).map((b) => b.textContent);
    for (const want of ["info", "alias", "rename", "remove"]) {
      assert.ok(labels.includes(want), `${type} row must offer ${want}, got ${JSON.stringify(labels)}`);
    }
    if (type === "llm") {
      assert.ok(labels.includes("use"), "an llm row keeps use");
    } else {
      assert.ok(!labels.includes("use"), `${type} row is not a chat model and must not offer use`);
    }
  });
}

test("models actions: alias, rename and remove on a diffusion row post to the existing routes", async () => {
  const calls = [];
  const model = { name: "flux-unet", model_type: "diffusion-unet", active: false, loaded: false, size_bytes: 10 };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([model], calls) });
  runScript(window, `promptText = async () => "fresh-name"; confirmDanger = (t, m, l, ok) => ok(); streamJob = async () => ({ status: "done" });`);
  const buttons = await rowButtons(window, "flux-unet");
  const click = async (label) => {
    await buttons.find((b) => b.textContent === label).onclick();
    await tick(3);
  };
  await click("alias");
  await click("rename");
  await click("remove");
  assert.deepEqual(calls, [
    { url: "/api/models/alias", body: { model: "flux-unet", alias: "fresh-name" } },
    { url: "/api/models/rename", body: { model: "flux-unet", new_name: "fresh-name" } },
    { url: "/api/models/remove", body: { model: "flux-unet" } },
  ]);
});

test("models actions: a loaded embedding row can be unloaded but is not offered reload or use", async () => {
  const model = { name: "bge", model_type: "embedding", active: false, loaded: true, size_bytes: 10 };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([model], []) });
  const labels = (await rowButtons(window, "bge")).map((b) => b.textContent);
  assert.ok(labels.includes("unload"));
  assert.ok(!labels.includes("reload") && !labels.includes("use"));
});
