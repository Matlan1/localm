// SPDX-License-Identifier: AGPL-3.0-or-later
// refreshMusicBackend (pages/music.js) reads GET /api/music/backend and shows
// which backend runs: the backend note, the ComfyUI workflow card (hidden for
// native) and the Steps/CFG placeholders, which name the running backend's
// own defaults.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function fetchWith(backend) {
  return async (url) => ({
    ok: true, status: 200, text: async () => "",
    json: async () => (String(url).includes("/api/music/backend")
      ? backend
      : { models: [], active: "", conversations: [], plugins: [] }),
  });
}

async function refreshed(backend) {
  const { window: win } = loadAppWithPages({ fetchImpl: fetchWith(backend) });
  runScript(win, "globalThis.__p = refreshMusicBackend();");
  await win.__p;
  const $ = (id) => win.document.getElementById(id);
  return {
    note: $("music-backend-note").textContent,
    steps: $("music-steps").placeholder,
    cfg: $("music-cfg").placeholder,
    cardHidden: $("music-workflow-card").hidden,
  };
}

const MODELS = { text_encoder: "te.gguf", dit: "dit.gguf", vae: "vae.gguf", lm: "lm.gguf" };
const GB = 1024 ** 3;

test("native: the placeholders are the native defaults and the workflow card is hidden", async () => {
  const r = await refreshed({
    choice: "auto", active: "native",
    native: { models: MODELS, missing: [], runtime: { installed: true, backend: "vulkan" } },
  });
  assert.equal(r.steps, "default (8)");
  assert.equal(r.cfg, "default (1.0)");
  assert.equal(r.cardHidden, true);
  assert.equal(r.note, "Backend: native ACE-Step 1.5 (KoboldCpp), runtime vulkan.");
});

test("comfy: the placeholders are the ComfyUI defaults and the workflow card shows", async () => {
  const r = await refreshed({ choice: "comfy", active: "comfy" });
  assert.equal(r.steps, "default (50)");
  assert.equal(r.cfg, "default (5.0)");
  assert.equal(r.cardHidden, false);
});

test("native with some default models missing says how many", async () => {
  const r = await refreshed({
    choice: "auto", active: "native",
    native: { models: { ...MODELS, dit: null }, missing: [{ size_bytes: 2 * GB }], runtime: {} },
  });
  assert.match(r.note, /Missing 1 of 4 default music models: Generate offers to download them \(2(\.0+)? GB\)\./);
});

test("native with every default model missing offers the default set", async () => {
  const none = { text_encoder: null, dit: null, vae: null, lm: null };
  const r = await refreshed({
    choice: "auto", active: "native",
    native: { models: none, missing: [1, 2, 3, 4].map(() => ({ size_bytes: GB })), runtime: {} },
  });
  assert.match(r.note, /No music models yet: Generate offers to download the default set/);
});
