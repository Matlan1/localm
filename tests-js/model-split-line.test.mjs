// SPDX-License-Identifier: AGPL-3.0-or-later
// The sidebar's loaded-model status shows the GPU split applied to the active
// model: per-device shares plus the source (auto, pinned or equal). The data
// comes from GET /api/models' active_gpu_split; absent means no split and the
// line stays hidden.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function makeFetch(split) {
  return async (url) => {
    if (String(url).startsWith("/api/models")) {
      const body = {
        models: [{ name: "model-a", active: true }],
        active: "model-a",
      };
      if (split) body.active_gpu_split = split;
      return { ok: true, status: 200, json: async () => body, text: async () => "" };
    }
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function render(win) {
  runScript(win, "refreshModels();");
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
}

test("an auto split renders per-device shares with the free-VRAM wording", async () => {
  const split = { source: "auto", devices: [
    { index: 0, share: 1 / 3 }, { index: 1, share: 2 / 3 }] };
  const { window: win } = loadApp({ fetchImpl: makeFetch(split) });
  await render(win);
  const line = win.document.getElementById("model-split-line");
  assert.ok(line, "#model-split-line exists in the sidebar");
  assert.equal(line.hidden, false, "visible when a split was applied");
  assert.match(line.textContent, /GPU 0[^%]*33%/);
  assert.match(line.textContent, /GPU 1[^%]*67%/);
  assert.match(line.textContent, /free VRAM/,
    "the auto source is attributed so the user knows WHY the shares differ");
});

test("pinned and equal sources are labelled as such", async () => {
  for (const [source, wording] of [["pinned", /pinned/], ["equal", /equal/]]) {
    const split = { source, devices: [
      { index: 0, share: 0.5 }, { index: 1, share: 0.5 }] };
    const { window: win } = loadApp({ fetchImpl: makeFetch(split) });
    await render(win);
    const line = win.document.getElementById("model-split-line");
    assert.equal(line.hidden, false);
    assert.match(line.textContent, wording, `${source} split labelled`);
  }
});

test("no split keeps the line hidden (single-GPU boxes unchanged)", async () => {
  const { window: win } = loadApp({ fetchImpl: makeFetch(null) });
  await render(win);
  const line = win.document.getElementById("model-split-line");
  assert.ok(line, "#model-split-line exists in the sidebar");
  assert.equal(line.hidden, true, "hidden when active_gpu_split is absent");
});

test("a split appearing then disappearing hides the line again", async () => {
  // A model switch from a split model to a single-GPU one.
  let split = { source: "auto", devices: [
    { index: 0, share: 0.5 }, { index: 1, share: 0.5 }] };
  const { window: win } = loadApp({
    fetchImpl: async (url) => makeFetch(split)(url) });
  await render(win);
  assert.equal(win.document.getElementById("model-split-line").hidden, false);
  split = null;
  await render(win);
  assert.equal(win.document.getElementById("model-split-line").hidden, true);
});
