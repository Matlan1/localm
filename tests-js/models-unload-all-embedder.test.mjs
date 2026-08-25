// SPDX-License-Identifier: AGPL-3.0-or-later
// The Models page "Unload all" button's toast counts `unloaded_models` (chat
// engines) plus the `embedder_unloaded` flag on the same response, which covers
// the shared embedding model's separate lifecycle (localm.inference.embedder).

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const flush = async () => {
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
};

function makeUnloadFetch(unloadResponse) {
  return async (url) => {
    const u = String(url);
    if (u.startsWith("/api/models/unload")) {
      return { ok: true, status: 200, json: async () => unloadResponse, text: async () => "" };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models: [], active: null }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

async function clickUnloadAll(unloadResponse) {
  const { window } = loadAppWithPages({ fetchImpl: makeUnloadFetch(unloadResponse) });
  const btn = window.document.getElementById("models-unload-all-btn");
  assert.ok(btn, "the Unload all button exists");
  btn.click();
  await flush();
  return window.document.getElementById("toast").textContent;
}

test("unload-all toast counts a resident embedder even with no chat model loaded", async () => {
  const msg = await clickUnloadAll({
    status: "unloaded", model: "none", unloaded_models: [], embedder_unloaded: true,
  });
  assert.notEqual(msg, "Nothing was loaded",
    `must not claim nothing was loaded when the embedder WAS released (got: ${msg})`);
  assert.match(msg, /Unloaded 1 model/, `should count the embedder (got: ${msg})`);
});

test("unload-all toast counts chat models plus the embedder together", async () => {
  const msg = await clickUnloadAll({
    status: "unloaded", model: "model-a",
    unloaded_models: ["model-a", "model-b"], embedder_unloaded: true,
  });
  assert.match(msg, /Unloaded 3 model/, `should count 2 chat models + the embedder (got: ${msg})`);
});

test("unload-all toast still reports 'Nothing was loaded' when truly nothing was resident", async () => {
  const msg = await clickUnloadAll({
    status: "already_unloaded", model: "none", unloaded_models: [], embedder_unloaded: false,
  });
  assert.equal(msg, "Nothing was loaded", `negative control (got: ${msg})`);
});

test("unload-all toast tolerates a response with no embedder_unloaded field (older server)", async () => {
  const msg = await clickUnloadAll({
    status: "unloaded", model: "model-a", unloaded_models: ["model-a"],
  });
  assert.match(msg, /Unloaded 1 model/, `must not throw / miscount on a missing field (got: ${msg})`);
});
