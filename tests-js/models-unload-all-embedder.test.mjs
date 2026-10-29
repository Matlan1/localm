// SPDX-License-Identifier: AGPL-3.0-or-later
// The Models page "Unload all" button's toast counts `unloaded_models` (chat
// engines) plus `unloaded_other` (the embedding, reranker and speech models),
// or the `embedder_unloaded` flag from a server that sends no `unloaded_other`.

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

test("unload-all toast counts every embedding, reranker and speech model in unloaded_other", async () => {
  const msg = await clickUnloadAll({
    status: "unloaded", model: "model-a", unloaded_models: ["model-a"],
    unloaded_other: ["embedding model", "reranker model", "speech model"],
    embedder_unloaded: true,
  });
  assert.match(msg, /Unloaded 4 model/, `1 chat model + 3 others (got: ${msg})`);
});

function makeForceFlowFetch(first, forced, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models/unload")) {
      const body = opts.body ? JSON.parse(opts.body) : {};
      calls.push(body);
      const data = body.force ? forced : first;
      return { ok: true, status: 200, json: async () => data, text: async () => "" };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models: [], active: null }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

async function unloadAllThenConfirm(first, forced) {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeForceFlowFetch(first, forced, calls) });
  const doc = window.document;
  doc.getElementById("models-unload-all-btn").click();
  await flush();
  const confirmBtn = doc.querySelector("#modal-body .btn-danger");
  assert.ok(confirmBtn, "a busy model asks before unloading anyway");
  confirmBtn.click();
  await flush();
  return { calls, toast: doc.getElementById("toast").textContent };
}

const BUSY_REPLY = {
  status: "in_use", model: "none", unloaded_models: [], unloaded_other: [],
  embedder_unloaded: false, skipped_in_use: ["chat-a", "embedding model"],
  confirm_required: { "chat-a": "1 other active request", "embedding model": "1 active request" },
};

test("unload-all anyway counts the chat and embedding models it stopped", async () => {
  const { calls, toast } = await unloadAllThenConfirm(BUSY_REPLY, {
    status: "unloaded", model: "chat-a", unloaded_models: ["chat-a"],
    unloaded_other: ["embedding model"], embedder_unloaded: true,
  });
  assert.deepEqual(calls, [{}, { force: true }]);
  assert.equal(toast, "Unloaded 2 model(s)");
});

test("unload-all anyway reports a model that still stayed loaded", async () => {
  const { toast } = await unloadAllThenConfirm(BUSY_REPLY, {
    status: "unloaded", model: "chat-a", unloaded_models: ["chat-a"],
    unloaded_other: [], embedder_unloaded: false, skipped_in_use: ["embedding model"],
  });
  assert.equal(toast, "Unloaded 1 model(s); 1 still generating - try again once finished");
});

test("unload-all toast counts a stopped native image worker in unloaded_other", async () => {
  const msg = await clickUnloadAll({
    status: "unloaded", model: "model-a", unloaded_models: ["model-a"],
    unloaded_other: ["native image/video worker"], embedder_unloaded: false,
  });
  assert.match(msg, /Unloaded 2 model/, `1 chat model + the image worker (got: ${msg})`);
});

test("unload-all reports a busy media runtime and does not offer to force it", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeForceFlowFetch({
    status: "in_use", model: "none", unloaded_models: [], unloaded_other: [],
    embedder_unloaded: false, skipped_in_use: ["native image/video worker"],
    media_left_running: { "native image/video worker": "busy" },
  }, {}, calls) });
  const doc = window.document;
  doc.getElementById("models-unload-all-btn").click();
  await flush();
  assert.equal(doc.getElementById("toast").textContent,
    "1 model(s) still generating - try again once finished");
  assert.equal(doc.querySelector("#modal-body .btn-danger"), null,
    "force cannot stop a media job, so no unload-anyway confirm is shown");
  assert.deepEqual(calls, [{}]);
});

test("unload-all says a media runtime could not be stopped, not that it is generating", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeForceFlowFetch({
    status: "in_use", model: "none", unloaded_models: [], unloaded_other: [],
    embedder_unloaded: false, skipped_in_use: ["native image/video worker"],
    media_left_running: { "native image/video worker": "failed" },
  }, {}, calls) });
  const doc = window.document;
  doc.getElementById("models-unload-all-btn").click();
  await flush();
  const msg = doc.getElementById("toast").textContent;
  assert.equal(msg, "Could not stop the native image/video worker; see the server log");
  assert.equal(doc.querySelector("#modal-body .btn-danger"), null);
  assert.deepEqual(calls, [{}]);
});

test("unload-all offers force only for the chat model beside a busy media runtime", async () => {
  const { calls } = await unloadAllThenConfirm({
    status: "in_use", model: "none", unloaded_models: [], unloaded_other: [],
    embedder_unloaded: false, skipped_in_use: ["chat-a", "native music runtime"],
    confirm_required: { "chat-a": "1 other active request" },
    media_left_running: { "native music runtime": "busy" },
  }, { status: "unloaded", model: "chat-a", unloaded_models: ["chat-a"] });
  assert.deepEqual(calls, [{}, { force: true }]);
});
