// SPDX-License-Identifier: AGPL-3.0-or-later
// A page loaded while the server is still loading a model learns the load is in
// flight from /api/models ("loading") and re-polls every 2 s until it finishes,
// instead of waiting for the 30 s poll with "No model loaded".
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function makeFetch(state) {
  return async (url) => {
    const u = String(url);
    if (u.startsWith("/api/models?type=llm")) {
      state.modelCalls += 1;
      const body = { models: [{ name: "model-a", size_bytes: 1000, active: state.active === "model-a" }],
                     active: state.active };
      if (state.loading) body.loading = state.loading;
      return { ok: true, status: 200, json: async () => body, text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

const tick = (ms) => new Promise((r) => setTimeout(r, ms));

test("a model load in flight on the server shows as loading and is re-polled within seconds", async () => {
  const state = { active: "", loading: "model-a", modelCalls: 0 };
  const { window: win } = loadApp({ fetchImpl: makeFetch(state) });
  runScript(win, "refreshModels();");
  await tick(50);
  const pill = win.document.getElementById("status-text");
  assert.match(pill.textContent, /loading model-a/);
  assert.equal(win.document.getElementById("model-status").hidden, false);

  const before = state.modelCalls;
  state.active = "model-a";
  state.loading = "";
  await tick(2600);
  assert.ok(state.modelCalls > before, "the page polled again without waiting for the 30 s interval");
  assert.doesNotMatch(pill.textContent, /loading/, "the loading state is cleared once the load ends");
  assert.equal(win.document.getElementById("model-select").value, "model-a");
  runScript(win, "window.__active = modelCache.active;");
  assert.equal(win.__active, "model-a");
});

test("sendChat tells the user the model is still loading rather than that none is loaded", async () => {
  const state = { active: "", loading: "model-a", modelCalls: 0 };
  const { window: win } = loadApp({ fetchImpl: makeFetch(state) });
  runScript(win, "refreshModels();");
  await tick(50);
  win.document.getElementById("chat-input").value = "hello";
  await win.sendChat();
  assert.match(win.document.getElementById("toast").textContent, /model-a is still loading/);
});
