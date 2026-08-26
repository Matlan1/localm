// SPDX-License-Identifier: AGPL-3.0-or-later
// Drives the real switchModel/onchange -> sendChat sequence.
import { test } from "node:test";
import assert from "node:assert/strict";

import { loadApp, runScript } from "./harness.mjs";

// a fetch stub that answers the model load and records every call
function _stub(window, { loadedName = "my-model" } = {}) {
  const calls = [];
  window.fetch = (url, opts) => {
    const u = String(url);
    calls.push(u);
    if (u.includes("/api/models/load")) {
      return Promise.resolve({
        ok: true, status: 200,
        json: async () => ({ status: "loaded", model: loadedName }),
        text: async () => "",
      });
    }
    return Promise.resolve({
      ok: true, status: 200, body: null,
      json: async () => ({}), text: async () => "",
    });
  };
  return calls;
}

const activeOf = (window) => {
  runScript(window, "window.__active = modelCache.active;");
  return window.__active;
};

test("switchModel updates modelCache.active as soon as the load lands", async () => {
  const { window } = loadApp();
  runScript(window, "modelCache.active = '';");     // fresh start / gui --no-model
  _stub(window);

  assert.equal(activeOf(window), "", "precondition: no model loaded");
  await window.switchModel("my-model");

  assert.equal(activeOf(window), "my-model",
    "a successful load must publish the active model, not wait for the 30s poll");
});

test("switchModel records the model the SERVER reports, not the requested name", async () => {
  const { window } = loadApp();
  runScript(window, "modelCache.active = '';");
  _stub(window, { loadedName: "resolved-alias" });

  await window.switchModel("some-alias");

  assert.equal(activeOf(window), "resolved-alias");
});

test("a superseded load does NOT claim to be the active model", async () => {
  // another model was selected while this one loaded, so the server aborted it
  const { window } = loadApp();
  runScript(window, "modelCache.active = '';");
  window.fetch = () => Promise.resolve({
    ok: true, status: 200,
    json: async () => ({ status: "superseded", model: "abandoned" }),
    text: async () => "",
  });

  await window.switchModel("abandoned");

  assert.equal(activeOf(window), "",
    "a superseded load must not publish itself as active");
});

test("picking a model in the sidebar lets the very next chat send through", async () => {
  // no model -> pick one in the sidebar -> send immediately, with no
  // refreshModels() poll in between
  const { window } = loadApp();
  runScript(window, "modelCache.active = '';");
  const calls = _stub(window);

  const select = window.document.getElementById("model-select");
  const opt = window.document.createElement("option");
  opt.value = "my-model";
  select.appendChild(opt);
  select.value = "my-model";
  runScript(window, "window.__onchange = modelSelect.onchange;");
  await window.__onchange();                     // the real sidebar handler

  window.document.getElementById("chat-input").value = "hello there";
  await window.sendChat().catch(() => {});       // stubbed stream may throw

  assert.ok(calls.some((u) => u.includes("/chat/completions")),
    "chat must not be blocked right after the sidebar loaded a model");
});
