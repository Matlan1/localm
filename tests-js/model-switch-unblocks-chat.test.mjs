// SPDX-License-Identifier: AGPL-3.0-or-later
// Drives the real switchModel/onchange -> sendChat sequence.
import { test } from "node:test";
import assert from "node:assert/strict";

import { loadApp, runScript } from "./harness.mjs";

// a fetch stub that answers the model load and records every call
function _stub(window, { loadedName = "my-model" } = {}) {
  const calls = [];
  window.fetch = (url, _opts) => {
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

// switch_engine (http_server.py) answers HTTP 200 with status:"confirm_required"
// when something is in the way (a busy peer that will not clear, or a load
// that would degrade to CPU offload) and the caller did not pass force. That
// is resolved inside switchModel itself via confirmDangerAsync - never
// returned to the caller as a final outcome.
function _confirmRequiredStub(window, calls) {
  window.fetch = (url, opts = {}) => {
    const u = String(url);
    calls.push(u);
    if (u.includes("/api/models/load")) {
      const body = opts.body ? JSON.parse(opts.body) : {};
      if (!body.force) {
        return Promise.resolve({
          ok: true, status: 200,
          json: async () => ({
            status: "confirm_required", model: "busy-model",
            detail: "'other-model' is still generating",
          }),
          text: async () => "",
        });
      }
      return Promise.resolve({
        ok: true, status: 200,
        json: async () => ({ status: "loaded", model: "busy-model" }),
        text: async () => "",
      });
    }
    return Promise.resolve({ ok: true, status: 200, json: async () => ({}), text: async () => "" });
  };
}

test("declining a confirm_required switch does not publish the new model as active", async () => {
  const { window } = loadApp();
  runScript(window, "modelCache.active = 'other-model';");
  const calls = [];
  _confirmRequiredStub(window, calls);
  runScript(window, "confirmDangerAsync = async () => false;");   // decline

  const res = await window.switchModel("busy-model");

  assert.equal(calls.filter((u) => u.includes("/api/models/load")).length, 1,
    "declining must not retry with force");
  assert.equal(res.status, "cancelled", "a declined confirm reports as cancelled, not loaded");
  assert.equal(activeOf(window), "other-model",
    "a declined switch must not publish the requested model as active");
});

test("confirming a confirm_required switch retries with force and publishes the new active model", async () => {
  const { window } = loadApp();
  runScript(window, "modelCache.active = 'other-model';");
  const calls = [];
  _confirmRequiredStub(window, calls);
  runScript(window, "confirmDangerAsync = async () => true;");   // confirm

  const res = await window.switchModel("busy-model");

  const loadCalls = calls.filter((u) => u.includes("/api/models/load"));
  assert.equal(loadCalls.length, 2, "confirming retries exactly once, with force");
  assert.equal(res.status, "loaded");
  assert.equal(activeOf(window), "busy-model",
    "confirming and forcing through must publish the new model as active");
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
