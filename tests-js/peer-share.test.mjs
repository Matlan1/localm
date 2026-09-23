// SPDX-License-Identifier: AGPL-3.0-or-later
// Using a model another localm instance on this machine already has loaded:
// a peer in open mode is used without asking for a key it does not have, a
// keyed peer's key is asked for and sent, and the chosen model stays selected
// across the sidebar's periodic model refresh while the route holds.
import { test } from "node:test";
import assert from "node:assert/strict";

import { loadApp, runScript } from "./harness.mjs";

const PEER = { instance_id: "peer-1", host: "127.0.0.1", port: 9555, scheme: "http",
               model: "shared" };

function setup({ requiresKey, modelsPayload }) {
  const calls = [];
  const state = { models: modelsPayload };
  const impl = async (url, opts = {}) => {
    const u = String(url);
    let body;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = opts.body; }
    calls.push({ url: u, method: opts.method || "GET", body });
    const json = (obj) => ({ ok: true, status: 200, json: async () => obj, text: async () => "" });
    if (u === "/v1/models/shared/peer-offer") {
      return json({ available: true, peer: { ...PEER, requires_key: requiresKey } });
    }
    if (u === "/v1/models/shared/peer-route") {
      return json({ status: "routed", model: "shared", peer: PEER });
    }
    if (u.startsWith("/api/models")) return json(state.models);
    return json({});
  };
  const { window } = loadApp({ fetchImpl: impl });
  return { window, calls, state };
}

async function tick(n = 5) {
  for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 20));
}

function modalButton(doc, label) {
  return [...doc.querySelectorAll("#modal button")].find((b) => b.textContent === label);
}

test("an open-mode peer is used without asking for a key", async () => {
  const { window, calls } = setup({ requiresKey: false, modelsPayload: { models: [] } });
  const doc = window.document;
  const pending = window.switchModel("shared");
  await tick();
  assert.equal(doc.querySelector("#modal input[type=password]"), null,
    "no key field for a peer that has no key");
  modalButton(doc, "Use it").click();
  const res = await pending;
  assert.equal(res.status, "routed");
  const accept = calls.find((c) => c.url === "/v1/models/shared/peer-route");
  assert.ok(accept, "the route was accepted");
  assert.equal(accept.body.api_key, "");
  assert.equal(calls.filter((c) => c.url === "/api/models/load").length, 0,
    "no local copy was loaded");
});

test("a keyed peer asks for its key and sends it", async () => {
  const { window, calls } = setup({ requiresKey: true, modelsPayload: { models: [] } });
  const doc = window.document;
  const pending = window.switchModel("shared");
  await tick();
  const input = doc.querySelector("#modal input[type=password]");
  assert.ok(input, "a key field for a peer that needs its key");
  input.value = "peer-key";
  modalButton(doc, "Route").click();
  await pending;
  const accept = calls.find((c) => c.url === "/v1/models/shared/peer-route");
  assert.equal(accept.body.api_key, "peer-key");
});

test("the peer-routed model stays selected across the model refresh while routed", async () => {
  const withRoute = {
    models: [{ name: "local-one", active: true, loaded: true }, { name: "shared" }],
    active: "local-one",
    peer_routes: { shared: { host: "127.0.0.1", port: 9555, instance_id: "peer-1" } },
  };
  const { window, state } = setup({ requiresKey: false, modelsPayload: withRoute });
  const doc = window.document;
  const pending = window.switchModel("shared");
  await tick();
  modalButton(doc, "Use it").click();
  await pending;

  await window.refreshModels();
  assert.equal(doc.getElementById("model-select").value, "shared",
    "the poll must not snap the selection back to the local model");
  runScript(window, "window.__active = modelCache.active;");
  assert.equal(window.__active, "shared");

  state.models = { models: withRoute.models, active: "local-one" };
  await window.refreshModels();
  assert.equal(doc.getElementById("model-select").value, "local-one",
    "once the route is gone the local model is selected again");
});
