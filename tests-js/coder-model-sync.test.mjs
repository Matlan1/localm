// SPDX-License-Identifier: AGPL-3.0-or-later
// A sidebar model switch repoints every local-engine coder session (a busy one
// once its task ends), never a session on another backend, and a model switch
// made from the coder refreshes the sidebar when the session shares the engine.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const MODEL_URL = /^\/api\/coder\/sessions\/([^/]+)\/model$/;

function setup(sessions, { busyOnce = new Set() } = {}) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    let body = null;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = opts.body; }
    calls.push({ url: u, body });
    const m = MODEL_URL.exec(u);
    if (m) {
      if (busyOnce.has(m[1])) {
        busyOnce.delete(m[1]);
        return { ok: false, status: 409, statusText: "Conflict",
                 json: async () => ({ detail: "Session is busy; cannot switch models mid-task" }) };
      }
      const info = sessions.find((s) => s.info.id === m[1]).info;
      return { ok: true, status: 200,
               json: async () => ({ ...info, model: body.model, model_pinned: body.pin }) };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };
  const { window } = loadApp({ fetchImpl });
  window.__sessions = sessions;
  runScript(window, `
    for (const s of window.__sessions) {
      s.feedEl = document.createElement("div");
      coder.sessions.set(s.info.id, s);
    }
    coder.activeId = window.__sessions[0].info.id;
  `);
  return { window, calls };
}

async function flush() {
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 10));
}

const posts = (calls) => calls.filter((c) => MODEL_URL.test(c.url));
const refreshes = (calls) => calls.filter((c) => c.url.startsWith("/api/models"));
const local = (id, extra = {}) => ({
  info: { id, model: "X", model_pinned: false, backend_info: { backend: "local" } },
  busy: false, ...extra,
});
const url = (id) => ({
  info: { id, model: "qwen2.5-coder:7b", model_pinned: false,
          backend_info: { backend: "url", leaves_machine: false,
                          target: "http://localhost:11434/v1" } },
  busy: false,
});

function emit(window, model) {
  window.dispatchEvent(new window.CustomEvent("localm:model-switched", { detail: { model } }));
}

test("a loopback url-backend session is not repointed by the sidebar", async () => {
  const { window, calls } = setup([url("u1")]);
  emit(window, "gemma3-4b");
  await flush();
  assert.equal(posts(calls).length, 0);
});

test("session controls offer a free-text model for a loopback url-backend session", () => {
  const { window } = setup([url("u1")]);
  window.openSessionControls();
  const input = window.document.getElementById("ctl-model");
  assert.equal(input.tagName, "INPUT");
  assert.equal(input.value, "qwen2.5-coder:7b");
});

test("session controls offer a model list for a local session", () => {
  const { window } = setup([local("a")]);
  window.openSessionControls();
  assert.equal(window.document.getElementById("ctl-model").tagName, "SELECT");
});

test("every idle local session follows the sidebar; a busy one follows when its task ends", async () => {
  const a = local("a");
  const b = local("b", { busy: true });
  const { window, calls } = setup([a, b]);
  emit(window, "Y");
  await flush();
  assert.deepEqual(posts(calls).map((c) => c.url), ["/api/coder/sessions/a/model"]);
  assert.equal(posts(calls)[0].body.pin, false);

  runScript(window, `handleCoderEvent(coder.sessions.get("b"),
    { type: "final", ok: true, turns: 1, total_tokens: 1, changed_files: [] });`);
  await flush();
  assert.deepEqual(posts(calls).map((c) => c.url),
    ["/api/coder/sessions/a/model", "/api/coder/sessions/b/model"]);
  assert.equal(posts(calls)[1].body.model, "Y");
  assert.equal(posts(calls)[1].body.pin, false);
});

test("a pending follow is retried when the server still reports the session busy", async () => {
  const b = local("b", { busy: true });
  const { window, calls } = setup([b], { busyOnce: new Set(["b"]) });
  emit(window, "Y");
  runScript(window, `handleCoderEvent(coder.sessions.get("b"),
    { type: "final", ok: true, turns: 1, total_tokens: 1, changed_files: [] });`);
  for (let i = 0; i < 20; i++) await new Promise((r) => setTimeout(r, 40));
  assert.equal(posts(calls).length, 2, "one refused as busy, one applied");
  assert.equal(window.__sessions[0].info.model, "Y");
});

test("a pinned session never follows, even after its task ends", async () => {
  const b = local("b", { busy: true });
  b.info.model_pinned = true;
  const { window, calls } = setup([b]);
  emit(window, "Y");
  runScript(window, `handleCoderEvent(coder.sessions.get("b"),
    { type: "final", ok: true, turns: 1, total_tokens: 1, changed_files: [] });`);
  await flush();
  assert.equal(posts(calls).length, 0);
});

test("switching a local session's model from the coder refreshes the model sidebar", async () => {
  const { window, calls } = setup([local("a")]);
  await window.switchActiveSessionModel("Z");
  assert.equal(posts(calls).length, 1);
  assert.ok(refreshes(calls).length >= 1, "the sidebar model list was re-read");
});

test("switching a url-backend session's model does not refresh the model sidebar", async () => {
  const { window, calls } = setup([url("u1")]);
  await window.switchActiveSessionModel("llama3");
  assert.equal(posts(calls).length, 1);
  assert.equal(refreshes(calls).length, 0);
});
