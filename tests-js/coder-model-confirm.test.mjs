// SPDX-License-Identifier: AGPL-3.0-or-later
// A coder session start or model switch whose model load needs confirmation
// answers 409 with detail {status: "confirm_required"}. The coder page asks
// with the model picker's dialog and re-posts with force only on consent.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const CONFIRM_409 = [409, {
  detail: { status: "confirm_required", model: "model-b",
            detail: "loading needs to free model-a" },
}];

function reply([status, data]) {
  return { ok: status < 400, status, json: async () => data, text: async () => "" };
}

function makeFetch(calls, { create = [], model = [] } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    if (u === "/api/coder/sessions" && method === "POST") {
      calls.push({ url: u, body: JSON.parse(opts.body) });
      return reply(create.shift());
    }
    if (/^\/api\/coder\/sessions\/[^/]+\/model$/.test(u) && method === "POST") {
      calls.push({ url: u, body: JSON.parse(opts.body) });
      return reply(model.shift());
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

function tick() { return new Promise((r) => setTimeout(r, 0)); }

test("start: a confirm_required 409 asks, and a confirmed retry carries force", async () => {
  const calls = [];
  const asked = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, {
    create: [CONFIRM_409, [200, { id: "s1", cwd: "/tmp/project", model: "model-b", notes: [] }]],
  }) });
  win.confirmDangerAsync = async (title, message) => { asked.push(message); return true; };
  win.document.getElementById("setup-cwd").value = "/tmp/project";

  await win.startCoderSession({ model: "model-b" });
  await tick();

  assert.deepEqual(asked, ["loading needs to free model-a"], "the server's detail is what the user is asked");
  assert.equal(calls.length, 2, "one retry after consent");
  assert.equal(calls[0].body.force, undefined, "the first post does not force");
  assert.equal(calls[1].body.force, true, "the retry forces");
  assert.equal(calls[1].body.model, "model-b");
  assert.doesNotMatch(win.document.getElementById("toast").textContent, /Failed to start session/);
});

test("start: declining the confirm posts nothing more and shows no failure", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { create: [CONFIRM_409] }) });
  win.confirmDangerAsync = async () => false;
  win.document.getElementById("setup-cwd").value = "/tmp/project";

  await win.startCoderSession({ model: "model-b" });
  await tick();

  assert.equal(calls.length, 1, "no forced retry without consent");
  assert.doesNotMatch(win.document.getElementById("toast").textContent, /Failed to start session/);
  assert.equal(win.document.getElementById("setup-start").disabled, false, "the start button is usable again");
});

test("start: an ordinary error detail is still shown as text", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, {
    create: [[503, { detail: "Model load was superseded by a newer request: model-c" }]],
  }) });
  win.confirmDangerAsync = async () => assert.fail("only confirm_required asks");
  win.document.getElementById("setup-cwd").value = "/tmp/project";

  await win.startCoderSession({ model: "model-b" });
  await tick();

  assert.equal(calls.length, 1);
  assert.match(win.document.getElementById("toast").textContent,
    /Failed to start session: Model load was superseded by a newer request: model-c/);
});

test("model switch: a confirm_required 409 asks, and a confirmed retry carries force", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, {
    model: [CONFIRM_409, [200, { id: "s1", model: "model-b" }]],
  }) });
  win.confirmDangerAsync = async () => true;

  const updated = await win.postSessionModel("s1", "model-b");

  assert.deepEqual(calls.map((c) => c.body),
    [{ model: "model-b", pin: true }, { model: "model-b", pin: true, force: true }]);
  assert.equal(updated.model, "model-b");
});

test("model switch: declining the confirm returns null and posts nothing more", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { model: [CONFIRM_409] }) });
  win.confirmDangerAsync = async () => false;

  const updated = await win.postSessionModel("s1", "model-b");

  assert.equal(updated, null);
  assert.equal(calls.length, 1);
});

// A running session s1 for /p/alpha on model-a, made the active one.
function seedSession(win) {
  runScript(win, `
    coder.sessions.set("s1", { info: { id: "s1", cwd: "/p/alpha", model: "model-a",
                                       total_tokens: 0, turns: 0, patch_mode: false,
                                       backend_info: { backend: "local" } },
                               busy: false, feedEl: document.createElement("div") });
    activateSession("s1");
  `);
}

function sessionInfo(win) {
  runScript(win, `window.__s1info = coder.sessions.get("s1").info;`);
  return win.__s1info;
}

test("session controls: declining the confirm keeps the session as it was, with no error", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { model: [CONFIRM_409] }) });
  seedSession(win);
  win.confirmDangerAsync = async () => false;

  await win.switchActiveSessionModel("model-b");
  await tick();

  const info = sessionInfo(win);
  assert.ok(info, "the session keeps its info");
  assert.equal(info.model, "model-a");
  assert.equal(calls.length, 1);
  assert.doesNotMatch(win.document.getElementById("toast").textContent,
    /Could not switch model|Model switched/);
});

test("resume onto an open session: declining the model confirm keeps it as it was, with no error", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { model: [CONFIRM_409] }) });
  seedSession(win);
  win.confirmDangerAsync = async () => false;
  win.document.getElementById("setup-cwd").value = "/p/alpha";

  await win.startCoderSession({ resume: true, model: "model-b" });
  for (let i = 0; i < 5; i++) await tick();

  const info = sessionInfo(win);
  assert.ok(info, "the session keeps its info");
  assert.equal(info.model, "model-a");
  assert.deepEqual(calls.map((c) => c.body), [{ model: "model-b", pin: true }]);
  assert.doesNotMatch(win.document.getElementById("toast").textContent, /Could not switch model/);
});

test("model switch: a busy 409 rejects with the server's text", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, {
    model: [[409, { detail: "Session is busy; cannot switch models mid-task" }]],
  }) });
  win.confirmDangerAsync = async () => assert.fail("only confirm_required asks");

  await assert.rejects(win.postSessionModel("s1", "model-b"),
    /Session is busy; cannot switch models mid-task/);
  assert.equal(calls.length, 1);
});
