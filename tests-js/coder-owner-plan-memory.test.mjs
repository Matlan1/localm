// SPDX-License-Identifier: AGPL-3.0-or-later
// Coder page: a non-owner key is told lessons and hooks are the owner's, not
// that there are none; remembering a note that is already there says so; and
// the session controls switch plan mode on and off.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function reply(status, data) {
  return { ok: status < 400, status, json: async () => data, text: async () => "" };
}

function makeFetch(calls, routes) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    calls.push({ url: u, method, body: opts.body ? JSON.parse(opts.body) : null });
    for (const [match, fn] of routes) {
      if (match(u, method)) return fn(opts.body ? JSON.parse(opts.body) : null);
    }
    return reply(200, {});
  };
}

function modalText(win) {
  return win.document.getElementById("modal-body").textContent;
}

function buttons(win) {
  return [...win.document.querySelectorAll("#modal-body button")].map((b) => b.textContent.trim());
}

function seedSession(win, info) {
  runScript(win, `
    coder.sessions.set("s1", { info: ${JSON.stringify(info)}, busy: false,
                               feedEl: document.createElement("div"), pendingCards: [] });
    activateSession("s1");
  `);
}

const settle = () => new Promise((r) => setTimeout(r, 0));

test("lessons: a non-owner key is told they are the owner's, with nothing to act on", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, [
    [(u) => u.startsWith("/api/coder/episodes?"),
     () => reply(200, { episodes: [], cwd: null, authorized: false })],
  ]) });
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.openEpisodesModal();
  const text = modalText(win);
  assert.match(text, /Lessons are shown only to the server owner/);
  assert.doesNotMatch(text, /No lessons stored/);
  assert.deepEqual(buttons(win).filter((b) => /consolidate|erase/i.test(b)), [],
    JSON.stringify(buttons(win)));
});

test("lessons: an owner with none stored still sees the empty state and the actions", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, [
    [(u) => u.startsWith("/api/coder/episodes?"),
     () => reply(200, { episodes: [], cwd: "/tmp/project" })],
  ]) });
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.openEpisodesModal();
  const text = modalText(win);
  assert.match(text, /No lessons stored for this project yet/);
  assert.doesNotMatch(text, /only to the server owner/);
  assert.ok(buttons(win).length > 2, JSON.stringify(buttons(win)));
});

test("hooks: a non-owner key is told they are the owner's, not that none exist", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, [
    [(u) => u.startsWith("/api/coder/hooks?"),
     () => reply(200, { configured: false, cwd: null, authorized: false })],
  ]) });
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.openHooksModal();
  const text = modalText(win);
  assert.match(text, /Hooks are shown only to the server owner/);
  assert.doesNotMatch(text, /No tool hooks or MCP servers/);
});

test("memory: a note already in the file is said to be there, a new one is remembered", async () => {
  const calls = [];
  let added = false;
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, [
    [(u, m) => u.endsWith("/memory") && m === "POST",
     () => reply(200, { exists: true, text: "- use tabs", added })],
    [(u) => u.endsWith("/memory"), () => reply(200, { exists: true, text: "- use tabs" })],
  ]) });
  const toasts = [];
  win.toast = (msg) => toasts.push(String(msg));
  seedSession(win, { id: "s1", cwd: "/tmp/project" });
  await win.openCoderMemoryModal();
  win.document.getElementById("mem-add").value = "use tabs";
  const btn = [...win.document.querySelectorAll("#modal-body button")]
    .find((b) => b.textContent.trim() === "remember");
  await btn.onclick();
  await settle();
  added = true;
  await win.openCoderMemoryModal();
  win.document.getElementById("mem-add").value = "use spaces";
  await [...win.document.querySelectorAll("#modal-body button")]
    .find((b) => b.textContent.trim() === "remember").onclick();
  await settle();
  assert.deepEqual(toasts, [
    "Already remembered - the project memory has that note",
    "Remembered - the agent has it from the next turn",
  ]);
});

test("controls: plan mode is switched on and off with the state the server reports", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, [
    [(u, m) => u.endsWith("/settings") && m === "POST",
     (body) => reply(200, { id: "s1", cwd: "/tmp/project", plan_mode: body.plan_mode,
                            backend_info: { backend: "local" } })],
  ]) });
  const toasts = [];
  win.toast = (msg) => toasts.push(String(msg));
  seedSession(win, { id: "s1", cwd: "/tmp/project", plan_mode: false,
                     backend_info: { backend: "local" } });
  win.openSessionControls();
  const box = win.document.getElementById("ctl-plan-mode");
  assert.ok(box, "the plan mode switch renders");
  assert.equal(box.checked, false);
  box.checked = true;
  await box.onchange();
  assert.equal(box.checked, true);
  box.checked = false;
  await box.onchange();
  const posts = calls.filter((c) => c.url.endsWith("/settings")).map((c) => c.body);
  assert.deepEqual(JSON.parse(JSON.stringify(posts)), [{ plan_mode: true }, { plan_mode: false }]);
  assert.deepEqual(toasts, [
    "Plan mode on: the agent reads and searches until you approve its plan",
    "Plan mode off",
  ]);
});

test("controls: no plan mode switch when the server does not report plan mode", () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, []) });
  seedSession(win, { id: "s1", cwd: "/tmp/project", backend_info: { backend: "local" } });
  win.openSessionControls();
  assert.equal(win.document.getElementById("ctl-plan-mode"), null);
});
