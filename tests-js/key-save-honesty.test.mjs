// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

// Settings -> API key (workflow.js's gui-key-save handler), over
// POST /api/session (login) and POST /api/session/logout (sign out).

const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms));

function setup(routes) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const method = (opts.method || "GET").toUpperCase();
    const path = String(url).replace(/^https?:\/\/[^/]+/, "");
    const key = `${method} ${path}`;
    calls.push(key);
    const route = routes[key];
    if (route) {
      const res = route(opts);
      return { ok: res.status < 400, status: res.status,
               json: async () => res.body || {}, text: async () => res.text || "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  try {
    Object.defineProperty(window.location, "reload", { configurable: true, value: () => {} });
  } catch { /* jsdom no-op nav */ }
  return { window, calls };
}

// Records every setTimeout delay scheduled after this point.
function armReloadTimerSpy(window) {
  const delays = [];
  const realSetTimeout = window.setTimeout;
  window.setTimeout = (fn, ms, ...rest) => { delays.push(ms); return realSetTimeout(fn, ms, ...rest); };
  return delays;
}

async function keySave(routes, keyValue) {
  const { window, calls } = setup(routes);
  await tick(0);
  const toasts = [];
  window.toast = (msg, isError) => toasts.push({ msg: String(msg), isError: !!isError });
  await tick(50);
  const delays = armReloadTimerSpy(window);

  window.document.getElementById("gui-api-key").value = keyValue;
  window.document.getElementById("gui-key-save").click();
  await tick(50);

  return { calls, toasts, reloadScheduled: delays.filter((d) => d === 600) };
}

test("key save: an accepted key is reported saved and reloads (control)", async () => {
  const { calls, toasts, reloadScheduled } = await keySave({
    "POST /api/session": () => ({ status: 200, body: { csrf: "tok" } }),
  }, "a-good-key");

  assert.ok(calls.includes("POST /api/session"), "the login call never went out");
  assert.ok(toasts.some((t) => /key saved/i.test(t.msg) && !t.isError),
    `expected a non-error 'key saved' toast, got: ${JSON.stringify(toasts)}`);
  assert.deepEqual(reloadScheduled, [600], "an accepted key must still reload");
});

test("key save: a rejected key must not be reported as saved", async () => {
  const { calls, toasts, reloadScheduled } = await keySave({
    "POST /api/session": () => ({ status: 401, body: { detail: "bad key" } }),
  }, "a-wrong-key");

  assert.ok(calls.includes("POST /api/session"), "the login call never went out");
  assert.ok(!toasts.some((t) => /key saved/i.test(t.msg)),
    `a rejected key must not be reported as saved, got: ${JSON.stringify(toasts)}`);
  assert.ok(toasts.some((t) => t.isError),
    `a rejected key must produce an error toast, got: ${JSON.stringify(toasts)}`);
  assert.deepEqual(reloadScheduled, [], "a rejected key must not reload");
});

test("key save: signing out is not reported as saving a key", async () => {
  const { calls, toasts } = await keySave({
    "POST /api/session/logout": () => ({ status: 200, body: {} }),
  }, "");

  assert.ok(calls.includes("POST /api/session/logout"), "the sign-out call never went out");
  assert.ok(!toasts.some((t) => /key saved/i.test(t.msg)),
    `signing out must not be reported as saving a key, got: ${JSON.stringify(toasts)}`);
});

test("key save: a failed sign out is reported as an error, not silently discarded", async () => {
  const { calls, toasts, reloadScheduled } = await keySave({
    "POST /api/session/logout": () => ({ status: 500, body: {} }),
  }, "");

  assert.ok(calls.includes("POST /api/session/logout"), "the sign-out call never went out");
  assert.ok(toasts.some((t) => t.isError),
    `a failed sign out must produce an error toast, got: ${JSON.stringify(toasts)}`);
  assert.deepEqual(reloadScheduled, [], "a failed sign out must not reload");
});
