// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the key sign-in screen and the model sidebar: a failed
// sign-in says why and stays on the screen, a 403 on the model list names its
// real cause, a model being switched to counts as loading, "load failed" and a
// peer route stay shown, and a server that stops shows the reconnect screen.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, timeout = 2000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await settle(15); }
  return false;
}
const res = (status, body, headers = {}) => ({
  ok: status >= 200 && status < 300, status,
  headers: { get: (k) => headers[k] ?? null },
  json: async () => body, text: async () => JSON.stringify(body),
});
const pill = (win) => win.document.getElementById("status-text").textContent;

function gateWith(fetchImpl) {
  const { window } = loadApp({ fetchImpl });
  let reloads = 0;
  try { Object.defineProperty(window.location, "reload", { configurable: true, value: () => { reloads++; } }); }
  catch { /* jsdom */ }
  window.showKeyGate("This LocaLM server requires an API key.");
  return { window, reloads: () => reloads };
}

for (const [label, answer, expected] of [
  ["a wrong key", () => res(401, { detail: "Invalid API key" }), /was not accepted/],
  ["an unreachable server", () => Promise.reject(new TypeError("Failed to fetch")), /Could not reach the LocaLM server/],
  ["an open-mode server", () => res(400, { detail: "This server runs in open mode (no API key configured); there is nothing to log into." }), /open mode/],
]) {
  test(`the sign-in screen stays and says why for ${label}`, async () => {
    const { window, reloads } = gateWith(async (url) =>
      (String(url).includes("/api/session") ? answer() : res(200, { models: [], active: "" })));
    window.document.getElementById("key-gate-input").value = "typo";
    await window.submitKeyGate();
    const line = window.document.getElementById("key-gate-error");
    assert.ok(line && !line.hidden, "an error line is shown");
    assert.match(line.textContent, expected);
    assert.equal(reloads(), 0, "the page does not reload into the same screen");
  });
}

test("an empty sign-in asks for a key and sends nothing", async () => {
  const calls = [];
  const { window } = gateWith(async (url) => { calls.push(String(url)); return res(200, {}); });
  calls.length = 0;
  window.document.getElementById("key-gate-input").value = "  ";
  await window.submitKeyGate();
  assert.equal(calls.filter((u) => u.includes("/api/session")).length, 0);
  assert.match(window.document.getElementById("key-gate-error").textContent, /Enter your API key/);
});

test("loginWithKeyResult returns the status and the server's reason", async () => {
  const { window } = gateWith(async (url) =>
    (String(url).includes("/api/session") ? res(400, { detail: "nothing to log into" }) : res(200, {})));
  const r = await window.loginWithKeyResult("k");
  assert.deepEqual({ ok: r.ok, status: r.status, detail: r.detail },
    { ok: false, status: 400, detail: "nothing to log into" });
  assert.equal(await window.loginWithKey("k"), false);
});

test("showing the sign-in screen again hides a revealed key", () => {
  const { window } = gateWith(async () => res(200, {}));
  const input = window.document.getElementById("key-gate-input");
  input.type = "text";
  window.showKeyGate("again");
  assert.equal(input.type, "password");
});

for (const [label, headers, shell, expected] of [
  ["a scope refusal", {}, "tok", "this key cannot list models"],
  ["a stale shell token", { "X-Localm-Shell-Rejected": "1" }, "tok", "page out of date - reload to reconnect"],
  ["a page served no shell token", { "X-Localm-Shell-Rejected": "1" }, "", "this server needs an API key to manage models from here"],
]) {
  test(`a 403 on the model list for ${label} says so`, async () => {
    const fetchImpl = async (url) => (String(url).includes("/api/models")
      ? res(403, { detail: "refused" }, headers) : res(200, {}));
    const { window } = loadApp({ fetchImpl, shellToken: shell || undefined });
    await window.refreshModels();
    assert.equal(pill(window), expected);
  });
}

test("a 401 after the model list was read once says the sign-in ended", async () => {
  let deny = false;
  const fetchImpl = async (url) => (String(url).includes("/api/models")
    ? (deny ? res(401, { detail: "Invalid or missing API key" }) : res(200, { models: [], active: "" }))
    : res(200, {}));
  const { window } = loadApp({ fetchImpl });
  await window.refreshModels();
  deny = true;
  await window.refreshModels();
  assert.match(window.document.getElementById("key-gate-msg").textContent, /revoked or may have expired/);
});

test("a model being switched to counts as loading at once, also through a poll", async () => {
  let release;
  const gate = new Promise((r) => { release = r; });
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u.includes("/api/models/load")) { await gate; return res(200, { status: "loaded", model: "m2" }); }
    if (u.includes("/api/models")) return res(200, { models: [{ name: "m1" }, { name: "m2" }], active: "m1" });
    return res(200, {});
  };
  const { window } = loadApp({ fetchImpl });
  const p = window.switchModel("m2", { local: true });
  await settle(20);
  runScript(window, "globalThis.__loading = modelCache.loading;");
  assert.equal(window.__loading, "m2");
  await window.refreshModels();
  runScript(window, "globalThis.__loading = modelCache.loading;");
  assert.equal(window.__loading, "m2", "a poll during the switch keeps the loading mark");
  release();
  await p;
  runScript(window, "globalThis.__loading = modelCache.loading;");
  assert.ok(!window.__loading, "the mark ends with the load");
});

test("load failed stays shown through the re-read that follows it", async () => {
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u.includes("/api/models/load")) return res(500, { detail: "boom" });
    if (u.includes("/api/models")) return res(200, { models: [{ name: "m1" }], active: "" });
    return res(200, {});
  };
  const { window } = loadApp({ fetchImpl });
  await window.switchModel("m1", { local: true }).catch(() => {});
  await settle(30);
  await window.refreshModels();
  assert.equal(pill(window), "load failed");
  assert.equal(window.document.getElementById("model-status").hidden, false);
});

test("a server that stops while the page is open shows the reconnect screen", async () => {
  let down = false;
  const fetchImpl = async (url) => {
    if (down) throw new TypeError("Failed to fetch");
    return res(200, { models: [], active: "" });
  };
  const { window } = loadApp({ fetchImpl });
  let overlay = 0;
  window.serverReachable = async () => false;
  window.onServerUnreachable = () => { overlay++; };
  down = true;
  await window.refreshModels();
  assert.equal(overlay, 1);
  assert.equal(pill(window), "server unreachable");
});
