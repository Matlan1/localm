// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, timeout = 3000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await settle(15); }
  return false;
}

const DE = { "nav.settings": "Einstellungen" };

const ok = (obj) => ({
  ok: true, status: 200, statusText: "OK", headers: { get: () => null },
  json: async () => obj, text: async () => JSON.stringify(obj),
});
const refused = (status, detail) => ({
  ok: false, status, statusText: detail, headers: { get: () => null },
  json: async () => ({ detail }), text: async () => detail,
});

const isStatic = (u) => u.includes("/i18n/") || u.includes("/static/") || u.endsWith(".js");

function server(calls, state) {
  return async (url, opts = {}) => {
    const u = String(url);
    calls.push({ u, method: (opts.method || "GET").toUpperCase() });
    if (u.includes("/i18n/de.json")) return ok(DE);
    if (u.includes("/i18n/")) return refused(404, "no");
    if (state.keyed) return refused(401, "Invalid or missing API key");
    if (u === "/api/capabilities") {
      return ok({ core: { chat: true }, plugins: state.plugins });
    }
    if (u === "/api/prompts" && !state.plugins.some((p) => p.name === "chat")) {
      return refused(403, "Key lacks required scope: chat");
    }
    if (u.startsWith("/api/models")) return ok({ models: [], active: "" });
    if (u === "/api/prompts") return ok({ prompts: [] });
    return ok({});
  };
}

const promptReads = (calls) => calls.filter((c) => c.u === "/api/prompts").length;

test("U8: a remembered non-English language fires no authenticated read behind the key gate", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: server(calls, { keyed: true, plugins: [] }),
    seedLocalStorage: { "localm.language": "de" },
  });
  await waitFor(() => window.document.documentElement.lang === "de");
  await settle(300);
  const gate = window.document.getElementById("key-gate");
  assert.notEqual(gate.style.display, "none", "the key gate is shown");
  const reads = calls.filter((c) => !isStatic(c.u)).map((c) => c.u);
  assert.deepEqual(reads.filter((u) => !u.startsWith("/api/models")), [],
    "only the boot probe reached the server");
  assert.equal(reads.filter((u) => u.startsWith("/api/models")).length, 1,
    "the model list was read once (the probe)");
});

test("U8: a key without the chat plugin reads no persona list at load or on a language change", async () => {
  const calls = [];
  const state = { keyed: false, plugins: [{ name: "coder", active: true }] };
  const { window } = loadAppWithPages({ fetchImpl: server(calls, state) });
  await waitFor(() => calls.some((c) => c.u === "/api/capabilities"));
  await settle(200);
  assert.equal(promptReads(calls), 0, "no persona read at load");
  window.eval('applyLanguage("de")');
  await waitFor(() => window.document.documentElement.lang === "de");
  await settle(300);
  assert.equal(promptReads(calls), 0, "no persona read on a language change");
});

test("U8: a key with the chat plugin still reads the persona list at load and after a language change", async () => {
  const calls = [];
  const state = { keyed: false, plugins: [{ name: "chat", active: true }] };
  const { window } = loadAppWithPages({ fetchImpl: server(calls, state) });
  await waitFor(() => promptReads(calls) >= 1);
  await settle(200);
  const before = promptReads(calls);
  assert.ok(before >= 1, "personas read at load");
  window.eval('applyLanguage("de")');
  await waitFor(() => promptReads(calls) > before);
  assert.ok(promptReads(calls) > before, "personas re-read for the new language");
});

test("U8: a key created while the page is open in open mode says so at the gate", async () => {
  const calls = [];
  const state = { keyed: false, plugins: [{ name: "chat", active: true }] };
  const { window } = loadAppWithPages({
    fetchImpl: server(calls, state), shellToken: "shell-token-1",
  });
  await waitFor(() => promptReads(calls) >= 1);
  await settle(200);
  const gate = window.document.getElementById("key-gate");
  assert.equal(gate.style.display, "none", "no gate while the page is allowed in");
  state.keyed = true;
  await window.fetch("/v1/config/schema", { headers: { Authorization: "Bearer shell-token-1" } });
  await settle(50);
  assert.notEqual(gate.style.display, "none", "the key gate shows");
  assert.match(window.document.getElementById("key-gate-msg").textContent,
    /key was created for this server/i);
  assert.equal(window.document.getElementById("app").style.display, "none",
    "the app is hidden behind the gate");
});

test("U8: a 401 on the capabilities read after the boot probe keeps language reloads closed", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ u, method: (opts.method || "GET").toUpperCase() });
    if (u.includes("/i18n/de.json")) return ok(DE);
    if (u === "/api/capabilities") return refused(401, "Invalid or missing API key");
    if (u.startsWith("/api/models")) return ok({ models: [], active: "" });
    return ok({});
  };
  const { window } = loadAppWithPages({ fetchImpl, shellToken: "shell-token-1" });
  await waitFor(() => calls.some((c) => c.u === "/api/capabilities"));
  await settle(200);
  assert.notEqual(window.document.getElementById("key-gate").style.display, "none", "locked");
  const before = calls.length;
  window.eval('applyLanguage("de")');
  await waitFor(() => window.document.documentElement.lang === "de");
  await settle(300);
  const after = calls.slice(before).filter((c) => !isStatic(c.u));
  assert.deepEqual(after, [], "no read after the lock");
});

test("U8: a refused sign-in attempt at /api/session does not lock the page again", async () => {
  const calls = [];
  const state = { keyed: false, plugins: [{ name: "chat", active: true }] };
  const { window } = loadAppWithPages({ fetchImpl: server(calls, state) });
  await waitFor(() => promptReads(calls) >= 1);
  await settle(200);
  state.keyed = true;
  await window.fetch("/api/session", { method: "POST" });
  await settle(50);
  assert.equal(window.document.getElementById("key-gate").style.display, "none");
});
