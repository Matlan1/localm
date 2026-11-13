// SPDX-License-Identifier: AGPL-3.0-or-later
// A server restart on the same port gives the open-mode shell a new token. The
// page reads the new token from a fresh page shell once and recovers every
// refused call without reloading; a reload the page starts itself while Settings
// holds unsaved edits asks first in the page, never through the browser's prompt.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import vm from "node:vm";

import { loadApp, runScript } from "./harness.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const STATIC = join(ROOT, "localm", "plugins", "gui", "static");

const OLD = "OLD-SHELL-TOKEN";
const NEW = "NEW-SHELL-TOKEN";
const REJECTED = { get: (n) => (String(n).toLowerCase() === "x-localm-shell-rejected" ? "1" : null) };
const NO_HEADERS = { get: () => null };

const ok = (body = { models: [], active: "" }) => ({
  ok: true, status: 200, headers: NO_HEADERS, json: async () => body, text: async () => "",
});
const refused = () => ({
  ok: false, status: 403, headers: REJECTED,
  json: async () => ({ detail: "Open-mode management requires the localm GUI shell" }),
  text: async () => "",
});
const shellPage = (token) => ({
  ok: true, status: 200, headers: NO_HEADERS, json: async () => ({}),
  text: async () => '<!doctype html><html><head><script nonce="n">window.__LOCALM_SHELL_TOKEN__='
    + JSON.stringify(token) + ';</script><title>LocaLM</title></head><body></body></html>',
});

const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, timeout = 2000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await settle(15); }
  return false;
}

// Counts location.reload() calls: jsdom reports each as a not-implemented
// navigation on the window's virtual console.
function countReloads(dom) {
  const state = { reloads: 0 };
  dom.virtualConsole.on("jsdomError", (e) => {
    if (/navigation/i.test(String(e && e.message))) state.reloads += 1;
  });
  return state;
}

function recoveryStarted(window) {
  runScript(window, "window.__recoveryStarted = _shellRecoveryStarted;");
  return window.__recoveryStarted;
}

// A server that, once restarted, refuses OLD and accepts NEW; GET / serves the
// page shell with *shellToken()*.
function restartServer(shellToken = () => NEW) {
  const s = { restarted: false, shellReads: 0, refusedUrls: [] };
  s.fetchImpl = async (url, init) => {
    const u = String(url);
    if (u === "/") { s.shellReads += 1; return shellPage(s.restarted ? shellToken() : OLD); }
    const auth = init && init.headers && init.headers.Authorization;
    if (s.restarted && auth === "Bearer " + OLD) { s.refusedUrls.push(u); return refused(); }
    if (s.restarted && auth && auth !== "Bearer " + NEW) { s.refusedUrls.push(u); return refused(); }
    return ok();
  };
  return s;
}

async function bootOpen(server) {
  const { dom, window } = loadApp({ fetchImpl: server.fetchImpl, shellToken: OLD });
  const reload = countReloads(dom);
  assert.ok(await waitFor(() => window.__localmLocked === false), "premise: the page booted unlocked");
  return { dom, window, reload };
}

test("a same-port restart: every refused call recovers with the renewed token and no reload", async () => {
  const server = restartServer();
  const { window, reload } = await bootOpen(server);
  server.restarted = true;
  const urls = ["/api/models?t=1", "/v1/config?t=1", "/api/live?t=1", "/api/stats?t=1", "/api/activity?t=1"];
  const results = await Promise.all(urls.map((u) => window.fetch(u, { headers: window.authHeaders() })));
  assert.deepEqual(results.map((r) => r.status), urls.map(() => 200), "every refused call answered 200 after the retry");
  assert.equal(server.shellReads, 1, "the page shell was read once for the whole burst");
  for (const u of urls) {
    assert.equal(server.refusedUrls.filter((x) => x === u).length, 1, `${u} was refused once, then retried once`);
  }
  assert.equal(window.authHeaders().Authorization, "Bearer " + NEW, "later calls send the new token");
  const refusedSoFar = server.refusedUrls.length;
  const later = await window.fetch("/api/stats?t=2", { headers: window.authHeaders() });
  assert.equal(later.status, 200);
  await settle(100);
  assert.equal(server.refusedUrls.length, refusedSoFar, "no refusal after the renewal: no burst of repeated 403s");
  assert.equal(reload.reloads, 0, "the page did not reload");
  assert.equal(recoveryStarted(window), false, "the reload recovery never started");
});

test("a call that still carries the old token after the renewal is retried with the new one", async () => {
  const server = restartServer();
  const { window } = await bootOpen(server);
  const oldHeaders = window.authHeaders();
  server.restarted = true;
  await window.fetch("/api/stats?t=1", { headers: window.authHeaders() });
  const late = await window.fetch("/api/activity?t=1", { headers: oldHeaders });
  assert.equal(late.status, 200, "a late call with the old token is answered after one retry");
  assert.equal(server.shellReads, 1, "it reuses the token already read");
});

test("a page shell that brings no new token falls back to the reload recovery", async () => {
  const server = restartServer(() => OLD);
  const { window } = await bootOpen(server);
  server.restarted = true;
  const results = await Promise.all(["/api/stats?t=1", "/api/activity?t=1"].map(
    (u) => window.fetch(u, { headers: window.authHeaders() })));
  assert.deepEqual(results.map((r) => r.status), [403, 403], "the refusal reaches the caller");
  assert.equal(server.shellReads, 1, "the page shell was read once");
  assert.equal(recoveryStarted(window), true, "the reload recovery took over");
});

test("a reload with unsaved Settings asks in the page and the browser prompt stays away", async () => {
  const { dom, window } = loadApp({ fetchImpl: async () => ok(), shellToken: OLD });
  const reload = countReloads(dom);
  window.settingsDirty = () => true;
  const asked = window.reloadApp();
  const modal = window.document.getElementById("modal");
  assert.ok(await waitFor(() => modal.style.display === "flex"), "the in-page dialog is shown");
  assert.equal(reload.reloads, 0, "nothing reloads before the answer");
  const before = new window.Event("beforeunload", { cancelable: true });
  window.dispatchEvent(before);
  assert.equal(before.defaultPrevented, true, "premise: an unconfirmed leave still raises the browser prompt");
  window.document.getElementById("modal-body").querySelector(".btn-danger").click();
  assert.equal(await asked, true);
  assert.equal(reload.reloads, 1, "the page reloads once the person chose to");
  const after = new window.Event("beforeunload", { cancelable: true });
  window.dispatchEvent(after);
  assert.equal(after.defaultPrevented, false, "the confirmed reload raises no browser prompt");
});

test("staying on the page keeps it, and a restart with unsaved Settings shows the page again", async () => {
  const server = restartServer();
  const { window, reload } = await bootOpen(server);
  const poll = {};
  window.setInterval = (fn) => { poll.fn = fn; return 7; };
  window.clearInterval = () => {};
  window.settingsDirty = () => true;
  server.restarted = true;
  window.onServerUnreachable({ sawDown: true });
  assert.ok(poll.fn, "the reconnect poll was armed");
  const tick = poll.fn();
  const modal = window.document.getElementById("modal");
  assert.ok(await waitFor(() => modal.style.display === "flex"), "the server answered: the in-page dialog asks");
  const ov = window.document.getElementById("reconnect-overlay");
  assert.equal(ov.style.display, "none", "the overlay is out of the dialog's way");
  window.document.getElementById("modal-body").querySelector(".btn-secondary:not(.btn-danger)").click();
  await tick;
  assert.ok(await waitFor(() => window.__localmLocked === false), "the page is shown again");
  assert.equal(reload.reloads, 0, "no reload");
  assert.equal(window.authHeaders().Authorization, "Bearer " + NEW, "the shell token was renewed for the restarted server");
});

test("every reload init.js starts goes through reloadApp", () => {
  const src = readFileSync(join(STATIC, "app", "init.js"), "utf8");
  assert.equal((src.match(/location\.reload\(/g) || []).length, 1, "init.js calls location.reload() in one place");
  const body = src.slice(src.indexOf("export async function reloadApp"));
  assert.match(body.slice(0, body.indexOf("\n}")), /location\.reload\(/, "and that place is reloadApp");
});

test("the service worker passes a no-store request to the network and caches nothing", async () => {
  const handlers = {};
  const cached = [];
  const ctx = {
    self: { addEventListener: (n, fn) => { handlers[n] = fn; }, location: { origin: "http://localhost:8642" }, clients: {} },
    caches: {
      match: async () => shellPage(OLD), open: async () => ({ put: (r) => cached.push(r) }),
      keys: async () => [], delete: async () => true,
    },
    fetch: async () => shellPage(NEW),
    URL, Request: class { constructor(u) { this.url = u; } }, Promise, console,
  };
  vm.runInNewContext(readFileSync(join(STATIC, "sw.js"), "utf8"), ctx);
  const event = (cache) => {
    const ev = { request: { method: "GET", url: "http://localhost:8642/", mode: "cors", cache }, responded: null };
    ev.respondWith = (p) => { ev.responded = p; };
    return ev;
  };
  const fresh = event("no-store");
  handlers.fetch(fresh);
  assert.equal(fresh.responded, null, "a no-store GET of / is left to the network");
  const normal = event("default");
  handlers.fetch(normal);
  assert.ok(normal.responded, "premise: an ordinary GET of / is answered from the cache");
  assert.equal(cached.length, 0);
});

test("staying after the shut down overlay's Reload shows the page again", async () => {
  const server = restartServer();
  const { window, reload } = await bootOpen(server);
  window.settingsDirty = () => true;
  window.onServerUnreachable({ reason: "shutdown" });
  const ov = window.document.getElementById("reconnect-overlay");
  ov.querySelector("button").click();
  const modal = window.document.getElementById("modal");
  assert.ok(await waitFor(() => modal.style.display === "flex"), "the server answers: the in-page dialog asks");
  window.document.getElementById("modal-body").querySelector(".btn-secondary:not(.btn-danger)").click();
  assert.ok(await waitFor(() => window.__localmLocked === false), "the page is shown again, not left blank");
  assert.equal(reload.reloads, 0);
});

test("staying after the stale page overlay's Reload keeps that overlay on screen", async () => {
  const { dom, window } = loadApp({ fetchImpl: async () => ok(), shellToken: OLD });
  const reload = countReloads(dom);
  window.settingsDirty = () => true;
  window.showShellStaleOverlay();
  const ov = window.document.getElementById("shell-stale-overlay");
  ov.querySelector("button").click();
  const modal = window.document.getElementById("modal");
  assert.ok(await waitFor(() => modal.style.display === "flex"));
  assert.equal(ov.style.display, "none", "hidden while the dialog asks");
  window.document.getElementById("modal-body").querySelector(".btn-secondary:not(.btn-danger)").click();
  assert.ok(await waitFor(() => ov.style.display === "flex"), "shown again after staying");
  assert.equal(reload.reloads, 0);
});

test("a token read that brought no new token is not kept: a later restart is still renewed", async () => {
  let served = OLD;
  const server = restartServer(() => served);
  const { window } = await bootOpen(server);
  server.restarted = true;
  assert.equal(await window.renewShellToken(OLD), "", "premise: the shell still carries the same token");
  const reads = server.shellReads;
  served = NEW;
  assert.equal(await window.renewShellToken(OLD), "", "no new read right after a miss");
  assert.equal(server.shellReads, reads);
  const realNow = window.Date.now;
  window.Date.now = () => realNow() + 6000;
  try {
    assert.equal(await window.renewShellToken(OLD), NEW, "a later read finds the restarted server's token");
  } finally {
    window.Date.now = realNow;
  }
  assert.equal(window.authHeaders().Authorization, "Bearer " + NEW);
});
