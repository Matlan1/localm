// SPDX-License-Identifier: AGPL-3.0-or-later
// GUI fetch failures are surfaced rather than swallowed:
// deleteConversationRemote, the debounced conversation PUT, model detail
// lookup, and the Models page's 401/403 handling.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

/** A working, in-memory localStorage stand-in, for swapping a broken one out
 *  mid-test once a scenario needs a save to actually succeed. */
function workingStorage() {
  const data = {};
  return {
    getItem: (k) => (Object.prototype.hasOwnProperty.call(data, k) ? data[k] : null),
    setItem: (k, v) => { data[k] = String(v); },
    removeItem: (k) => { delete data[k]; },
    clear: () => { for (const k of Object.keys(data)) delete data[k]; },
    key: () => null,
    get length() { return Object.keys(data).length; },
  };
}

const drain = async (n = 12) => {
  for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0));
};

const OK = { ok: true, status: 200, json: async () => ({}), text: async () => "" };

test("a failed server delete is surfaced, not silently swallowed", async () => {
  const { window } = loadApp({ fetchImpl: async (url, opts) => {
    if (String(url).startsWith("/api/conversations/") && opts && opts.method === "DELETE") {
      return { ok: false, status: 500, json: async () => ({}), text: async () => "boom" };
    }
    return OK;
  } });
  runScript(window, "chat.persist = true; deleteConversationRemote('c9');");
  await drain();
  const toastEl = window.document.getElementById("toast");
  assert.ok(/may reappear/.test(toastEl.textContent),
    "the user is told the delete did not stick");
  assert.ok(toastEl.className.includes("error"), "shown as an error toast");
  assert.ok((window.__localmClientLog || []).some((l) => l.includes("conversation delete failed")),
    "the failure lands in the client log ring");
});

test("a successful server delete stays silent", async () => {
  const { window } = loadApp({ fetchImpl: async () => OK });
  runScript(window, "chat.persist = true; deleteConversationRemote('c1');");
  await drain();
  assert.equal(window.document.getElementById("toast").textContent, "",
    "no toast on the happy path");
  assert.ok(!(window.__localmClientLog || []).some((l) => l.includes("delete failed")),
    "nothing logged on the happy path");
});

test("a resolved-but-failed conversation save is logged once, not per save", async () => {
  let puts = 0;
  const { window } = loadApp({ fetchImpl: async (url, opts) => {
    if (opts && opts.method === "PUT") {
      puts++;
      return { ok: false, status: 500, json: async () => ({}), text: async () => "" };
    }
    return OK;
  } });
  runScript(window, `
    chat.persist = true;
    window.__conv = { id: "cx", title: "T", pinned: false, folder: null,
                      branches: [], messages: [{ role: "user", content: "hi" }] };
    pushConversation(window.__conv);
  `);
  // Rides out the 600ms debounce.
  await new Promise((r) => setTimeout(r, 1000));
  runScript(window, "pushConversation(window.__conv);");
  await new Promise((r) => setTimeout(r, 1000));
  assert.equal(puts, 2, "both saves reached the server");
  const lines = (window.__localmClientLog || []).filter((l) => l.includes("conversation save failed"));
  assert.equal(lines.length, 1, "the breakage is logged once, not per debounce tick");
});

test("a network exception on the conversation save is surfaced, not silently swallowed", async () => {
  const { window } = loadApp({ fetchImpl: async (url, opts) => {
    if (opts && opts.method === "PUT") throw new Error("network unreachable");
    return OK;
  } });
  runScript(window, `
    chat.persist = true;
    window.__conv = { id: "cy", title: "T", pinned: false, folder: null,
                      branches: [], messages: [{ role: "user", content: "hi" }] };
    pushConversation(window.__conv);
  `);
  await new Promise((r) => setTimeout(r, 1000));
  const lines = window.__localmClientLog || [];
  assert.ok(lines.some((l) => l.includes("could not reach the server")),
    "a thrown fetch must be logged, not silently ignored");
});

test("both local and remote save failing shows a visible warning, never a false " +
     "'local copy is intact' claim, and clears only once a write is confirmed", async () => {
  let putCalls = 0;
  const { window } = loadApp({
    breakStorage: true,   // localStorage.setItem throws on every attempt, full and reduced alike
    fetchImpl: async (url, opts) => {
      if (opts && opts.method === "PUT") {
        putCalls++;
        if (putCalls === 1) return { ok: false, status: 500, json: async () => ({}), text: async () => "" };
        if (putCalls === 2) throw new Error("network unreachable");
        return OK;
      }
      return OK;
    },
  });
  runScript(window, `
    chat.persist = true;
    chat.modeKnown = true;
    window.__conv = { id: "cz", title: "T", pinned: false, folder: null,
                      branches: [], messages: [{ role: "user", content: "hi" }] };
    chat.conversations = [window.__conv];
    saveConversations(window.__conv);
  `);
  await new Promise((r) => setTimeout(r, 1000));   // ride out the debounce - PUT #1 (500)

  let lines = window.__localmClientLog || [];
  assert.ok(!lines.some((l) => /local copy is intact/i.test(l)),
    "must never claim the local copy is intact when it could not be saved either");
  assert.ok(lines.some((l) => /conversation save failed/.test(l)), "the HTTP failure is surfaced");
  let banner = window.document.getElementById("conv-unsaved-warning");
  assert.ok(banner, "a persistent 'not saved' warning must show once neither path has landed");
  assert.equal(banner.textContent, "not saved - history is only in this tab");

  runScript(window, "saveConversations(window.__conv);");
  await new Promise((r) => setTimeout(r, 1000));   // PUT #2 (network exception)
  lines = window.__localmClientLog || [];
  assert.ok(!lines.some((l) => /local copy is intact/i.test(l)),
    "still no false claim after the network-exception attempt");
  banner = window.document.getElementById("conv-unsaved-warning");
  assert.ok(banner, "the warning must stay visible - nothing has actually been saved yet");

  // Recovery: both paths work now - the warning must clear.
  Object.defineProperty(window, "localStorage", { value: workingStorage(), configurable: true });
  runScript(window, "saveConversations(window.__conv);");
  await new Promise((r) => setTimeout(r, 1000));   // PUT #3 (ok)
  assert.equal(window.document.getElementById("conv-unsaved-warning"), null,
    "the warning must clear once a save actually lands both locally and remotely");
});

test("a plain-text 500 on model detail still shows the error toast", async () => {
  const { window } = loadAppWithPages({ fetchImpl: async (url) => {
    if (String(url).startsWith("/v1/models/")) {
      return { ok: false, status: 500, statusText: "Internal Server Error",
               json: async () => { throw new Error("Unexpected token I"); },
               text: async () => "Internal Server Error" };
    }
    return OK;
  } });
  runScript(window, "showModelDetail('m1');");
  await drain();
  assert.ok(/Lookup failed/.test(window.document.getElementById("toast").textContent),
    "the error toast survives a non-JSON error body");
});

test("a 403 on the models page surfaces an honest error, not 'No models yet'", async () => {
  // A 403 with a {detail:...} body, which parses as valid JSON.
  const { window } = loadAppWithPages({ fetchImpl: async (url) => {
    if (String(url).startsWith("/api/models")) {
      return { ok: false, status: 403, statusText: "Forbidden",
               json: async () => ({ detail: "forbidden" }), text: async () => "" };
    }
    return OK;
  } });
  runScript(window, "refreshModelsPage();");
  await drain();
  const box = window.document.getElementById("models-table");
  assert.ok(/Could not load models \(HTTP 403\)/.test(box.textContent),
    "the box shows the real HTTP status, not an empty state");
  assert.ok(!/No models yet/.test(box.textContent),
    "the misleading 'No models yet' empty state is NOT shown on a 403");
});

test("a 401 on the models page shows the key gate, not 'No models yet'", async () => {
  const { window } = loadAppWithPages({ fetchImpl: async (url) => {
    if (String(url).startsWith("/api/models")) {
      return { ok: false, status: 401, statusText: "Unauthorized",
               json: async () => ({ detail: "unauthorized" }), text: async () => "" };
    }
    return OK;
  } });
  runScript(window, "refreshModelsPage();");
  await drain();
  assert.equal(window.document.getElementById("key-gate").style.display, "flex",
    "an expired/absent session opens the in-page key gate");
  const box = window.document.getElementById("models-table");
  assert.ok(!/No models yet/.test(box.textContent),
    "the misleading 'No models yet' empty state is NOT shown on a 401");
});
