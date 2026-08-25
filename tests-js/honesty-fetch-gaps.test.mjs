// SPDX-License-Identifier: AGPL-3.0-or-later
// GUI fetch failures are surfaced rather than swallowed:
// deleteConversationRemote, the debounced conversation PUT, model detail
// lookup, and the Models page's 401/403 handling.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

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
