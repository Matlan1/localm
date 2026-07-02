// SPDX-License-Identifier: AGPL-3.0-or-later
// HONESTY-0702: GUI failures must be surfaced, not swallowed (AGENTS.md rule 5).
//   1. deleteConversationRemote: a failed server-side DELETE toasts + logs
//      (the copy would otherwise silently resurrect on the next load).
//   2. models.js: r.json() tolerates a non-JSON error body (a plain-text 500)
//      so the failure toast still fires instead of an unhandled rejection.
//   3. pushConversation: a resolved-but-failed save (HTTP 500) reaches the
//      client error log via console.error instead of total silence.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

const drain = async (times = 12) => {
  for (let i = 0; i < times; i++) await new Promise((r) => setTimeout(r, 0));
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// A 500 whose body is NOT JSON (FastAPI's unhandled-exception response is
// plain text) - json() must reject like the real Response would.
const plainText500 = {
  ok: false, status: 500, statusText: "Internal Server Error",
  json: async () => { throw new SyntaxError("Unexpected token 'I'"); },
  text: async () => "Internal Server Error",
};
const okEmpty = {
  ok: true, status: 200, text: async () => "",
  json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
};

test("a failed server-side conversation delete is surfaced, not silent", async () => {
  const calls = [];
  const { window } = loadApp({
    fetchImpl: async (url, opts = {}) => {
      calls.push((opts.method || "GET") + " " + url);
      if ((opts.method || "GET") === "DELETE") return plainText500;
      return okEmpty;
    },
  });
  runScript(window, 'chat.persist = true; deleteConversationRemote("conv-x");');
  await drain();
  assert.ok(calls.some((c) => c.startsWith("DELETE ")), "the DELETE was sent");
  const toastEl = window.document.getElementById("toast");
  assert.match(toastEl.textContent, /Could not delete the conversation/i,
    "the failure reaches the user as a toast");
  assert.ok(toastEl.className.includes("error"), "shown as an error toast");
});

test("model detail lookup with a plain-text 500 still toasts 'Lookup failed'", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: async (url) => {
      if (String(url).startsWith("/v1/models/")) return plainText500;
      return okEmpty;
    },
  });
  runScript(window, 'showModelDetail("some-model");');
  await drain();
  const toastEl = window.document.getElementById("toast");
  assert.match(toastEl.textContent, /Lookup failed/i,
    "the non-JSON error body must not kill the toast as an unhandled rejection");
});

test("a resolved-but-failed conversation push lands in the client error log", async () => {
  const { window } = loadApp({
    fetchImpl: async (url, opts = {}) =>
      (opts.method === "PUT" ? plainText500 : okEmpty),
  });
  runScript(window, `
    window.__errs = [];
    console.error = (...a) => window.__errs.push(a.join(" "));
    chat.persist = true;
    pushConversation({ id: "c1", title: "T", pinned: false, folder: null,
                       branches: [], messages: [{ role: "user", content: "x" }] });
  `);
  await sleep(750);   // the push is debounced by 600ms
  await drain();
  assert.ok(
    window.__errs.some((m) => m.includes("conversation push failed")),
    "the failed save is reported to the client error log, not swallowed: "
      + JSON.stringify(window.__errs));
});
