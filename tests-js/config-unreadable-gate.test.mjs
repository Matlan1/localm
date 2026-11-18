// SPDX-License-Identifier: AGPL-3.0-or-later

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, timeout = 2000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await settle(15); }
  return false;
}
const reply = (status, body, headers = {}) => Promise.resolve({
  ok: status >= 200 && status < 300, status, json: async () => body, text: async () => "",
  headers: { get: (k) => headers[k] ?? null },
});

test("a 401 on a server locked by a damaged config.json says so at the gate", async () => {
  const fetchImpl = () => reply(401, {}, { "X-Localm-Config-Unreadable": "1" });
  const { window } = loadApp({ fetchImpl });
  assert.ok(await waitFor(() => window.__localmLocked === true), "the gate was shown");
  assert.ok(await waitFor(() => /config\.json/.test(
    window.document.getElementById("key-gate-msg").textContent)),
    "the gate names config.json");
  assert.doesNotMatch(window.document.getElementById("key-gate-msg").textContent,
    /requires an API key/);
});

test("an ordinary 401 keeps the API-key message", async () => {
  const fetchImpl = () => reply(401, {});
  const { window } = loadApp({ fetchImpl });
  assert.ok(await waitFor(() => window.__localmLocked === true));
  assert.match(window.document.getElementById("key-gate-msg").textContent, /requires an API key/);
});
