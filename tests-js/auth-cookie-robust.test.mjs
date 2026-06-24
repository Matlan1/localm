// SPDX-License-Identifier: AGPL-3.0-or-later
// ROOT CAUSE of the phone showing "Can't reach the LocaLM server" on a perfectly
// reachable server: readCookie() decoded the cookie with decodeURIComponent,
// which throws a URIError on any malformed percent-encoding. That made
// authHeaders() throw, so every `fetch(url, {headers: authHeaders()})` rejected
// before the request was sent, and bootAuthProbe reported the (reachable) server
// as unreachable. A bad cookie must NEVER brick the client into a false offline
// state - it is independent of the key, the IP, or anything the server does.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

test("a malformed cookie value does not make authHeaders() throw", () => {
  const { window } = loadApp();
  // A value that is NOT valid percent-encoding (a stray %), which decodeURIComponent
  // throws on. Any corrupt/odd cookie can be this.
  runScript(window, `document.cookie = "localm_csrf=ab%cd";`);
  let headers;
  assert.doesNotThrow(() => { headers = window.authHeaders(); });
  // Falls back to the raw stored value rather than crashing.
  assert.equal(headers["X-CSRF-Token"], "ab%cd");
});

test("a malformed cookie does NOT report a reachable server as unreachable", async () => {
  const calls = [];
  const { window } = loadApp({
    fetchImpl: async (url) => {
      calls.push(String(url));
      return { ok: true, status: 200, json: async () => ({ models: [] }), text: async () => "" };
    },
  });
  // The exact phone condition: a bad cookie + a reachable server.
  runScript(window, `document.cookie = "localm_csrf=ab%cd";`);

  const authed = await window.bootAuthProbe();

  assert.equal(authed, true, "a 200 from a reachable server -> authed, not 'unreachable'");
  assert.ok(calls.some((u) => u.includes("/api/models")),
    "the probe actually reached the server (the request was sent, not thrown away)");
  assert.equal(window.document.getElementById("reconnect-overlay"), null,
    "the 'server unreachable' overlay must NOT be shown when the server is reachable");
});

test("a clean cookie still decodes normally (percent-encoded values round-trip)", () => {
  const { window } = loadApp();
  runScript(window, `document.cookie = "localm_csrf=a%20b";`);   // %20 -> space
  assert.equal(window.authHeaders()["X-CSRF-Token"], "a b");
});
