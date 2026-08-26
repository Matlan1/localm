// SPDX-License-Identifier: AGPL-3.0-or-later
// authHeaders() reads no cookies; the CSRF token comes from window.__LOCALM_CSRF__,
// populated from GET /api/session.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));

test("authHeaders sends the in-memory session CSRF token", async () => {
  const { window } = loadApp();
  await tick();
  window.__LOCALM_CSRF__ = "session-derived-token";
  const h = window.authHeaders();
  assert.equal(h["X-CSRF-Token"], "session-derived-token");
});

test("a malformed cookie is IRRELEVANT to authHeaders (it reads no cookies)", async () => {
  const { window } = loadApp();
  await tick();
  // "ab%cd" is not valid percent-encoding.
  runScript(window, `document.cookie = "localm_csrf=ab%cd";`);
  window.__LOCALM_CSRF__ = "";
  let headers;
  assert.doesNotThrow(() => { headers = window.authHeaders(); });
  assert.ok(!("X-CSRF-Token" in headers), "no cookie is read, so no token from it");
});

test("a malformed cookie does NOT report a reachable server as unreachable", async () => {
  const calls = [];
  const { window } = loadApp({
    fetchImpl: async (url) => {
      calls.push(String(url));
      return { ok: true, status: 200, json: async () => ({ models: [] }), text: async () => "" };
    },
  });
  runScript(window, `document.cookie = "localm_csrf=ab%cd";`);

  const authed = await window.bootAuthProbe();

  assert.equal(authed, true, "a 200 from a reachable server -> authed, not 'unreachable'");
  assert.ok(calls.some((u) => u.includes("/api/models")),
    "the probe actually reached the server (the request was sent, not thrown away)");
  assert.equal(window.document.getElementById("reconnect-overlay"), null,
    "the 'server unreachable' overlay must NOT be shown when the server is reachable");
});

test("a cookie-authed write that 403s on a stale CSRF token self-heals and retries once", async () => {
  // The stub 403s the unload unless the sent token is "FRESH"; /api/session serves "FRESH".
  let unloadCalls = 0;
  const { window } = loadApp({
    fetchImpl: async (url, init) => {
      const u = String(url);
      if (u.includes("/api/session")) {
        return { ok: true, status: 200, json: async () => ({ authed: true, csrf: "FRESH" }), text: async () => "" };
      }
      if (u.includes("/v1/models/unload")) {
        unloadCalls += 1;
        const sent = init && init.headers && init.headers["X-CSRF-Token"];
        const ok = sent === "FRESH";
        return { ok, status: ok ? 200 : 403, json: async () => ({}), text: async () => "" };
      }
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    },
  });
  await tick();
  window.__LOCALM_CSRF__ = "STALE";

  const res = await window.fetch("/v1/models/unload", {
    method: "POST",
    headers: window.authHeaders(),   // carries X-CSRF-Token: STALE
  });

  assert.equal(res.status, 200, "the retry with the refreshed token succeeded");
  assert.equal(unloadCalls, 2, "exactly one retry (the stale attempt + the healed one)");
  assert.equal(window.__LOCALM_CSRF__, "FRESH", "the token was refreshed from /api/session");
});
