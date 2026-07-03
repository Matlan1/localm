// SPDX-License-Identifier: AGPL-3.0-or-later
// authHeaders() no longer reads any cookie: the CSRF token is DERIVED from the
// session server-side and delivered via GET /api/session (stashed in
// window.__LOCALM_CSRF__), so it can never desync from the session (the old
// "missing CSRF token on every action" bug) and a malformed cookie can never brick
// the client into a false offline state. These tests pin that new contract plus the
// 403-CSRF self-heal (refetch the token and retry a write once).

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
  // A value that is NOT valid percent-encoding used to make readCookie throw and
  // brick authHeaders. authHeaders no longer touches cookies, so this cannot happen.
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
  // Simulate the server rotating its per-process CSRF secret (a restart): the first
  // write with the stale token 403s; the client refetches the token from
  // /api/session and retries ONCE, which succeeds. Safe: a 403 is rejected before
  // the handler runs, so nothing is duplicated.
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
  window.__LOCALM_CSRF__ = "STALE";   // the token the client currently believes in

  const res = await window.fetch("/v1/models/unload", {
    method: "POST",
    headers: window.authHeaders(),   // carries X-CSRF-Token: STALE
  });

  assert.equal(res.status, 200, "the retry with the refreshed token succeeded");
  assert.equal(unloadCalls, 2, "exactly one retry (the stale attempt + the healed one)");
  assert.equal(window.__LOCALM_CSRF__, "FRESH", "the token was refreshed from /api/session");
});
