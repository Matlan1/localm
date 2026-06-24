// SPDX-License-Identifier: AGPL-3.0-or-later
// Defence-in-depth for the "server unreachable" page (after the readCookie root
// fix): the overlay must only appear when the server is GENUINELY unreachable,
// and a stuck client must always have a manual way out.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

// fetchImpl that fails any AUTHED call (one carrying headers) but answers a
// header-free reachability probe - i.e. the server is UP but the authed request
// could not be made (a client-side problem).
const reachableButAuthFails = async (url, opts = {}) => {
  if (opts.headers) throw new TypeError("authed call failed");
  return { ok: true, status: 200, json: async () => ({ models: [] }), text: async () => "" };
};
const totallyDown = async () => { throw new TypeError("server down"); };

test("a reachable server is NOT reported unreachable when only the authed probe fails", async () => {
  const { window } = loadApp({ fetchImpl: reachableButAuthFails });
  const authed = await window.bootAuthProbe();
  assert.equal(authed, false, "authed probe failed -> not authed");
  assert.equal(window.document.getElementById("reconnect-overlay"), null,
    "the header-free probe reaches the server -> the unreachable overlay must NOT show");
});

test("a genuinely unreachable server DOES show the reconnect overlay", async () => {
  const { window } = loadApp({ fetchImpl: totallyDown });
  const authed = await window.bootAuthProbe();
  assert.equal(authed, false);
  assert.ok(window.document.getElementById("reconnect-overlay"),
    "both the authed and the header-free probe threw -> genuinely down -> overlay shown");
});

test("the reconnect overlay offers a reset escape hatch", () => {
  const { window } = loadApp({ fetchImpl: totallyDown });
  window.onServerUnreachable();
  const btn = window.document.querySelector("#reconnect-overlay .reconnect-reset");
  assert.ok(btn, "a reset button is present on the unreachable overlay");
  assert.match(btn.textContent, /re-enter key/i);
});

test("reset clears local client state (so nothing local can permanently wedge it)", async () => {
  const { window } = loadApp();
  runScript(window, `localStorage.setItem("a", "1"); sessionStorage.setItem("b", "2");`);
  try { await window.resetClientState(); } catch (e) { /* jsdom location.reload is a noop */ }
  assert.equal(window.localStorage.length, 0, "localStorage cleared");
  assert.equal(window.sessionStorage.length, 0, "sessionStorage cleared");
});
