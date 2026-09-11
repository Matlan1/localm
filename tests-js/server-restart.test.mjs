// SPDX-License-Identifier: AGPL-3.0-or-later
// The Settings "Restart server" button confirms, then POSTs /v1/server/restart
// and shows the reconnect overlay.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

// Polls until fn() is true. The timeout is a failure bound, not a delay: the
// restart handler awaits /whoami before it POSTs, then arms an 800 ms timer
// before it shows the reconnect overlay.
const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, timeout = 2000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await settle(15); }
  return false;
}
const restartPosts = (posts) =>
  posts.filter((p) => p.url === "/v1/server/restart" && p.method === "POST");

function makeFetch(posts) {
  return async (url, opts = {}) => {
    posts.push({ url: String(url), method: opts.method || "GET" });
    if (String(url) === "/v1/server/restart") {
      return { ok: true, status: 200, json: async () => ({ restarting: true }), text: async () => "" };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

test("R18: the restart button confirms then POSTs /v1/server/restart", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  const btn = window.document.getElementById("server-restart");
  assert.ok(btn, "the Settings Server section has a Restart button");
  let overlay = 0;
  window.onServerUnreachable = () => { overlay += 1; };   // overlay shown on restart
  window.confirmDanger = (_t, _m, _l, onConfirm) => onConfirm();   // auto-confirm
  btn.click();
  assert.ok(await waitFor(() => restartPosts(posts).length > 0),
    "confirming the dialog POSTs the restart endpoint");
  assert.equal(restartPosts(posts).length, 1, "exactly one restart POST per confirmation");
  assert.ok(await waitFor(() => overlay > 0),
    "the reconnect overlay is shown so it can auto-reconnect");
  assert.equal(overlay, 1, "the overlay is shown once per restart");
});

test("R18: declining the confirmation does not restart", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  const btn = window.document.getElementById("server-restart");
  window.confirmDanger = () => {};   // user dismisses; onConfirm never called
  btn.click();
  // Positive control in the same window: a CONFIRMED click reaches the fetch
  // stub, so a POST from the declined click above would have been recorded
  // ahead of it.
  window.confirmDanger = (_t, _m, _l, onConfirm) => onConfirm();
  btn.click();
  assert.ok(await waitFor(() => restartPosts(posts).length > 0),
    "positive control: a confirmed click POSTs the restart endpoint");
  assert.equal(restartPosts(posts).length, 1,
    "no restart POST without confirmation: only the confirmed click reached the server");
});
