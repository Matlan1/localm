// SPDX-License-Identifier: AGPL-3.0-or-later
// The Settings "Restart server" button confirms, then POSTs /v1/server/restart
// and shows the reconnect overlay.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

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
  await new Promise((r) => setTimeout(r, 0));
  const hit = posts.filter((p) => p.url === "/v1/server/restart" && p.method === "POST");
  assert.equal(hit.length, 1, "confirming the dialog POSTs the restart endpoint");
  await new Promise((r) => setTimeout(r, 900));
  assert.equal(overlay, 1, "the reconnect overlay is shown so it can auto-reconnect");
});

test("R18: declining the confirmation does not restart", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  window.confirmDanger = () => {};   // user dismisses; onConfirm never called
  window.document.getElementById("server-restart").click();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(posts.filter((p) => p.url === "/v1/server/restart").length, 0,
    "no restart POST without confirmation");
});
