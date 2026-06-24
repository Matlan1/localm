// SPDX-License-Identifier: AGPL-3.0-or-later
// R18 (shutdown half): a Settings "Shut down server" button that confirms, then
// POSTs the existing /v1/server/shutdown endpoint, so the user is not left
// force-closing the window. (A true in-app restart needs a server re-exec
// endpoint that does not exist yet - tracked separately.)

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(posts) {
  return async (url, opts = {}) => {
    posts.push({ url: String(url), method: opts.method || "GET" });
    if (String(url) === "/v1/server/shutdown") {
      return { ok: true, status: 200, json: async () => ({ stopping: true }), text: async () => "" };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

test("R18: the shutdown button confirms then POSTs /v1/server/shutdown", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  const btn = window.document.getElementById("server-shutdown");
  assert.ok(btn, "the Settings Server section has a Shut down button");
  window.confirmDanger = (_t, _m, _l, onConfirm) => onConfirm();   // auto-confirm
  btn.click();
  await new Promise((r) => setTimeout(r, 0));
  const shut = posts.filter((p) => p.url === "/v1/server/shutdown" && p.method === "POST");
  assert.equal(shut.length, 1, "confirming the dialog POSTs the shutdown endpoint");
});

test("R18: declining the confirmation does not shut down", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  window.confirmDanger = () => {};   // user dismisses; onConfirm never called
  window.document.getElementById("server-shutdown").click();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(posts.filter((p) => p.url === "/v1/server/shutdown").length, 0,
    "no shutdown POST without confirmation");
});
