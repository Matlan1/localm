// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings "Export logs" picks a folder and POSTs it to /api/logs/export.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(posts) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u === "/api/logs/export") {
      posts.push(JSON.parse(opts.body || "{}"));
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ copied: 4, dest: "/chosen/localm-logs-20260624" }) };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function flush() {
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
}

test("R30: export posts the picked folder and shows the count", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  window.pickDirectory = async () => "/chosen";          // user picks a folder
  window.document.getElementById("logs-export").click();
  await flush();
  assert.equal(posts.length, 1, "one export POST");
  assert.equal(posts[0].dest, "/chosen");
  const out = window.document.getElementById("logs-result");
  assert.equal(out.hidden, false);
  assert.match(out.textContent, /4 log file/);
  assert.match(out.textContent, /localm-logs-/);
});

test("R30: dismissing the folder picker does not export", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  window.pickDirectory = async () => null;               // dismissed
  window.document.getElementById("logs-export").click();
  await flush();
  assert.equal(posts.length, 0, "no POST when dismissed");
});
