// SPDX-License-Identifier: AGPL-3.0-or-later
// R47: a Settings "Report a bug" control that POSTs a description (and an
// optional "attach recent log" flag) to /api/bug-report, which saves an
// editable report file the user can send to the maintainer.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(posts) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u === "/api/bug-report") {
      posts.push(JSON.parse(opts.body || "{}"));
      return {
        ok: true, status: 200, text: async () => "",
        json: async () => ({ saved: true, filename: "bug-x.md",
          path: "/home/bug-reports/bug-x.md", maintainer: "owner@example.com" }),
      };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function flush() {
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
}

test("R47: Save report POSTs description + include_log and shows the saved path", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  const doc = window.document;
  doc.getElementById("bug-desc").value = "Mic button does nothing";
  doc.getElementById("bug-include-log").checked = true;
  doc.getElementById("bug-send").click();
  await flush();
  assert.equal(posts.length, 1, "one bug-report POST");
  assert.equal(posts[0].description, "Mic button does nothing");
  assert.equal(posts[0].include_log, true);
  const out = doc.getElementById("bug-result");
  assert.equal(out.hidden, false, "result line is shown");
  assert.match(out.textContent, /bug-x\.md/);
  assert.match(out.textContent, /owner@example\.com/);
  assert.equal(doc.getElementById("bug-desc").value, "", "textarea cleared after send");
});

test("R47: a blank description does not POST", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  window.document.getElementById("bug-desc").value = "   ";
  window.document.getElementById("bug-send").click();
  await flush();
  assert.equal(posts.length, 0, "no POST for an empty description");
});
