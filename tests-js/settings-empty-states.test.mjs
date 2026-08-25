// SPDX-License-Identifier: AGPL-3.0-or-later
// The Settings page's Issues and Uploads lists render an empty state built by
// the emptyState(icon, text, hint) helper: a centred icon, a line of text, and
// a one-line hint. Asserted on the DOM structure, not on the copy.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch({ issues = null, issuesError = null, uploads = [] } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    if (u === "/api/issues") {
      const body = issuesError ? { error: issuesError } : { issues: issues || [] };
      return { ok: true, status: 200, text: async () => "", json: async () => body };
    }
    if (u === "/api/uploads" && method === "GET") {
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ items: uploads, dir: "/home/uploads" }) };
    }
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [],
                                  plugins: [], items: [] }) };
  };
}

const emptyStates = (root) => root.querySelectorAll(".empty-state");

test("rule 7: an empty Issues list renders a designed empty state, not a bare line", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ issues: [] }) });
  await window.issuesRefresh();
  const out = window.document.getElementById("issues-list");

  assert.equal(emptyStates(out).length, 1,
               "exactly one .empty-state block for an empty issues list");
  // The helper's three parts: a centred icon, a line of text, and a hint.
  assert.equal(out.querySelectorAll(".empty-state-ic").length, 1, "a centred icon");
  assert.equal(out.querySelectorAll(".empty-state-text").length, 1, "a line of text");
  assert.equal(out.querySelectorAll(".empty-state-hint").length, 1, "a do-this-next hint");
});

test("rule 7: an empty Uploads list renders an empty state instead of nothing", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ uploads: [] }) });
  await window.refreshUploadsList();
  const list = window.document.getElementById("upload-list");

  assert.equal(list.querySelectorAll("li").length, 0, "no rows to show");
  assert.equal(emptyStates(list).length, 1,
               "an empty uploads list must not be a blank scroll area");
  assert.equal(list.querySelectorAll(".empty-state-ic").length, 1, "a centred icon");
});

test("a NON-empty list still renders rows and NO empty state", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({
      issues: [{ number: 7, state: "open", title: "something broke" }],
      uploads: [{ name: "a.txt", bytes: 10 }],
    }),
  });

  await window.issuesRefresh();
  const out = window.document.getElementById("issues-list");
  assert.equal(emptyStates(out).length, 0, "no empty state when an issue exists");
  assert.match(out.textContent, /something broke/);

  await window.refreshUploadsList();
  const list = window.document.getElementById("upload-list");
  assert.equal(emptyStates(list).length, 0, "no empty state when a file exists");
  assert.equal(list.querySelectorAll("li").length, 1, "the row is still rendered");
});

test("a FAILED issues fetch still says it could not load - not 'no issues'", async () => {
  // A fetch error renders an error message, not the empty state.
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({ issuesError: "upstream refused" }),
  });
  await window.issuesRefresh();
  const out = window.document.getElementById("issues-list");

  assert.equal(emptyStates(out).length, 0,
               "a fetch error must NOT render the empty state");
  assert.match(out.textContent, /Could not load/);
  assert.match(out.textContent, /upstream refused/);
});
