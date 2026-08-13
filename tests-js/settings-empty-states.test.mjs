// SPDX-License-Identifier: AGPL-3.0-or-later
// gui-design.md rule 7: "Every list renders a real empty state (a centered icon
// + a one-line 'do this next' hint), never a blank scroll area or a lone .sub
// line. Use the emptyState(icon, text, hint) helper."
//
// Two lists on the Settings page broke that rule (findings F6/F7 of the
// conformance sweep) while a third on the same page - the keys list - already
// used the helper:
//   - Issues rendered the bare string "No issues." into a .sub div;
//   - Uploads rendered NOTHING at all, the literal blank area the rule names.
//
// These tests pin the fix at the DOM level rather than on the copy, so a later
// wording change does not break them but deleting the empty state does.

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
  // The helper's three parts: a centred ICON, a line of text, and a hint. The
  // icon is what a lone .sub line can never have, so assert it specifically.
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
  // The control that keeps the two fixes honest: an empty state that renders
  // unconditionally would satisfy both tests above and be a worse bug than the
  // one being fixed.
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
  // "there are none" and "I could not ask" are different answers, and rule 7's
  // empty state is only correct for the first. Collapsing them would be the
  // AGENTS.md rule 5 fault: a failure wearing a success-shaped result.
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
