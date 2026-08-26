// SPDX-License-Identifier: AGPL-3.0-or-later
// "Show changelog" fetches GET /api/changelog and renders the version history in
// the shared modal. marked/DOMPurify are identity-stubbed in the harness, so the
// raw markdown lands verbatim in #modal.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const CHANGELOG_MD =
  "# Changelog\n\n## [Unreleased]\n\n### Added\n- brand new feature\n\n" +
  "## [0.1.0] - 2026-07-04\n\nFirst tagged release.\n";

function makeFetch(routes) {
  return async (url, opts = {}) => {
    const u = String(url);
    for (const [frag, resp] of routes) {
      if (u.includes(frag)) {
        const body = typeof resp === "function" ? resp(opts) : resp;
        return { ok: true, status: 200, text: async () => "", json: async () => body };
      }
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function flush() {
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
}

test("Changelog: 'Show changelog' fetches /api/changelog and renders version history", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/changelog", (opts) => { calls.push(opts.method || "GET"); return {
      available: true, version: "0.1.1", markdown: CHANGELOG_MD }; }],
  ]) });
  const doc = window.document;
  doc.getElementById("changelog-show").click();
  await flush();
  assert.equal(calls[0], "GET", "it GETs /api/changelog");
  const modal = doc.getElementById("modal");
  assert.notEqual(modal.style.display, "none", "the changelog modal opened");
  // Both the newest and an older release section are present.
  assert.match(modal.textContent, /\[Unreleased\]/);
  assert.match(modal.textContent, /\[0\.1\.0\]/);
  assert.match(modal.textContent, /First tagged release\./);
});

test("Changelog: a missing changelog is reported, never a blank modal", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/changelog", { available: false }],
  ]) });
  const doc = window.document;
  doc.getElementById("changelog-show").click();
  await flush();
  assert.match(doc.getElementById("modal").textContent, /No changelog/i);
});
