// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings action buttons report their outcome at the button: the status
// element follows the button's row in the DOM, is a polite live region, and the
// button shows a busy state while the action runs.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const PAIRS = [
  ["update-check", "update-status"],
  ["update-rollback", "update-rollback-status"],
  ["rebuild-launcher", "rebuild-launcher-status"],
  ["runtime-update-check", "runtime-update-status"],
  ["runtime-rollback", "runtime-rollback-status"],
  ["doctor-run", "doctor-status"],
  ["bug-send", "bug-result"],
  ["logs-export", "logs-result"],
  ["upload-send", "upload-result"],
];

const FOLLOWING = 4;   // Node.DOCUMENT_POSITION_FOLLOWING

function makeFetch(overrides = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (overrides[u]) return overrides[u](opts);
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [],
                                  plugins: [], items: [] }) };
  };
}

test("every action's status element follows its button row and is a polite live region", () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch() });
  const doc = window.document;
  for (const [btnId, outId] of PAIRS) {
    const btn = doc.getElementById(btnId), out = doc.getElementById(outId);
    assert.ok(btn && out, `${btnId} and ${outId} exist`);
    assert.ok(btn.compareDocumentPosition(out) & FOLLOWING,
      `${outId} comes after ${btnId} in document order`);
    assert.equal(btn.closest("section"), out.closest("section"),
      `${outId} is in the same card as ${btnId}`);
    assert.equal(out.getAttribute("aria-live"), "polite", `${outId} is a live region`);
  }
});

test("the runtime check result sits directly below its action row, not above the controls", () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch() });
  const doc = window.document;
  const row = doc.getElementById("runtime-update-check").closest(".actions");
  const out = doc.getElementById("runtime-update-status");
  assert.equal(row.nextElementSibling, out,
    "the status is the element right after the Check for runtime update row");
  const backend = doc.getElementById("runtime-backend");
  assert.ok(backend.compareDocumentPosition(out) & FOLLOWING,
    "the status is below the Backend and Build controls");
});

test("Check for runtime update shows the result at the button and is busy while it runs", async () => {
  let release;
  const gate = new Promise((r) => { release = r; });
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({
    "/api/runtime/check": async () => {
      await gate;
      return { ok: true, status: 200, text: async () => "", json: async () => ({
        installed: true, backend: "cpu", current: "b1", newer: false }) };
    },
  }) });
  const doc = window.document;
  const btn = doc.getElementById("runtime-update-check");
  const out = doc.getElementById("runtime-update-status");

  btn.click();
  assert.equal(btn.disabled, true, "disabled while the check runs");
  assert.equal(btn.getAttribute("aria-busy"), "true", "marked busy while the check runs");
  assert.equal(out.hidden, false, "the progress line is shown at the button");

  release();
  for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
  assert.equal(btn.disabled, false, "enabled again afterwards");
  assert.equal(btn.hasAttribute("aria-busy"), false, "busy mark removed afterwards");
  assert.match(out.textContent, /b1/, "the outcome replaced the progress line");
  assert.equal(out.previousElementSibling, btn.closest(".actions"));
});

test("Rebuild launcher is busy while it runs and reports in its status line", async () => {
  let release;
  const gate = new Promise((r) => { release = r; });
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({
    "/api/app/rebuild-launcher?force=true": async () => {
      await gate;
      return { ok: true, status: 200, text: async () => "", json: async () => ({
        ok: true, path: "x", notes: [] }) };
    },
  }) });
  const doc = window.document;
  const btn = doc.getElementById("rebuild-launcher");
  const out = doc.getElementById("rebuild-launcher-status");

  btn.click();
  assert.equal(btn.getAttribute("aria-busy"), "true");
  assert.equal(out.hidden, false);
  release();
  for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
  assert.equal(btn.hasAttribute("aria-busy"), false);
  assert.equal(btn.disabled, false);
  assert.notEqual(out.textContent, "");
  assert.equal(out.previousElementSibling, btn.closest(".actions"));
});
