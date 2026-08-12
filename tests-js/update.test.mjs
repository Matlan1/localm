// SPDX-License-Identifier: AGPL-3.0-or-later
// The Settings "Updates" + "Issues" cards: check surfaces an available update and
// reveals "Update now"; apply reports restart / rollback honestly; issues list is
// rendered read-only (textContent, never innerHTML).

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

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

test("Updates: check surfaces an available update and reveals 'Update now'", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/update/check", { available: true, newer: true, current: "0.1.0",
      latest: "v0.2.0", notes: "cool", asset: { id: 3 } }],
  ]) });
  const doc = window.document;
  doc.getElementById("update-check").click();
  await flush();
  assert.match(doc.getElementById("update-status").textContent, /Update available: v0\.2\.0/);
  assert.equal(doc.getElementById("update-apply").hidden, false, "Update now revealed");
});

test("Updates: up to date hides 'Update now'", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/update/check", { available: true, newer: false, current: "0.2.0", latest: "0.2.0" }],
  ]) });
  const doc = window.document;
  doc.getElementById("update-check").click();
  await flush();
  assert.match(doc.getElementById("update-status").textContent, /up to date/);
  assert.equal(doc.getElementById("update-apply").hidden, true);
});

test("Updates: an UNCOMPARABLE tag is reported as 'could not tell', never as up to date", async () => {
  // IDEA part 1's second dead zone. is_newer() returns False both for a genuine
  // tie/older release AND for a tag it could not parse at all (nightly, stable,
  // release-5), so `newer:false` alone cannot tell those apart - and the GUI
  // used to print "up to date" for both. The CLI already branched on
  // `comparable` (cli/maintenance.py); a GUI user was the one left falsely
  // reassured, including UNPROMPTED via the throttled startup auto-surface.
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/update/check", { available: true, newer: false, comparable: false,
      current: "0.1.4", latest: "nightly" }],
  ]) });
  const doc = window.document;
  doc.getElementById("update-check").click();
  await flush();
  const text = doc.getElementById("update-status").textContent;
  assert.match(text, /Could not tell|unrecognized version format/);
  assert.doesNotMatch(text, /up to date/,
    "a tag we could not order must never be reported as up to date");
  assert.equal(doc.getElementById("update-apply").hidden, true,
    "nothing to apply - there is no ordered newer release");
});

test("Updates: 'Update now' POSTs apply and shows restarting", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/update/apply", (opts) => { calls.push(opts.method); return {
      applied: true, version: "0.2.0", klass: "reboot", restarting: true }; }],
  ]) });
  const doc = window.document;
  doc.getElementById("update-apply").click();
  await flush();
  assert.equal(calls[0], "POST");
  assert.match(doc.getElementById("update-status").textContent, /Updated to 0\.2\.0\. Restarting/);
});

test("Updates: a failed apply is reported as rolled back (never faked success)", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/update/apply", { applied: false,
      error: "the post-update step failed; rolled back: uv exited 1" }],
  ]) });
  const doc = window.document;
  doc.getElementById("update-apply").click();
  await flush();
  assert.match(doc.getElementById("update-status").textContent, /rolled back/);
});

test("Issues: refresh lists open + closed issues", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/issues", { available: true, issues: [
      { number: 5, state: "open", title: "bug a", html_url: "u5" },
      { number: 6, state: "closed", title: "bug b" }] }],
  ]) });
  const doc = window.document;
  doc.getElementById("issues-refresh").click();
  await flush();
  const out = doc.getElementById("issues-list");
  assert.match(out.textContent, /#5/);
  assert.match(out.textContent, /#6/);
});
