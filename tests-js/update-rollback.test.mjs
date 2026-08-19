// SPDX-License-Identifier: AGPL-3.0-or-later
// The Settings > Updates "Roll back" sub-block: the GUI form of
// `localm update --rollback`.
//
// The block is revealed ONLY by the read-only probe (GET /api/update/rollback), so
// an install that never updated never sees it. The action is a separate POST behind
// a danger confirm, and the server restarts itself afterwards.
//
// Both verbs share ONE url, so every mock here discriminates on opts.method and
// records the sequence: a harness that identified the call by url alone could not
// tell the probe from the action, which is the exact thing these tests are about
// (diff-review-discipline.md item 20).

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(rollbackGet, rollbackPost, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    if (u.includes("/api/update/rollback")) {
      calls.push(method);
      const spec = method === "POST" ? rollbackPost : rollbackGet;
      const body = typeof spec === "function" ? spec() : spec;
      const status = (body && body.__status) || 200;
      return { ok: status < 400, status, text: async () => "",
        json: async () => body };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function flush() {
  for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0));
}

test("Roll back: the block stays hidden when there is no backup", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({ available: false, backup: null, version: null,
      current: "0.2.0" }, {}, calls) });
  const doc = window.document;
  await window.__localmRollbackCheck();
  await flush();
  assert.equal(doc.getElementById("app-rollback-block").hidden, true,
    "no backup: the control must not be offered at all");
  assert.equal(calls.includes("POST"), false,
    "checking whether a rollback is possible must never perform one");
});

test("Roll back: an available backup reveals the block and names the build", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({ available: true, backup: "/h/updates/backup",
      version: "0.1.4", current: "0.2.0" }, {}, calls) });
  const doc = window.document;
  await window.__localmRollbackCheck();
  await flush();
  assert.equal(doc.getElementById("app-rollback-block").hidden, false);
  const text = doc.getElementById("update-rollback-status").textContent;
  assert.match(text, /restores 0\.1\.4/,
    "the user must be told WHICH build this puts back before pressing it");
  assert.match(text, /running 0\.2\.0/);
  assert.equal(calls.includes("POST"), false, "the probe must not POST");
});

test("Roll back: the button confirms, POSTs, and reports the restart", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch(
      { available: true, version: "0.1.4", current: "0.2.0" },
      { rolled_back: true, version: "0.1.4", restarting: true }, calls) });
  const doc = window.document;
  await window.__localmRollbackCheck();
  await flush();

  doc.getElementById("update-rollback").click();
  await flush();
  const ok = doc.querySelector("#modal-body .btn-danger");
  assert.ok(ok, "replacing the running code is a danger action, so it confirms first");
  assert.equal(calls.includes("POST"), false, "nothing happens until the confirm");

  ok.click();
  await flush();
  assert.equal(calls.filter((m) => m === "POST").length, 1);
  const text = doc.getElementById("update-rollback-status").textContent;
  assert.match(text, /Rolled back to 0\.1\.4/);
  assert.match(text, /Restarting/);
  assert.equal(doc.getElementById("update-rollback").hidden, true,
    "the server is re-execing; the reconnect overlay takes over from here");
});

test("Roll back: a refusal is reported and leaves the button usable", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch(
      { available: true, version: "0.1.4", current: "0.2.0" },
      { __status: 403, detail: "requires an owner (admin) key" }, calls) });
  const doc = window.document;
  await window.__localmRollbackCheck();
  await flush();
  doc.getElementById("update-rollback").click();
  await flush();
  doc.querySelector("#modal-body .btn-danger").click();
  await flush();
  const text = doc.getElementById("update-rollback-status").textContent;
  assert.match(text, /Roll back failed/,
    "a refused rollback must say so, never fall silent or read as done");
  assert.match(text, /owner/);
  assert.equal(doc.getElementById("update-rollback").disabled, false,
    "nothing happened, so the control stays usable");
  assert.equal(doc.getElementById("update-rollback").hidden, false);
});
