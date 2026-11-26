// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings > Updates: the backup an older localm's update left that is kept because it
// cannot be proven to be this install's, with its Remove button (the GUI form of
// `localm update --remove-old-backup`). The block shows only while there is one
// (GET /api/update/old-backup); Remove asks first and posts to its own route.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(state, removeReply, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    if (u.startsWith("/api/update/old-backup")) {
      calls.push(`${method} ${u}`);
      const body = u.endsWith("/remove") ? removeReply : state;
      const status = (body && body.__status) || 200;
      return { ok: status < 400, status, text: async () => "", json: async () => body };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function flush() {
  for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0));
}

test("the block stays hidden when no backup is kept", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ kept: false }, {}, calls) });
  await window.__localmOldBackupCheck();
  assert.equal(window.document.getElementById("app-old-backup-block").hidden, true);
  assert.deepEqual(calls, ["GET /api/update/old-backup"]);
});

test("a kept backup is shown with its version, size and why it is kept", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({ kept: true, version: "0.3.2", bytes: 12 * 1024 * 1024 }, {}, calls) });
  await window.__localmOldBackupCheck();
  const doc = window.document;
  assert.equal(doc.getElementById("app-old-backup-block").hidden, false);
  assert.equal(doc.getElementById("update-old-backup-text").textContent,
    "Backup of the update from localm 0.3.2, 12 MB, kept because it cannot be proven to be this install's.");
  assert.equal(doc.getElementById("update-old-backup-remove").hidden, false);
});

test("Remove asks first, posts once and says it was removed", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({ kept: true, version: "0.3.2", bytes: 5 }, { removed: true }, calls) });
  const doc = window.document;
  await window.__localmOldBackupCheck();
  doc.getElementById("update-old-backup-remove").click();
  await flush();
  const ok = doc.querySelector("#modal-body .btn-danger");
  assert.ok(ok, "removing a backup cannot be undone, so it confirms first");
  assert.match(doc.getElementById("modal-body").textContent, /localm 0\.3\.2/);
  assert.equal(calls.filter((c) => c.startsWith("POST")).length, 0);
  ok.click();
  await flush();
  assert.deepEqual(calls.filter((c) => c.startsWith("POST")), ["POST /api/update/old-backup/remove"]);
  assert.equal(doc.getElementById("update-old-backup-status").textContent, "The backup was removed.");
  assert.equal(doc.getElementById("update-old-backup-remove").hidden, true);
});

test("a refused removal says why and leaves the button usable", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({ kept: true, version: "0.3.2", bytes: 5 },
      { __status: 409, detail: "an update watchdog of this install may still run" }, calls) });
  const doc = window.document;
  await window.__localmOldBackupCheck();
  doc.getElementById("update-old-backup-remove").click();
  await flush();
  doc.querySelector("#modal-body .btn-danger").click();
  await flush();
  assert.match(doc.getElementById("update-old-backup-status").textContent,
    /Not removed: an update watchdog/);
  assert.equal(doc.getElementById("update-old-backup-remove").disabled, false);
});
