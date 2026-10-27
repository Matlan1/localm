// SPDX-License-Identifier: AGPL-3.0-or-later
// The Settings > Updates "Choose a version" picker: the GUI form of
// `localm update --list` / `--to <version>`.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const RELEASES = {
  available: true, current: "0.3.0",
  releases: [
    { version: "v0.4.0", direction: "newer", installable: true, prerelease: false,
      published_at: "2026-11-01T00:00:00Z", warnings: [] },
    { version: "v0.3.0", direction: "same", installable: true, prerelease: false,
      warnings: ["this is the installed version: it would be reinstalled"] },
    { version: "v0.2.0", direction: "older", installable: true, prerelease: false,
      warnings: ["older than the installed 0.3.0: this is a downgrade"] },
    { version: "v0.1.0", direction: "older", installable: false, prerelease: false,
      reason: "the release has no build attached", warnings: [] },
  ],
};

function makeFetch(applyReplies, posts) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.includes("/api/update/releases")) {
      return { ok: true, status: 200, text: async () => "", json: async () => RELEASES };
    }
    if (u.includes("/api/update/apply")) {
      posts.push(JSON.parse(opts.body || "{}"));
      const body = applyReplies[posts.length - 1] || {};
      return { ok: true, status: 200, text: async () => "", json: async () => body };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function flush() {
  for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
}

async function openPicker(window) {
  window.document.getElementById("update-versions").click();
  await flush();
}

test("Choose a version: lists releases, disables one with no build, explains the pick", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([], posts) });
  const doc = window.document;
  await openPicker(window);
  assert.equal(doc.getElementById("update-version-picker").hidden, false);
  const opts = [...doc.querySelectorAll("#update-version-select option")];
  assert.deepEqual(opts.map((o) => o.value), ["v0.4.0", "v0.3.0", "v0.2.0", "v0.1.0"]);
  assert.match(opts[2].textContent, /older/);
  assert.equal(opts[3].disabled, true, "a release with no build cannot be picked");
  const sel = doc.getElementById("update-version-select");
  sel.value = "v0.2.0";
  sel.onchange();
  assert.match(doc.getElementById("update-version-info").textContent, /downgrade/);
  assert.equal(posts.length, 0, "listing never installs anything");
});

test("Choose a version: a downgrade confirms, then an unverified build warns with the reason, then installs that build", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    { applied: false, needs_confirmation: "unverified", version: "v0.2.0",
      reason: "the build is not signed by a trusted release key", sha256: "b".repeat(64) },
    { applied: true, version: "0.2.0", klass: "reboot", restarting: true },
  ], posts) });
  const doc = window.document;
  await openPicker(window);
  const sel = doc.getElementById("update-version-select");
  sel.value = "v0.2.0";
  sel.onchange();

  doc.getElementById("update-install-version").click();
  await flush();
  assert.match(doc.getElementById("modal-title").textContent, /Downgrade to v0\.2\.0/);
  assert.equal(posts.length, 0, "nothing happens until the downgrade is confirmed");
  doc.querySelector("#modal-body .btn-danger").click();
  await flush();
  assert.deepEqual(posts[0], { version: "v0.2.0" });

  assert.match(doc.getElementById("modal-body").textContent, /not signed by a trusted release key/,
    "the warning names why the signature does not verify");
  doc.querySelector("#modal-body .btn-danger").click();
  await flush();
  assert.deepEqual(posts[1], { version: "v0.2.0", allow_unverified: true, sha256: "b".repeat(64) });
  assert.match(doc.getElementById("update-version-status").textContent, /Installed 0\.2\.0/);
});

test("Choose a version: declining the unverified warning installs nothing", async () => {
  const posts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    { applied: false, needs_confirmation: "unverified", version: "v0.4.0",
      reason: "the release has no signature", sha256: "c".repeat(64) },
  ], posts) });
  const doc = window.document;
  await openPicker(window);
  doc.getElementById("update-install-version").click();
  await flush();
  assert.match(doc.getElementById("modal-title").textContent, /Install v0\.4\.0/);
  doc.querySelector("#modal-body .btn-danger").click();
  await flush();
  const cancel = [...doc.querySelectorAll("#modal-body button")].find((b) => !b.classList.contains("btn-danger"));
  cancel.click();
  await flush();
  assert.equal(posts.length, 1, "declining must not send the allow_unverified request");
  assert.match(doc.getElementById("update-version-status").textContent, /Not installed/);
});
