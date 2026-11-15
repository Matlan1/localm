// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings > Updates, Update now: the line after an applied update names a git
// clone's changed files, and a setup-class update names the setup script of
// every platform.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(applyReply) {
  return async (url) => {
    const u = String(url);
    if (u.includes("/api/update/apply")) {
      return { ok: true, status: 200, text: async () => "", json: async () => applyReply };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function flush() {
  for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
}

async function applyWith(reply) {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(reply) });
  const doc = window.document;
  const btn = doc.getElementById("update-apply");
  assert.ok(btn, "the Update now button exists");
  btn.click();
  await flush();
  return doc.getElementById("update-status").textContent;
}

test("an update applied to a git clone says git lists the replaced files", async () => {
  const text = await applyWith({ applied: true, version: "0.3.5", klass: "reboot",
    git_clone: true });
  assert.match(text, /Updated to 0\.3\.5\./);
  assert.match(text, /git clone, so git now lists the files the update replaced as changed/);
});

test("an update applied to an install that is not a git clone says nothing about git", async () => {
  const text = await applyWith({ applied: true, version: "0.3.5", klass: "reboot",
    git_clone: false });
  assert.match(text, /Updated to 0\.3\.5\./);
  assert.doesNotMatch(text, /git/);
});

test("a refused update shows the server's own words, not a claim that it was rolled back", async () => {
  const text = await applyWith({ applied: false,
    error: "this localm runs from its container image, so no update was made" });
  assert.match(text, /Update failed: this localm runs from its container image/);
  assert.doesNotMatch(text, /rolled back/);
});

test("a setup-class update names the setup script of each platform", async () => {
  const text = await applyWith({ applied: true, version: "0.3.5", klass: "setup" });
  assert.match(text, /setup\.bat on Windows, setup\.sh on Linux and macOS/);
});
