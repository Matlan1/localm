// SPDX-License-Identifier: AGPL-3.0-or-later
// The "Add a model" card's #pull-sha256 field reaches the POST body, is omitted
// rather than sent as "" when left blank, and clears on a successful pull like
// the other fields in the row.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function makeFetch(calls) {
  return async (url, opts = {}) => {
    if (url === "/api/models/pull") {
      calls.push({ url, body: opts.body ? JSON.parse(opts.body) : null });
      return { ok: true, status: 200, json: async () => ({ job_id: "j1" }), text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

test("pull-sha256: a typed digest reaches the POST body", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  runScript(win, `streamJob = async () => ({ status: "done" });`);

  win.document.getElementById("pull-spec").value = "owner/repo:m.gguf";
  win.document.getElementById("pull-sha256").value =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85";
  win.document.getElementById("pull-start").click();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(calls.length, 1, "exactly one /api/models/pull request was made");
  assert.equal(calls[0].body.sha256,
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85",
    "the typed digest is forwarded as sha256 in the pull request body");
});

test("pull-sha256: an empty field omits sha256 entirely rather than sending \"\"", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  runScript(win, `streamJob = async () => ({ status: "done" });`);

  win.document.getElementById("pull-spec").value = "owner/repo";
  win.document.getElementById("pull-sha256").value = "";
  win.document.getElementById("pull-start").click();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(calls.length, 1, "exactly one /api/models/pull request was made");
  assert.ok(!("sha256" in calls[0].body),
    "no sha256 key is sent at all when the field was left blank");
});

test("pull-sha256: a successful pull clears the field, like spec/name/store", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  runScript(win, `streamJob = async () => ({ status: "done" });`);

  const sha = win.document.getElementById("pull-sha256");
  win.document.getElementById("pull-spec").value = "owner/repo:m.gguf";
  sha.value = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85";
  win.document.getElementById("pull-start").click();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(sha.value, "", "the sha256 field is cleared after a successful pull");
});
