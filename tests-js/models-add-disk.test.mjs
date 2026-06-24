// SPDX-License-Identifier: AGPL-3.0-or-later
// add-models-disk: adding a model already on disk used to require knowing you
// could paste a path into the "Pull a model" field. The Models page now has a
// Browse button that opens the directory picker and drops the chosen folder into
// the spec field (the /api/models/pull endpoint already accepts a local path).

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const okFetch = async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => "" });

test("add-models-disk: Browse drops the chosen folder into the model spec field", async () => {
  const { window } = loadAppWithPages({ fetchImpl: okFetch });
  // Stub the directory picker to return a chosen folder.
  window.pickDirectory = async () => "D:/models/my-gguf";
  const spec = window.document.getElementById("pull-spec");
  const browse = window.document.getElementById("pull-browse");
  assert.ok(browse, "the Browse button exists on the Models page");
  spec.value = "";
  browse.click();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(spec.value, "D:/models/my-gguf", "the browsed folder path fills the spec");
});

test("add-models-disk: a cancelled Browse leaves the spec unchanged", async () => {
  const { window } = loadAppWithPages({ fetchImpl: okFetch });
  window.pickDirectory = async () => null;   // user dismissed the picker
  const spec = window.document.getElementById("pull-spec");
  spec.value = "owner/repo";
  window.document.getElementById("pull-browse").click();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(spec.value, "owner/repo", "cancelling the picker does not clear the field");
});
