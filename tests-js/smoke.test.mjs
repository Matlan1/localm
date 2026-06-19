// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

test("harness loads app.js and exposes top-level functions", () => {
  const { window } = loadApp();
  assert.equal(typeof window.renderNav, "function", "renderNav should be on window");
  assert.equal(typeof window.runCompletion, "function", "runCompletion should be on window");
  // the nav slot from index.html exists
  assert.ok(window.document.getElementById("nav-plugin-slot"), "nav-plugin-slot present");
});
