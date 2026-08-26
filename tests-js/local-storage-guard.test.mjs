// SPDX-License-Identifier: AGPL-3.0-or-later
// readStoredJSON (localm/plugins/gui/static/app/helpers.js) reads a JSON value
// from localStorage: the parsed value when well-formed, the fallback for an
// absent key, and the fallback plus a warning for a corrupt entry. It never
// throws.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

test("readStoredJSON returns the parsed value for well-formed JSON", () => {
  const { window } = loadApp();
  window.localStorage.setItem("k", JSON.stringify([1, 2, 3]));
  // Spread the window-realm array into a node array: cross-realm
  // deepStrictEqual fails on the differing Array constructor.
  assert.deepEqual([...window.readStoredJSON("k", [])], [1, 2, 3]);
});

test("readStoredJSON returns the fallback for an ABSENT key (no warning)", () => {
  const { window } = loadApp();
  const warns = [];
  window.console.warn = (...a) => warns.push(a.join(" "));
  assert.deepEqual(window.readStoredJSON("missing", []), []);
  assert.equal(warns.length, 0, "an absent key is the normal case, not a warning");
});

test("readStoredJSON returns the fallback for a CORRUPT value and WARNS (does not throw)", () => {
  const { window } = loadApp();
  const warns = [];
  window.console.warn = (...a) => warns.push(a.join(" "));
  window.localStorage.setItem("localm.conversations", "{oops not json");
  let out;
  assert.doesNotThrow(() => { out = window.readStoredJSON("localm.conversations", []); });
  assert.deepEqual(out, [], "falls back to the blank default");
  assert.equal(warns.length, 1, "corruption is surfaced, not silently swallowed");
  assert.match(warns[0], /corrupt/i);
  // the fallback survives caller-side wrapping such as new Set(...)
  assert.deepEqual([...new Set(out)], [], "wrapping the fallback in new Set(...) stays safe");
});
