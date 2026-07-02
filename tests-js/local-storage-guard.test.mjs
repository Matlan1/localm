// SPDX-License-Identifier: AGPL-3.0-or-later
// readStoredJSON (localm/plugins/gui/static/app/helpers.js) must never let a
// CORRUPT localStorage entry crash the caller. Both chat-state reads (chat.js:19
// conversations, chat.js:399 convUI.collapsed) run at MODULE EVALUATION time, so a
// thrown SyntaxError there aborts the whole ES-module graph and boots the app to a
// blank shell. The guard branches missing-vs-corrupt (AGENTS.md rule 5: surface,
// do not collapse the two) and falls back instead of throwing.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

test("readStoredJSON returns the parsed value for well-formed JSON", () => {
  const { window } = loadApp();
  window.localStorage.setItem("k", JSON.stringify([1, 2, 3]));
  // spread the window-realm array into a node array (cross-realm deepStrictEqual
  // otherwise fails on the differing Array constructor).
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
});

test("a distinct fallback object is honoured (e.g. Set-backed collapsed state)", () => {
  const { window } = loadApp();
  window.localStorage.setItem("localm.convCollapsed", "&&&broken");
  const arr = window.readStoredJSON("localm.convCollapsed", []);
  assert.deepEqual([...new Set(arr)], [], "wrapping in new Set(...) stays safe");
});
