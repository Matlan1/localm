// SPDX-License-Identifier: AGPL-3.0-or-later
// Regression test for app/main.js's window-export loop: `export let` bindings
// (modelCache and ~20 siblings across app/*+pages/*) are REASSIGNED at
// runtime, not mutated, so a one-time `window[name] = mod[name]` snapshot
// freezes forever at the first-load value - the actual bug this loop fixes
// (window.modelCache stayed {models:[],active:""} even after a model loaded).
//
// main.js can't be imported directly here: it statically imports the whole
// 20-module app/pages graph, and several of those modules touch
// `document`/`window` at TOP LEVEL (e.g. models-sidebar.js's
// `modelSelect = $("model-select")`) - only jsdom's harness.mjs can satisfy
// that, and jsdom does not execute `<script type="module">` at all, which is
// exactly why harness.mjs converts every module to a classic script sharing
// one lexical scope instead of running the real import graph. That
// conversion cannot exercise main.js's actual mechanism (a live getter into
// an ES module namespace object), so this test takes a different real-code
// approach instead of a mock: it extracts the REAL trailing window-export
// loop straight out of the shipped main.js file (the loop is provably the
// file's last statement) and runs that exact extracted code against a real,
// live ES module fixture - not a hand-copied reimplementation of the loop -
// so an accidental revert to a one-time copy is caught here, not just in a
// human review.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const MAIN_JS = readFileSync(
  join(HERE, "..", "localm", "plugins", "gui", "static", "app", "main.js"), "utf8");

const LOOP_ANCHOR = "for (const mod of [";
const loopStart = MAIN_JS.indexOf(LOOP_ANCHOR);
assert.notEqual(loopStart, -1,
  "main.js's window-export loop anchor not found - did its shape change? update this test's anchor to match.");
const LOOP_SRC = MAIN_JS.slice(loopStart); // the loop is the file's last statement

// main.js iterates a fixed list of 25 imported namespace objects (m0, m1,
// mI18nEn, mI18n, mIcons, mPk, m2..m20); alias every one of them to the same
// fixture module so the extracted loop runs, unmodified, against a real live
// namespace.
const MOD_PARAM_NAMES = ["m0", "m1", "mI18nEn", "mI18n", "mIcons", "mPk", "m2", "m3", "m4", "m5", "m6", "m7",
  "m8", "m9", "m10", "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20"];

test("window.X tracks a reassigned `export let` binding (RED against a one-time value copy, GREEN against a live getter)", async () => {
  const fixture = await import("./fixtures/reassignable-export.mjs");
  const window = {};
  const run = new Function(...MOD_PARAM_NAMES, "window", LOOP_SRC);
  run(...MOD_PARAM_NAMES.map(() => fixture), window);

  assert.deepEqual(window.counter, { n: 0 }, "initial value should come through");
  fixture.bump();
  assert.deepEqual(window.counter, { n: 1 },
    "window.counter must track the module's live reassignment, not freeze at the first snapshot");
  fixture.bump();
  assert.deepEqual(window.counter, { n: 2 }, "a second reassignment must also be reflected live");
});
