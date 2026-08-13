// SPDX-License-Identifier: AGPL-3.0-or-later
// gui-design.md rule 3: two button tiers plus danger, one vocabulary.
// "Never ship a class-less el("button", "", ...)."
//
// WHY THIS TEST EXISTS, and it is the useful part. Three settings-page buttons
// shipped as el("button", "btn", ...) - "Warm up now", "Synthesize now" and
// "Keep as is". `.btn` has NO rule anywhere in style.css, so those rendered as
// fully NATIVE buttons: light grey fill, black text, a 2px black border and
// square corners, inside a dark app, with "Keep as is" sitting directly beside
// an accent-filled .btn-primary.
//
// Nothing caught it:
//   - jsdom parses and builds a DOM but never LAYS OUT or PAINTS (see the blind
//     spot note at the top of harness.mjs), so every DOM assertion passed;
//   - a grep for the literal el("button", "") missed it, because the class was
//     "btn" rather than "". An instrument that matches one exact string cannot
//     see a defect written with a different one.
// It was found by driving the real page in a browser and reading COMPUTED
// STYLES. This test is the cheap standing guard that replaces that.
//
// It is deliberately a SOURCE scan rather than a rendered-DOM check: the defect
// is invisible in a rendered DOM (the element exists and has a class either
// way), and many of these buttons only appear after a modal or a job stream.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const STATIC = join(ROOT, "localm", "plugins", "gui", "static");

const SETTINGS_SOURCES = [
  join("pages", "settings.js"),
  join("app", "settings-perf.js"),
];

/** Every class token that style.css actually defines a rule for. */
function definedClasses() {
  const css = readFileSync(join(STATIC, "style.css"), "utf-8");
  const out = new Set();
  for (const m of css.matchAll(/\.([A-Za-z][\w-]*)/g)) out.add(m[1]);
  return out;
}

/** Every el("button", "<classes>", ...) call site in a source file. */
function buttonClassSites(rel) {
  const src = readFileSync(join(STATIC, rel), "utf-8");
  const sites = [];
  const lines = src.split(/\r?\n/);
  lines.forEach((line, i) => {
    for (const m of line.matchAll(/el\(\s*"button"\s*,\s*"([^"]*)"/g)) {
      sites.push({ file: rel, line: i + 1, classes: m[1] });
    }
  });
  return sites;
}

test("rule 3: no settings-page button is built with an empty class", () => {
  const bad = SETTINGS_SOURCES.flatMap(buttonClassSites)
    .filter((s) => s.classes.trim() === "");
  assert.deepEqual(bad, [],
    "el(\"button\", \"\", ...) ships an unstyled native button: " + JSON.stringify(bad));
});

// NAVIGATION buttons, which gui-design.md governs under rule 2 (hover + an
// inset accent bar, the settings-nav treatment) rather than rule 3's action
// tiers. Both carry full styling of their own; demanding a .btn-* tier of them
// would be a false positive, and a gate that cries wolf gets switched off.
// Asserted to be exactly these two so the carve-out cannot grow unnoticed.
const NAV_BUTTON_CLASSES = ["settings-nav-link", "nav-group-parent"];

test("rule 3: every settings-page ACTION button carries a real tier", () => {
  const sites = SETTINGS_SOURCES.flatMap(buttonClassSites);
  assert.ok(sites.length > 20, `expected many button sites, found ${sites.length}`);

  const isNav = (s) => NAV_BUTTON_CLASSES.some(
    (c) => new RegExp(`\\b${c}\\b`).test(s.classes));
  assert.equal(sites.filter(isNav).length, NAV_BUTTON_CLASSES.length,
    "the navigation carve-out should cover exactly the two nav buttons");

  const untiered = sites.filter((s) => !isNav(s)
    && !/\b(btn-primary|btn-secondary|btn-danger)\b/.test(s.classes));
  assert.deepEqual(untiered, [],
    "these buttons carry no .btn-primary/.btn-secondary/.btn-danger tier, so "
    + "they fall back to the browser's native rendering: " + JSON.stringify(untiered));
});

test("no settings-page button relies ONLY on classes style.css never defines", () => {
  // The property `.btn` actually violated, and the sharpest form of it: a class
  // name that LOOKS deliberate but matches no rule renders exactly like no class
  // at all, and reads as intentional in review.
  //
  // The check is "at least ONE token has a rule", not "every token has one":
  // several buttons legitimately carry a hook class used only as a querySelector
  // (.settings-section-save, .comfy-managed-update-btn, .media-save, ...), and
  // requiring a rule for those would flag 11 correct sites. That version of this
  // test was written first and is exactly the false-positive gate the project's
  // rules warn about, so it was narrowed rather than shipped.
  const defined = definedClasses();
  const unstyled = [];
  for (const s of SETTINGS_SOURCES.flatMap(buttonClassSites)) {
    const tokens = s.classes.split(/\s+/).filter(Boolean);
    if (!tokens.some((t) => defined.has(t))) {
      unstyled.push(`${s.file}:${s.line} "${s.classes}"`);
    }
  }
  assert.deepEqual(unstyled, [],
    "buttons whose every class is undefined in style.css, so they render "
    + "natively: " + JSON.stringify(unstyled));
});
