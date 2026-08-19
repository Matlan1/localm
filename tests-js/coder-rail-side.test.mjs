import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadApp } from "./harness.mjs";

const css = () => readFileSync(
  fileURLToPath(new URL("../localm/plugins/gui/static/style.css", import.meta.url)), "utf8")
  .replace(/\/\*[\s\S]*?\*\//g, "");

test("rail side: the left variant flips the row rather than moving the element", () => {
  const c = css();
  // row-reverse, NOT a DOM move: the DOM order is what a screen reader and the tab
  // key follow, and the session list is secondary to the session in both. Moving
  // the element to change its visual side would reorder the reading sequence as a
  // side effect nobody asked for.
  assert.match(c, /#view-coder\.active\[data-rail="left"\]\s*\{[^}]*flex-direction:\s*row-reverse/,
    "left rail must be a row-reverse, not a reordered DOM");
});

test("rail side: the divider moves with the rail", () => {
  const c = css();
  const m = c.match(/#view-coder\[data-rail="left"\] #coder-sessions\s*\{([^}]*)\}/);
  assert.ok(m, "a left-rail rule for #coder-sessions exists");
  // A left rail keeping border-left draws a line against the window edge and none
  // against the content it separates - the border has to swap sides with it.
  assert.match(m[1], /border-left:\s*none/, "the old left border must be cleared");
  assert.match(m[1], /border-right:\s*1px solid/, "and redrawn on the content side");
});

// ASYNC, and it drains a tick before asserting. loadApp boots the real app, whose
// startup fetches resolve on a later microtask; if the test returns first, jsdom is
// torn down underneath them and the continuation dies on an undefined `document`.
// node:test reports that as "asynchronous activity after the test ended", which
// reads like a defect in the code under test and is not one - the same
// adjacent-question shape as everything else in this repo's rules.
const settle = () => new Promise((r) => setTimeout(r, 0));

test("rail side: 'left' is applied, and anything else falls back to the default", async () => {
  const { window } = loadApp({ fetchImpl: async () => ({
    ok: true, status: 200, json: async () => ({}), text: async () => "",
    headers: { get: () => null },
  }) });
  await settle();
  const view = window.document.getElementById("view-coder");
  assert.ok(view, "the coder view exists");

  window.applyCoderRailSide("left");
  assert.equal(view.dataset.rail, "left");

  window.applyCoderRailSide("right");
  assert.equal(view.dataset.rail, undefined, "right is the CSS default, so no attribute");

  // The cases that matter for an older server or a partial config payload: an
  // unknown value must lay the page out correctly rather than leave a rail on
  // neither side. Each is checked from the LEFT state, so a no-op would be caught.
  for (const bad of [undefined, null, "", "sideways", 0]) {
    window.applyCoderRailSide("left");
    window.applyCoderRailSide(bad);
    assert.equal(view.dataset.rail, undefined,
      `unexpected value ${JSON.stringify(bad)} must fall back to the default side`);
  }
});
