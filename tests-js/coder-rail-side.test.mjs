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
  assert.match(c, /#view-coder\.active\[data-rail="left"\]\s*\{[^}]*flex-direction:\s*row-reverse/,
    "left rail must be a row-reverse, not a reordered DOM");
});

test("rail side: the divider moves with the rail", () => {
  const c = css();
  const m = c.match(/#view-coder\[data-rail="left"\] #coder-sessions\s*\{([^}]*)\}/);
  assert.ok(m, "a left-rail rule for #coder-sessions exists");
  assert.match(m[1], /border-left:\s*none/, "the old left border must be cleared");
  assert.match(m[1], /border-right:\s*1px solid/, "and redrawn on the content side");
});

// drains a tick so the app's startup fetches resolve before the test returns
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

  // each unknown value is applied from the LEFT state
  for (const bad of [undefined, null, "", "sideways", 0]) {
    window.applyCoderRailSide("left");
    window.applyCoderRailSide(bad);
    assert.equal(view.dataset.rail, undefined,
      `unexpected value ${JSON.stringify(bad)} must fall back to the default side`);
  }
});
