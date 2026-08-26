import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

// Comments are stripped so text inside them cannot match a declaration assertion.
const css = () => readFileSync(
  fileURLToPath(new URL("../localm/plugins/gui/static/style.css", import.meta.url)), "utf8")
  .replace(/\/\*[\s\S]*?\*\//g, "");

// Returns the declaration block of the given selector's rule.
const rule = (sel) => {
  const text = css();
  const at = text.indexOf("\n" + sel + " {");
  assert.ok(at !== -1, sel + " rule found in style.css");
  const open = text.indexOf("{", at);
  const close = text.indexOf("\n}", open);
  assert.ok(close !== -1, sel + " rule is closed");
  return text.slice(open + 1, close);
};

test("coder setup: the card is no longer pinned to a narrow column", () => {
  const r = rule("#coder-setup");
  assert.doesNotMatch(r, /max-width:\s*680px/,
    "680px left the form a narrow column with its flag row crushed beside it");
  assert.match(r, /max-width:\s*\d{4}px/,
    "a cap still exists - an unbounded form line on an ultrawide reads worse, not better");
});

test("coder setup: the six flags are a wrapping grid, not one flex row", () => {
  const r = rule("#coder-setup .checks");
  assert.match(r, /display:\s*grid/,
    "as a single flex row, six labelled checkboxes shared the leftover width and every label wrapped at once");
  assert.match(r, /grid-template-columns:\s*repeat\(auto-fit,\s*minmax\(/,
    "auto-fit + minmax gives each label a floor and drops COLUMNS on a narrow window instead of crushing all six");
});

test("coder setup: a wrapped flag label keeps its checkbox on the first line", () => {
  const r = rule("#coder-setup .checks label");
  assert.match(r, /align-items:\s*start/,
    "centering floats the checkbox to the middle of a two-line label");
});

test("coder setup: the API key field is styled like every other field on the form", () => {
  // The width/border/padding rule selects by input TYPE, so it silently excludes
  // any type nobody thought to list. The model-server API key is the form's only
  // password input, and it rendered as a bare unstyled box a third the width of
  // its neighbours until password was added - invisible to a jsdom test, which
  // applies no CSS, and caught only by looking at the rendered page.
  const text = css();
  const at = text.indexOf('#coder-setup input[type="text"]');
  assert.ok(at !== -1, "the shared field rule exists");
  const selector = text.slice(at, text.indexOf("{", at));
  assert.match(selector, /input\[type="password"\]/,
    "a password input on this form must share the field styling, not inherit browser defaults");
});
