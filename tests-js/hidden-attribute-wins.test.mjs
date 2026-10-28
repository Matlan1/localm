// SPDX-License-Identifier: AGPL-3.0-or-later
// The hidden attribute must hide an element even when a class rule in
// style.css gives it a display value (`.card .actions { display: flex }` kept
// Settings > Updates "Show all" on screen with no release notes).
//
// jsdom resolves display:none for every [hidden] element whatever the
// stylesheet says, so getComputedStyle cannot show this defect. The tests
// resolve `display` from the shipped style.css with a small cascade: author
// !important, then author normal (beating the user-agent `[hidden]` rule),
// ordered by specificity and then source order. Rules inside at-rules
// (@media and friends) are left out: the desktop layout has no media query
// that matches.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";
import { loadAppWithPages } from "./harness.mjs";

const STYLE_CSS = readFileSync(
  new URL("../localm/plugins/gui/static/style.css", import.meta.url), "utf8");
const BROWSER_CSS = readFileSync(
  new URL("../localm/plugins/builtin/browser/static/browser.css", import.meta.url), "utf8");

/** Split a selector list on top-level commas. */
function splitTop(text) {
  const out = [];
  let depth = 0, cur = "";
  for (const ch of text) {
    if (ch === "(" || ch === "[") depth++;
    else if (ch === ")" || ch === "]") depth--;
    if (ch === "," && depth === 0) { out.push(cur.trim()); cur = ""; } else cur += ch;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const cmp = (a, b) => (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]);
const LEGACY_PSEUDO_ELEMENTS = new Set(["before", "after", "first-line", "first-letter"]);

/** Selectors Level 4 specificity of one complex selector as [ids, classes, types]. */
function specificity(sel) {
  let s = [0, 0, 0];
  let i = 0;
  const ident = () => {
    let name = "";
    while (i < sel.length && /[-\w\\]/.test(sel[i])) name += sel[i++];
    return name;
  };
  const parens = () => {
    if (sel[i] !== "(") return null;
    let depth = 0, body = "";
    for (; i < sel.length; i++) {
      if (sel[i] === "(") { if (depth++ === 0) continue; }
      else if (sel[i] === ")") { if (--depth === 0) { i++; break; } }
      body += sel[i];
    }
    return body;
  };
  while (i < sel.length) {
    const ch = sel[i];
    if (ch === "#") { i++; ident(); s = add(s, [1, 0, 0]); }
    else if (ch === ".") { i++; ident(); s = add(s, [0, 1, 0]); }
    else if (ch === "[") { while (i < sel.length && sel[i] !== "]") i++; i++; s = add(s, [0, 1, 0]); }
    else if (ch === ":") {
      i++;
      if (sel[i] === ":") { i++; ident(); parens(); s = add(s, [0, 0, 1]); continue; }
      const name = ident().toLowerCase();
      const body = parens();
      if (["not", "is", "has", "matches"].includes(name) && body !== null) {
        s = add(s, splitTop(body).map(specificity).reduce((m, x) => (cmp(x, m) > 0 ? x : m), [0, 0, 0]));
      } else if (name === "where") {
        // zero specificity
      } else if (LEGACY_PSEUDO_ELEMENTS.has(name)) {
        s = add(s, [0, 0, 1]);
      } else {
        s = add(s, [0, 1, 0]);
      }
    } else if (/[a-zA-Z]/.test(ch)) { ident(); s = add(s, [0, 0, 1]); }
    else i++;
  }
  return s;
}

/** Top-level style rules that declare `display`, in source order. */
function displayRules(cssText) {
  const sheet = new JSDOM(`<style>${cssText}</style>`).window.document.styleSheets[0];
  const rules = [];
  for (const rule of sheet.cssRules) {
    if (!rule.selectorText || !rule.style) continue;
    const value = rule.style.getPropertyValue("display");
    if (!value) continue;
    rules.push({
      selectors: splitTop(rule.selectorText),
      value: value.trim(),
      important: rule.style.getPropertyPriority("display") === "important",
    });
  }
  return rules;
}

const RULES = displayRules(STYLE_CSS);

/** The `display` the cascade gives `el` under style.css, or "ua-none" /
 *  "ua-default" when no author rule declares one. */
function resolvedDisplay(el) {
  let best = null;
  RULES.forEach((rule, order) => {
    for (const sel of rule.selectors) {
      let hit = false;
      try { hit = el.matches(sel); } catch { hit = false; }
      if (!hit) continue;
      const cand = { value: rule.value, important: rule.important, spec: specificity(sel), order };
      if (!best
          || (cand.important && !best.important)
          || (cand.important === best.important
              && (cmp(cand.spec, best.spec) > 0
                  || (cmp(cand.spec, best.spec) === 0 && cand.order > best.order)))) {
        best = cand;
      }
    }
  });
  const inline = el.style && el.style.getPropertyValue("display");
  if (inline) {
    const inlineImportant = el.style.getPropertyPriority("display") === "important";
    if (inlineImportant || !best || !best.important) return inline;
  }
  if (best) return best.value;
  return el.hidden ? "ua-none" : "ua-default";
}

const isHiddenDisplay = (d) => d === "none" || d === "ua-none";

function makeFetch(check) {
  return async (url) => {
    const u = String(url);
    if (u.includes("/api/update/check")) {
      return { ok: true, status: 200, text: async () => "", json: async () => check };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function flush() {
  for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
}

test("resolver self-check: a class display rule beats a bare [hidden]", () => {
  const { window } = new JSDOM(
    '<div class="card"><div class="actions" id="a" hidden></div></div>');
  const el = window.document.getElementById("a");
  assert.equal(resolvedDisplay(el), "none", "style.css as shipped hides it");
  const saved = RULES.splice(0, RULES.length,
    ...displayRules(".card .actions { display: flex; }"));
  try {
    assert.equal(resolvedDisplay(el), "flex",
      "without a [hidden] author rule the class rule must win, as in a browser");
  } finally {
    RULES.splice(0, RULES.length, ...saved);
  }
  assert.deepEqual(specificity(".card .actions"), [0, 2, 0]);
  assert.deepEqual(specificity("#a:not(.b, #c) span::before"), [2, 0, 2]);
  assert.deepEqual(specificity(":where(#a) [hidden]"), [0, 1, 0]);
});

test("test_hidden_attribute_beats_every_display_rule: every [hidden] element in the page resolves to display none", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({}) });
  await flush();
  const hidden = [...window.document.querySelectorAll("[hidden]")];
  assert.ok(hidden.length > 20, `expected the shipped page's hidden elements, got ${hidden.length}`);
  const shown = hidden
    .map((el) => ({ el, display: resolvedDisplay(el) }))
    .filter(({ display }) => !isHiddenDisplay(display))
    .map(({ el, display }) => `${el.tagName.toLowerCase()}#${el.id}.${el.className}: ${display}`);
  assert.deepEqual(shown, [], "a [hidden] element still gets a display value");
});

test("no stylesheet rule can out-rank [hidden] with an important non-none display", () => {
  for (const [name, css] of [["style.css", STYLE_CSS], ["browser.css", BROWSER_CSS]]) {
    const offenders = displayRules(css)
      .filter((r) => r.important && r.value !== "none")
      .map((r) => r.selectors.join(", "));
    assert.deepEqual(offenders, [], `${name} has an important display that beats [hidden]`);
  }
});

test("Updates: no 'Show all' before a check, when up to date, or with no build attached", async () => {
  const cases = [
    ["up to date", { available: true, newer: false, current: "0.3.2", latest: "v0.3.2", notes: "" }],
    ["no build attached", { available: true, newer: true, current: "0.3.1", latest: "v0.3.2",
      notes: Array.from({ length: 12 }, (_, i) => `- item ${i}`).join("\n") }],
    ["short notes", { available: true, newer: true, current: "0.3.1", latest: "v0.3.2",
      notes: "Small fix.", asset: { id: 7 } }],
  ];
  for (const [label, check] of cases) {
    const { window } = loadAppWithPages({ fetchImpl: makeFetch(check) });
    await flush();
    const row = window.document.getElementById("update-notes-actions");
    assert.equal(row.hidden, true, `${label}: hidden before any check`);
    assert.ok(isHiddenDisplay(resolvedDisplay(row)), `${label}: not displayed before any check`);
    window.document.getElementById("update-check").click();
    await flush();
    assert.equal(row.hidden, true, `${label}: hidden after the check`);
    assert.ok(isHiddenDisplay(resolvedDisplay(row)),
      `${label}: not displayed after the check (got ${resolvedDisplay(row)})`);
  }
});

test("Updates: long notes show 'Show all', displayed as the flex action row", async () => {
  const notes = Array.from({ length: 12 }, (_, i) => `- change ${i}`).join("\n");
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(
    { available: true, newer: true, current: "0.3.1", latest: "v0.3.2", notes, asset: { id: 7 } }) });
  await flush();
  window.document.getElementById("update-check").click();
  await flush();
  const row = window.document.getElementById("update-notes-actions");
  assert.equal(row.hidden, false);
  assert.equal(resolvedDisplay(row), "flex", "the row keeps its .card .actions layout when shown");
  const box = window.document.getElementById("update-notes");
  window.document.getElementById("update-notes-more").click();
  assert.ok(!box.classList.contains("collapsed"), "Show all expands the notes");
});
