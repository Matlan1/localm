// SPDX-License-Identifier: AGPL-3.0-or-later
// One record in a .data-table is ONE row, with ONE continuous separator line.
//
// WHY THIS TEST EXISTS. `.data-table .name-cell` carried `display: flex` - on the
// <td> itself. A td with display:flex is no longer a table-cell: the browser wraps
// it in an anonymous cell, and the real td then sizes to its OWN content instead of
// stretching to the row height. Its border-bottom draws there too, so the row's
// separator line broke partway across, at a different height on every row. Measured
// on the Models page before the fix: the name cell's bottom edge sat 82px above
// every other cell's in the same row, and the actions cell stacked its six controls
// one per line, turning a single model into a 163px-tall block of staggered rules.
//
// NOTHING CAUGHT IT, and could not have:
//   - jsdom parses and builds a DOM but never lays out or paints, so no rendered-DOM
//     assertion can see a border in the wrong place;
//   - every existing structural test passed, because the elements were all present
//     and correctly nested - the defect was one declaration on one of them.
// It was found by driving the real page in a browser and measuring the cells'
// bounding boxes. This source scan is the cheap standing guard that replaces that,
// and it deliberately sits at BOTH layers the defect can come back through: the
// stylesheet declaration, and the markup pairing every call site has to get right.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const CSS = readFileSync(join(ROOT, "localm/plugins/gui/static/style.css"), "utf8");

// Every module that builds a .data-table name cell. A new one added here without
// the .cell-line wrapper reintroduces the broken separator on that page alone,
// which is exactly the shape that ships unnoticed.
const NAME_CELL_SOURCES = [
  "localm/plugins/gui/static/pages/models.js",
  "localm/plugins/gui/static/pages/plugins.js",
  "localm/plugins/gui/static/pages/knowledge.js",
  "localm/plugins/builtin/jobs/static/jobs.js",
];

// Every `selector { body }` pair in the sheet, as flat text. Good enough for this
// file's own hand-written CSS (no nested at-rule bodies are inspected, and the
// selectors checked below do not appear inside one).
function rules(css) {
  const out = [];
  const re = /([^{}]+)\{([^{}]*)\}/g;
  let m;
  while ((m = re.exec(css)) !== null) out.push({ sel: m[1].trim(), body: m[2] });
  return out;
}

// Rules that target the name cell ELEMENT itself - i.e. some selector in the list
// ends at `.name-cell`, rather than continuing into a descendant like
// `.name-cell .cell-line`. Those are the only ones that can style the <td>.
function rulesOnTheNameCell(css) {
  return rules(css).filter((r) => r.sel.split(",").some(
    (one) => /\.name-cell(?::[a-z-]+(?:\([^)]*\))?)*\s*$/.test(one.trim())));
}

test("the name cell's flex row is on .cell-line, never on the <td> itself", () => {
  for (const r of rulesOnTheNameCell(CSS)) {
    assert.ok(!/display\s*:\s*(flex|grid|inline-flex|inline-grid)/.test(r.body),
      "a display:flex/grid <td> stops being a table-cell, so its border-bottom draws "
      + "under its own content and the row separator breaks partway across the row. "
      + "Put the flex on the inner .cell-line instead. Offending rule: "
      + r.sel + " {" + r.body + "}");
  }
  // NOT a vacuous pass: the replacement has to be present, so this cannot go green
  // merely because somebody deleted the rule and left the cell with no layout.
  assert.ok(/\.data-table \.name-cell \.cell-line\s*\{[^}]*display\s*:\s*flex/.test(CSS),
    ".data-table .name-cell .cell-line carries the flex row");
});

test("every name cell in the app pairs its <td> with a .cell-line wrapper", () => {
  for (const rel of NAME_CELL_SOURCES) {
    const src = readFileSync(join(ROOT, rel), "utf8");
    // el("td", "name-cell") / el("td", "name-cell shrink-cell") / ...
    const tds = src.match(/el\(\s*["']td["']\s*,\s*["'][^"']*\bname-cell\b[^"']*["']/g) || [];
    const lines = src.match(/el\(\s*["']span["']\s*,\s*["']cell-line["']/g) || [];
    assert.ok(tds.length > 0, `${rel} builds at least one name cell`);
    assert.equal(lines.length, tds.length,
      `${rel}: ${tds.length} name cell(s) but ${lines.length} .cell-line wrapper(s). `
      + "Each name-cell <td> needs its own inner .cell-line to carry the flex row.");
  }
});

test("a row's action controls never wrap onto a second line", () => {
  const bodies = rules(CSS).filter((r) => /\.actions-cell\s*$/.test(r.sel)).map((r) => r.body);
  assert.ok(bodies.length > 0, "there is an .actions-cell rule");
  assert.ok(bodies.some((b) => /white-space\s*:\s*nowrap/.test(b)),
    "the actions cell pins white-space:nowrap, or its buttons stack one per line "
    + "and a single record becomes a multi-line block");

  // The inline style it replaced. An inline textAlign on a table action cell means
  // that cell missed the shared class, so it also missed the nowrap above.
  for (const rel of NAME_CELL_SOURCES) {
    const src = readFileSync(join(ROOT, rel), "utf8");
    assert.ok(!/\.style\.textAlign\s*=\s*["']right["']/.test(src),
      `${rel}: a row action cell right-aligns inline instead of using .actions-cell, `
      + "so it does not get the never-wrap rule that keeps the record on one row");
  }
});

test("the row <select> is not left on the full-width .card form-input styling", () => {
  // `.card select { width: 100% }` beats a lone `.model-type-select` class on
  // specificity, which is why the compact styling written for the row control never
  // applied: it rendered 36px tall and as wide as its cell, and it was the widest
  // thing in the row. The reset is on `.data-table select` so the next row control
  // to appear does not walk into the same trap.
  assert.ok(/\.data-table select\s*\{[^}]*width\s*:\s*auto/.test(CSS),
    ".data-table select resets the .card select width:100%");
  assert.ok(/\.card[^{]*\bselect\b[^{]*\{[^}]*width\s*:\s*100%/.test(CSS),
    "the .card select rule this exists to beat is still there - if it is gone, "
    + "the reset above is dead weight and this test is guarding nothing");
});
