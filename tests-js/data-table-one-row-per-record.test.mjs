// SPDX-License-Identifier: AGPL-3.0-or-later
// Source scan over the stylesheet and the page modules: a .data-table record
// renders as one row with one continuous separator line.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const CSS = readFileSync(join(ROOT, "localm/plugins/gui/static/style.css"), "utf8");

// Every module that builds a .data-table name cell.
const NAME_CELL_SOURCES = [
  "localm/plugins/gui/static/pages/models.js",
  "localm/plugins/gui/static/pages/plugins.js",
  "localm/plugins/gui/static/pages/knowledge.js",
  "localm/plugins/builtin/jobs/static/jobs.js",
];

// Every `selector { body }` pair in the sheet, as flat text. Nested at-rule
// bodies are not inspected.
function rules(css) {
  const out = [];
  const re = /([^{}]+)\{([^{}]*)\}/g;
  let m;
  while ((m = re.exec(css)) !== null) out.push({ sel: m[1].trim(), body: m[2] });
  return out;
}

// Rules with a selector ending at `.name-cell`, i.e. targeting the <td> itself
// rather than a descendant such as `.name-cell .cell-line`.
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

  // An inline right-align means the cell missed the shared .actions-cell class.
  for (const rel of NAME_CELL_SOURCES) {
    const src = readFileSync(join(ROOT, rel), "utf8");
    assert.ok(!/\.style\.textAlign\s*=\s*["']right["']/.test(src),
      `${rel}: a row action cell right-aligns inline instead of using .actions-cell, `
      + "so it does not get the never-wrap rule that keeps the record on one row");
  }
});

test("the row <select> is not left on the full-width .card form-input styling", () => {
  assert.ok(/\.data-table select\s*\{[^}]*width\s*:\s*auto/.test(CSS),
    ".data-table select resets the .card select width:100%");
  assert.ok(/\.card[^{]*\bselect\b[^{]*\{[^}]*width\s*:\s*100%/.test(CSS),
    "the .card select rule this exists to beat is still there - if it is gone, "
    + "the reset above is dead weight and this test is guarding nothing");
});
