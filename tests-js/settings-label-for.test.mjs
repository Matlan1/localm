// SPDX-License-Identifier: AGPL-3.0-or-later
// Every schema-driven Settings control is the control its own label names, so
// assistive tech (and anything that finds a control by its label) reaches the
// Settings "Session persistence" select, not the Coder tab's select that
// carries the same label text.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const MODE_FIELD = { key: "mode", widget: "select", label: "Session persistence", help: "",
  group: "Privacy", owner: "core", options: ["privacy", "log", "full"],
  default: "privacy", shipped_default: "privacy" };

test("each Settings label names its own control, with ids unique across renders", () => {
  const { window } = loadAppWithPages();
  runScript(window, `
    window.__ctrls = [
      buildSettingControl(${JSON.stringify(MODE_FIELD)}),
      buildSettingControl(${JSON.stringify(MODE_FIELD)}),
      buildSettingControl({ key: "require_auth", widget: "toggle", label: "Require auth",
        help: "", group: "Security", owner: "core", default: false }),
    ];
    for (const c of window.__ctrls) document.body.appendChild(c.node);
  `);
  const ids = new Set();
  for (const ctrl of window.__ctrls) {
    const label = ctrl.node.querySelector("label");
    const input = ctrl.node.querySelector(`[data-key="${ctrl.field.key}"]`);
    assert.ok(input, "the control element is rendered");
    assert.equal(label.control, input, `the "${ctrl.field.label}" label names its control`);
    ids.add(input.id);
  }
  assert.equal(ids.size, 3, "two renders of the same field never share an id");

  const coderSelect = window.document.getElementById("setup-mode");
  assert.ok(coderSelect, "precondition: the Coder tab has its own Session persistence select");
  const settingsLabels = window.__ctrls.slice(0, 2).map((c) => c.node.querySelector("label"));
  for (const label of settingsLabels) assert.notEqual(label.control, coderSelect);
});
