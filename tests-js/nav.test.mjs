// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

// renderNav() builds the dynamic nav from the top-level `pluginState`, a `let`
// in the realm's global lexical env rather than a window property, so it is
// seeded from an injected script. rebuildViews / reconcileActiveView are
// stubbed out.
function renderWith(plugins) {
  const { window } = loadApp();
  runScript(window, `
    rebuildViews = function () {};
    reconcileActiveView = function () {};
    pluginState = ${JSON.stringify(plugins)};
    renderNav();
  `);
  return window;
}

test("LBUG-1: two active plugins sharing a tab do not produce duplicate nav ids", () => {
  const win = renderWith([
    { name: "a", active: true, tab: "extras", group: "", icon: "code", label: "A" },
    { name: "b", active: true, tab: "extras", group: "", icon: "code", label: "B" },
  ]);
  const slot = win.document.getElementById("nav-plugin-slot");
  const matches = slot.querySelectorAll('[id="nav-extras"]');
  assert.equal(matches.length, 1, "exactly one nav node for the shared tab");
});

test("LGAP-1: a studio plugin with an unknown tab is still rendered as a child", () => {
  const win = renderWith([
    { name: "img", active: true, tab: "images", group: "studio", icon: "image", label: "Images" },
    { name: "threed", active: true, tab: "threed", group: "studio", icon: "image", label: "3D" },
  ]);
  const slot = win.document.getElementById("nav-plugin-slot");
  assert.ok(slot.querySelector('[id="nav-threed"]'),
    "an unknown-tab studio plugin must still be rendered, not silently dropped");
});

test("jobs plugin gets its clock SVG icon, not the generic fallback", () => {
  const win = renderWith([
    { name: "jobs", active: true, tab: "jobs", group: "", icon: "clock", label: "Jobs" },
  ]);
  const node = win.document.getElementById("nav-jobs");
  assert.ok(node, "jobs nav node rendered");
  assert.ok(node.querySelector('.nav-ic[data-icon-name="clock"] svg'),
    "the clock SVG icon is shown");
  assert.ok(!node.querySelector('[data-icon-name="plugins"]'),
    "not the generic plugins fallback");
  assert.ok(node.textContent.includes("Jobs"), "label shown");
  assert.ok(!node.textContent.includes("•"), "no bullet fallback text");
});

test("rebuildViews (unstubbed) puts every CORE_VIEWS entry into VIEWS", () => {
  // Every other test in this file stubs rebuildViews out entirely, which is
  // exactly how a hardcoded second copy of the core view list (instead of
  // deriving it from CORE_VIEWS) went untested: VIEWS.push("chat", "models",
  // "plugins", "settings") silently dropped any view CORE_VIEWS gained later.
  const { window } = loadApp();
  runScript(window, `
    reconcileActiveView = function () {};
    pluginState = ${JSON.stringify([
      { name: "coder", active: true, tab: "coder", group: "", icon: "code", label: "Coder" },
    ])};
    renderNav();
    window.__views = [...VIEWS];
    window.__coreViews = [...CORE_VIEWS];
  `);
  const views = window.__views;
  for (const v of window.__coreViews) {
    assert.ok(views.includes(v), `VIEWS is missing core view "${v}": ${views}`);
  }
  assert.ok(views.includes("coder"), "an active plugin's own tab must also survive");
  assert.equal(views[0], "chat", "chat stays the static first anchor");
});

test("renderNav still renders ordinary flat + studio tabs", () => {
  const win = renderWith([
    { name: "coder", active: true, tab: "coder", group: "", icon: "code", label: "Coder" },
    { name: "img", active: true, tab: "images", group: "studio", icon: "image", label: "Images" },
    { name: "music", active: true, tab: "music", group: "studio", icon: "music", label: "Music" },
  ]);
  const slot = win.document.getElementById("nav-plugin-slot");
  assert.ok(slot.querySelector('[id="nav-coder"]'), "flat coder tab rendered");
  assert.ok(slot.querySelector('[id="nav-images"]'), "studio child images rendered");
  assert.ok(slot.querySelector('[id="nav-music"]'), "studio child music rendered");
});
