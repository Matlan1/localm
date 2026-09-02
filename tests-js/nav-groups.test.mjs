// SPDX-License-Identifier: AGPL-3.0-or-later
// Nav categories are data, not one hardcoded name. "studio" was spelled into
// renderNav directly, so a second category (Coder, holding the agent and the
// browser) could not exist without duplicating the whole renderer.
//
// The hybrid shape is unchanged and is what these pin: nothing for 0 enabled
// members, a flat tab for exactly 1, a collapsible parent for 2+.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

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

const CODER = { name: "coder", active: true, tab: "coder", group: "coder", icon: "code", label: "Coder" };
const BROWSER = { name: "browser", active: true, tab: "browser", group: "coder", icon: "web", label: "Browser" };
const IMAGES = { name: "image", active: true, tab: "images", group: "studio", icon: "image", label: "Images" };
const MUSIC = { name: "music", active: true, tab: "music", group: "studio", icon: "music", label: "Music" };

test("one member of a category renders a flat tab, not a parent", () => {
  const win = renderWith([CODER]);
  const slot = win.document.getElementById("nav-plugin-slot");
  assert.ok(slot.querySelector('[id="nav-coder"]'), "the flat tab is there");
  assert.equal(slot.querySelector(".nav-group-parent"), null,
    "one member must not be collapsed under a parent");
});

test("two members collapse under the category's own parent", () => {
  const win = renderWith([CODER, BROWSER]);
  const slot = win.document.getElementById("nav-plugin-slot");
  const parent = slot.querySelector(".nav-group-parent");
  assert.ok(parent, "a parent is rendered for 2+ members");
  assert.match(parent.textContent, /Coder/,
    "the parent takes the category's own label, not Studio's");
  assert.ok(slot.querySelector('[id="nav-coder"]'), "child: the agent");
  assert.ok(slot.querySelector('[id="nav-browser"]'), "child: the browser");
});

test("the studio category still behaves exactly as before", () => {
  const win = renderWith([IMAGES, MUSIC]);
  const slot = win.document.getElementById("nav-plugin-slot");
  const parent = slot.querySelector(".nav-group-parent");
  assert.ok(parent);
  assert.match(parent.textContent, /Studio/);
  assert.ok(slot.querySelector('[id="nav-images"]'));
  assert.ok(slot.querySelector('[id="nav-music"]'));
});

test("two categories render side by side without borrowing each other's members", () => {
  const win = renderWith([CODER, BROWSER, IMAGES, MUSIC]);
  const slot = win.document.getElementById("nav-plugin-slot");
  const parents = [...slot.querySelectorAll(".nav-group-parent")]
    .map((n) => n.textContent);
  assert.equal(parents.length, 2, parents);
  assert.ok(parents.some((t) => /Coder/.test(t)), parents);
  assert.ok(parents.some((t) => /Studio/.test(t)), parents);
  // Each parent holds exactly its own two children.
  for (const wrap of slot.querySelectorAll(".nav-group")) {
    const kids = wrap.querySelectorAll(".nav-child");
    assert.equal(kids.length, 2, wrap.textContent);
  }
});

test("an unknown tab in a category is still rendered rather than dropped", () => {
  const win = renderWith([
    CODER, BROWSER,
    { name: "x", active: true, tab: "novel", group: "coder", icon: "code", label: "Novel" },
  ]);
  const slot = win.document.getElementById("nav-plugin-slot");
  assert.ok(slot.querySelector('[id="nav-novel"]'),
    "a member with an unlisted tab counts toward the group and must render");
});

test("a plugin in no category is still a flat tab", () => {
  const win = renderWith([
    { name: "jobs", active: true, tab: "jobs", group: "", icon: "clock", label: "Jobs" },
  ]);
  const slot = win.document.getElementById("nav-plugin-slot");
  assert.ok(slot.querySelector('[id="nav-jobs"]'));
  assert.equal(slot.querySelector(".nav-group-parent"), null);
});

test("the collapse state is remembered per category, not shared", () => {
  // NAV_GROUPS is a top-level const, so it lives in the realm's lexical env
  // rather than on window; read it from inside the realm.
  const { window: win } = loadApp();
  runScript(win, "window.__groups = JSON.stringify(NAV_GROUPS);");
  const groups = JSON.parse(win.__groups);
  assert.ok(groups.coder && groups.studio, Object.keys(groups).join(","));
  assert.notEqual(groups.coder.key, groups.studio.key,
    "one storage key for both would collapse them together");
});
