// SPDX-License-Identifier: AGPL-3.0-or-later
// A page opened from a palette, link or reload shows its own nav button and is
// restored before its plugin tab is in VIEWS; a dialog builder runs with the
// dialog already visible (so focus() lands); the activity details follow an
// operation started in this tab, not only one found at boot.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));

const IMAGES = { name: "image", active: true, tab: "images", group: "studio", icon: "image", label: "Images" };
const MUSIC = { name: "music", active: true, tab: "music", group: "studio", icon: "music", label: "Music" };

test("a page in a collapsed group opens the group and shows its nav button", () => {
  const { window: win } = loadApp({ seedLocalStorage: { "localm.studioOpen": "0" } });
  runScript(win, `
    rebuildViews = function () {};
    reconcileActiveView = function () {};
    pluginState = ${JSON.stringify([IMAGES, MUSIC])};
    renderNav();
  `);
  const wrap = win.document.querySelector(".nav-group");
  const kids = wrap.querySelector(".nav-children");
  assert.equal(kids.style.display, "none", "precondition: the group starts closed");

  runScript(win, `showView("images");`);
  assert.equal(kids.style.display, "block");
  assert.ok(wrap.querySelector(".nav-group-parent").classList.contains("open"));
  assert.ok(win.document.getElementById("nav-images").classList.contains("active"));
});

test("a plugin page restored before its tab is in VIEWS becomes the active page", () => {
  const { window: win } = loadApp();
  runScript(win, `showView("images");`);
  assert.ok(win.document.getElementById("view-images").classList.contains("active"));
  assert.ok(!win.document.getElementById("view-chat").classList.contains("active"));
});

test("openModal shows the dialog before its builder runs", () => {
  const { window: win } = loadApp();
  let shown = null;
  win.openModal("t", () => { shown = win.document.getElementById("modal").style.display; });
  assert.equal(shown, "flex");
});

test("the activity details of an operation started in this tab carry its log lines", async () => {
  const fetchImpl = async (url) => {
    if (String(url) === "/api/activity") {
      return { ok: true, status: 200, json: async () => ({ operations: [
        { id: "j9", kind: "pull", label: "Model pull", status: "running" }] }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  runScript(win, `
    streamJob = async (id, onLine) => { onLine("fetching shard 1"); return new Promise(() => {}); };
  `);
  await tick(); await tick(); await tick();
  await win.pollActivity();
  win.showActivityDetails();
  assert.match(win.document.getElementById("modal-body").textContent, /fetching shard 1/);
});
