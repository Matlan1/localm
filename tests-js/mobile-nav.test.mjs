// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

// Issues 3/5: on a phone the 230px sidebar squeezes the content to one word per
// line. It becomes an off-canvas drawer toggled by a hamburger; the backdrop or
// any navigation closes it. (CSS @media does the slide; this pins the JS logic.)

const allOk = () => Promise.resolve({
  ok: true, status: 200, json: async () => ({ models: [], active: "" }), text: async () => "",
});

test("drawer: hamburger opens, backdrop closes, aria-expanded tracks state", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  const app = window.document.getElementById("app");
  const toggle = window.document.getElementById("nav-toggle");
  const backdrop = window.document.getElementById("sidebar-backdrop");

  assert.ok(!app.classList.contains("nav-open"), "starts closed");
  toggle.click();
  assert.ok(app.classList.contains("nav-open"), "hamburger opens the drawer");
  assert.equal(toggle.getAttribute("aria-expanded"), "true");

  backdrop.click();
  assert.ok(!app.classList.contains("nav-open"), "backdrop closes the drawer");
  assert.equal(toggle.getAttribute("aria-expanded"), "false");
});

test("drawer: navigating to a view closes it", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  const app = window.document.getElementById("app");
  window.setNavOpen(true);
  assert.ok(app.classList.contains("nav-open"));
  window.showView("models");
  assert.ok(!app.classList.contains("nav-open"), "showView closes the drawer");
});

// The phone top bar carries its own new-chat button (the sidebar + is behind a
// closed drawer there). It starts a fresh conversation, lands on chat, and
// closes the drawer if it was open.
test("mobile top-bar new-chat: starts a chat, shows it, closes the drawer", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  const doc = window.document;
  const app = doc.getElementById("app");
  const mtbNew = doc.getElementById("mtb-new");
  assert.ok(mtbNew, "the mobile top bar has a new-chat button");

  window.showView("models");
  window.setNavOpen(true);
  assert.ok(app.classList.contains("nav-open"));

  mtbNew.click();

  assert.ok(doc.getElementById("view-chat").classList.contains("active"),
    "new-chat lands on the chat view");
  assert.ok(!app.classList.contains("nav-open"),
    "new-chat closes the drawer");
});
