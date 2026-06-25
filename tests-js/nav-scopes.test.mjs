// SPDX-License-Identifier: AGPL-3.0-or-later
// KEY-SCOPE (frontend): the nav must show ONLY the tabs the current key may use.
// chat is the baseline anchor and is always visible; models/plugins/settings
// render only when the key grants models:read / plugins:read / config:read; a
// plugin tab renders only when the key grants that plugin's scope. Driven by
// GET /api/capabilities (.core + scope-filtered .plugins). No show-then-"no access".

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 60));

// fetch mock: serve a given /api/capabilities payload; everything else is a benign
// 200 so boot (bootAuthProbe -> /api/models) unlocks the app.
function fetchFor(caps) {
  return async (url) => {
    const u = String(url);
    if (u.includes("/api/capabilities")) {
      return { ok: true, status: 200, json: async () => caps, text: async () => "" };
    }
    return { ok: true, status: 200,
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
             text: async () => "" };
  };
}

const navShown = (window, view) => {
  const n = window.document.getElementById("nav-" + view);
  return !!n && n.style.display !== "none";
};

const caps = (core, plugins = []) => ({
  scopes: [], open: false, core, plugins, suggest_plugins: true,
});

test("chat-only key: only the chat tab shows (models/plugins/settings hidden)", async () => {
  const { window } = loadApp({ fetchImpl: fetchFor(
    caps({ chat: true, models: false, plugins: false, settings: false })) });
  await tick(); await tick();
  assert.ok(navShown(window, "chat"), "chat is baseline - always shown");
  assert.equal(navShown(window, "models"), false, "no models:read -> models tab hidden");
  assert.equal(navShown(window, "plugins"), false, "no plugins:read -> plugins tab hidden");
  assert.equal(navShown(window, "settings"), false, "no config:read -> settings tab hidden");
});

test("models:read: models tab shows, settings/plugins stay hidden", async () => {
  const { window } = loadApp({ fetchImpl: fetchFor(
    caps({ chat: true, models: true, plugins: false, settings: false })) });
  await tick(); await tick();
  assert.ok(navShown(window, "models"));
  assert.equal(navShown(window, "plugins"), false);
  assert.equal(navShown(window, "settings"), false);
});

test("config:read shows settings; plugins:read shows plugins", async () => {
  const { window } = loadApp({ fetchImpl: fetchFor(
    caps({ chat: true, models: false, plugins: true, settings: true })) });
  await tick(); await tick();
  assert.ok(navShown(window, "settings"));
  assert.ok(navShown(window, "plugins"));
  assert.equal(navShown(window, "models"), false);
});

test("owner / open mode: every core tab shows", async () => {
  const { window } = loadApp({ fetchImpl: fetchFor(
    { scopes: [], open: true,
      core: { chat: true, models: true, plugins: true, settings: true },
      plugins: [], suggest_plugins: true }) });
  await tick(); await tick();
  for (const v of ["chat", "models", "plugins", "settings"]) {
    assert.ok(navShown(window, v), `${v} must show`);
  }
});

test("a plugin tab renders only when capabilities lists it (server scope-filtered)", async () => {
  const { window } = loadApp({ fetchImpl: fetchFor(
    caps({ chat: true, models: false, plugins: false, settings: false },
      [{ name: "image", active: true, tab: "images", label: "Images",
         icon: "image", group: "studio", scope: "image", commands: [] }])) });
  await tick(); await tick();
  assert.ok(window.document.getElementById("nav-images"),
    "granted plugin (image) gets its nav tab");
});

test("an active tab that becomes inaccessible falls back to chat (no parking on a hidden view)", async () => {
  const core = { chat: true, models: false, plugins: false, settings: false };
  const { window } = loadApp({ fetchImpl: fetchFor(caps(core)) });
  await tick(); await tick();
  // Simulate the user having been on Settings (e.g. a remembered view), then a
  // capability refresh that no longer grants it.
  window.showView("settings");
  window.applyCoreTabVisibility(core);
  const settingsActive = window.document
    .getElementById("view-settings").classList.contains("active");
  assert.equal(settingsActive, false, "redirected off the now-hidden settings view");
  assert.ok(window.document.getElementById("view-chat").classList.contains("active"),
    "fell back to chat");
});
