// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - tabs (split from app.js). Classic script: it
   shares the one global lexical environment with the other app/* and
   pages/* scripts, so every cross-section reference resolves by bare
   name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { chat } from "./chat.js";
import { $ } from "./helpers.js";

/* ================================================================ */
/*  Tabs                                                             */
/* ================================================================ */

// Kernel pages are always present; plugin views (coder, images, music, video,
// knowledge) are added to VIEWS by renderNav() while their plugin is active.
export const CORE_VIEWS = ["chat", "models", "plugins", "settings"];
export let VIEWS = [...CORE_VIEWS];

// Toggle the .active class on the view sections + nav buttons. Split out of
// showView so the nav rebuild (reconcileActiveView) can re-assert the highlight
// on freshly-created buttons WITHOUT re-running onViewShown - re-firing
// onViewShown for chat/coder calls refreshPluginCommands, which calls renderNav
// -> reconcileActiveView -> showView -> onViewShown, an infinite /api/plugins
// loop.
export function _applyActiveClasses(name) {
  for (const v of VIEWS) {
    const view = $("view-" + v), nav = $("nav-" + v);
    if (view) view.classList.toggle("active", v === name);
    if (nav) nav.classList.toggle("active", v === name);
  }
}

/** R09/R10: is the Settings view currently the active one? */
export function isSettingsView() {
  const v = document.querySelector(".view.active");
  return !!v && v.id === "view-settings";
}
window.isSettingsView = isSettingsView;

export function showView(name) {
  // Fall back to chat for an unknown name OR a view whose section is gone
  // (e.g. a remembered tab whose plugin was since uninstalled). Tolerating a
  // missing nav/view element is what lets the nav rail be rebuilt at runtime.
  if (!$("view-" + name)) name = "chat";
  // R10: leaving Settings with unsaved edits warns first (returning to Settings
  // re-renders the form from server state, so the edits would be silently lost).
  if (name !== "settings" && isSettingsView() &&
      window.settingsDirty && window.settingsDirty()) {
    if (!confirm("You have unsaved settings changes. Leave without saving?")) return;
  }
  _applyActiveClasses(name);
  // Remembered across reloads - but never in privacy mode (no traces).
  if (!chat.privacy) localStorage.setItem("localm.activeView", name);
  // On a phone the sidebar is an off-canvas drawer; navigating closes it.
  closeNav();
  // Lazy page refreshes live in pages.js
  if (window.onViewShown) window.onViewShown(name);
}
// Kernel nav buttons are static; plugin nav buttons get their handler in
// renderNav() as they are (re)created from the active-plugin set.
for (const v of CORE_VIEWS) $("nav-" + v).onclick = () => showView(v);

// --- mobile sidebar drawer ------------------------------------------------
// On a narrow screen the sidebar is off-canvas; the hamburger in the top bar
// toggles it, the backdrop or any navigation closes it. No-ops on desktop,
// where the sidebar is always visible and the toggle/backdrop are hidden.
export function setNavOpen(open) {
  const app = $("app");
  if (app) app.classList.toggle("nav-open", open);
  const toggle = $("nav-toggle");
  if (toggle) toggle.setAttribute("aria-expanded", open ? "true" : "false");
}
export function closeNav() { setNavOpen(false); }
if ($("nav-toggle")) {
  $("nav-toggle").onclick = () => {
    const app = $("app");
    setNavOpen(!(app && app.classList.contains("nav-open")));
  };
}
if ($("sidebar-backdrop")) $("sidebar-backdrop").onclick = closeNav;
// Close the drawer if the viewport grows back to desktop width (e.g. rotate).
window.addEventListener("resize", () => {
  if (window.innerWidth > 760) closeNav();
});

