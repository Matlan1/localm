// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - tabs. */
"use strict";

// --- ES module imports ---
import { lsSetScoped } from "./chat.js";
import { $ } from "./helpers.js";

// Kernel pages are always present; plugin views (coder, images, music, video,
// knowledge) are added to VIEWS by renderNav() while their plugin is active.
export const CORE_VIEWS = ["chat", "models", "plugins", "settings", "setup"];
export let VIEWS = [...CORE_VIEWS];

// Toggle the .active class on the view sections and nav buttons, without
// running onViewShown.
export function _applyActiveClasses(name) {
  for (const v of VIEWS) {
    const view = $("view-" + v), nav = $("nav-" + v);
    if (view) view.classList.toggle("active", v === name);
    if (nav) nav.classList.toggle("active", v === name);
  }
}

/** Whether the Settings view is the active one. */
export function isSettingsView() {
  const v = document.querySelector(".view.active");
  return !!v && v.id === "view-settings";
}
window.isSettingsView = isSettingsView;

export function showView(name) {
  // Fall back to chat for an unknown name or a view whose section is missing.
  if (!$("view-" + name)) name = "chat";
  // Leaving Settings with unsaved edits asks for confirmation first.
  if (name !== "settings" && isSettingsView() &&
      window.settingsDirty && window.settingsDirty()) {
    if (!confirm("You have unsaved settings changes. Leave without saving?")) return;
  }
  _applyActiveClasses(name);
  // Remembered across reloads.
  lsSetScoped("localm.activeView", name);
  // Navigating closes the mobile drawer.
  closeNav();
  if (window.onViewShown) window.onViewShown(name);
}
// Kernel nav buttons are static; plugin nav buttons get their handler in
// renderNav().
for (const v of CORE_VIEWS) $("nav-" + v).onclick = () => showView(v);

// --- mobile sidebar drawer ------------------------------------------------
// On a narrow screen the sidebar is off-canvas: the hamburger toggles it, the
// backdrop or any navigation closes it. No-op on desktop.
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
// Close the drawer when the viewport grows back to desktop width.
window.addEventListener("resize", () => {
  if (window.innerWidth > 760) closeNav();
});

