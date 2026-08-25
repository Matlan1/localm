// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - theme. */
"use strict";

// --- ES module imports ---
import { $ } from "./helpers.js";

/* ================================================================ */
/*  Theme                                                            */
/* ================================================================ */

export function applyTheme(name) {
  document.documentElement.dataset.theme = name;
  localStorage.setItem("localm.theme", name);
}
applyTheme(localStorage.getItem("localm.theme") || "dark");
$("theme-toggle").onclick = () =>
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

