// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - theme. */
"use strict";

// --- ES module imports ---
import { $, safeStorageGet, safeStorageSet } from "./helpers.js";

/* ================================================================ */
/*  Theme                                                            */
/* ================================================================ */

export function applyTheme(name) {
  document.documentElement.dataset.theme = name;
  safeStorageSet("localm.theme", name);
}
applyTheme(safeStorageGet("localm.theme") || "dark");
$("theme-toggle").onclick = () =>
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

