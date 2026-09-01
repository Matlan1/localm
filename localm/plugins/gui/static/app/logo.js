// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - logo style. */
"use strict";

// --- ES module imports ---
import { $, authHeaders } from "./helpers.js";
import { t } from "./i18n.js";

// The three sidebar wordmark styles. The choice is stored in server config
// (logo_style) and cached in localStorage. The blue half is rendered into the
// <span> (#logo span / .logo-tile span); the rest inherits the text colour.
export const LOGO_STYLES = [
  { id: "local-m", white: "LocaL", blue: "M"  },   // default
  { id: "loca-lm", white: "Loca",  blue: "LM" },
  { id: "localm",  white: "local", blue: "m"  },
];
export const LOGO_DEFAULT = LOGO_STYLES[0].id;

// Draw a wordmark into el as text plus an accent-coloured <span>.
export function drawWordmark(el, style) {
  el.textContent = style.white;
  const span = document.createElement("span");
  span.textContent = style.blue;
  el.appendChild(span);
}

// Render the wordmark, mark the active picker tile, and cache the choice in
// localStorage. Does not touch the server.
export function applyLogoStyle(id) {
  const style = LOGO_STYLES.find((s) => s.id === id) || LOGO_STYLES[0];
  drawWordmark($("logo"), style);
  localStorage.setItem("localm.logoStyle", style.id);
  for (const tile of document.querySelectorAll("#logo-style-picker .logo-tile")) {
    tile.classList.toggle("active", tile.dataset.style === style.id);
  }
  return style.id;
}

// Apply a pick locally, then persist it to the server config.
export async function setLogoStyle(id) {
  const applied = applyLogoStyle(id);
  try {
    await fetch("/v1/config", {
      method: "PATCH", headers: authHeaders(),
      body: JSON.stringify({ logo_style: applied }),
    });
  } catch (e) { /* ignored */ }
}

// Apply the wordmark style recorded in the server config.
export async function syncLogoStyleFromConfig() {
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (!r.ok) return;
    const cfg = await r.json();
    if (cfg && typeof cfg.logo_style === "string") applyLogoStyle(cfg.logo_style);
  } catch (e) { /* ignored */ }
}

// Render the preview tiles into the Settings -> GUI card. Clicking a tile
// applies and persists that style.
export function renderLogoPicker() {
  const wrap = $("logo-style-picker");
  if (!wrap) return;
  wrap.textContent = "";
  const current = localStorage.getItem("localm.logoStyle") || LOGO_DEFAULT;
  for (const style of LOGO_STYLES) {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.className = "logo-tile" + (style.id === current ? " active" : "");
    tile.dataset.style = style.id;
    tile.title = t("appearance.logoStyle.tile", { name: style.white + style.blue });
    drawWordmark(tile, style);
    tile.onclick = () => setLogoStyle(style.id);
    wrap.appendChild(tile);
  }
}

applyLogoStyle(localStorage.getItem("localm.logoStyle") || LOGO_DEFAULT);
renderLogoPicker();

