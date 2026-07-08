// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - logo style (split from app.js). Classic script sharing the one
   global lexical scope with the other app/* + pages/* scripts (bare-name refs). */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { $, authHeaders } from "./helpers.js";

// The sidebar wordmark can be drawn three ways. The choice is SHARED via server
// config (logo_style), so the web GUI and the desktop launcher agree; the
// localStorage copy is only a no-flash cache for the next page load. The blue
// half always goes in the <span> (#logo span / .logo-tile span = var(--accent));
// the rest inherits the white/ink text colour. The console command, app icon,
// and desktop shortcut are fixed and unaffected.
export const LOGO_STYLES = [
  { id: "local-m", white: "LocaL", blue: "M"  },   // default: single blue M, matches the icon (L white, M blue)
  { id: "loca-lm", white: "Loca",  blue: "LM" },
  { id: "localm",  white: "local", blue: "m"  },
];
export const LOGO_DEFAULT = LOGO_STYLES[0].id;

// Draw a wordmark into el as white text + an accent-coloured <span>. The parts
// are constant strings, but build via DOM nodes (no innerHTML) to stay clear of
// the no-raw-HTML house style.
export function drawWordmark(el, style) {
  el.textContent = style.white;
  const span = document.createElement("span");
  span.textContent = style.blue;
  el.appendChild(span);
}

// Render the wordmark, reflect the active picker tile, and cache the choice
// locally so the next load paints instantly. Does NOT touch the server.
export function applyLogoStyle(id) {
  const style = LOGO_STYLES.find((s) => s.id === id) || LOGO_STYLES[0];
  drawWordmark($("logo"), style);
  localStorage.setItem("localm.logoStyle", style.id);
  for (const tile of document.querySelectorAll("#logo-style-picker .logo-tile")) {
    tile.classList.toggle("active", tile.dataset.style === style.id);
  }
  return style.id;
}

// Apply a pick locally, then persist it to the shared server config so the
// launcher (and other browsers) follow. Offline: the cached style still shows.
export async function setLogoStyle(id) {
  const applied = applyLogoStyle(id);
  try {
    await fetch("/v1/config", {
      method: "PATCH", headers: authHeaders(),
      body: JSON.stringify({ logo_style: applied }),
    });
  } catch (e) { /* server unreachable - the cached style still applies */ }
}

// Reconcile the cached wordmark with the shared server truth on load (the
// launcher or another browser may have changed it). Best-effort.
export async function syncLogoStyleFromConfig() {
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (!r.ok) return;
    const cfg = await r.json();
    if (cfg && typeof cfg.logo_style === "string") applyLogoStyle(cfg.logo_style);
  } catch (e) { /* server unreachable - keep the cached style */ }
}

// Render the three preview tiles into the Settings -> GUI card. Each tile shows
// the wordmark in its own style; clicking one applies + persists it.
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
    tile.title = "Use the " + style.white + style.blue + " wordmark";
    drawWordmark(tile, style);
    tile.onclick = () => setLogoStyle(style.id);
    wrap.appendChild(tile);
  }
}

applyLogoStyle(localStorage.getItem("localm.logoStyle") || LOGO_DEFAULT);
renderLogoPicker();

