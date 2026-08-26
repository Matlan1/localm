// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - shared inline-SVG icon set.

   24x24 stroke SVGs drawn with stroke="currentColor", so every icon inherits
   its context's text colour.

   Two ways to use it:
     - JS-built DOM: iconEl("send", "cls") -> a <span class="cls"> with the SVG.
     - Static markup: put <span data-icon="send"></span> in index.html; this
       module hydrates every [data-icon] on load (and hydrateIcons(root) can be
       called again after a fragment is inserted). */
"use strict";

// --- ES module imports ---
import { el } from "./helpers.js";

// Wrap paths in a 24x24 currentColor-stroked <svg>. The names iconMarkup and
// APP_ICONS must not collide with picker.js's locals: the app modules share one
// global scope.
function iconMarkup(paths) {
  return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
    'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" ' +
    'aria-hidden="true">' + paths + "</svg>";
}

const APP_ICONS = {
  // --- primary nav ---
  chat:     iconMarkup('<path d="M21 12a8 8 0 0 1-11.7 7.1L4 20.5l1.4-5A8 8 0 1 1 21 12z"/>'),
  models:   iconMarkup('<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5"/><path d="M12 12v9"/>'),
  plugins:  iconMarkup('<path d="M9 4a2 2 0 1 1 4 0v2h3a1 1 0 0 1 1 1v3h2a2 2 0 1 1 0 4h-2v3a1 1 0 0 1-1 1h-3v-2a2 2 0 1 0-4 0v2H6a1 1 0 0 1-1-1v-3H4a2 2 0 1 1 0-4h1V7a1 1 0 0 1 1-1h3z"/>'),
  settings: iconMarkup('<path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"/><path d="M19.4 13.5a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-2.9 1.2V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-2.9-1.2l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0-1.2-2.9H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.2-2.9l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 2.9-1.2V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 2.9 1.2l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.9z"/>'),
  // --- plugin nav tabs (renderNav) ---
  coder:    iconMarkup('<path d="M8 9l-3 3 3 3"/><path d="M16 9l3 3-3 3"/><path d="M13 6l-2 12"/>'),
  image:    iconMarkup('<path d="M4 5h16v14H4z"/><path d="M4 15l4-4 4 4 3-3 5 5"/><path d="M9 9a1.4 1.4 0 1 1-2.8 0 1.4 1.4 0 0 1 2.8 0z"/>'),
  music:    iconMarkup('<path d="M9 18V6l10-2v12"/><path d="M9 18a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z"/><path d="M19 16a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z"/>'),
  video:    iconMarkup('<path d="M4 6h11v12H4z"/><path d="M15 10l5-3v10l-5-3z"/>'),
  book:     iconMarkup('<path d="M5 4h9a3 3 0 0 1 3 3v13a2.5 2.5 0 0 0-2.5-2.5H5z"/><path d="M5 4v13"/>'),
  clock:    iconMarkup('<path d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z"/><path d="M12 7v5l3 2"/>'),
  studio:   iconMarkup('<path d="M4 4h7v7H4z"/><path d="M13 4h7v7h-7z"/><path d="M13 13h7v7h-7z"/><path d="M4 13h7v7H4z"/>'),
  // --- composer ---
  attach:   iconMarkup('<path d="M20 11l-8 8a4.5 4.5 0 0 1-6.4-6.4l8-8a3 3 0 0 1 4.3 4.3l-8 8a1.5 1.5 0 0 1-2.1-2.1L13 8"/>'),
  mic:      iconMarkup('<path d="M12 4a2.5 2.5 0 0 1 2.5 2.5v5a2.5 2.5 0 0 1-5 0v-5A2.5 2.5 0 0 1 12 4z"/><path d="M6 11a6 6 0 0 0 12 0"/><path d="M12 17v3"/>'),
  camera:   iconMarkup('<path d="M4 8h3l2-2h6l2 2h3v11H4z"/><path d="M12 16a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"/>'),
  send:     iconMarkup('<path d="M4 12l16-7-7 16-2.5-6.5z"/><path d="M10.5 14.5L20 5"/>'),
  stop:     iconMarkup('<path d="M7 7h10v10H7z" fill="currentColor" stroke="none"/>'),
  play:     iconMarkup('<path d="M8 5.5v13l11-6.5z" fill="currentColor" stroke="none"/>'),
  // --- chat params + composer row ---
  web:      iconMarkup('<path d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z"/><path d="M3 12h18"/><path d="M12 3a13 13 0 0 1 0 18 13 13 0 0 1 0-18z"/>'),
  memory:   iconMarkup('<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-1 5.8V16a3 3 0 0 0 4 2.8V4z"/><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 1 5.8V16a3 3 0 0 1-4 2.8V4z"/>'),
  speak:    iconMarkup('<path d="M5 9v6h4l5 4V5L9 9z"/><path d="M17 8a5 5 0 0 1 0 8"/>'),
  voice:    iconMarkup('<path d="M8 8v8"/><path d="M12 5v14"/><path d="M16 9v6"/><path d="M4 11v2"/><path d="M20 11v2"/>'),
  sliders:  iconMarkup('<path d="M5 6h14"/><path d="M5 12h14"/><path d="M5 18h14"/><path d="M9 4v4"/><path d="M15 10v4"/><path d="M8 16v4"/>'),
  export:   iconMarkup('<path d="M12 4v10"/><path d="M8 12l4 4 4-4"/><path d="M5 20h14"/>'),
  compact:  iconMarkup('<path d="M8 4l4 4 4-4"/><path d="M8 20l4-4 4 4"/><path d="M4 12h16"/>'),
  // --- global chrome / actions ---
  theme:    iconMarkup('<path d="M12 3a9 9 0 1 0 9 9 7 7 0 0 1-9-9z"/>'),
  close:    iconMarkup('<path d="M6 6l12 12"/><path d="M18 6L6 18"/>'),
  refresh:  iconMarkup('<path d="M4 12a8 8 0 0 1 13.7-5.7L20 8"/><path d="M20 4v4h-4"/><path d="M20 12a8 8 0 0 1-13.7 5.7L4 16"/><path d="M4 20v-4h4"/>'),
  plus:     iconMarkup('<path d="M12 5v14"/><path d="M5 12h14"/>'),
  menu:     iconMarkup('<path d="M4 7h16"/><path d="M4 12h16"/><path d="M4 17h16"/>'),
  search:   iconMarkup('<path d="M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14z"/><path d="M20 20l-3.5-3.5"/>'),
  download: iconMarkup('<path d="M12 4v10"/><path d="M8 12l4 4 4-4"/><path d="M5 20h14"/>'),
  trash:    iconMarkup('<path d="M5 7h14"/><path d="M9 7V5h6v2"/><path d="M7 7l1 13h8l1-13"/>'),
  folder:   iconMarkup('<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'),
  file:     iconMarkup('<path d="M14 3v5h5"/><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>'),
  key:      iconMarkup('<path d="M14 7a4 4 0 1 0 0 8 4 4 0 0 0-3.9-3H4v3H2"/><path d="M10 12H8v2"/>'),
  heart:    iconMarkup('<path d="M12 20l-6.8-6.8a4 4 0 0 1 5.7-5.7l1.1 1.1 1.1-1.1a4 4 0 0 1 5.7 5.7z"/>'),
  pin:      iconMarkup('<path d="M12 21s7-7.5 7-12a7 7 0 1 0-14 0c0 4.5 7 12 7 12z"/><path d="M12 11a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"/>'),
  // --- status / markers ---
  warning:  iconMarkup('<path d="M12 3L22 21H2Z"/><path d="M12 9v5"/><path d="M12 17h.01"/>'),
  check:    iconMarkup('<path d="M5 12l4 4 10-10"/>'),
  // Filled/outline pair sharing one geometry.
  dot:      iconMarkup('<path d="M12 18a6 6 0 1 0 0-12 6 6 0 0 0 0 12z" fill="currentColor" stroke="none"/>'),
  ring:     iconMarkup('<path d="M12 18a6 6 0 1 0 0-12 6 6 0 0 0 0 12z"/>'),
  caret:    iconMarkup('<path d="M12 8l5 8H7z" fill="currentColor" stroke="none"/>'),
};

/** The raw SVG string for `name` (falls back to a generic file icon). */
export function iconSvg(name) {
  return APP_ICONS[name] || APP_ICONS.file;
}

// Parse a static SVG string from APP_ICONS into DOM nodes. The input must be
// trusted markup, never user input.
function svgFragment(str) {
  return document.createRange().createContextualFragment(str);
}

/** A <span class=cls> wrapping the icon. The name is recorded on data-icon-name,
 *  not data-icon, so hydrateIcons never re-processes it. */
export function iconEl(name, cls) {
  const s = el("span", cls || "ic");
  s.dataset.iconName = name;
  s.appendChild(svgFragment(iconSvg(name)));
  return s;
}
window.iconEl = iconEl;

/** An empty state: a centred icon, a line of text, and an optional hint. */
export function emptyState(iconName, text, hint) {
  const box = el("div", "empty-state");
  box.appendChild(iconEl(iconName, "empty-state-ic"));
  box.appendChild(el("div", "empty-state-text", text));
  if (hint) box.appendChild(el("div", "empty-state-hint", hint));
  return box;
}
window.emptyState = emptyState;

/** Replace every <span data-icon="NAME"> placeholder under `root` with its SVG,
 *  once each. Can be re-run after a fragment is inserted. */
export function hydrateIcons(root) {
  const scope = root || document;
  for (const n of scope.querySelectorAll("[data-icon]")) {
    if (n.dataset.iconDone) continue;
    n.replaceChildren(svgFragment(iconSvg(n.dataset.icon)));
    n.dataset.iconDone = "1";
  }
}
window.hydrateIcons = hydrateIcons;

// Hydrate the static placeholders on load.
hydrateIcons(document);
