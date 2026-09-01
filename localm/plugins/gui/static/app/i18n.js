// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - interface language.
 *
 * English is the source language and ships as a statically imported module
 * (app/i18n-en.js), so t() resolves synchronously from the first call and
 * cannot depend on a fetch. Every other language is a JSON catalog under
 * /i18n/<id>.json, fetched when that language is selected. A key the active
 * catalog does not carry falls back to English.
 */
"use strict";

// --- ES module imports ---
import { $, authHeaders, toast } from "./helpers.js";
import { I18N_EN } from "./i18n-en.js";

/* ================================================================ */
/*  Language registry                                                */
/* ================================================================ */

// id: the config.json `language` value and the /i18n/<id>.json basename.
// label: the language's own name, shown untranslated in the picker.
// Pinned to settings_schema.py's LANGUAGE_IDS by
// test_language_ids_match_the_gui_registry.
export const LANGUAGES = [
  { id: "en", label: "English" },
  { id: "de", label: "Deutsch" },
];
export const LANGUAGE_DEFAULT = "en";
export const LANGUAGE_STORAGE_KEY = "localm.language";

export function isKnownLanguage(id) {
  return LANGUAGES.some((l) => l.id === id);
}

/* ================================================================ */
/*  Lookup                                                           */
/* ================================================================ */

let _language = LANGUAGE_DEFAULT;
// The active translation catalog. Empty for English, which reads I18N_EN.
let _catalog = {};

export function currentLanguage() { return _language; }

/** Substitute {name} placeholders from `params`. An unknown placeholder is
 *  left as written. */
function interpolate(text, params) {
  return text.replace(/\{(\w+)\}/g, (whole, name) =>
    (Object.prototype.hasOwnProperty.call(params, name) ? String(params[name]) : whole));
}

/**
 * The translation for `key` in the active language, falling back to English.
 *
 * A key that is in NEITHER catalog is a typo rather than a missing
 * translation: it is warned about and the key itself is returned.
 * tests-js/i18n.test.mjs source-scans every t() call site and every
 * data-i18n attribute against app/i18n-en.js, so that branch is a backstop.
 */
export function t(key, params) {
  let text = _catalog[key];
  if (typeof text !== "string" || text === "") text = I18N_EN[key];
  if (typeof text !== "string") {
    console.warn(`localm: no i18n entry for "${key}" - add it to app/i18n-en.js.`);
    return key;
  }
  return params ? interpolate(text, params) : text;
}

/** Whether a key resolves in the active catalog or in English. */
function hasKey(key) {
  return typeof _catalog[key] === "string" || typeof I18N_EN[key] === "string";
}

/** The CLDR plural category `count` selects in the active language. Falls back
 *  to the English one/other split where Intl.PluralRules is unavailable. */
export function pluralCategory(count) {
  try {
    return new Intl.PluralRules(_language).select(count);
  } catch (e) {
    return count === 1 ? "one" : "other";
  }
}

/**
 * The plural form of `key` for `count`, looked up as `<key>.<category>` with
 * `<key>.other` as the fallback form. `count` is available to the string as
 * {count} on top of any `params`.
 *
 * Categories come from Intl.PluralRules, so a language with more than two
 * forms gets the right one as soon as its catalog carries them; a catalog that
 * only carries `.other` still reads correctly.
 */
export function tn(key, count, params) {
  const merged = Object.assign({ count }, params || {});
  const specific = `${key}.${pluralCategory(count)}`;
  return t(hasKey(specific) ? specific : `${key}.other`, merged);
}

/* ================================================================ */
/*  Applying a catalog to the DOM                                    */
/* ================================================================ */

// [marker attribute, its dataset property, the attribute it sets].
const I18N_ATTRS = [
  ["data-i18n-title", "i18nTitle", "title"],
  ["data-i18n-placeholder", "i18nPlaceholder", "placeholder"],
  ["data-i18n-aria-label", "i18nAriaLabel", "aria-label"],
];

/** Replace an element's own text while leaving its element children in place:
 *  the first non-blank direct text node takes the new text and any later one
 *  is emptied. An element with no text node of its own gets one.
 *
 *  The replaced node's leading and trailing whitespace is kept, so catalog
 *  entries are trimmed and the markup's own spacing around sibling elements
 *  survives translation. */
export function setI18nText(el, text) {
  const own = [];
  for (const node of el.childNodes) {
    if (node.nodeType === 3 && node.nodeValue.trim() !== "") own.push(node);
  }
  if (!own.length) {
    el.appendChild(el.ownerDocument.createTextNode(text));
    return;
  }
  const was = own[0].nodeValue;
  own[0].nodeValue = /^\s*/.exec(was)[0] + text + /\s*$/.exec(was)[0];
  for (let i = 1; i < own.length; i++) own[i].nodeValue = "";
}

// The inline tags a data-i18n-rich string may carry. A tag carrying any
// attribute does not match and stays literal text, and a matched tag is built
// with createElement plus textContent. See "setI18nRichText builds only
// b/strong/code, and rich text is attribute-free" in tests-js/i18n.test.mjs.
const RICH_RE = /<(b|strong|code)>([^<]*)<\/\1>/g;

/** Rebuild an element's whole content from a string that may carry <b>,
 *  <strong> and <code>. Every part is built with createTextNode /
 *  createElement, never innerHTML. Caller must not use this on an element
 *  whose children are referenced elsewhere: it replaces all of them. */
export function setI18nRichText(el, text) {
  const doc = el.ownerDocument;
  const frag = doc.createDocumentFragment();
  let last = 0;
  let m;
  RICH_RE.lastIndex = 0;
  while ((m = RICH_RE.exec(text)) !== null) {
    if (m.index > last) frag.appendChild(doc.createTextNode(text.slice(last, m.index)));
    const tag = doc.createElement(m[1]);
    tag.textContent = m[2];
    frag.appendChild(tag);
    last = m.index + m[0].length;
  }
  if (last < text.length) frag.appendChild(doc.createTextNode(text.slice(last)));
  el.replaceChildren(frag);
}

/** Translate every data-i18n* element under `root` (the document by default). */
export function applyI18n(root) {
  const scope = root || document;
  for (const el of scope.querySelectorAll("[data-i18n]")) {
    setI18nText(el, t(el.dataset.i18n));
  }
  for (const el of scope.querySelectorAll("[data-i18n-rich]")) {
    setI18nRichText(el, t(el.dataset.i18nRich));
  }
  for (const [marker, prop, attr] of I18N_ATTRS) {
    for (const el of scope.querySelectorAll(`[${marker}]`)) {
      el.setAttribute(attr, t(el.dataset[prop]));
    }
  }
}

/* ================================================================ */
/*  Switching language                                               */
/* ================================================================ */

/** Fetch a language's catalog. Returns {} for English (which reads I18N_EN)
 *  and null when the catalog could not be loaded, so the caller can keep the
 *  language it already has rather than switching to a blank one. */
export async function fetchCatalog(id) {
  if (id === LANGUAGE_DEFAULT) return {};
  if (!isKnownLanguage(id)) return null;
  try {
    const r = await fetch(`/i18n/${id}.json`);
    if (!r.ok) return null;
    const data = await r.json();
    return (data && typeof data === "object" && !Array.isArray(data)) ? data : null;
  } catch (e) {
    return null;
  }
}

/**
 * Load `id`'s catalog, translate the document, and remember the choice in this
 * browser. Does not touch the server; setLanguage() does that.
 *
 * A catalog that cannot be loaded leaves the interface in the language it
 * already had and reports false, rather than blanking every translated string.
 */
export async function applyLanguage(id) {
  const want = isKnownLanguage(id) ? id : LANGUAGE_DEFAULT;
  const catalog = await fetchCatalog(want);
  if (catalog === null) {
    console.warn(`localm: could not load the ${want} interface catalog - ` +
      `staying in ${_language}.`);
    return false;
  }
  _language = want;
  _catalog = catalog;
  document.documentElement.lang = want;
  applyI18n(document);
  try { localStorage.setItem(LANGUAGE_STORAGE_KEY, want); }
  catch (e) { /* storage full or blocked - the server config still holds it */ }
  document.dispatchEvent(new CustomEvent("localm:language", { detail: { language: want } }));
  return true;
}

/** Read the language cached in this browser and apply it. Runs at load so a
 *  returning visitor never sees English first. */
export function storedLanguage() {
  let id = null;
  try { id = localStorage.getItem(LANGUAGE_STORAGE_KEY); }
  catch (e) { return LANGUAGE_DEFAULT; }
  return isKnownLanguage(id) ? id : LANGUAGE_DEFAULT;
}

/* ================================================================ */
/*  Server config + the Settings picker                              */
/* ================================================================ */

/**
 * Apply `id` and persist it to the server config.
 *
 * Returns whether the choice was SAVED. The interface switches either way, but
 * a save that did not land is reported to the user rather than passed off as
 * success: only the server copy survives a new browser or a reinstall.
 */
export async function setLanguage(id) {
  if (!(await applyLanguage(id))) return false;
  try {
    const r = await fetch("/v1/config", {
      method: "PATCH", headers: authHeaders(),
      body: JSON.stringify({ language: currentLanguage() }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
  } catch (e) {
    toast(t("appearance.language.saveFailed"), true);
    return false;
  }
  return true;
}

/** Apply the language recorded in the server config. */
export async function syncLanguageFromConfig() {
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (!r.ok) return;
    const cfg = await r.json();
    if (cfg && typeof cfg.language === "string" && cfg.language !== _language) {
      await applyLanguage(cfg.language);
    }
  } catch (e) { /* ignored */ }
  renderLanguagePicker();
}

/** Fill the Settings > Appearance language select. Each option is named in its
 *  own language, so it stays readable whatever the interface is set to. */
export function renderLanguagePicker() {
  const sel = $("lang-select");
  if (!sel) return;
  sel.replaceChildren();
  for (const lang of LANGUAGES) {
    const opt = document.createElement("option");
    opt.value = lang.id;
    opt.textContent = lang.label;
    sel.appendChild(opt);
  }
  sel.value = currentLanguage();
  sel.onchange = () => setLanguage(sel.value);
}

const _booted = storedLanguage();
if (_booted !== LANGUAGE_DEFAULT) applyLanguage(_booted);
else document.documentElement.lang = LANGUAGE_DEFAULT;
renderLanguagePicker();
