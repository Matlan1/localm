// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom harness for the localm GUI. Loads the real index.html so every element
// the app's $() expects exists, stubs the browser and vendor globals the app
// touches at top level, and injects the app scripts as classic scripts, which
// jsdom executes in the window context. init.js's own boot sequence has no
// DOMContentLoaded/readyState gate, so it runs unconditionally on injection;
// the root after-hook below drains its dangling async chains before closing
// each window.
//
// jsdom builds a DOM but never lays out or paints, so nothing here observes
// rendered geometry or composed on-screen text.

import { JSDOM } from "jsdom";
import { after } from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const STATIC = join(ROOT, "localm", "plugins", "gui", "static");

const read = (p) => readFileSync(join(STATIC, p), "utf-8");

// The GUI ships as native ES modules, which jsdom does not execute. Strip the
// generated import/export lines back to classic form so the module bodies can be
// injected as classic scripts sharing one realm.
const _IMPORT_BLOCK = /^\/\/ --- ES module imports.*\r?\n(?:import [^\n]*\r?\n)*\r?\n?/m;
function moduleToClassic(code) {
  return code
    .replace(_IMPORT_BLOCK, "")     // drop the auto-generated import block
    .replace(/^import [^\n]*\r?\n/gm, "")  // any stray import line
    .replace(/^export\s+/gm, "");   // drop the `export ` declaration prefix
}
const readClassic = (p) => moduleToClassic(read(p));

// pretendToBeVisual: true runs an internal requestAnimationFrame loop that keeps
// node's event loop alive after the tests finish, so every window is tracked and
// closed in a root after-hook.
const _openWindows = new Set();
after(async () => {
  // Drains dangling boot-chain promises (init.js's unawaited async IIFE and
  // its downstream fetch chains) against a still-valid document before
  // win.close() nulls it out from under them.
  if (_openWindows.size > 0) {
    for (let i = 0; i < 20; i++) await new Promise((r) => setTimeout(r, 50));
  }
  for (const win of _openWindows) {
    try { win.close(); } catch (e) { /* already torn down */ }
  }
  _openWindows.clear();
});

// Stubs for the vendored browser libs the app references at top level.
function installStubs(win, { fetchImpl } = {}) {
  win.marked = { setOptions() {}, parse: (s) => s };
  win.DOMPurify = { sanitize: (s) => s };
  win.hljs = { highlightElement() {}, highlightAuto: (s) => ({ value: s }) };
  win.katex = { renderToString: (s) => s };
  win.renderMathInElement = () => {};
  win.scrollTo = () => {};
  win.requestAnimationFrame = (cb) => setTimeout(cb, 0);
  win.cancelAnimationFrame = (id) => clearTimeout(id);
  if (!win.matchMedia) {
    win.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
  }
  win.EventSource = class { constructor() {} close() {} addEventListener() {} };
  if (!win.AbortController) win.AbortController = AbortController;   // node global
  // jsdom does not expose these on window; readSSE() needs TextDecoder to decode
  // a streamed fetch body.
  if (!win.TextDecoder) win.TextDecoder = TextDecoder;   // node global
  if (!win.TextEncoder) win.TextEncoder = TextEncoder;   // node global
  win.fetch = fetchImpl
    || (async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => "" }));
  // speechSynthesis: speak() calls are recorded in win.__spoken.
  win.__spoken = [];
  win.speechSynthesis = { speak: (u) => win.__spoken.push(u), cancel() {}, getVoices: () => [] };
  win.SpeechSynthesisUtterance = class { constructor(t) { this.text = t; } };
}

// The app/<name>.js sections, in load order. They share one global lexical
// environment, so this order must be preserved.
const APP_SCRIPTS = [
  "client-log", "helpers", "i18n-en", "i18n", "icons", "picker", "theme", "logo", "tabs", "models-sidebar",
  "chat", "media-gallery", "cmdk", "settings-perf", "coder", "slash", "init",
];

/**
 * Builds a jsdom window with the app scripts loaded and the DOMContentLoaded
 * init not run. Returns { dom, window }. Pass fetchImpl to control network.
 */
export function loadApp({ fetchImpl, url, shellToken, seedLocalStorage } = {}) {
  const html = read("index.html");
  const dom = new JSDOM(html, {
    // Pass url: "https://..." to exercise the HTTPS-only paths.
    url: url || "http://localhost:8642/",
    runScripts: "dangerously",   // execute scripts we inject (vendor src tags are not loaded)
    pretendToBeVisual: true,
  });
  const win = dom.window;
  _openWindows.add(win);   // closed in the after-hook so the process can exit
  installStubs(win, { fetchImpl });
  // Seeded before the app scripts run: helpers.js reads
  // window.__LOCALM_SHELL_TOKEN__ into a const at load.
  if (shellToken) win.__LOCALM_SHELL_TOKEN__ = shellToken;
  // Also seeded before the app scripts run: some module-level state reads
  // localStorage at eval time.
  if (seedLocalStorage) {
    for (const [k, v] of Object.entries(seedLocalStorage)) win.localStorage.setItem(k, v);
  }

  // Injected in order as classic scripts: top-level function declarations land
  // on the window, top-level const/let in the shared global lexical environment.
  // The HTML's own <script src> tags are never fetched, so nothing auto-runs and
  // the DOMContentLoaded init does not re-run.
  for (const name of APP_SCRIPTS) {
    const script = win.document.createElement("script");
    script.textContent = readClassic(`app/${name}.js`);
    win.document.body.appendChild(script);
  }
  return { dom, window: win };
}

/**
 * Runs a snippet of code as a classic script in the window's realm. Top-level
 * `let`/`const` (pluginState, chat, ...) live in the shared global lexical
 * environment rather than on `window`, so they can only be set from another
 * classic script in the same realm.
 */
export function runScript(win, code) {
  const s = win.document.createElement("script");
  s.textContent = code;
  win.document.body.appendChild(s);
}

// The pages/<name>.js scripts, in load order. They share the realm's global
// lexical environment with the app scripts, so helpers ($, el, authHeaders,
// toast, ...) resolve by bare name.
const PAGE_SCRIPTS = [
  "dispatch", "models", "images", "plugins", "settings",
  "workflow", "music", "video", "knowledge", "setup",
];

/**
 * Loads the app scripts via loadApp, then the per-page scripts in the same
 * realm. The page scripts rely on helpers from the app scripts, so they run
 * after them, and their top-level `$("...").onclick = ...` wiring runs against
 * the real index.html elements. Takes and forwards the same options as loadApp.
 * window.onViewShown, which the boot deep-link/restore path ends in, is
 * installed only by pages/dispatch.js, so that path needs this loader.
 */
export function loadAppWithPages(opts = {}) {
  const { dom, window } = loadApp(opts);
  for (const name of PAGE_SCRIPTS) {
    runScript(window, readClassic(`pages/${name}.js`));
  }
  return { dom, window };
}
