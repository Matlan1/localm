// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom harness for the localm GUI (localm/plugins/gui/static/app.js).
//
// app.js is a 3200-line top-level "use strict" script (not a module). It defines
// top-level functions (renderNav, runCompletion, ...) that become properties of
// the window when run as a script, and wires a DOMContentLoaded handler that
// fetches /api/plugins etc. on load. We load the real index.html so every
// element app.js's $() expects exists, stub the browser/vendor globals app.js
// touches at top level, run app.js as an injected <script> (jsdom executes it in
// the window context) WITHOUT firing DOMContentLoaded again - parsing already
// completed, so the network-driven init never runs - then drive functions.

import { JSDOM } from "jsdom";
import { after } from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const STATIC = join(ROOT, "localm", "plugins", "gui", "static");

const read = (p) => readFileSync(join(STATIC, p), "utf-8");

// The GUI ships as native ES modules (app/*.js, pages/*.js, entry app/main.js).
// jsdom does NOT execute <script type="module">, and the conversion only ADDED
// import/export lines (bodies are byte-identical to the pre-module classic
// split). So for the unit tests we strip those generated lines back to the
// classic form and inject the modules as classic scripts in the same realm -
// exactly what jsdom can run. This tests the real module BODIES (identical to
// the browser) and keeps the shared-global test access (window.foo,
// runScript("chat...")) working; the real module GRAPH/resolution is covered by
// node --check + the `localm gui` browser smoke, which jsdom could never run.
const _IMPORT_BLOCK = /^\/\/ --- ES module imports.*\r?\n(?:import [^\n]*\r?\n)*\r?\n?/m;
function moduleToClassic(code) {
  return code
    .replace(_IMPORT_BLOCK, "")     // drop the auto-generated import block
    .replace(/^import [^\n]*\r?\n/gm, "")  // any stray import line
    .replace(/^export\s+/gm, "");   // drop the `export ` declaration prefix
}
const readClassic = (p) => moduleToClassic(read(p));

// Every loadApp() builds a jsdom window. With pretendToBeVisual: true jsdom
// runs an internal requestAnimationFrame timer loop that keeps node's event
// loop alive AFTER the tests finish, so a bare `node --test` would hang forever
// (only `npm test`, which passes --test-force-exit, escaped it). Track the
// windows and close them in a root after-hook so the process exits on its own,
// whatever command launched it - no footgun for the next person (or agent).
const _openWindows = new Set();
after(() => {
  for (const win of _openWindows) {
    try { win.close(); } catch (e) { /* already torn down */ }
  }
  _openWindows.clear();
});

// Minimal stubs for the vendored browser libs app.js references at top level
// (it calls marked.setOptions on load). We do not need their real behaviour for
// the nav / abort logic under test.
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
  win.fetch = fetchImpl
    || (async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => "" }));
  // speechSynthesis (speak()) - record calls so abort tests can assert it was NOT called.
  win.__spoken = [];
  win.speechSynthesis = { speak: (u) => win.__spoken.push(u), cancel() {}, getVoices: () => [] };
  win.SpeechSynthesisUtterance = class { constructor(t) { this.text = t; } };
}

// app.js was split per section into app/<name>.js (same load order). Inject in
// this exact order to preserve the original top-level execution order; they share
// one global lexical environment so cross-section references resolve as before.
const APP_SCRIPTS = [
  "client-log", "helpers", "icons", "picker", "theme", "logo", "tabs", "models-sidebar",
  "chat", "cmdk", "settings-perf", "coder", "slash", "init",
];

/**
 * Build a jsdom window with the app scripts loaded but the DOMContentLoaded init
 * NOT run. Returns { dom, window }. Pass fetchImpl to control network.
 */
export function loadApp({ fetchImpl, url, shellToken, seedLocalStorage } = {}) {
  const html = read("index.html");
  const dom = new JSDOM(html, {
    // Pass url: "https://..." to exercise the HTTPS-only paths (e.g. the
    // built-in-TLS "Install certificate" link on the key gate).
    url: url || "http://localhost:8642/",
    runScripts: "dangerously",   // execute scripts we inject (vendor src tags are not loaded)
    pretendToBeVisual: true,
  });
  const win = dom.window;
  _openWindows.add(win);   // closed in the after-hook so the process can exit
  installStubs(win, { fetchImpl });
  // Seed the open-mode shell token BEFORE the app scripts run, since helpers.js
  // reads window.__LOCALM_SHELL_TOKEN__ into a const at load. Pass shellToken to
  // exercise the open-mode (loopback shell) auth path.
  if (shellToken) win.__LOCALM_SHELL_TOKEN__ = shellToken;
  // Seed localStorage BEFORE the app scripts run: some module-level state
  // (e.g. chat.js's `conversations`) reads localStorage at import/eval time, so
  // a test simulating "this origin already had cached data from a PRIOR page
  // load" (AUD-INSTANCEID: a different backend's cache left behind at the same
  // origin) must seed it before injection - mirroring a real browser, where
  // localStorage persists across page loads at the same origin.
  if (seedLocalStorage) {
    for (const [k, v] of Object.entries(seedLocalStorage)) win.localStorage.setItem(k, v);
  }

  // app.js was split per section into app/<name>.js (same load order). Inject
  // each as a classic script in order: jsdom executes them in the window context;
  // top-level function declarations land on the window and the shared global
  // lexical environment holds the top-level const/let exactly as a single app.js
  // did. The HTML's own <script src> tags are not fetched (no resource loader),
  // so this does not auto-run and the DOMContentLoaded init (already fired during
  // parse) does not re-run.
  for (const name of APP_SCRIPTS) {
    const script = win.document.createElement("script");
    script.textContent = readClassic(`app/${name}.js`);
    win.document.body.appendChild(script);
  }
  return { dom, window: win };
}

/**
 * Run a snippet of code as a classic script in the window's realm. Top-level
 * `let`/`const` in app.js (pluginState, chat, ...) live in the shared global
 * lexical environment - not on `window` - so the only way to set them from a
 * test is to assign them from another classic script in the same realm. Use
 * this to seed state (e.g. `pluginState = [...]`) and invoke functions.
 */
export function runScript(win, code) {
  const s = win.document.createElement("script");
  s.textContent = code;
  win.document.body.appendChild(s);
}

// pages.js was split per page into pages/<name>.js (same load order). They are
// classic scripts that share the realm's global lexical environment with app.js,
// so their helpers ($, el, authHeaders, toast, ...) resolve by bare name. Inject
// in this exact order to preserve the original top-level execution order.
const PAGE_SCRIPTS = [
  "dispatch", "models", "images", "plugins", "settings",
  "workflow", "music", "video", "knowledge",
];

/**
 * Load app.js (via loadApp) and then the per-page scripts in the same realm. They
 * hold the Models / Images / Plugins / Settings page logic (refreshSettingsPage,
 * the config-save click handler, ...) and rely on helpers from app.js, so they
 * run AFTER app.js. Their top-level `$("...").onclick = ...` wiring runs against
 * the real index.html elements.
 */
export function loadAppWithPages({ fetchImpl, url } = {}) {
  const { dom, window } = loadApp({ fetchImpl, url });
  for (const name of PAGE_SCRIPTS) {
    runScript(window, readClassic(`pages/${name}.js`));
  }
  return { dom, window };
}
