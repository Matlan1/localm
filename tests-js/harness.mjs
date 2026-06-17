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
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const STATIC = join(ROOT, "localm", "plugins", "gui", "static");

const read = (p) => readFileSync(join(STATIC, p), "utf-8");

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

/**
 * Build a jsdom window with app.js loaded but its DOMContentLoaded init NOT run.
 * Returns { dom, window }. Pass fetchImpl to control network.
 */
export function loadApp({ fetchImpl } = {}) {
  const html = read("index.html");
  const dom = new JSDOM(html, {
    url: "http://localhost:8642/",
    runScripts: "dangerously",   // execute scripts we inject (vendor src tags are not loaded)
    pretendToBeVisual: true,
  });
  const win = dom.window;
  installStubs(win, { fetchImpl });

  // Run app.js as an injected inline script. jsdom executes it in the window
  // context; top-level function declarations land on the window. The HTML's own
  // <script src> tags are not fetched (no resource loader), so app.js does not
  // auto-run and the DOMContentLoaded init (already fired during parse) does not
  // re-run.
  const script = win.document.createElement("script");
  script.textContent = read("app.js");
  win.document.body.appendChild(script);
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

/**
 * Load app.js (via loadApp) and then pages.js in the same realm. pages.js holds
 * the Models / Images / Plugins / Settings page logic (refreshSettingsPage, the
 * config-save click handler, ...) and relies on helpers from app.js ($, el,
 * authHeaders, toast, ...), so it must run AFTER app.js. Its top-level
 * `$("...").onclick = ...` wiring runs against the real index.html elements.
 */
export function loadAppWithPages({ fetchImpl } = {}) {
  const { dom, window } = loadApp({ fetchImpl });
  runScript(window, read("pages.js"));
  return { dom, window };
}
