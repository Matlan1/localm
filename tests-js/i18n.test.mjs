// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the interface-language runtime (app/i18n.js) and the two
// catalogs it reads: the built-in English source language (app/i18n-en.js) and
// the fetched German translation (static/i18n/de.json).
//
// Several of these are drift gates rather than behaviour tests: every
// data-i18n* key in index.html must resolve and must still carry the English it
// was written from, every t()/tn() key in the migrated modules must exist, and
// the two catalogs must hold the same keys. Without them a renamed key or an
// edited English string degrades silently to a raw key on screen.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const STATIC = join(ROOT, "localm", "plugins", "gui", "static");
const read = (...p) => readFileSync(join(STATIC, ...p), "utf-8");

const DE = JSON.parse(read("i18n", "de.json"));
const INDEX_HTML = read("index.html");

const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, timeout = 1000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await settle(10); }
  return false;
}

// Serves the real German catalog. /api/models answers 401 so init.js's boot
// chain stops at the key gate: its tail is fire-and-forget, and a continuation
// still in flight when the harness closes the window reads a torn-down document.
function makeFetch(calls, { deStatus = 200, deBody = null } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    calls.push({ u, method: (opts.method || "GET").toUpperCase(), body: opts.body });
    if (u.includes("/i18n/de.json")) {
      return { ok: deStatus === 200, status: deStatus,
               json: async () => (deBody === null ? DE : deBody) };
    }
    if (u.includes("/i18n/")) return { ok: false, status: 404, json: async () => ({}) };
    if (u.includes("/api/models")) return { ok: false, status: 401, json: async () => ({}) };
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

/** A window whose app modules booted with `lang` cached in localStorage. */
async function loadIn(lang, opts = {}) {
  const calls = [];
  const seed = lang ? { "localm.language": lang } : undefined;
  const { window } = (opts.withPages ? loadAppWithPages : loadApp)({
    fetchImpl: makeFetch(calls, opts), seedLocalStorage: seed,
  });
  if (lang && lang !== "en") {
    await waitFor(() => window.document.documentElement.lang === lang
                        || calls.some((c) => c.u.includes("/i18n/")));
    await settle(20);
  }
  return { window, calls };
}

const evalIn = (window, expr) => {
  runScript(window, `window.__r = (${expr});`);
  return window.__r;
};

/* ================================================================ */
/*  Lookup and fallback                                              */
/* ================================================================ */

test("t() resolves from the English catalog when English is active", async () => {
  const { window } = await loadIn(null);
  assert.equal(evalIn(window, 't("nav.settings")'), "Settings");
  assert.equal(evalIn(window, "currentLanguage()"), "en");
});

test("t() resolves from the active catalog once German is applied", async () => {
  const { window } = await loadIn("de");
  assert.equal(evalIn(window, "currentLanguage()"), "de");
  assert.equal(evalIn(window, 't("nav.settings")'), "Einstellungen");
});

test("t() falls back to English for a key the active catalog does not carry", async () => {
  const { window } = await loadIn("de");
  assert.equal(evalIn(window, "currentLanguage()"), "de");
  runScript(window, `
    I18N_EN["zz.only.in.english"] = "English only";
    window.__missing = t("zz.only.in.english");`);
  assert.equal(window.__missing, "English only",
    "a key absent from the German catalog must read as English, never as the raw key");
});

test("t() interpolates {name} params and leaves an unknown placeholder alone", async () => {
  const { window } = await loadIn(null);
  assert.equal(evalIn(window, 't("sidebar.status.loading", { model: "tiny.gguf" })'),
    "loading tiny.gguf\u2026");
  runScript(window, 'I18N_EN["zz.unknown.ph"] = "a {b} c {d}";');
  assert.equal(evalIn(window, 't("zz.unknown.ph", { b: "B" })'), "a B c {d}");
});

test("an unknown key is warned about and returns the key rather than a blank", async () => {
  const { window } = await loadIn(null);
  const warned = [];
  window.console.warn = (m) => warned.push(String(m));
  assert.equal(evalIn(window, 't("zz.no.such.key.at.all")'), "zz.no.such.key.at.all");
  assert.ok(warned.some((w) => w.includes("zz.no.such.key.at.all")),
    "a missing key must be reported, not silently swallowed");
});

/* ================================================================ */
/*  Plurals                                                          */
/* ================================================================ */

test("tn() selects the one/other form from Intl.PluralRules in both languages", async () => {
  const en = (await loadIn(null)).window;
  assert.equal(evalIn(en, 'tn("chat.share.imagesIn", 1)'), "1 image shared into chat");
  assert.equal(evalIn(en, 'tn("chat.share.imagesIn", 3)'), "3 images shared into chat");
  const de = (await loadIn("de")).window;
  assert.equal(evalIn(de, 'tn("chat.share.imagesIn", 1)'), "1 Bild in den Chat geteilt");
  assert.equal(evalIn(de, 'tn("chat.share.imagesIn", 3)'), "3 Bilder in den Chat geteilt");
});

test("tn() falls back to the .other form when the selected category is absent", async () => {
  const { window } = await loadIn(null);
  runScript(window, 'I18N_EN["zz.plural.other"] = "{count} things";');
  assert.equal(evalIn(window, 'tn("zz.plural", 1)'), "1 things",
    "a catalog carrying only .other must still produce a string");
});

/* ================================================================ */
/*  Writing into the DOM                                             */
/* ================================================================ */

test("setI18nText replaces the element's own text and keeps its child elements", async () => {
  const { window } = await loadIn(null);
  runScript(window, `
    const host = document.createElement("button");
    host.innerHTML = '<span class="nav-ic"></span>Chat';
    setI18nText(host, "Unterhaltung");
    window.__html = host.innerHTML;
    window.__kids = host.querySelectorAll("span").length;`);
  assert.equal(window.__kids, 1, "the icon span must survive translation");
  assert.equal(window.__html, '<span class="nav-ic"></span>Unterhaltung');
});

test("setI18nText keeps the whitespace around the text it replaces", async () => {
  const { window } = await loadIn(null);
  runScript(window, `
    const host = document.createElement("label");
    host.innerHTML = '<input type="checkbox">\\n  Speak replies aloud\\n';
    setI18nText(host, "Antworten vorlesen");
    window.__html = host.innerHTML;`);
  assert.equal(window.__html, '<input type="checkbox">\n  Antworten vorlesen\n');
});

test("setI18nRichText builds only b/strong/code, and rich text is attribute-free", async () => {
  const { window } = await loadIn(null);
  runScript(window, `
    const host = document.createElement("div");
    setI18nRichText(host, "a <b>bold</b> and <code>code</code> and <em>no</em>");
    window.__html = host.innerHTML;
    const host2 = document.createElement("div");
    setI18nRichText(host2, '<b onclick="x()" class="c">click</b><img src=y onerror=z()>');
    window.__html2 = host2.innerHTML;
    window.__attrs2 = Array.from(host2.querySelectorAll("*"))
      .flatMap((e) => Array.from(e.attributes).map((a) => a.name));
    window.__imgs2 = host2.querySelectorAll("img").length;
    window.__bolds2 = host2.querySelectorAll("b").length;`);
  assert.equal(window.__html, "a <b>bold</b> and <code>code</code> and &lt;em&gt;no&lt;/em&gt;",
    "a tag outside the whitelist stays literal text");
  assert.deepEqual(Array.from(window.__attrs2), [],
    "no element built from rich text may carry an attribute");
  assert.equal(window.__imgs2, 0, "rich text may never build an element outside the whitelist");
  assert.equal(window.__bolds2, 0,
    "a tag written with an attribute must stay literal text, not become an element");
  assert.ok(!window.__html2.includes("<b "),
    "an attribute written into a catalog string must not survive as markup");
});

/* ================================================================ */
/*  Switching language                                               */
/* ================================================================ */

test("applying German translates the shell, sets <html lang>, and announces itself", async () => {
  const { window } = await loadIn("de");
  const doc = window.document;
  assert.equal(doc.documentElement.lang, "de");
  assert.equal(doc.getElementById("nav-settings").textContent.trim(), "Einstellungen");
  assert.equal(doc.getElementById("chat-send").getAttribute("title"), "Senden");
  assert.equal(doc.getElementById("chat-input").getAttribute("placeholder"),
    "Nachricht an das Modell\u2026");
  assert.equal(doc.getElementById("nav-toggle").getAttribute("aria-label"),
    "Men\u00fc \u00f6ffnen");
  assert.equal(doc.querySelectorAll("#nav-chat .nav-ic").length, 1,
    "the nav icon must survive translation");
});

test("the choice is cached in this browser and re-applied on the next load", async () => {
  const first = await loadIn(null);
  runScript(first.window, 'window.__p = applyLanguage("de");');
  await first.window.__p;
  assert.equal(first.window.localStorage.getItem("localm.language"), "de");
  const second = await loadIn("de");
  assert.equal(second.window.document.getElementById("nav-models").textContent.trim(), "Modelle");
});

test("a language change fires localm:language for dynamic surfaces", async () => {
  const { window } = await loadIn(null);
  runScript(window, `
    window.__seen = [];
    document.addEventListener("localm:language", (e) => window.__seen.push(e.detail.language));
    window.__p = applyLanguage("de");`);
  await window.__p;
  assert.ok(window.__seen.length > 0, "the event must fire");
  assert.deepEqual(Array.from(window.__seen), ["de"]);
});

test("a catalog that will not load keeps the current language instead of blanking it", async () => {
  const { window } = await loadIn(null, { deStatus: 500 });
  const warned = [];
  window.console.warn = (m) => warned.push(String(m));
  runScript(window, 'applyLanguage("de").then((ok) => { window.__ok = ok; });');
  assert.ok(await waitFor(() => window.__ok !== undefined));
  assert.equal(window.__ok, false, "a failed load must report failure, not success");
  assert.equal(evalIn(window, "currentLanguage()"), "en");
  assert.equal(window.document.getElementById("nav-settings").textContent.trim(), "Settings");
  assert.ok(warned.some((w) => w.includes("catalog")), "the failure must be reported");
});

test("an unknown stored language falls back to English and fetches nothing", async () => {
  const { window, calls } = await loadIn("../../../etc/passwd");
  await settle(30);
  assert.equal(evalIn(window, "currentLanguage()"), "en");
  assert.equal(window.document.documentElement.lang, "en");
  assert.deepEqual(calls.filter((c) => c.u.includes("i18n")), [],
    "only a registered language id may reach the catalog URL");
});

/* ================================================================ */
/*  The Settings control                                             */
/* ================================================================ */

test("the Appearance picker lists every language in its own name and selects the active one",
  async () => {
    const { window } = await loadIn("de", { withPages: true });
    runScript(window, "renderLanguagePicker();");
    const sel = window.document.getElementById("lang-select");
    assert.deepEqual(Array.from(sel.options).map((o) => [o.value, o.textContent]),
      [["en", "English"], ["de", "Deutsch"]]);
    assert.equal(sel.value, "de");
  });

test("choosing a language persists it with a PATCH to the server config", async () => {
  const { window, calls } = await loadIn(null);
  runScript(window, 'setLanguage("de").then((ok) => { window.__saved = ok; });');
  assert.ok(await waitFor(() => window.__saved !== undefined));
  const patch = calls.find((c) => c.method === "PATCH" && c.u.includes("/v1/config"));
  assert.ok(patch, "the choice must be written to the server, not only to this browser");
  assert.deepEqual(JSON.parse(patch.body), { language: "de" });
  assert.equal(window.__saved, true);
  assert.equal(evalIn(window, "currentLanguage()"), "de");
});

test("a server that refuses the save says so instead of reporting success", async () => {
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u.includes("/i18n/de.json")) return { ok: true, status: 200, json: async () => DE };
    if (u.includes("/api/models")) return { ok: false, status: 401, json: async () => ({}) };
    if (u.includes("/v1/config") && (opts.method || "GET") === "PATCH") {
      return { ok: false, status: 403, json: async () => ({}) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl });
  runScript(window, 'setLanguage("de").then((ok) => { window.__saved = ok; });');
  assert.ok(await waitFor(() => window.__saved !== undefined));
  assert.equal(window.__saved, false, "an unsaved choice must never be reported as saved");
  assert.equal(evalIn(window, "currentLanguage()"), "de",
    "the interface still switches; only the persistence failed");
});

test("the server config wins over this browser's cache at boot", async () => {
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u.includes("/i18n/de.json")) return { ok: true, status: 200, json: async () => DE };
    if (u.includes("/api/models")) return { ok: false, status: 401, json: async () => ({}) };
    if (u.endsWith("/v1/config") && (opts.method || "GET") === "GET") {
      return { ok: true, status: 200, json: async () => ({ language: "de" }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl });
  // Awaited in full: a pending continuation that outlives the test reaches a
  // window the harness has already closed.
  runScript(window, "window.__sync = syncLanguageFromConfig();");
  await window.__sync;
  assert.equal(window.document.documentElement.lang, "de");
  assert.equal(window.document.getElementById("nav-models").textContent.trim(), "Modelle");
});

test("text painted by JS is redrawn when the language changes", async () => {
  const { window } = await loadIn(null, { withPages: true });
  runScript(window, "renderChat();");
  const blurb = () => window.document.getElementById("chat-messages").textContent;
  assert.ok(blurb().includes("Everything stays on this machine"),
    `expected the English empty state, got ${JSON.stringify(blurb().slice(0, 120))}`);
  runScript(window, 'window.__p = applyLanguage("de");');
  await window.__p;
  assert.ok(blurb().includes("Alles bleibt auf diesem Rechner"),
    `the chat empty state must be rebuilt in German, got ${JSON.stringify(blurb().slice(0, 120))}`);
});

test("the privacy hint the app itself paints follows a later language change", async () => {
  // Driven through refreshCtxLimit, the real code path that creates the hint:
  // it is built once and never repainted, so it only follows the language if
  // the app marked it with its catalog keys when it was created.
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u.includes("/i18n/de.json")) return { ok: true, status: 200, json: async () => DE };
    if (u.includes("/api/models")) return { ok: false, status: 401, json: async () => ({}) };
    if (u.endsWith("/v1/config") && (opts.method || "GET") === "GET") {
      return { ok: true, status: 200, json: async () => ({ effective_mode: "privacy" }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl });
  runScript(window, "window.__p = refreshCtxLimit();");
  await window.__p;
  const hint = () => window.document.getElementById("privacy-hint");
  assert.ok(hint(), "refreshCtxLimit must have painted the privacy hint");
  assert.equal(hint().textContent, "privacy mode - this session only");
  runScript(window, 'window.__q = applyLanguage("de");');
  await window.__q;
  assert.equal(hint().textContent, "Privatmodus - nur diese Sitzung");
  assert.ok(hint().getAttribute("title").startsWith("Der Server l"),
    `the hint's tooltip must follow too, got ${hint().getAttribute("title")}`);
});

test("the model box is refreshed when the language changes", async () => {
  const { window, calls } = await loadIn(null);
  const models = () => calls.filter((c) => c.u.includes("/api/models")).length;
  const before = models();
  runScript(window, 'window.__p = applyLanguage("de");');
  await window.__p;
  await settle(30);
  assert.ok(models() > before,
    "the placeholder and status pill are only rewritten by refreshModels");
});

test("the persona and knowledge dropdowns rebuild their (none) option in German",
  async () => {
    const fetchImpl = async (url, opts = {}) => {
      const u = String(url);
      if (u.includes("/i18n/de.json")) return { ok: true, status: 200, json: async () => DE };
      if (u.includes("/api/models")) return { ok: false, status: 401, json: async () => ({}) };
      if (u.includes("/v1/prompts")) {
        return { ok: true, status: 200, json: async () => ({ prompts: [] }) };
      }
      if (u.includes("/collections") || u.includes("/rag")) {
        return { ok: true, status: 200, json: async () => ({ collections: [] }) };
      }
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    };
    const { window } = loadApp({ fetchImpl });
    runScript(window, "window.__p = refreshPersonas();");
    await window.__p;
    const opt = () => window.document.getElementById("p-persona").options[0].textContent;
    assert.equal(opt(), "(none)", "the rebuilt option starts in English");
    runScript(window, 'window.__q = applyLanguage("de");');
    await window.__q;
    assert.ok(await waitFor(() => opt() === "(keine)"),
      `the rebuilt option must follow the language, got ${opt()}`);
  });

test("the settings nav redraws its labels when the language changes", async () => {
  const { window } = await loadIn(null, { withPages: true });
  runScript(window, "buildSettingsNav();");
  const labels = () =>
    Array.from(window.document.querySelectorAll("#settings-nav .settings-nav-link"))
      .map((b) => b.textContent.trim());
  assert.ok(labels().includes("Security"), `expected the English nav, got ${labels()}`);
  runScript(window, 'window.__p = applyLanguage("de");');
  await window.__p;
  assert.ok(labels().includes("Sicherheit"),
    `the nav must redraw in German, got ${labels()}`);
});

/** A single protected, active builtin plugin - enough to exercise a status
 *  pill and a table header without pulling in a full plugin-refresh fixture. */
function pluginsPayload() {
  return {
    plugins: [
      { name: "chat", label: "Chat", builtin: true, installed: true,
        enabled: true, active: true, protected: true, description: "core",
        requires: [], missing_requires: [] },
    ],
  };
}

test("the plugins page redraws its status pills and table headers when the language changes",
  async () => {
    const fetchImpl = async (url) => {
      const u = String(url);
      if (u.includes("/i18n/de.json")) return { ok: true, status: 200, json: async () => DE };
      if (u.includes("/i18n/")) return { ok: false, status: 404, json: async () => ({}) };
      if (u.includes("/api/models")) return { ok: false, status: 401, json: async () => ({}) };
      if (u === "/api/plugins") {
        return { ok: true, status: 200, json: async () => pluginsPayload(), text: async () => "" };
      }
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    };
    const { window } = loadAppWithPages({ fetchImpl });

    async function settled() {
      for (let i = 0; i < 30; i++) {
        await new Promise((r) => setTimeout(r, 0));
        const s = window.document.querySelector(".catalog-status");
        if (s && !/Loading/.test(s.textContent)) return;
      }
    }
    const pill = () => window.document.querySelector("#catalog-table .job-state");
    const descHeader = () => window.document.querySelectorAll("#catalog-table th")[2];

    runScript(window, "_catalogStaggerMs = 0; renderCatalogPlugins();");
    await settled();
    assert.equal(pill() && pill().textContent, "protected", "starts in English");
    assert.equal(descHeader() && descHeader().textContent, "Description");

    runScript(window, 'window.__p = applyLanguage("de");');
    await window.__p;
    await settled();
    assert.equal(pill() && pill().textContent, "geschützt",
      "the status pill must be rebuilt in German");
    assert.equal(descHeader() && descHeader().textContent, "Beschreibung",
      "the table header must be rebuilt in German");
  });

test("the models page redraws its table headers and row badges when the language changes",
  async () => {
    const fetchImpl = async (url) => {
      const u = String(url);
      if (u.includes("/i18n/de.json")) return { ok: true, status: 200, json: async () => DE };
      if (u.includes("/i18n/")) return { ok: false, status: 404, json: async () => ({}) };
      if (u === "/api/models" || u.startsWith("/api/models?")) {
        return {
          ok: true, status: 200,
          json: async () => ({
            models: [{ name: "demo-model", active: true, loaded: true,
              model_type: "llm", size_bytes: 100, vision: true }],
            active: "demo-model",
          }),
        };
      }
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    };
    const { window } = loadAppWithPages({ fetchImpl });

    await window.refreshModelsPage();
    await settle(20);

    // A module-level MODEL_COLUMNS.label (the header) and a module-level
    // CAP_LABEL.vision (the pill, via visionBadge()) - both hold catalog key
    // names rather than resolved text, so this also covers the Finding-B
    // freeze-avoidance fix, not only the ordinary per-render t() calls.
    const roleHeader = () => window.document.querySelectorAll("#models-table th")[1];
    const activeTag = () => window.document.querySelector("#models-table .active-tag");
    const visionPill = () => window.document.querySelector("#models-table .cap-badge.cap-vision");

    assert.equal(roleHeader() && roleHeader().textContent, "Role", "starts in English");
    assert.equal(activeTag() && activeTag().textContent, "active");
    assert.equal(visionPill() && visionPill().textContent, "vision");

    runScript(window, 'window.__p = applyLanguage("de");');
    await window.__p;
    assert.ok(await waitFor(() => roleHeader() && roleHeader().textContent === "Rolle"),
      "the table must be redrawn after the language change");

    assert.equal(roleHeader() && roleHeader().textContent, "Rolle",
      "the table header must be rebuilt in German");
    assert.equal(activeTag() && activeTag().textContent, "aktiv",
      "the active tag must be rebuilt in German");
    assert.equal(visionPill() && visionPill().textContent, "Vision",
      "the vision capability pill must be rebuilt in German too");
  });

/* ================================================================ */
/*  Drift gates                                                      */
/* ================================================================ */

/** Every data-i18n* key named in index.html. */
function htmlKeys() {
  const re = /data-i18n(?:-rich|-title|-placeholder|-aria-label)?="([^"]+)"/g;
  return Array.from(INDEX_HTML.matchAll(re), (m) => m[1]);
}

test("every data-i18n key in index.html resolves in the English catalog", async () => {
  const { window } = await loadIn(null);
  const en = evalIn(window, "I18N_EN");
  const keys = htmlKeys();
  assert.ok(keys.length > 50, `expected the annotated shell, found ${keys.length} keys`);
  assert.deepEqual(keys.filter((k) => typeof en[k] !== "string"), [],
    "a data-i18n key with no catalog entry renders as the raw key");
});

// The migrated modules. A file added here must have its t() keys in the catalog.
const T_CALL_FILES = ["app/chat.js", "app/models-sidebar.js", "app/logo.js",
                      "app/settings-perf.js", "app/helpers.js", "app/media-gallery.js",
                      "app/slash.js", "app/cmdk.js", "app/picker.js",
                      "pages/settings.js", "pages/plugins.js", "pages/images.js",
                      "pages/models.js"];

/** Key-shaped string literals inside a balanced t(...) / tn(...) call. */
function tCallKeys(src) {
  const keys = [];
  for (const m of src.matchAll(/(?<![\w.])(t|tn)\(/g)) {
    const start = m.index + m[0].length;
    let depth = 1;
    let i = start;
    for (; i < src.length && depth > 0; i++) {
      if (src[i] === "(") depth++;
      else if (src[i] === ")") depth--;
    }
    for (const lit of src.slice(start, i).match(/"[^"\\]*"/g) || []) {
      const s = lit.slice(1, -1);
      if (/^[a-z][A-Za-z0-9]*(\.[A-Za-z0-9]+)+$/.test(s)) keys.push(s);
    }
  }
  return keys;
}

test("every t() key in the migrated modules resolves in the English catalog", async () => {
  const { window } = await loadIn(null);
  const en = evalIn(window, "I18N_EN");
  const bad = [];
  let seen = 0;
  for (const f of T_CALL_FILES) {
    for (const k of tCallKeys(read(...f.split("/")))) {
      seen++;
      // A plural base resolves through its .other form.
      if (typeof en[k] !== "string" && typeof en[`${k}.other`] !== "string") {
        bad.push(`${f}: ${k}`);
      }
    }
  }
  assert.ok(seen > 30, `expected the migrated call sites, found ${seen}`);
  assert.deepEqual(bad, []);
});

test("the settings nav group ids all have a catalog entry", async () => {
  const { window } = await loadIn(null, { withPages: true });
  const ids = Array.from(evalIn(window, "SETTINGS_GROUPS.map((g) => g.id)"));
  const en = evalIn(window, "I18N_EN");
  assert.ok(ids.length > 0);
  assert.deepEqual(ids.filter((id) => typeof en[`settings.group.${id}`] !== "string"), []);
});

test("the German catalog carries exactly the English keys", async () => {
  const { window } = await loadIn(null);
  const en = evalIn(window, "I18N_EN");
  const missing = Object.keys(en).filter((k) => !(k in DE));
  // A language with more plural categories than English legitimately carries
  // extra `<base>.<category>` keys; anything else is an orphan.
  const orphan = Object.keys(DE).filter((k) => {
    if (k in en) return false;
    return typeof en[`${k.replace(/\.[a-z]+$/, "")}.other`] !== "string";
  });
  assert.deepEqual(missing, [],
    "an English key with no German entry falls back and reads English");
  assert.deepEqual(orphan, [], "a German key with no English key is dead weight or a typo");
});

test("the German catalog keeps each string's placeholders and inline markup", async () => {
  const { window } = await loadIn(null);
  const en = evalIn(window, "I18N_EN");
  const ph = (s) => (s.match(/\{\w+\}/g) || []).sort().join(",");
  const tags = (s) => (s.match(/<\/?(?:b|strong|code)>/g) || []).sort().join(",");
  const bad = [];
  for (const [k, enText] of Object.entries(en)) {
    const deText = DE[k];
    if (typeof deText !== "string") continue;
    if (ph(enText) !== ph(deText)) bad.push(`${k}: placeholders ${ph(enText)} vs ${ph(deText)}`);
    if (tags(enText) !== tags(deText)) bad.push(`${k}: markup ${tags(enText)} vs ${tags(deText)}`);
  }
  assert.deepEqual(bad, []);
});

test("no catalog string carries an em-dash or en-dash", async () => {
  const { window } = await loadIn(null);
  const en = evalIn(window, "I18N_EN");
  const offenders = [];
  for (const [name, cat] of [["en", en], ["de", DE]]) {
    for (const [k, v] of Object.entries(cat)) {
      if (/[\u2013\u2014]/.test(v)) offenders.push(`${name}: ${k}`);
    }
  }
  assert.deepEqual(offenders, []);
});

test("index.html still carries the English each data-i18n key was written from", async () => {
  // Applying English to a document that has never been translated must leave the
  // rendered text unchanged: that is what proves the catalog and the markup have
  // not drifted apart.
  const { window } = await loadIn(null);
  const doc = window.document;
  const norm = (s) => s.replace(/\s+/g, " ").trim();
  const textBefore = Array.from(doc.querySelectorAll("[data-i18n]"),
    (el) => [el, el.dataset.i18n, norm(el.textContent)]);
  const richBefore = Array.from(doc.querySelectorAll("[data-i18n-rich]"),
    (el) => [el, el.dataset.i18nRich, norm(el.textContent)]);
  const attrBefore = [];
  for (const [marker, attr] of [["data-i18n-title", "title"],
                                ["data-i18n-placeholder", "placeholder"],
                                ["data-i18n-aria-label", "aria-label"]]) {
    for (const el of doc.querySelectorAll(`[${marker}]`)) {
      attrBefore.push([el, attr, el.getAttribute(attr)]);
    }
  }
  assert.ok(textBefore.length > 30 && richBefore.length > 0 && attrBefore.length > 15);

  runScript(window, "applyI18n(document);");

  const drift = [];
  for (const [el, key, was] of textBefore.concat(richBefore)) {
    const now = norm(el.textContent);
    if (now !== was) drift.push(`${key}: ${JSON.stringify(was)} -> ${JSON.stringify(now)}`);
  }
  for (const [el, attr, was] of attrBefore) {
    const now = el.getAttribute(attr);
    if (now !== was) drift.push(`${attr}: ${JSON.stringify(was)} -> ${JSON.stringify(now)}`);
  }
  assert.deepEqual(drift, []);
});

test("the service worker precaches the runtime and every language catalog", () => {
  const shell = /const SHELL = \[(.*?)\];/s.exec(read("sw.js"));
  assert.ok(shell, "SHELL array not found in sw.js");
  assert.ok(shell[1].includes('"/i18n/de.json"'),
    "a catalog missing from SHELL cannot be selected offline");
  assert.ok(shell[1].includes('"/app/i18n.js"') && shell[1].includes('"/app/i18n-en.js"'));
});
