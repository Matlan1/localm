// SPDX-License-Identifier: AGPL-3.0-or-later
// tts.js has exactly ONE function that ever touches the DOM: requestDownloadConsent()
// (the "Download voice model?" confirmation dialog). It takes zero arguments, and
// every window.el(...) call inside it uses a hardcoded literal string - never a
// value read from /api/tts/config or vendor/voices.json. tts-util.js has NO
// DOM/window surface at all (grep -nE "innerHTML|appendChild|createElement|
// document\.|window\." on tts-util.js returns nothing; its own module docstring
// says so too), so there is nothing in it to drive through jsdom.
//
// That claim had only ever been read, never driven. This drives it: every
// server/model-controlled field tts.js consumes (cfg.model, cfg.library,
// cfg.voice, cfg.wasm_paths, and each vendor/voices.json voice's name/language/
// gender/grade) is set to hostile markup, net_mode is forced to "ask" so the one
// real dialog actually opens, and the DOM is asked whether any of it was parsed
// into an element, or even rendered as text at all.
//
// Unlike jobs.js's battery (plugin-render-injection.test.mjs), which expects a
// hostile job name to survive as ESCAPED TEXT because jobs.js legitimately
// renders job data, this dialog renders NONE of these values, so the correct
// assertion here is ABSENCE, not escaped presence.
//
// The battery carries its own positive control (the last test): the same
// detector, pointed at a document built by parsing that markup, must report
// INJECTED. Without it, a clean sweep here could mean the payloads are safe OR
// that the detector cannot see anything at all.

import { test } from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const TTS_JS = join(HERE, "..", "localm", "plugins", "builtin", "tts", "static", "tts.js");

const PAYLOADS = [
  '<img src=x onerror="window.__INJECTED=1">',
  '<svg onload="window.__INJECTED=1"></svg>',
  '"><b>bold</b>',
  '<iframe src="javascript:window.__INJECTED=1"></iframe>',
];

// The detector: did the payload become ELEMENTS, or stay out of the DOM entirely?
function injectedElements(root) {
  return root.querySelectorAll("img, script, svg, b, iframe").length;
}

// Faithful stand-ins for app/helpers.js's $ / el / openModal, backed by a real
// jsdom document, adapted verbatim from tts-net-gate.test.mjs. el() sets
// textContent only, never innerHTML (helpers.js:847-852 confirms this is also
// true of the real implementation, not just the test double).
function installModalShell() {
  const dom = new JSDOM(
    `<!DOCTYPE html><html><body>
       <div id="modal" style="display:none">
         <div id="modal-title"></div>
         <div id="modal-body"></div>
       </div>
     </body></html>`,
    { url: "http://localhost:8642/" });
  const win = dom.window;
  win.$ = (id) => win.document.getElementById(id);
  win.el = (tag, cls, text) => {
    const n = win.document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };
  win.openModal = (title, bodyBuilder) => {
    win.$("modal-title").textContent = title;
    const body = win.$("modal-body");
    body.replaceChildren();
    bodyBuilder(body);
    win.$("modal").style.display = "flex";
  };
  global.window = win;
  global.document = win.document;
  delete global.confirm;                 // force the with-shell path
  // register() unconditionally builds its playback queue's `new Audio()`,
  // which jsdom does not implement - a no-op stub, since this battery never
  // reaches speak()/playNext(), which are the only things that touch it.
  win.Audio = class {};
  global.Audio = win.Audio;
  return win;
}

function clickButtonNamed(win, text) {
  const btn = Array.from(win.document.querySelectorAll("#modal-body button"))
    .find((b) => b.textContent === text);
  assert.ok(btn, `no "${text}" button rendered in the modal`);
  btn.onclick();
}

async function importFresh(path) {
  return import(pathToFileURL(path).href + `?t=${Date.now()}_${Math.random()}`);
}

function makeCtx() {
  const calls = { toasts: [], registerTTS: [] };
  const ctx = {
    authHeaders: () => ({}),
    toast: (msg, isError) => calls.toasts.push({ msg, isError }),
    registerTTS: (provider) => calls.registerTTS.push(provider),
  };
  return { ctx, calls };
}

// net_mode=ask forces planModelFetch() to "confirm" (net_mode=off refuses before
// ever offering a dialog; net_mode=allow proceeds without one), so it is the
// only mode that actually opens requestDownloadConsent(). Every server/model
// field tts.js reads carries the SAME payload. jsdom defines no global `caches`,
// so modelCached() resolves false, which is the "never downloaded before" state
// that makes the dialog open unconditionally.
function installHostileFetchEnv(win, payload) {
  win.fetch = async (url) => {
    const u = String(url);
    if (u.includes("/api/tts/config")) {
      return { ok: true, status: 200, json: async () => ({
        net_mode: "ask",
        model: payload,
        library: payload,
        voice: payload,
        wasm_paths: payload,
      }) };
    }
    if (u.includes("voices.json")) {
      return { ok: true, status: 200, json: async () => ({
        v1: { name: payload, language: payload, gender: payload, grade: payload },
      }) };
    }
    return { ok: false, status: 404, json: async () => ({}) };
  };
  global.fetch = win.fetch;
}

test("the tts download-consent dialog never renders server/model-supplied text", async () => {
  for (const payload of PAYLOADS) {
    const win = installModalShell();
    installHostileFetchEnv(win, payload);
    const { ctx, calls } = makeCtx();
    const { register } = await importFresh(TTS_JS);
    await register(ctx);

    const provider = calls.registerTTS[0];
    assert.ok(provider, "register() must call ctx.registerTTS with a provider");

    const readyPromise = provider.ready();
    // Let the gate's own fetch + modelCached microtasks settle before the
    // dialog is expected to be open (same idiom as tts-net-gate.test.mjs).
    await new Promise((r) => setTimeout(r, 0));

    assert.equal(win.$("modal-title").textContent, "Download voice model?",
      `net_mode=ask must offer the dialog even with a hostile cfg: ${payload}`);
    assert.equal(injectedElements(win.document.body), 0,
      `payload was parsed into DOM elements: ${payload}`);
    assert.equal(win.__INJECTED, undefined, `payload executed: ${payload}`);
    assert.equal(win.$("modal-body").textContent.includes(payload), false,
      `the dialog must never render this hostile value at all: ${payload}`);
    assert.match(win.$("modal-body").textContent, /huggingface\.co/,
      "the fixed copy must still render normally despite the hostile cfg");

    clickButtonNamed(win, "Not now");
    await assert.rejects(readyPromise);
  }
});

test("POSITIVE CONTROL: the same detector reports elements when they exist", async () => {
  // Proves the detector can report INJECTED. Without this, "0 elements" could
  // mean the payloads are safe OR that the query is simply blind. Built with
  // DOMParser rather than an unsafe assignment, so the proof costs the repo no
  // dangerous sink of its own.
  const win = installModalShell();
  for (const payload of PAYLOADS) {
    const parsed = new win.DOMParser().parseFromString(
      `<body>${payload}</body>`, "text/html");
    assert.ok(injectedElements(parsed.body) > 0,
      `the detector must see an element parsed out of: ${payload}`);
  }
});
