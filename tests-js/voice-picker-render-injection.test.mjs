// SPDX-License-Identifier: AGPL-3.0-or-later
// populateVoicePicker() (localm/plugins/gui/static/app/settings-perf.js) fills
// the chat voice picker (<select id="p-voice">) from the active TTS provider's
// voices() list - which, for the tts plugin, is built from each
// vendor/voices.json entry's name/language/gender/grade (see tts.js's
// register()). Reading the source shows it only ever does
// document.createElement("option") + el.textContent = o.label, never
// innerHTML, so a hostile label can at most appear as escaped text. This
// drives that claim through jsdom with a hostile payload battery, the same
// pattern as tests-js/plugin-render-injection.test.mjs and
// tests-js/tts-render-injection.test.mjs.
//
// Unlike the tts.js download-consent dialog (which renders NO voice data at
// all), this picker legitimately displays each voice's label, so the correct
// assertion here is ESCAPED PRESENCE, like jobs.js's battery, not absence.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const PAYLOADS = [
  '<img src=x onerror="window.__INJECTED=1">',
  '<svg onload="window.__INJECTED=1"></svg>',
  '"><b>bold</b>',
  '<iframe src="javascript:window.__INJECTED=1"></iframe>',
];

// The detector: did the payload become ELEMENTS, or stay text?
function injectedElements(root) {
  return root.querySelectorAll("img, script, svg, b, iframe").length;
}

// JSON.stringify safely embeds the payload as a JS string literal in the
// injected script source (escaping the quotes/brackets the payloads carry) -
// this is test-harness plumbing, not the app's own escaping under test.
function fakeProviderScript(payload) {
  return `
    registerTTS({
      name: "Fake",
      voices: () => [{ id: "v1", label: ${JSON.stringify(payload)} }],
      getVoice: () => "v1",
      setVoice: () => {},
      speaking: () => false,
      ready: () => Promise.resolve(),
      speak: () => {},
      stop: () => {},
      applyConfig: () => {},
    });
  `;
}

test("the chat voice picker never parses a hostile voice label into DOM elements", () => {
  for (const payload of PAYLOADS) {
    const { window: win } = loadApp();
    runScript(win, fakeProviderScript(payload));

    const sel = win.document.getElementById("p-voice");
    assert.ok(sel, "the voice picker select must exist");
    // Scoped to the select itself, not the whole document: the real
    // index.html shell renders unrelated inline <svg> icons elsewhere on the
    // page, which a page-wide scan would misreport as injected by this
    // payload. A <select> can only ever hold <option>/<optgroup> children.
    assert.equal(injectedElements(sel), 0,
      `payload was parsed into DOM elements inside the picker: ${payload}`);
    assert.equal(win.__INJECTED, undefined, `payload executed: ${payload}`);
    const opt = sel.querySelector("option");
    assert.ok(opt, "a voice option must have been rendered");
    assert.equal(opt.textContent, payload,
      `the hostile label must survive as literal text, not vanish: ${payload}`);
  }
});

test("POSITIVE CONTROL: the same detector reports elements when they exist", () => {
  const { window: win } = loadApp();
  for (const payload of PAYLOADS) {
    const parsed = new win.DOMParser().parseFromString(
      `<body>${payload}</body>`, "text/html");
    assert.ok(injectedElements(parsed.body) > 0,
      `the detector must see an element parsed out of: ${payload}`);
  }
});
