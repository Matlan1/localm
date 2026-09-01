// SPDX-License-Identifier: AGPL-3.0-or-later
// The "Speak this reply aloud" message button had no active/playing state and
// no visible way to tell it would stop playback (chat.js's speak() already
// stops on a second click - settings-perf.js speak(): "if
// (ttsProvider.speaking()) { ttsProvider.stop(); if (opts.toggle) return; }" -
// this was purely a missing indicator).
//
// speakToggle() (chat.js) keeps the CLICKED button's own .speaking class and
// icon in sync with what actually happens: starting, an explicit stop, an
// interruption by a different message's button, and the utterance ending on
// its own via speak()'s opts.onEnd. addMessageRow's action loop must pass the
// button itself to the action handler for speakToggle to have anything to
// mark - test_addMessageRow_action_receives_its_own_button pins that half.

import { test } from "node:test";
import assert from "node:assert/strict";

import { loadApp, runScript } from "./harness.mjs";

// A controllable fake Kokoro-shaped provider. stop() fires whatever onEnd was
// registered by the most recent speak() call, mirroring tts.js's real
// endCallback/fireEnd() contract (see tts.js stop()/speak()).
const FAKE_PROVIDER = `
  window.__speaking = false;
  window.__endCb = null;
  window.__speakCalls = [];
  window.__stopCalls = 0;
  registerTTS({
    name: "Fake",
    voices: () => [],
    getVoice: () => "x",
    setVoice: () => {},
    speaking: () => window.__speaking,
    ready: () => Promise.resolve(),
    speak: (text, opts) => {
      window.__speaking = true;
      window.__speakCalls.push(text);
      window.__endCb = (opts && opts.onEnd) || null;
    },
    stop: () => {
      window.__speaking = false;
      window.__stopCalls++;
      const cb = window.__endCb;
      window.__endCb = null;
      if (cb) cb();
    },
  });
`;

function speakActions(win, text) {
  return [["Speak this reply aloud", (btn) => win.speakToggle(btn, text), "speak"]];
}

function speakBtn(box) {
  return box.querySelector(".msg-meta button.action");
}

function iconNameOf(btn) {
  return btn.querySelector("[data-icon-name]").dataset.iconName;
}

test("addMessageRow's action handler receives the button it belongs to", () => {
  const { window: win } = loadApp();
  const box = win.document.getElementById("chat-messages");
  let got = null;
  win.addMessageRow(box, "assistant", "hi", {
    actions: [["do it", (btn) => { got = btn; }]],
    final: true,
  });
  box.querySelector(".msg-meta button.action").click();
  assert.notEqual(got, null, "the action handler must receive an argument");
  assert.equal(got.tagName, "BUTTON", "the argument must be the button element itself");
});

test("clicking speak starts playback, marks the button active, and swaps to the stop icon", () => {
  const { window: win } = loadApp();
  runScript(win, FAKE_PROVIDER);
  const box = win.document.getElementById("chat-messages");
  win.addMessageRow(box, "assistant", "hello", { actions: speakActions(win, "hello"), final: true });

  const btn = speakBtn(box);
  assert.equal(btn.classList.contains("speaking"), false, "not active before any click");
  btn.click();

  assert.equal(win.__speakCalls.length, 1);
  assert.equal(win.__speakCalls[0], "hello");
  assert.ok(btn.classList.contains("speaking"), "the clicked button must show it is playing");
  assert.equal(iconNameOf(btn), "stop", "the icon swaps to stop while active");
});

test("clicking the SAME active button again stops it and clears the indicator", () => {
  const { window: win } = loadApp();
  runScript(win, FAKE_PROVIDER);
  const box = win.document.getElementById("chat-messages");
  win.addMessageRow(box, "assistant", "hello", { actions: speakActions(win, "hello"), final: true });

  const btn = speakBtn(box);
  btn.click();                                    // start
  assert.ok(btn.classList.contains("speaking"));
  btn.click();                                     // stop (toggle)

  assert.equal(win.__stopCalls, 1, "the second click must stop, not start a new utterance");
  assert.equal(win.__speakCalls.length, 1, "must not start a second utterance");
  assert.equal(btn.classList.contains("speaking"), false,
    "clicking the active button again must clear its own indicator");
  assert.equal(iconNameOf(btn), "speak", "the icon reverts once stopped");
});

test("the utterance ending on its own (onEnd) clears the indicator, with no further click", () => {
  const { window: win } = loadApp();
  runScript(win, FAKE_PROVIDER);
  const box = win.document.getElementById("chat-messages");
  win.addMessageRow(box, "assistant", "hello", { actions: speakActions(win, "hello"), final: true });

  const btn = speakBtn(box);
  btn.click();
  assert.ok(btn.classList.contains("speaking"));

  runScript(win, "window.__speaking = false; window.__endCb();");   // simulate natural completion
  assert.equal(btn.classList.contains("speaking"), false,
    "the button must clear itself once the provider reports it is done");
});

test("starting a different message's speak clears the previously active button's indicator", () => {
  const { window: win } = loadApp();
  runScript(win, FAKE_PROVIDER);
  const box = win.document.getElementById("chat-messages");
  win.addMessageRow(box, "assistant", "first", { actions: speakActions(win, "first"), final: true });
  win.addMessageRow(box, "assistant", "second", { actions: speakActions(win, "second"), final: true });

  const [btnA, btnB] = [...box.querySelectorAll(".msg-meta button.action")];
  btnA.click();
  assert.ok(btnA.classList.contains("speaking"));

  // Existing contract (settings-perf.js speak()): a toggle click while
  // something else is speaking only STOPS it - it never starts the new one on
  // the same click. So B does not light up here, but A's indicator must clear.
  btnB.click();
  assert.equal(win.__stopCalls, 1);
  assert.equal(win.__speakCalls.length, 1, "B's own utterance did not start on this click");
  assert.equal(btnA.classList.contains("speaking"), false, "A must no longer read as active");
  assert.equal(btnB.classList.contains("speaking"), false, "B never actually started");
});

// A THIRD-PARTY provider only has to implement the documented shape
// (speaking()/speak()/stop()) - onEnd is an addition to speak()'s opts that an
// older or unaware plugin is free to ignore entirely. The active/playing
// indicator must still work correctly from settings-perf.js's speak() return
// value alone, never relying on a provider that does not know about onEnd.
const DUMB_PROVIDER = `
  window.__speaking = false;
  window.__speakCalls = [];
  window.__stopCalls = 0;
  registerTTS({
    name: "Dumb",
    voices: () => [],
    getVoice: () => "x",
    setVoice: () => {},
    speaking: () => window.__speaking,
    ready: () => Promise.resolve(),
    speak: (text) => { window.__speaking = true; window.__speakCalls.push(text); },
    stop: () => { window.__speaking = false; window.__stopCalls++; },
  });
`;

test("a provider that never calls onEnd still gets its OLD button cleared when a different one is clicked", () => {
  const { window: win } = loadApp();
  runScript(win, DUMB_PROVIDER);
  const box = win.document.getElementById("chat-messages");
  win.addMessageRow(box, "assistant", "first", { actions: speakActions(win, "first"), final: true });
  win.addMessageRow(box, "assistant", "second", { actions: speakActions(win, "second"), final: true });

  const [btnA, btnB] = [...box.querySelectorAll(".msg-meta button.action")];
  btnA.click();
  assert.ok(btnA.classList.contains("speaking"));

  btnB.click();                 // stops A (existing toggle contract); B does not start
  assert.equal(win.__stopCalls, 1);
  assert.equal(btnA.classList.contains("speaking"), false,
    "must clear via the click path itself, not depend on the provider calling onEnd");
});

test("the browser-fallback voice (no Kokoro plugin) also updates the button via onend/onerror", () => {
  const { window: win } = loadApp();       // no FAKE_PROVIDER: ttsProvider stays null
  const box = win.document.getElementById("chat-messages");
  win.addMessageRow(box, "assistant", "hi", { actions: speakActions(win, "hi"), final: true });

  const btn = speakBtn(box);
  btn.click();
  assert.equal(win.__spoken.length, 1, "the browser SpeechSynthesisUtterance path was used");
  assert.ok(btn.classList.contains("speaking"));

  win.__spoken[0].onend();                 // simulate the browser finishing the utterance
  assert.equal(btn.classList.contains("speaking"), false,
    "the fallback path must clear the indicator too, via the same onEnd wiring");
});
