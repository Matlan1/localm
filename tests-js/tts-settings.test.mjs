// SPDX-License-Identifier: AGPL-3.0-or-later
// The Text-to-speech settings section (the tts plugin's own config block) and
// the voice-picker reconciliation around it.
//
// Before this, config["plugins"]["tts"] had NO write surface: the chat voice
// picker wrote browser localStorage ("localm.ttsVoice"), a DIFFERENT store from
// the server-side `voice` key, so a user who picked a voice believed they had
// changed the server default and had not (2026-07-22 settings-exposure audit).
// These tests pin the fix: the server-side block is editable in Settings, the
// per-browser override is NAMED and clearable, and a stored voice the provider
// no longer offers is ignored instead of being handed to the model.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = { fields: [
  { key: "host", widget: "text", label: "Host", help: "", group: "Server",
    owner: "core", default: "127.0.0.1" },
]};

const MEDIA = { plugins: [] };

function ttsField(key, extra = {}) {
  return {
    key, widget: "text", label: key, help: "", value: "", is_override: false,
    default: "", gui: true, advanced: false, admin_only: false, ...extra,
  };
}

function ttsPayload({ active = true, voice = "af_heart" } = {}) {
  return {
    plugin: "tts", active,
    fields: [
      ttsField("voice", {
        widget: "select", label: "Default voice", value: voice,
        default: "af_heart", is_override: voice !== "af_heart",
        options: ["af_heart", "am_onyx"],
        option_labels: ["Heart (en-us, Female, A)", "Onyx (en-us, Male, D)"],
      }),
      ttsField("speed", { widget: "number", label: "Speaking speed", value: 1,
                          default: 1, min: 0.5, max: 2, step: 0.05 }),
      ttsField("model", {
        widget: "select", label: "Voice model",
        value: "onnx-community/Kokoro-82M-v1.0-ONNX",
        default: "onnx-community/Kokoro-82M-v1.0-ONNX",
        options: ["onnx-community/Kokoro-82M-v1.0-ONNX",
                  "onnx-community/Kokoro-82M-v1.0-ONNX-timestamped"],
        option_labels: ["Kokoro 82M v1.0 (default)",
                        "Kokoro 82M v1.0, with word timestamps"],
      }),
      ttsField("device", { widget: "select", label: "Compute device", value: "auto",
                           default: "auto", options: ["auto", "webgpu", "wasm"],
                           advanced: true }),
      ttsField("dtype", { widget: "select", label: "Model precision", value: "auto",
                          default: "auto", options: ["auto", "fp32", "fp16", "q8"],
                          advanced: true }),
      // Never rendered: one engine is implemented, and the two script-URL keys
      // are an owner-only install escape hatch.
      ttsField("engine", { widget: "select", value: "kokoro", default: "kokoro",
                           options: ["kokoro"], gui: false }),
      ttsField("library", { value: "vendor/kokoro.min.js",
                            default: "vendor/kokoro.min.js",
                            gui: false, admin_only: true }),
      ttsField("wasm_paths", { gui: false, admin_only: true }),
    ],
  };
}

function makeFetch({ tts = ttsPayload(), ttsFails = false, posts = [] } = {}) {
  return async (url, opts = {}) => {
    const method = opts.method || "GET";
    const json = (body) => ({ ok: true, status: 200, json: async () => body,
                              text: async () => "" });
    if (url === "/v1/config/schema") return json(SCHEMA);
    if (url === "/v1/media/config") return json(MEDIA);
    if (url === "/v1/comfy/status")
      return json({ alive: false, launched_by_localm: false });
    if (url === "/v1/tts/config" && method === "GET") {
      if (ttsFails) return { ok: false, status: 503, statusText: "Service Unavailable",
                             json: async () => ({}), text: async () => "" };
      return json(tts);
    }
    if (url === "/v1/tts/config" && method === "POST") {
      const body = JSON.parse(opts.body);
      posts.push(body);
      const next = ttsPayload({ voice: body.voice || tts.fields[0].value });
      return json(next);
    }
    return json({ models: [], active: "", conversations: [], plugins: [] });
  };
}

async function render(win) {
  runScript(win, "refreshSettingsPage();");
  for (let i = 0; i < 16; i++) await new Promise((r) => setTimeout(r, 0));
}

const section = (win) => win.document.getElementById("settings-sec-tts");
const ctrl = (sec, key) => sec.querySelector(`[data-field-key="${key}"]`);

// ---- the section itself ------------------------------------------------- //

test("the section renders the user-facing fields and hides the deployment ones", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  const sec = section(win);
  assert.ok(sec, "the Text-to-speech section rendered");
  assert.equal(sec.dataset.group, "plugins", "it belongs to the Plugins nav group");
  for (const key of ["voice", "speed", "model", "device", "dtype"]) {
    assert.ok(ctrl(sec, key), `${key} must be editable in the GUI`);
  }
  for (const key of ["engine", "library", "wasm_paths"]) {
    assert.equal(ctrl(sec, key), null,
      `${key} must NOT render (single engine / owner-only script URL)`);
  }
});

test("device + precision render in the Advanced box, not next to the voice", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  const sec = section(win);
  const adv = [...sec.querySelectorAll(".media-comfy-box")]
    .find((b) => (b.querySelector("h4") || {}).textContent === "Advanced");
  assert.ok(adv, "the Advanced box rendered");
  assert.ok(adv.querySelector('[data-field-key="device"]'));
  assert.ok(adv.querySelector('[data-field-key="dtype"]'));
  assert.equal(adv.querySelector('[data-field-key="voice"]'), null);
});

test("the voice select keeps the ids as values but shows the friendly names", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  const sel = ctrl(section(win), "voice").querySelector("select");
  assert.deepEqual([...sel.options].map((o) => o.value), ["af_heart", "am_onyx"]);
  assert.deepEqual([...sel.options].map((o) => o.textContent),
                   ["Heart (en-us, Female, A)", "Onyx (en-us, Male, D)"]);
  assert.equal(sel.value, "af_heart", "the current server value is selected");
});

test("the model select offers the curated repos plus a Custom option", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  const sel = ctrl(section(win), "model").querySelector("select");
  const values = [...sel.options].map((o) => o.value);
  assert.ok(values.includes("onnx-community/Kokoro-82M-v1.0-ONNX"));
  assert.ok(values.includes("onnx-community/Kokoro-82M-v1.0-ONNX-timestamped"));
  assert.ok(values.includes("__custom__"), "a free-text escape hatch must exist");
  assert.equal(sel.value, "onnx-community/Kokoro-82M-v1.0-ONNX");
});

test("picking Custom reveals a text box, hidden again on picking a curated repo", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  const wrap = ctrl(section(win), "model");
  const sel = wrap.querySelector("select");
  const box = wrap.querySelector("input[type=text]");
  assert.ok(box.hidden, "the custom box starts hidden");
  sel.value = "__custom__";
  sel.dispatchEvent(new win.Event("change", { bubbles: true }));
  assert.ok(!box.hidden, "picking Custom reveals the text box");
  sel.value = "onnx-community/Kokoro-82M-v1.0-ONNX-timestamped";
  sel.dispatchEvent(new win.Event("change", { bubbles: true }));
  assert.ok(box.hidden, "picking a curated repo hides it again");
});

test("typing a custom repo id POSTs the typed value, not the sentinel", async () => {
  const posts = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch({ posts }) });
  await render(win);
  const sec = section(win);
  const wrap = ctrl(sec, "model");
  const sel = wrap.querySelector("select");
  const box = wrap.querySelector("input[type=text]");
  sel.value = "__custom__";
  sel.dispatchEvent(new win.Event("change", { bubbles: true }));
  box.value = "someorg/some-other-kokoro-onnx";
  box.dispatchEvent(new win.Event("input", { bubbles: true }));
  sec.querySelector(".settings-section-save").click();
  for (let i = 0; i < 16; i++) await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(posts, [{ model: "someorg/some-other-kokoro-onnx" }]);
});

test("picking Custom with nothing typed yet does not POST an accidental clear", async () => {
  const posts = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch({ posts }) });
  await render(win);
  const sec = section(win);
  const sel = ctrl(sec, "model").querySelector("select");
  sel.value = "__custom__";
  sel.dispatchEvent(new win.Event("change", { bubbles: true }));
  // Saving some OTHER field must not treat the untouched Custom box as a change.
  sec.querySelector('[data-field-key="speed"] input').value = "1.5";
  sec.querySelector(".settings-section-save").click();
  for (let i = 0; i < 16; i++) await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(posts, [{ speed: 1.5 }],
    "an empty Custom box must not clear the model override");
});

test("no section when the tts plugin is not active (its settings would do nothing)", async () => {
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch({ tts: ttsPayload({ active: false }) }) });
  await render(win);
  assert.equal(section(win), null);
});

test("a failed fetch SHOWS the failure instead of silently dropping the section", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch({ ttsFails: true }) });
  await render(win);
  const sec = section(win);
  assert.ok(sec, "a failure must still render a section (binding rule 5)");
  assert.match(sec.textContent, /Could not load the text-to-speech settings/i);
});

// ---- saving -------------------------------------------------------------- //

test("saving POSTs only the fields the user actually changed", async () => {
  const posts = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch({ posts }) });
  await render(win);
  const sec = section(win);
  const sel = ctrl(sec, "voice").querySelector("select");
  sel.value = "am_onyx";
  sec.querySelector(".settings-section-save").click();
  for (let i = 0; i < 16; i++) await new Promise((r) => setTimeout(r, 0));
  assert.equal(posts.length, 1, "one POST to /v1/tts/config");
  assert.deepEqual(posts[0], { voice: "am_onyx" },
    "untouched fields must not be sent (they would pin an override)");
});

test("saving another field never wipes a value the dropdown cannot show", async () => {
  // A <select> whose value is absent from its options reads back as "" = clear
  // this override, so saving ONLY the speed used to silently delete the voice.
  // Two shapes of that: a value outside the list, and no list at all (the
  // shipped voice list could not be read, where the server falls back to a
  // shape check - the GUI must stay usable rather than render an empty,
  // value-destroying dropdown).
  const outside = ttsPayload();
  outside.fields[0] = { ...outside.fields[0], value: "hand_edited", is_override: true };
  const posts = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch({ tts: outside, posts }) });
  await render(win);
  const sec = section(win);
  const sel = sec.querySelector('[data-field-key="voice"] select');
  assert.ok([...sel.options].some((o) => o.value === "hand_edited"),
    "the current value must remain selectable so it survives a save");
  assert.equal(sel.value, "hand_edited");
  sec.querySelector('[data-field-key="speed"] input').value = "1.5";
  sec.querySelector(".settings-section-save").click();
  for (let i = 0; i < 16; i++) await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(posts, [{ speed: 1.5 }], "only the speed changed");

  const noList = ttsPayload();
  noList.fields[0] = { ...noList.fields[0], options: undefined,
                       option_labels: undefined, value: "af_heart" };
  const posts2 = [];
  const { window: win2 } = loadAppWithPages({
    fetchImpl: makeFetch({ tts: noList, posts: posts2 }) });
  await render(win2);
  const sec2 = section(win2);
  const voice2 = sec2.querySelector('[data-field-key="voice"]');
  assert.equal(voice2.querySelector("select"), null,
    "with no voice list the field falls back to a text box, not an empty select");
  assert.equal(voice2.querySelector("input").value, "af_heart");
  sec2.querySelector('[data-field-key="speed"] input').value = "1.5";
  sec2.querySelector(".settings-section-save").click();
  for (let i = 0; i < 16; i++) await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(posts2, [{ speed: 1.5 }], "the voice must not be cleared");
});

test("an overridden field offers (inherit) so a dropdown override is clearable", async () => {
  const overridden = ttsPayload({ voice: "am_onyx" });      // is_override: true
  const posts = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch({ tts: overridden, posts }) });
  await render(win);
  const sec = section(win);
  const sel = sec.querySelector('[data-field-key="voice"] select');
  const inherit = [...sel.options].find((o) => o.value === "");
  assert.ok(inherit, "an overridden select must offer a way back to the default");
  assert.equal(inherit.textContent, "(inherit)");
  sel.value = "";
  sec.querySelector(".settings-section-save").click();
  for (let i = 0; i < 16; i++) await new Promise((r) => setTimeout(r, 0));
  // null and "" both mean "clear this override" to validate_tts_block.
  assert.deepEqual(posts, [{ voice: null }], "blank clears the server-side override");
});

test("saving nothing does not POST", async () => {
  const posts = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch({ posts }) });
  await render(win);
  section(win).querySelector(".settings-section-save").click();
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
  assert.equal(posts.length, 0);
});

// ---- the per-browser override -------------------------------------------- //

test("a per-browser voice override is named, not left to look like a broken setting", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  win.localStorage.setItem("localm.ttsVoice", "am_onyx");   // picked in chat
  await render(win);
  const note = section(win).querySelector(".tts-browser-override");
  assert.ok(note, "the browser override must be surfaced");
  assert.match(note.textContent, /Onyx \(en-us, Male, D\)/,
    "named by its friendly label, the same one the chat picker shows");
  assert.ok(note.querySelector(".tts-clear-override"), "and be clearable");
});

test("no override note when this browser follows the server default", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  win.localStorage.setItem("localm.ttsVoice", "af_heart");  // same as the server
  await render(win);
  assert.equal(section(win).querySelector(".tts-browser-override"), null);

  const { window: win2 } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win2);                                        // nothing stored
  assert.equal(section(win2).querySelector(".tts-browser-override"), null);
});

test("a stored voice the picker ignores is NOT announced as in effect", async () => {
  // The chat picker discards a stored voice the provider no longer offers, so
  // Settings must not tell the user "this browser plays <that voice>" - it
  // plays the server default, and the live-apply on save must not be blocked
  // by a pick that is not in force.
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  win.localStorage.setItem("localm.ttsVoice", "zz_gone");
  runScript(win, FAKE_PROVIDER);                 // offers af_heart + am_onyx only
  await render(win);
  assert.equal(section(win).querySelector(".tts-browser-override"), null,
    "no override note for a voice nothing actually plays");
  runScript(win, "window.__live = applyServerTtsConfig({ voice: 'am_onyx' });");
  assert.equal(win.__live, true, "a stale stored id must not block the live apply");
});

test("clearing the override drops the stored key so the server default applies", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  win.localStorage.setItem("localm.ttsVoice", "am_onyx");
  await render(win);
  section(win).querySelector(".tts-clear-override").click();
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
  assert.equal(win.localStorage.getItem("localm.ttsVoice"), null);
});

// ---- the picker's own reconciliation ------------------------------------- //

const FAKE_PROVIDER = `
  window.__setVoiceCalls = [];
  window.__applied = null;
  registerTTS({
    name: "Fake",
    voices: () => [{ id: "af_heart", label: "Heart" }, { id: "am_onyx", label: "Onyx" }],
    getVoice: () => "af_heart",
    setVoice: (id) => { window.__setVoiceCalls.push(id); },
    speaking: () => false,
    ready: () => Promise.resolve(),
    speak: () => {},
    stop: () => {},
    applyConfig: (c) => { window.__applied = c; },
  });
`;

// Values built inside the jsdom realm have that realm's prototypes, so
// node:assert/strict's deepEqual rejects them as "not reference-equal" even when
// the structure matches. Re-create them in this realm before comparing.
const calls = (win) => [...win.__setVoiceCalls];

test("a stored voice the provider no longer offers is ignored, not fed to the model", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  win.localStorage.setItem("localm.ttsVoice", "zz_gone");
  runScript(win, FAKE_PROVIDER);
  assert.deepEqual(calls(win), ["af_heart"],
    "the provider's own voice is used, never the unknown stored id");
  assert.equal(win.document.getElementById("p-voice").value, "af_heart");
});

test("a stored voice the provider offers still wins (the browser's explicit pick)", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  win.localStorage.setItem("localm.ttsVoice", "am_onyx");
  runScript(win, FAKE_PROVIDER);
  assert.deepEqual(calls(win), ["am_onyx"]);
  assert.equal(win.document.getElementById("p-voice").value, "am_onyx");
});

test("a saved server voice applies live only when this browser has no override", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  runScript(win, FAKE_PROVIDER);
  runScript(win, "window.__live = applyServerTtsConfig({ voice: 'am_onyx', speed: 1.5 });");
  assert.equal(win.__live, true);
  assert.equal(win.__applied.voice, "am_onyx");
  assert.equal(win.__applied.speed, 1.5);

  const { window: win2 } = loadAppWithPages({ fetchImpl: makeFetch() });
  win2.localStorage.setItem("localm.ttsVoice", "af_heart");   // explicit pick
  runScript(win2, FAKE_PROVIDER);
  runScript(win2, "window.__live = applyServerTtsConfig({ voice: 'am_onyx', speed: 1.5 });");
  assert.equal(win2.__live, false, "an explicit per-browser pick is never overwritten");
  assert.equal(win2.__applied.voice, undefined,
    "the browser's own voice is left alone");
  assert.equal(win2.__applied.speed, 1.5,
    "speed still applies - only the voice is the browser's own choice");
});
