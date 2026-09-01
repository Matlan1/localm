// SPDX-License-Identifier: AGPL-3.0-or-later
// The Settings "Avatars" panel: user_avatar / user_name / model_avatar_default /
// model_avatar_overrides are Widget.HIDDEN (skipped from the flat schema
// grid, same as logo_style), so this proves the bespoke picker UI in
// buildAvatarsSection actually renders, round-trips existing values, and
// saves through the ordinary saveSettingsSection("avatars") PATCH path.
//
// The per-model override row's model id is a <select> populated from
// /api/models (real installed model names), not a free-text box - several
// cases here guard that an override for a model that is not (or no longer)
// installed is still shown and still preserved on save, never silently
// dropped or rewritten just because the picker rendered.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = {
  fields: [
    { key: "n_ctx", widget: "number", label: "Context window", help: "",
      group: "Engine", owner: "core", min: 512, step: 512, default: 4096 },
    { key: "user_avatar", widget: "hidden", label: "Your icon", help: "",
      group: "Chat", owner: "chat", default: "" },
    { key: "user_name", widget: "hidden", label: "Your name", help: "",
      group: "Chat", owner: "chat", default: "" },
    { key: "model_avatar_default", widget: "hidden", label: "Model icon", help: "",
      group: "Chat", owner: "chat", default: "" },
    { key: "model_avatar_overrides", widget: "hidden", label: "Per-model icons",
      help: "", group: "Chat", owner: "chat", default: {} },
  ],
};

function makeFetch(config, patches, models) {
  return async (url, opts = {}) => {
    const method = opts.method || "GET";
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    }
    if (url === "/v1/config" && method === "GET") {
      return { ok: true, status: 200, json: async () => config, text: async () => "" };
    }
    if (url === "/v1/config" && method === "PATCH") {
      const body = JSON.parse(opts.body);
      patches.push(body);
      Object.assign(config, body);
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    if (url.startsWith("/api/models")) {
      return {
        ok: true, status: 200, text: async () => "",
        json: async () => ({ models: (models || []).map((name) => ({ name })), active: "" }),
      };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

async function render(win) {
  await new Promise((r) => setTimeout(r, 0));
  runScript(win, "refreshSettingsPage();");
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
}

test("the avatar keys are skipped from the flat schema grid", async () => {
  const config = { user_avatar: "", user_name: "", model_avatar_default: "", model_avatar_overrides: {} };
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(config, []) });
  await render(win);
  const doc = win.document;
  assert.equal(doc.querySelector('[data-key="user_avatar"]'), null,
    "user_avatar has no generic text-box control");
  assert.equal(doc.querySelector('[data-key="user_name"]'), null,
    "user_name has no generic text-box control either");
});

test("the Avatars panel renders and round-trips existing values", async () => {
  const config = {
    user_avatar: "\u{1F60E}",
    user_name: "Matt",
    model_avatar_default: "data:image/png;base64,iVBORw0KGgo=",
    model_avatar_overrides: { "model-x": "\u{1F419}" },
  };
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(config, [], ["model-x"]) });
  await render(win);
  const doc = win.document;

  const panel = doc.querySelector("#settings-sec-avatars");
  assert.ok(panel, "the Avatars section exists");
  assert.equal(panel.dataset.group, "model", "shows in the same nav tab as Chat");

  const glyphInputs = panel.querySelectorAll(".avatar-picker-glyph");
  assert.equal(glyphInputs[0].value, "\u{1F60E}", "user_avatar round-trips into the first picker");

  const nameInput = panel.querySelector(".avatar-name-input");
  assert.ok(nameInput, "the name field renders");
  assert.equal(nameInput.value, "Matt", "user_name round-trips");

  const previews = panel.querySelectorAll(".avatar-picker-preview");
  assert.notEqual(previews[1].querySelector("img"), null,
    "a data: value renders as an <img> preview, not text");

  const overrideRow = panel.querySelector(".avatar-override-row");
  assert.ok(overrideRow, "the existing override is rendered as a row");
  const idSelect = overrideRow.querySelector(".avatar-override-id");
  assert.equal(idSelect.tagName, "SELECT", "the model id is a picker, not free text");
  assert.equal(idSelect.value, "model-x");
});

test("Save Avatars PATCHes exactly the four keys, and a removed row is dropped", async () => {
  const config = {
    user_avatar: "", user_name: "",
    model_avatar_overrides: { "stale-model": "\u{1F916}" },
  };
  const patches = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(config, patches, ["stale-model", "new-model"]),
  });
  await render(win);
  const doc = win.document;
  const panel = doc.querySelector("#settings-sec-avatars");

  // Set the user glyph and name through the real inputs, exactly as a person would.
  const userGlyph = panel.querySelector(".avatar-field-row .avatar-picker-glyph");
  userGlyph.value = "\u{1F600}";
  userGlyph.dispatchEvent(new win.Event("input"));
  const nameInput = panel.querySelector(".avatar-name-input");
  nameInput.value = "  Matt  ";
  nameInput.dispatchEvent(new win.Event("input"));

  // Remove the pre-existing override row.
  panel.querySelector(".avatar-override-row .btn-secondary:last-child").click();

  // Add a fresh override row and pick an installed model from the dropdown.
  panel.querySelector(".avatar-overrides-box > .btn-secondary").click();
  const freshRow = panel.querySelectorAll(".avatar-override-row")[0];
  const freshSelect = freshRow.querySelector(".avatar-override-id");
  freshSelect.value = "new-model";
  freshSelect.dispatchEvent(new win.Event("change"));
  const freshGlyph = freshRow.querySelector(".avatar-picker-glyph");
  freshGlyph.value = "\u{1F98A}";
  freshGlyph.dispatchEvent(new win.Event("input"));

  panel.querySelector(".actions .btn-primary").click();
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(patches.length, 1, "exactly one PATCH /v1/config fired");
  assert.equal(patches[0].user_avatar, "\u{1F600}");
  assert.equal(patches[0].user_name, "Matt", "the name is trimmed on save");
  assert.equal(patches[0].model_avatar_default, "");
  assert.deepEqual(patches[0].model_avatar_overrides, { "new-model": "\u{1F98A}" },
    "the removed row is gone and the new row is included - no stray keys");
  assert.deepEqual(Object.keys(patches[0]).sort(),
    ["model_avatar_default", "model_avatar_overrides", "user_avatar", "user_name"]);
});

test("the override model-id picker lists real installed models and saves the selected model's own name", async () => {
  const config = { user_avatar: "", user_name: "", model_avatar_overrides: {} };
  const patches = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(config, patches, ["llama-3-8b", "qwen3-coder-30b"]),
  });
  await render(win);
  const doc = win.document;
  const panel = doc.querySelector("#settings-sec-avatars");

  panel.querySelector(".avatar-overrides-box > .btn-secondary").click();
  const row = panel.querySelector(".avatar-override-row");
  const select = row.querySelector(".avatar-override-id");
  const optionValues = [...select.options].map((o) => o.value);
  assert.deepEqual(optionValues, ["llama-3-8b", "qwen3-coder-30b"],
    "populated from /api/models, no free-text entry and no disabled placeholder");

  select.value = "qwen3-coder-30b";
  select.dispatchEvent(new win.Event("change"));
  const glyph = row.querySelector(".avatar-picker-glyph");
  glyph.value = "\u{1F916}";
  glyph.dispatchEvent(new win.Event("input"));

  panel.querySelector(".actions .btn-primary").click();
  await new Promise((r) => setTimeout(r, 0));

  assert.deepEqual(patches[0].model_avatar_overrides, { "qwen3-coder-30b": "\u{1F916}" },
    "the PATCH carries the model's real registry name, not a guessed string");
});

test("an override for a currently-uninstalled model is preserved, never silently dropped", async () => {
  const config = {
    user_avatar: "", user_name: "",
    model_avatar_overrides: { "uninstalled-model": "\u{1F419}" },
  };
  const patches = [];
  // /api/models does NOT include "uninstalled-model" - it is missing/uninstalled
  // right now, but the saved override must still round-trip untouched.
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(config, patches, ["model-x"]),
  });
  await render(win);
  const doc = win.document;
  const panel = doc.querySelector("#settings-sec-avatars");

  const row = panel.querySelector(".avatar-override-row");
  const select = row.querySelector(".avatar-override-id");
  assert.equal(select.value, "uninstalled-model",
    "the row keeps its saved model id as the selected option");
  const preservedOpt = [...select.options].find((o) => o.value === "uninstalled-model");
  assert.ok(preservedOpt, "the uninstalled model id is still a real, selectable option");
  assert.match(preservedOpt.textContent, /not installed/);

  // Save without touching this row at all.
  panel.querySelector(".actions .btn-primary").click();
  await new Promise((r) => setTimeout(r, 0));

  assert.deepEqual(patches[0].model_avatar_overrides, { "uninstalled-model": "\u{1F419}" },
    "an untouched save must not drop or rewrite the uninstalled model's override");
});

test("with no models installed, a fresh override row shows a disabled placeholder instead of an empty picker", async () => {
  const config = { user_avatar: "", user_name: "", model_avatar_overrides: {} };
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(config, [], []) });
  await render(win);
  const doc = win.document;
  const panel = doc.querySelector("#settings-sec-avatars");

  panel.querySelector(".avatar-overrides-box > .btn-secondary").click();
  const select = panel.querySelector(".avatar-override-row .avatar-override-id");
  assert.equal(select.options.length, 1);
  assert.equal(select.options[0].value, "");
  assert.equal(select.options[0].disabled, true);
  assert.match(select.options[0].textContent, /No models installed/);
});
