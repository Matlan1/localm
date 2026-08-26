// SPDX-License-Identifier: AGPL-3.0-or-later
// The Settings "Avatars" panel: user_avatar / model_avatar_default /
// model_avatar_overrides are Widget.HIDDEN (skipped from the flat schema
// grid, same as logo_style), so this proves the bespoke picker UI in
// buildAvatarsSection actually renders, round-trips existing values, and
// saves through the ordinary saveSettingsSection("avatars") PATCH path.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = {
  fields: [
    { key: "n_ctx", widget: "number", label: "Context window", help: "",
      group: "Engine", owner: "core", min: 512, step: 512, default: 4096 },
    { key: "user_avatar", widget: "hidden", label: "Your icon", help: "",
      group: "Chat", owner: "chat", default: "" },
    { key: "model_avatar_default", widget: "hidden", label: "Model icon", help: "",
      group: "Chat", owner: "chat", default: "" },
    { key: "model_avatar_overrides", widget: "hidden", label: "Per-model icons",
      help: "", group: "Chat", owner: "chat", default: {} },
  ],
};

function makeFetch(config, patches) {
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
  const config = { user_avatar: "", model_avatar_default: "", model_avatar_overrides: {} };
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(config, []) });
  await render(win);
  const doc = win.document;
  assert.equal(doc.querySelector('[data-key="user_avatar"]'), null,
    "user_avatar has no generic text-box control");
});

test("the Avatars panel renders and round-trips existing values", async () => {
  const config = {
    user_avatar: "\u{1F60E}",
    model_avatar_default: "data:image/png;base64,iVBORw0KGgo=",
    model_avatar_overrides: { "model-x": "\u{1F419}" },
  };
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(config, []) });
  await render(win);
  const doc = win.document;

  const panel = doc.querySelector("#settings-sec-avatars");
  assert.ok(panel, "the Avatars section exists");
  assert.equal(panel.dataset.group, "model", "shows in the same nav tab as Chat");

  const glyphInputs = panel.querySelectorAll(".avatar-picker-glyph");
  assert.equal(glyphInputs[0].value, "\u{1F60E}", "user_avatar round-trips into the first picker");

  const previews = panel.querySelectorAll(".avatar-picker-preview");
  assert.notEqual(previews[1].querySelector("img"), null,
    "a data: value renders as an <img> preview, not text");

  const overrideRow = panel.querySelector(".avatar-override-row");
  assert.ok(overrideRow, "the existing override is rendered as a row");
  assert.equal(overrideRow.querySelector(".avatar-override-id").value, "model-x");
});

test("Save Avatars PATCHes exactly the three keys, and a removed row is dropped", async () => {
  const config = {
    user_avatar: "", model_avatar_default: "",
    model_avatar_overrides: { "stale-model": "\u{1F916}" },
  };
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(config, patches) });
  await render(win);
  const doc = win.document;
  const panel = doc.querySelector("#settings-sec-avatars");

  // Set the user glyph through the real input, exactly as a person would.
  const userGlyph = panel.querySelector(".avatar-field-row .avatar-picker-glyph");
  userGlyph.value = "\u{1F600}";
  userGlyph.dispatchEvent(new win.Event("input"));

  // Remove the pre-existing override row.
  panel.querySelector(".avatar-override-row .btn-secondary:last-child").click();

  // Add a fresh override row.
  panel.querySelector(".avatar-overrides-box > .btn-secondary").click();
  const freshRow = panel.querySelectorAll(".avatar-override-row")[0];
  freshRow.querySelector(".avatar-override-id").value = "new-model";
  freshRow.querySelector(".avatar-override-id").dispatchEvent(new win.Event("input"));
  const freshGlyph = freshRow.querySelector(".avatar-picker-glyph");
  freshGlyph.value = "\u{1F98A}";
  freshGlyph.dispatchEvent(new win.Event("input"));

  panel.querySelector(".actions .btn-primary").click();
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(patches.length, 1, "exactly one PATCH /v1/config fired");
  assert.equal(patches[0].user_avatar, "\u{1F600}");
  assert.equal(patches[0].model_avatar_default, "");
  assert.deepEqual(patches[0].model_avatar_overrides, { "new-model": "\u{1F98A}" },
    "the removed row is gone and the new row is included - no stray keys");
});
