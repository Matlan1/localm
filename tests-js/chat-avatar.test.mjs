// SPDX-License-Identifier: AGPL-3.0-or-later
// Chat-turn avatars: a derived monogram fallback for the assistant turn, and
// a user_avatar / model_avatar_default / model_avatar_overrides override, all
// local-only (a short glyph or a data: URI - see settings_schema.py's
// _validate_avatar_value, which is what keeps a URL from ever reaching here).
//
// monogramFor / avatarInfoFor / buildAvatarEl are plain top-level function
// declarations in chat.js, so after the harness's module-to-classic strip
// they are callable directly as window.<name>(...) - no runScript needed for
// a return value (runScript only injects a <script>; it has none). `chat` is
// a `const`, so it is NOT a window property; write to it via runScript.
import { test } from "node:test";
import assert from "node:assert/strict";

import { loadApp, runScript } from "./harness.mjs";

test("monogramFor is stable and gives one initial per word for a multi-word label", () => {
  const { window } = loadApp();
  const a1 = window.monogramFor("Qwen3-Coder-30B");
  const a2 = window.monogramFor("Qwen3-Coder-30B");
  assert.deepEqual(a1, a2, "the same label must always derive the same mark");
  assert.equal(a1.letters, "QC");
  assert.equal(typeof a1.hue, "number");
  assert.ok(a1.hue >= 0 && a1.hue < 360);

  const different = window.monogramFor("Llama-3-8B");
  assert.notEqual(different.letters, a1.letters,
    "two different labels should not collide on the common case");
});

test("monogramFor takes the first two characters of a single-word label", () => {
  const { window } = loadApp();
  assert.equal(window.monogramFor("phi3").letters, "PH");
});

test("monogramFor never throws and falls back to a placeholder with no alphanumerics", () => {
  const { window } = loadApp();
  assert.equal(window.monogramFor("").letters, "?");
  assert.equal(window.monogramFor("---").letters, "?");
});

test("addMessageRow: the user turn has no head wrapper when user_avatar is unset (unchanged default)", () => {
  const { window } = loadApp();
  const doc = window.document;
  const box = doc.getElementById("chat-messages");
  runScript(window, "chat.userAvatar = '';");
  window.addMessageRow(box, "user", "hi");

  const row = box.querySelector(".msg-row.user");
  assert.equal(row.querySelector(":scope > .msg-head"), null,
    "no .msg-head wrapper when nothing is configured");
  assert.notEqual(row.querySelector(":scope > .msg-role"), null,
    "msg-role stays a direct child, unchanged from before this feature");
});

test("addMessageRow: the assistant turn always gets a head wrapper, monogram by default", () => {
  const { window } = loadApp();
  const doc = window.document;
  const box = doc.getElementById("chat-messages");
  runScript(window, "chat.modelAvatarDefault = ''; chat.modelAvatarOverrides = {};");
  window.addMessageRow(box, "assistant", "hi", { model: "Mixtral-8x7B", final: true });

  const row = box.querySelector(".msg-row.assistant");
  const avatar = row.querySelector(".msg-head > .msg-avatar");
  assert.notEqual(avatar, null,
    "the monogram is the zero-config default for the assistant turn, per design");
  assert.equal(avatar.querySelector("img"), null, "the monogram is text, never an <img>");
  assert.equal(avatar.textContent, window.monogramFor("Mixtral-8x7B").letters);
});

test("model_avatar_overrides beats model_avatar_default, which beats the monogram", () => {
  const { window } = loadApp();
  const doc = window.document;
  const box = doc.getElementById("chat-messages");

  runScript(window, "chat.modelAvatarDefault = '\u{1F916}'; chat.modelAvatarOverrides = {};");
  window.addMessageRow(box, "assistant", "a", { model: "model-a", final: true });
  assert.equal(box.querySelector(".msg-row.assistant:last-child .msg-avatar").textContent, "\u{1F916}");

  runScript(window, "chat.modelAvatarOverrides = { 'model-a': '\u{1F419}' };");
  window.addMessageRow(box, "assistant", "b", { model: "model-a", final: true });
  assert.equal(box.querySelector(".msg-row.assistant:last-child .msg-avatar").textContent, "\u{1F419}");

  // A different model with no override still gets the default, not the override.
  window.addMessageRow(box, "assistant", "c", { model: "model-b", final: true });
  assert.equal(box.querySelector(".msg-row.assistant:last-child .msg-avatar").textContent, "\u{1F916}");
});

test("a data: URI avatar renders as an <img>, never a bare remote src", () => {
  const { window } = loadApp();
  const doc = window.document;
  const box = doc.getElementById("chat-messages");
  const uri = "data:image/png;base64,iVBORw0KGgo=";
  runScript(window, `chat.userAvatar = ${JSON.stringify(uri)};`);

  window.addMessageRow(box, "user", "hello");
  const img = box.querySelector(".msg-row.user .msg-avatar img");
  assert.notEqual(img, null);
  assert.equal(img.getAttribute("src"), uri);
});

test("avatarInfoFor never returns a URL-shaped value - it only ever sees what the server already validated", () => {
  const { window } = loadApp();
  // Defence in depth at the render layer: even if chat.userAvatar somehow held
  // a URL (it cannot, per settings_schema.py's _validate_avatar_value), the
  // renderer treats anything not starting with "data:" as an opaque glyph -
  // textContent, never an <img src>. This is the mechanism that makes that true.
  runScript(window, "chat.userAvatar = 'http://evil.example/x.png';");
  const doc = window.document;
  const box = doc.getElementById("chat-messages");
  window.addMessageRow(box, "user", "hi");
  const avatar = box.querySelector(".msg-row.user .msg-avatar");
  assert.equal(avatar.querySelector("img"), null,
    "a non-data: value is rendered as text, never as an <img src>");
  assert.equal(avatar.textContent, "http://evil.example/x.png");
});
