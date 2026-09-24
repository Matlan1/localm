// SPDX-License-Identifier: AGPL-3.0-or-later
// Secret fields (e.g. hf_token, civitai_api_key) must clearly communicate their
// status without leaking plaintext secrets: (not set), (configured), or (from environment).
// When configured, a Clear button allows removing the stored secret on next save.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// A plain field in hf_token's OWN group (Models), so a save of hf_token's own
// section has something else to send - real schema shape, mirrors
// import_max_depth in settings_schema.py. Left at its shipped default so it
// renders blank and is omitted unless a test explicitly edits it.
const MODELS_COMPANION_FIELD = { key: "import_max_depth", widget: "number",
  label: "Folder import depth", help: "", group: "Models", owner: "core",
  default: 3, shipped_default: 3, min: 1, max: 10, step: 1 };

const SCHEMA_UNSET = {
  fields: [
    { key: "hf_token", widget: "secret", label: "Hugging Face API token",
      help: "Optional: raises rate limits.", group: "Models", owner: "core",
      applies: "live", secret: true, is_set: false, env_set: false },
    MODELS_COMPANION_FIELD,
  ],
};

const SCHEMA_CONFIGURED = {
  fields: [
    { key: "hf_token", widget: "secret", label: "Hugging Face API token",
      help: "Optional: raises rate limits.", group: "Models", owner: "core",
      applies: "live", secret: true, is_set: true, env_set: false },
    MODELS_COMPANION_FIELD,
  ],
};

// The server's shape when model_source_credentials.json exists but could not
// be read: whether a token is stored is unknown.
const SCHEMA_UNKNOWN = {
  fields: [
    { key: "hf_token", widget: "secret", label: "Hugging Face API token",
      help: "Optional: raises rate limits.", group: "Models", owner: "core",
      applies: "live", secret: true, is_set: false, env_set: false,
      status_unknown: true },
    MODELS_COMPANION_FIELD,
  ],
};

const SCHEMA_ENV = {
  fields: [
    { key: "hf_token", widget: "secret", label: "Hugging Face API token",
      help: "Optional: raises rate limits.", group: "Models", owner: "core",
      applies: "live", secret: true, is_set: true, env_set: true },
  ],
};

function makeFetch(schema, patches = []) {
  return async (url, opts = {}) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => schema, text: async () => "" };
    }
    if (url === "/v1/config" && (opts.method || "GET") === "PATCH") {
      patches.push(JSON.parse(opts.body));
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

async function render(win) {
  runScript(win, "refreshSettingsPage();");
  await new Promise((r) => setTimeout(r, 0));
}

// Gives a save of hf_token's own (Models) section something else to send, so
// the save is never a no-op and the section that is actually PATCHed is the
// one hf_token itself lives in.
function touchCompanionField(win) {
  const depth = win.document.querySelector('input[data-key="import_max_depth"]');
  depth.value = "5";
  depth.dispatchEvent(new win.Event("input"));
}

test("secret field when not set displays (not set) tag and optional placeholder with no Clear button", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_UNSET) });
  await render(win);
  const input = win.document.querySelector('input[data-key="hf_token"]');
  const wrap = input.closest("[data-field-key]");
  const tag = wrap.querySelector(".secret-status-tag");
  const clearBtn = wrap.querySelector(".secret-clear-btn");

  assert.ok(tag, "status tag exists");
  assert.equal(tag.textContent, "(not set)");
  assert.ok(!tag.classList.contains("is-set"), "not-set does not have is-set class");
  assert.equal(input.placeholder, "not set (optional)");
  assert.equal(clearBtn, null, "no Clear button when unset");
});

test("typing into an unset secret field marks it as (new value)", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_UNSET, patches) });
  await render(win);
  const input = win.document.querySelector('input[data-key="hf_token"]');
  const wrap = input.closest("[data-field-key]");
  const tag = wrap.querySelector(".secret-status-tag");

  input.value = "hf_newtoken123";
  input.dispatchEvent(new win.Event("input"));

  assert.equal(tag.textContent, "(new value)");
  assert.ok(tag.classList.contains("is-set"));

  // Clear input back to empty restores (not set)
  input.value = "";
  input.dispatchEvent(new win.Event("input"));
  assert.equal(tag.textContent, "(not set)");
  assert.ok(!tag.classList.contains("is-set"));
});

test("secret field when configured displays (configured) tag, contextual placeholder, and Clear button", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_CONFIGURED) });
  await render(win);
  const input = win.document.querySelector('input[data-key="hf_token"]');
  const wrap = input.closest("[data-field-key]");
  const tag = wrap.querySelector(".secret-status-tag");
  const clearBtn = wrap.querySelector(".secret-clear-btn");

  assert.ok(tag, "status tag exists");
  assert.equal(tag.textContent, "(configured)");
  assert.ok(tag.classList.contains("is-set"), "configured has is-set class");
  assert.equal(input.placeholder, "saved (enter new value to replace)");
  assert.ok(clearBtn, "Clear button is rendered");
  assert.equal(clearBtn.textContent, "Clear");
});

test("clicking Clear button toggles to will-clear state and back", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_CONFIGURED, patches) });
  await render(win);
  const input = win.document.querySelector('input[data-key="hf_token"]');
  const wrap = input.closest("[data-field-key]");
  const tag = wrap.querySelector(".secret-status-tag");
  const clearBtn = wrap.querySelector(".secret-clear-btn");

  // Click Clear
  clearBtn.click();
  assert.equal(tag.textContent, "(will clear on save)");
  assert.ok(tag.classList.contains("is-clearing"));
  assert.equal(input.placeholder, "will be removed on save");
  assert.equal(clearBtn.textContent, "Undo");

  // Save in will-clear state sends empty string to clear secret on backend
  const secId = input.closest(".settings-section").dataset.sec;
  runScript(win, `saveSettingsSection(${JSON.stringify(secId)});`);
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(patches.length, 1);
  assert.equal(patches[0].hf_token, "");

  // Click Undo restores configured state
  clearBtn.click();
  assert.equal(tag.textContent, "(configured)");
  assert.ok(tag.classList.contains("is-set"));
  assert.equal(input.placeholder, "saved (enter new value to replace)");
  assert.equal(clearBtn.textContent, "Clear");
});

test("secret field when set via environment displays (from environment) and no Clear button", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_ENV) });
  await render(win);
  const input = win.document.querySelector('input[data-key="hf_token"]');
  const wrap = input.closest("[data-field-key]");
  const tag = wrap.querySelector(".secret-status-tag");
  const clearBtn = wrap.querySelector(".secret-clear-btn");

  assert.ok(tag, "status tag exists");
  assert.equal(tag.textContent, "(from environment)");
  assert.ok(tag.classList.contains("is-set"), "env-configured has is-set class");
  assert.equal(input.placeholder, "set via environment variable");
  assert.equal(clearBtn, null, "no Clear button for env credentials");
});

test("secret field with an unreadable store displays (status unknown), not (not set), and no Clear button", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_UNKNOWN) });
  await render(win);
  const input = win.document.querySelector('input[data-key="hf_token"]');
  const wrap = input.closest("[data-field-key]");
  const tag = wrap.querySelector(".secret-status-tag");
  const clearBtn = wrap.querySelector(".secret-clear-btn");

  assert.ok(tag, "status tag exists");
  assert.equal(tag.textContent, "(status unknown)");
  assert.ok(tag.classList.contains("is-unknown"), "unknown status has is-unknown class");
  assert.ok(!tag.classList.contains("is-set"), "unknown status is not shown as set");
  assert.equal(input.placeholder, "stored credentials could not be read");
  assert.equal(clearBtn, null, "no Clear button when the stored value cannot be read");
});

test("typing into a status-unknown secret then emptying it restores (status unknown)", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_UNKNOWN) });
  await render(win);
  const input = win.document.querySelector('input[data-key="hf_token"]');
  const wrap = input.closest("[data-field-key]");
  const tag = wrap.querySelector(".secret-status-tag");

  input.value = "hf_newtoken123";
  input.dispatchEvent(new win.Event("input"));
  assert.equal(tag.textContent, "(new value)");

  input.value = "";
  input.dispatchEvent(new win.Event("input"));
  assert.equal(tag.textContent, "(status unknown)");
  assert.ok(tag.classList.contains("is-unknown"));
  assert.equal(input.placeholder, "stored credentials could not be read");
});

test("untouched secret field is omitted on save", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_UNSET, patches) });
  await render(win);
  touchCompanionField(win);
  const input = win.document.querySelector('input[data-key="hf_token"]');

  const secId = input.closest(".settings-section").dataset.sec;
  runScript(win, `saveSettingsSection(${JSON.stringify(secId)});`);
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(patches.length, 1);
  assert.ok(!("hf_token" in patches[0]), "untouched secret is omitted from patch");
});

test("secret field typed then cleared back to empty is omitted on save", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_UNSET, patches) });
  await render(win);
  touchCompanionField(win);
  const input = win.document.querySelector('input[data-key="hf_token"]');
  input.value = "hf_typed_then_removed";
  input.dispatchEvent(new win.Event("input"));
  input.value = "";
  input.dispatchEvent(new win.Event("input"));

  const secId = input.closest(".settings-section").dataset.sec;
  runScript(win, `saveSettingsSection(${JSON.stringify(secId)});`);
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(patches.length, 1);
  assert.ok(!("hf_token" in patches[0]), "typed-then-cleared secret is omitted from patch");
});

test("secret field with only whitespace typed is omitted on save", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_UNSET, patches) });
  await render(win);
  touchCompanionField(win);
  const input = win.document.querySelector('input[data-key="hf_token"]');
  input.value = "   ";
  input.dispatchEvent(new win.Event("input"));

  const secId = input.closest(".settings-section").dataset.sec;
  runScript(win, `saveSettingsSection(${JSON.stringify(secId)});`);
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(patches.length, 1);
  assert.ok(!("hf_token" in patches[0]), "whitespace-only secret is omitted from patch");
});

test("secret field Clear then Undo leaves it omitted on save", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_CONFIGURED, patches) });
  await render(win);
  touchCompanionField(win);
  const input = win.document.querySelector('input[data-key="hf_token"]');
  const wrap = input.closest("[data-field-key]");
  const clearBtn = wrap.querySelector(".secret-clear-btn");

  clearBtn.click();          // willClear -> true
  clearBtn.click();          // Undo -> willClear -> false

  const secId = input.closest(".settings-section").dataset.sec;
  runScript(win, `saveSettingsSection(${JSON.stringify(secId)});`);
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(patches.length, 1);
  assert.ok(!("hf_token" in patches[0]), "Clear-then-Undo secret is omitted from patch");
});

test("typing a new secret into a configured field updates status and sends new value on save", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_CONFIGURED, patches) });
  await render(win);
  const input = win.document.querySelector('input[data-key="hf_token"]');
  const wrap = input.closest("[data-field-key]");
  const tag = wrap.querySelector(".secret-status-tag");

  input.value = "hf_replacement_key";
  input.dispatchEvent(new win.Event("input"));

  assert.equal(tag.textContent, "(new value)");
  assert.ok(tag.classList.contains("is-set"));

  const secId = input.closest(".settings-section").dataset.sec;
  runScript(win, `saveSettingsSection(${JSON.stringify(secId)});`);
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(patches.length, 1);
  assert.equal(patches[0].hf_token, "hf_replacement_key");

  // If user clears the input back to empty, it reverts to (configured)
  input.value = "";
  input.dispatchEvent(new win.Event("input"));
  assert.equal(tag.textContent, "(configured)");
  assert.equal(input.placeholder, "saved (enter new value to replace)");
});

test("models section heading is Model Library & Sources", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_CONFIGURED) });
  await render(win);
  const input = win.document.querySelector('input[data-key="hf_token"]');
  const sec = input.closest(".settings-section");
  assert.ok(sec, "Models section exists");
  assert.equal(sec.dataset.group, "model");
  const head = sec.querySelector(".settings-section-head");
  assert.ok(head, "section heading exists");
  assert.match(head.textContent, /Model Library & Sources/);
});

