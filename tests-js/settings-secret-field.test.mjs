// SPDX-License-Identifier: AGPL-3.0-or-later
// Secret fields (e.g. hf_token, civitai_api_key) must clearly communicate their
// status without leaking plaintext secrets: (not set), (configured), or (from environment).
// When configured, a Clear button allows removing the stored secret on next save.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA_UNSET = {
  fields: [
    { key: "hf_token", widget: "secret", label: "Hugging Face API token",
      help: "Optional: raises rate limits.", group: "Models", owner: "core",
      applies: "live", secret: true, is_set: false, env_set: false },
    { key: "mdns_name", widget: "text", label: "Network name (mDNS)", help: "",
      group: "Server", owner: "core", default: "localm", shipped_default: "localm" },
  ],
};

const SCHEMA_CONFIGURED = {
  fields: [
    { key: "hf_token", widget: "secret", label: "Hugging Face API token",
      help: "Optional: raises rate limits.", group: "Models", owner: "core",
      applies: "live", secret: true, is_set: true, env_set: false },
  ],
};

const SCHEMA_ENV = {
  fields: [
    { key: "hf_token", widget: "secret", label: "Hugging Face API token",
      help: "Optional: raises rate limits.", group: "Models", owner: "core",
      applies: "live", secret: true, is_set: false, env_set: true },
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
  assert.equal(input.placeholder, "set via environment variable");
  assert.equal(clearBtn, null, "no Clear button for env credentials");
});

test("untouched secret field is omitted on save", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_UNSET, patches) });
  await render(win);
  const mdns = win.document.querySelector('input[data-key="mdns_name"]');
  mdns.value = "custom-box";
  mdns.dispatchEvent(new win.Event("input"));

  const secId = mdns.closest(".settings-section").dataset.sec;
  runScript(win, `saveSettingsSection(${JSON.stringify(secId)});`);
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(patches.length, 1);
  assert.ok(!("hf_token" in patches[0]), "untouched secret is omitted from patch");
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

