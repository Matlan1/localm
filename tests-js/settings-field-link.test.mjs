// SPDX-License-Identifier: AGPL-3.0-or-later
// A SettingField carrying a `link` (e.g. hf_token / civitai_api_key's "Get a
// token"/"Get a key" pointers - see settings_schema.py) renders as a real
// clickable anchor that opens in the caller's own browser tab, never an
// in-app embed. Same harness/fetch-mock style as
// settings-default-placeholder.test.mjs.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA_WITH_LINK = {
  fields: [
    { key: "hf_token", widget: "secret", label: "Hugging Face API token",
      help: "Optional: raises rate limits.", group: "Models", owner: "core",
      applies: "live", secret: true, admin_only: true,
      link: { url: "https://huggingface.co/settings/tokens", label: "Get a token" } },
    { key: "mdns_name", widget: "text", label: "Network name (mDNS)", help: "",
      group: "Server", owner: "core", default: "localm", shipped_default: "localm" },
  ],
};

function makeFetch(schema) {
  return async (url, opts = {}) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => schema, text: async () => "" };
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

test("a field with a link renders a real anchor with the given url and label", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_WITH_LINK) });
  await render(win);
  const wrap = win.document.querySelector('input[data-key="hf_token"]').closest("div");
  const a = wrap.querySelector("a.settings-field-link");
  assert.ok(a, "the link must render as a real <a>, not plain text");
  assert.equal(a.href, "https://huggingface.co/settings/tokens");
  assert.equal(a.textContent, "Get a token");
});

test("the link opens in a new tab, never embedded in-app", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_WITH_LINK) });
  await render(win);
  const wrap = win.document.querySelector('input[data-key="hf_token"]').closest("div");
  const a = wrap.querySelector("a.settings-field-link");
  assert.equal(a.target, "_blank");
  assert.equal(a.rel, "noopener");
});

test("a field with no link renders no link element", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_WITH_LINK) });
  await render(win);
  const wrap = win.document.querySelector('input[data-key="mdns_name"]').closest("div");
  assert.equal(wrap.querySelector("a.settings-field-link"), null);
});

test("the secret field itself still renders as a masked password input, never prefilled", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(SCHEMA_WITH_LINK) });
  await render(win);
  const input = win.document.querySelector('input[data-key="hf_token"]');
  assert.equal(input.type, "password");
  assert.equal(input.value, "", "never prefill a real secret");
});
