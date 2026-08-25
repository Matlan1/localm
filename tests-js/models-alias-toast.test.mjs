// SPDX-License-Identifier: AGPL-3.0-or-later
// The Models page "alias" control toasts the name the server stored. Aliases
// are sanitized server-side, so "daily driver" comes back as "daily-driver".

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function makeFetch(models, calls, aliasResponse) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models/alias")) {
      calls.push(opts.body ? JSON.parse(opts.body) : {});
      return { ok: true, status: 200, json: async () => aliasResponse, text: async () => "" };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return {
        ok: true, status: 200,
        json: async () => ({ models, active: null }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

const MODELS = [
  { name: "gemma3-12b", active: false, loaded: false, model_type: "llm", size_bytes: 100 },
];

async function clickAlias(window, typed) {
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));
  runScript(window, `promptText = async () => ${JSON.stringify(typed)};`);
  const btn = [...window.document.querySelectorAll("#models-table tbody tr button")]
    .find((b) => b.textContent === "alias");
  assert.ok(btn, "the row exposes an alias control");
  await btn.onclick();
  await new Promise((r) => setTimeout(r, 0));
}

test("models-alias: the toast names the SANITIZED alias the server stored", async () => {
  const calls = [];
  const toasts = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch(MODELS, calls,
      { status: "aliased", model: "gemma3-12b", alias: "daily-driver" }),
  });
  window.toast = (msg) => toasts.push(String(msg));

  await clickAlias(window, "daily driver");

  assert.deepEqual(calls, [{ model: "gemma3-12b", alias: "daily driver" }],
    "the raw text is still what gets sent; the server owns sanitizing");
  assert.ok(toasts.some((t) => t.includes("daily-driver")),
    `the toast must name the stored alias, got: ${JSON.stringify(toasts)}`);
  assert.ok(!toasts.some((t) => t.includes("daily driver")),
    "the toast must NOT claim the raw name, which is not a registry key");
});

test("models-alias: an already-safe alias still reads back unchanged", async () => {
  const toasts = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch(MODELS, [],
      { status: "aliased", model: "gemma3-12b", alias: "daily-driver" }),
  });
  window.toast = (msg) => toasts.push(String(msg));

  await clickAlias(window, "daily-driver");

  assert.ok(toasts.some((t) => t.includes("daily-driver")));
});
