// SPDX-License-Identifier: AGPL-3.0-or-later
// The Models page "rename" control toasts the name the server stored;
// rename_model sanitizes server-side.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function makeFetch(models, calls, renameResponse) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models/rename")) {
      calls.push(opts.body ? JSON.parse(opts.body) : {});
      return { ok: true, status: 200, json: async () => renameResponse, text: async () => "" };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models, active: null }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

const MODELS = [
  { name: "gemma3-12b", active: false, loaded: false, model_type: "llm", size_bytes: 100 },
];

async function clickRename(window, typed) {
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));
  runScript(window, `promptText = async () => ${JSON.stringify(typed)};`);
  const btn = [...window.document.querySelectorAll("#models-table tbody tr button")]
    .find((b) => b.textContent === "rename");
  assert.ok(btn, "the row exposes a rename control");
  await btn.onclick();
  await new Promise((r) => setTimeout(r, 0));
}

test("models-rename: the toast names the SANITIZED name the server stored", async () => {
  const calls = [];
  const toasts = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch(MODELS, calls,
      { status: "renamed", model: "gemma3-12b", new_name: "daily-driver" }),
  });
  window.toast = (msg) => toasts.push(String(msg));

  await clickRename(window, "daily driver");

  assert.deepEqual(calls, [{ model: "gemma3-12b", new_name: "daily driver" }],
    "the raw text is still what gets sent; the server owns sanitizing");
  assert.ok(toasts.some((t) => t.includes("daily-driver")),
    `the toast must name the stored new name, got: ${JSON.stringify(toasts)}`);
  assert.ok(!toasts.some((t) => t.includes("daily driver")),
    "the toast must NOT claim the raw name, which is not a registry key");
});

test("models-rename: server migration notes (e.g. the unreachable .localcoder case) reach the toast", async () => {
  const toasts = [];
  const NOTE = "A per-project .localcoder/config.toml 'model' setting (if any) "
    + "lives outside <data dir> and was NOT updated - fix it by hand in any "
    + "project that pinned this model.";
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch(MODELS, [],
      { status: "renamed", model: "gemma3-12b", new_name: "daily-driver", notes: [NOTE] }),
  });
  window.toast = (msg) => toasts.push(String(msg));

  await clickRename(window, "daily-driver");

  assert.ok(toasts.some((t) => t.includes(".localcoder") && t.includes("daily-driver")),
    `expected the toast to include both the new name and the migration note, got: ${JSON.stringify(toasts)}`);
});

test("models-rename: a response with no notes (or an older server) toasts cleanly, no 'undefined'", async () => {
  const toasts = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch(MODELS, [],
      { status: "renamed", model: "gemma3-12b", new_name: "daily-driver" }),   // no notes key
  });
  window.toast = (msg) => toasts.push(String(msg));

  await clickRename(window, "daily-driver");

  assert.deepEqual(toasts, ["Renamed to 'daily-driver'"]);
});

test("models-rename: a failed rename surfaces the server's error, not a silent no-op", async () => {
  const toasts = [];
  const { window } = loadAppWithPages({
    fetchImpl: async (url, opts = {}) => {
      const u = String(url);
      if (u.startsWith("/api/models/rename")) {
        return { ok: false, status: 409, json: async () => ({ detail: "Name already taken: taken" }) };
      }
      if (u === "/api/models" || u.startsWith("/api/models?")) {
        return { ok: true, status: 200, json: async () => ({ models: MODELS, active: null }) };
      }
      return { ok: true, status: 200, json: async () => ({}) };
    },
  });
  window.toast = (msg, isError) => toasts.push({ msg: String(msg), isError });

  await clickRename(window, "taken");

  assert.ok(toasts.some((t) => t.isError && t.msg.includes("Name already taken")),
    `expected the server's error text as an error toast, got: ${JSON.stringify(toasts)}`);
});

test("models-rename: typing the same name (or cancelling) sends no request", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch(MODELS, calls, { status: "renamed", model: "gemma3-12b", new_name: "gemma3-12b" }),
  });
  window.toast = () => {};

  await clickRename(window, "gemma3-12b");   // unchanged
  assert.deepEqual(calls, [], "renaming to the exact same name should be a client-side no-op");

  await clickRename(window, "");   // cancelled prompt
  assert.deepEqual(calls, [], "a cancelled/empty prompt must not fire a request");
});
