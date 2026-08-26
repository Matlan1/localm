// SPDX-License-Identifier: AGPL-3.0-or-later
// The Models page marks a model whose file has moved as missing and offers a
// relocate control. Covers the row rendering and the button's fetch/toast wiring.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function makeFetch(models, calls, relocateResponse) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models/relocate")) {
      calls.push(opts.body ? JSON.parse(opts.body) : {});
      if (relocateResponse.ok === false) {
        return {
          ok: false, status: relocateResponse.status || 400,
          json: async () => ({ detail: relocateResponse.detail }),
        };
      }
      return { ok: true, status: 200, json: async () => relocateResponse.body };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models, active: null }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

const MISSING_MODEL = {
  name: "moved-model", active: false, loaded: false, model_type: "llm",
  size_bytes: null, missing: true, last_path: "Z:/old/moved-model.gguf",
};
const HEALTHY_MODEL = {
  name: "healthy-model", active: false, loaded: false, model_type: "llm", size_bytes: 100,
};

function rowButtons(window, modelName) {
  const row = [...window.document.querySelectorAll("#models-table tbody tr")]
    .find((tr) => tr.querySelector(".name")?.textContent === modelName);
  assert.ok(row, `no row rendered for ${modelName}`);
  return [...row.querySelectorAll("button")];
}

test("models-relocate: a missing row shows the badge and a relocate control", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([MISSING_MODEL, HEALTHY_MODEL], [], { body: {} }),
  });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));

  const missingRow = [...window.document.querySelectorAll("#models-table tbody tr")]
    .find((tr) => tr.querySelector(".name")?.textContent === "moved-model");
  assert.ok(missingRow.querySelector(".missing-tag"), "missing row must carry the badge");
  assert.ok(rowButtons(window, "moved-model").some((b) => b.textContent === "relocate"),
    "missing row must offer a relocate control");

  const healthyRow = [...window.document.querySelectorAll("#models-table tbody tr")]
    .find((tr) => tr.querySelector(".name")?.textContent === "healthy-model");
  assert.ok(!healthyRow.querySelector(".missing-tag"), "a healthy row must not carry the badge");
  assert.ok(!rowButtons(window, "healthy-model").some((b) => b.textContent === "relocate"),
    "a healthy row has no relocate use and must not show the control");
});

test("models-relocate: the prompt is pre-filled with the last known path", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([MISSING_MODEL], [], { body: {} }),
  });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));

  let seenDefault = null;
  runScript(window, `promptText = async (msg, dflt) => { window.__seenDefault = dflt; return null; };`);
  const btn = rowButtons(window, "moved-model").find((b) => b.textContent === "relocate");
  await btn.onclick();
  await new Promise((r) => setTimeout(r, 0));
  seenDefault = window.__seenDefault;
  assert.equal(seenDefault, "Z:/old/moved-model.gguf");
});

test("models-relocate: success posts model+new_path, toasts, and refreshes", async () => {
  const calls = [];
  const toasts = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([MISSING_MODEL], calls,
      { body: { status: "relocated", model: "moved-model", path: "D:/new/moved-model.gguf" } }),
  });
  window.toast = (msg) => toasts.push(String(msg));
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));

  runScript(window, `promptText = async () => "D:/new/moved-model.gguf";`);
  const btn = rowButtons(window, "moved-model").find((b) => b.textContent === "relocate");
  await btn.onclick();
  await new Promise((r) => setTimeout(r, 0));

  assert.deepEqual(calls, [{ model: "moved-model", new_path: "D:/new/moved-model.gguf" }]);
  assert.ok(toasts.some((t) => t.includes("Relocated") && t.includes("moved-model")),
    `expected a success toast naming the model, got: ${JSON.stringify(toasts)}`);
});

test("models-relocate: a server-side rejection is reported, not swallowed as success", async () => {
  const toasts = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([MISSING_MODEL], [],
      { ok: false, status: 400, detail: "Not a GGUF model file: D:/junk.txt" }),
  });
  window.toast = (msg, isError) => toasts.push({ msg: String(msg), isError: !!isError });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));

  runScript(window, `promptText = async () => "D:/junk.txt";`);
  const btn = rowButtons(window, "moved-model").find((b) => b.textContent === "relocate");
  await btn.onclick();
  await new Promise((r) => setTimeout(r, 0));

  assert.ok(toasts.some((t) => t.isError && t.msg.includes("Not a GGUF model file")),
    `expected the server's own reason surfaced as an error toast, got: ${JSON.stringify(toasts)}`);
  assert.ok(!toasts.some((t) => !t.isError), "must not ALSO report success");
});

test("models-relocate: dismissing the prompt makes no request", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([MISSING_MODEL], calls, { body: {} }),
  });
  await window.refreshModelsPage();
  await new Promise((r) => setTimeout(r, 0));

  runScript(window, `promptText = async () => null;`);
  const btn = rowButtons(window, "moved-model").find((b) => b.textContent === "relocate");
  await btn.onclick();
  await new Promise((r) => setTimeout(r, 0));

  assert.deepEqual(calls, [], "dismissing the prompt must not call the route");
});
