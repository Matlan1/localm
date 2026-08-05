// SPDX-License-Identifier: AGPL-3.0-or-later
// The models-page Remove flow hardcoded "Remove failed" for ANY non-"done"
// streamJob() outcome, with no interpolation - the same false-claim shape
// the streamJob reconnect fix exists to prevent, just spelled differently:
// a lost SSE connection (streamJob's new "disconnected" status) told the
// user the remove had failed, though nothing about the remove itself was
// known to have gone wrong. See streamjob-reconnect.test.mjs for the
// helpers.js-level contract this renders, and download-speed-eta.test.mjs
// for the same fix applied to the pull bar.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const MODELS_PAYLOAD = {
  models: [{ name: "other-model", model_type: "llm", active: false, size_bytes: 1e9 }],
  active: "current-model",
};

function makeFetch() {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => MODELS_PAYLOAD, text: async () => "" };
    }
    if (u === "/api/models/remove" && (opts.method || "GET") === "POST") {
      return { ok: true, status: 200, json: async () => ({ job_id: "rm1" }), text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

async function tick(n = 1) { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); }

function findRemoveButton(win) {
  const row = [...win.document.querySelectorAll("#models-table tbody tr")]
    .find((r) => r.querySelector(".name")?.textContent === "other-model");
  return [...row.querySelectorAll("button")].find((b) => b.textContent === "remove");
}

test("remove: a lost connection (streamJob 'disconnected') is NOT reported as 'Remove failed'", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  runScript(win, `confirmDanger = (t, m, l, onConfirm) => onConfirm();`);
  runScript(win, `streamJob = async () => ({ status: "disconnected" });`);
  await win.refreshModelsPage();
  await tick();

  findRemoveButton(win).onclick();
  await tick(3);

  const toastEl = win.document.getElementById("toast");
  assert.doesNotMatch(toastEl.textContent, /Remove failed/i,
    "a lost connection must not be reported as the remove having failed");
  assert.match(toastEl.textContent, /connection/i,
    "the toast names the real fact - a lost connection");
});

test("remove: a genuine failure is still reported as 'Remove failed'", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  runScript(win, `confirmDanger = (t, m, l, onConfirm) => onConfirm();`);
  runScript(win, `streamJob = async () => ({ status: "failed", returncode: 1 });`);
  await win.refreshModelsPage();
  await tick();

  findRemoveButton(win).onclick();
  await tick(3);

  const toastEl = win.document.getElementById("toast");
  assert.match(toastEl.textContent, /Remove failed/i, "a real failure is still reported as failed");
});

test("remove: a successful remove still toasts the unchanged success message", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  runScript(win, `confirmDanger = (t, m, l, onConfirm) => onConfirm();`);
  runScript(win, `streamJob = async () => ({ status: "done" });`);
  await win.refreshModelsPage();
  await tick();

  findRemoveButton(win).onclick();
  await tick(3);

  const toastEl = win.document.getElementById("toast");
  assert.match(toastEl.textContent, /Removed 'other-model'/,
    "the disconnected-status branch must not have broken the existing success toast");
});
