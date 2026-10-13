// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings, Models and Plugins waits say what they are doing: the update
// check, a model Remove, the first Models load and the tail of the Settings
// page each show a status while the server answers.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const ok = (body) => ({ ok: true, status: 200, json: async () => body, text: async () => "" });
const tick = async (n = 5) => { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); };

/** A deferred promise: `wait` resolves when `release()` is called. */
function gate() {
  let release;
  const wait = new Promise((r) => { release = r; });
  return { wait, release };
}

test("Check for updates shows a checking line and disables the button until the answer", async () => {
  const g = gate();
  const fetchImpl = async (url) => {
    if (String(url) === "/api/update/check") {
      await g.wait;
      return ok({ available: true, newer: false, current: "0.2.0", latest: "0.2.0" });
    }
    return ok({ models: [], active: "", conversations: [], plugins: [] });
  };
  const { window } = loadAppWithPages({ fetchImpl });
  const doc = window.document;
  const btn = doc.getElementById("update-check");
  const out = doc.getElementById("update-status");
  btn.click();
  await tick();
  assert.equal(out.hidden, false);
  assert.equal(out.textContent, "Checking for updates…");
  assert.equal(btn.disabled, true);
  g.release();
  await tick();
  assert.match(out.textContent, /up to date/);
  assert.equal(btn.disabled, false);
});

test("a failed update check re-enables the button and replaces the checking line", async () => {
  const fetchImpl = async (url) => {
    if (String(url) === "/api/update/check") {
      return { ok: false, status: 502, statusText: "Bad Gateway",
               json: async () => ({ detail: "proxy down" }), text: async () => "" };
    }
    return ok({ models: [], active: "", conversations: [], plugins: [] });
  };
  const { window } = loadAppWithPages({ fetchImpl });
  const doc = window.document;
  doc.getElementById("update-check").click();
  await tick();
  assert.match(doc.getElementById("update-status").textContent, /Could not check: proxy down/);
  assert.equal(doc.getElementById("update-check").disabled, false);
});

const MODELS = {
  models: [{ name: "other-model", model_type: "llm", active: false, size_bytes: 1e9 }],
  active: "current-model",
};

test("the Models page shows a loading line on its first open and then the table", async () => {
  const g = gate();
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u === "/api/models") { await g.wait; return ok(MODELS); }
    if (u.startsWith("/api/models?")) return ok(MODELS);
    return ok({});
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  const box = win.document.getElementById("models-table");
  box.replaceChildren();
  const pending = win.refreshModelsPage();
  await tick();
  assert.match(box.textContent, /Loading models/);
  g.release();
  await pending;
  await tick();
  assert.doesNotMatch(box.textContent, /Loading models/);
  assert.match(box.textContent, /other-model/);
});

test("a refresh of an already-filled Models table keeps the table instead of a loading line", async () => {
  const g = gate();
  let hold = false;
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u === "/api/models") { if (hold) await g.wait; return ok(MODELS); }
    if (u.startsWith("/api/models?")) return ok(MODELS);
    return ok({});
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await win.refreshModelsPage();
  await tick();
  const box = win.document.getElementById("models-table");
  assert.match(box.textContent, /other-model/);
  hold = true;
  const pending = win.refreshModelsPage();
  await tick();
  assert.match(box.textContent, /other-model/);
  assert.doesNotMatch(box.textContent, /Loading models/);
  g.release();
  await pending;
});

test("Remove disables its button and shows Removing... until the job ends", async () => {
  const g = gate();
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u === "/api/models" || u.startsWith("/api/models?")) return ok(MODELS);
    if (u === "/api/models/remove" && opts.method === "POST") return ok({ job_id: "rm1" });
    return ok({});
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  runScript(win, "confirmDanger = (t, m, l, onConfirm) => onConfirm();");
  runScript(win, "streamJob = async () => { await window.__rmGate; return { status: 'done' }; };");
  win.__rmGate = g.wait;
  await win.refreshModelsPage();
  await tick();
  const row = [...win.document.querySelectorAll("#models-table tbody tr")]
    .find((r) => r.querySelector(".name")?.textContent === "other-model");
  const rm = [...row.querySelectorAll("button")].find((b) => b.textContent === "remove");
  rm.onclick();
  await tick();
  assert.equal(rm.disabled, true);
  assert.equal(rm.textContent, "Removing…");
  g.release();
  await tick(5);
  assert.match(win.document.getElementById("toast").textContent, /Removed 'other-model'/);
});

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

test("the Settings page shows a loading line while its late sections build, then removes it", async () => {
  const g = gate();
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u === "/v1/config/schema") return ok(SCHEMA);
    if (u === "/v1/config" && (opts.method || "GET") === "GET") return ok({ n_ctx: 4096 });
    if (u === "/api/models?type=llm") { await g.wait; return ok({ models: [], active: "" }); }
    return ok({ models: [], active: "", conversations: [], plugins: [] });
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await tick();
  runScript(win, "refreshSettingsPage();");
  await tick(15);
  const doc = win.document;
  assert.match(doc.getElementById("settings-content").textContent, /Loading more settings/);
  g.release();
  await tick(30);
  assert.doesNotMatch(doc.getElementById("settings-content").textContent, /Loading more settings/);
});
