// SPDX-License-Identifier: AGPL-3.0-or-later
// The optional model reload controls: the Models tab "reload" button on the
// running model's row, the Settings > Runtime & GPU "Save and reload model"
// button beside Save, and the Live tuning "Reload model" button. A plain save
// never reloads; every reload is unload then load of the same model.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, timeout = 1500) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await settle(10); }
  return false;
}

const ENGINE_FIELD = {
  key: "engine_probe", widget: "select", label: "Engine probe", help: "",
  group: "Engine", owner: "core", options: ["a", "b"], default: "a",
};

/** A fake server. `s.models` is the /api/models list, `s.active` the active
 *  model, `s.unloadStatus` the unload answer, `s.loadFails` makes load 500. */
function makeFetch(s, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    calls.push({ u, method, body: opts.body });
    const json = (body, status = 200) => ({
      ok: status < 400, status, statusText: "Server Error", json: async () => body,
      text: async () => "" });
    if (u === "/v1/config/schema") return json({ fields: [ENGINE_FIELD] });
    if (u === "/v1/media/config") return json({ plugins: [] });
    if (u === "/v1/comfy/status") return json({ alive: false, launched_by_localm: false });
    if (u === "/v1/tts/config") return json({ plugin: "tts", active: false, fields: [] });
    if (u === "/v1/plugins/settings") return json({ plugins: [] });
    if (u.includes("/api/vram-estimate")) return json({ needed: 1, weights: 1, kv_cache: 0, overhead: 0 });
    if (u.includes("/api/gpus")) return json({ gpus: [] });
    if (u.includes("/api/models/unload") && method === "POST")
      return json({ status: s.unloadStatus || "unloaded" });
    if (u.includes("/api/models/load") && method === "POST") {
      if (s.loadFails) return json({ detail: "boom" }, 500);
      return json({ status: "loaded", model: JSON.parse(opts.body).model });
    }
    if (u.endsWith("/v1/config") && method === "PATCH") return json({});
    if (u.endsWith("/v1/config")) return json({ n_ctx: 4096, n_gpu_layers: 99 });
    if (u.startsWith("/api/models")) return json({ models: s.models, active: s.active || "" });
    return json({ models: [], active: "", conversations: [], plugins: [] });
  };
}

const LLM = (name, extra = {}) => ({
  name, model_type: "llm", size_bytes: 100, active: false, loaded: false, ...extra,
});

async function boot(s) {
  const calls = [];
  const toasts = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(s, calls) });
  window.toast = (msg, isErr) => toasts.push({ msg: String(msg), isErr: !!isErr });
  await window.refreshModels();
  return { window, calls, toasts };
}

const rowButtons = (window, name) => {
  const row = [...window.document.querySelectorAll("#models-table tbody tr")]
    .find((tr) => tr.textContent.includes(name));
  return row ? [...row.querySelectorAll("button")].map((b) => b.textContent) : null;
};

async function showModels(window) {
  await window.refreshModelsPage();
  await settle(20);
}

const reloadButton = (window, name) => {
  const row = [...window.document.querySelectorAll("#models-table tbody tr")]
    .find((tr) => tr.textContent.includes(name));
  return [...row.querySelectorAll("button")].find((b) => b.textContent === "reload");
};

const sideEffects = (calls) => calls.filter((c) => c.method !== "GET").map((c) =>
  c.u.endsWith("/v1/config") ? "patch" : c.u.includes("/api/models/unload") ? "unload"
    : c.u.includes("/api/models/load") ? "load" : "other").filter((o) => o !== "other");

test("Models tab: the running model's row offers reload and not use, others keep use", async () => {
  const s = { active: "run", models: [LLM("run", { active: true, loaded: true }), LLM("idle")] };
  const { window } = await boot(s);
  await showModels(window);
  const running = rowButtons(window, "run");
  assert.ok(running.includes("reload"), `running row buttons: ${running}`);
  assert.ok(!running.includes("use"), "the running model has no use button");
  const idle = rowButtons(window, "idle");
  assert.ok(idle.includes("use") && !idle.includes("reload"), `idle row buttons: ${idle}`);
});

test("Models tab: reload unloads then loads the same model, then toasts the result", async () => {
  const s = { active: "run", models: [LLM("run", { active: true, loaded: true })] };
  const { window, calls, toasts } = await boot(s);
  await showModels(window);
  await reloadButton(window, "run").onclick();
  assert.deepEqual(sideEffects(calls), ["unload", "load"]);
  assert.equal(JSON.parse(calls.find((c) => c.u.includes("/api/models/unload")).body).model, "run");
  assert.equal(JSON.parse(calls.find((c) => c.u.includes("/api/models/load")
    && c.method === "POST").body).model, "run");
  assert.ok(toasts.some((x) => /Reloading run/.test(x.msg)), JSON.stringify(toasts));
  assert.ok(toasts.some((x) => /Model switched to run/.test(x.msg)), JSON.stringify(toasts));
});

test("Models tab: a busy model is not reloaded and the toast says so", async () => {
  const s = { active: "run", unloadStatus: "in_use",
              models: [LLM("run", { active: true, loaded: true })] };
  const { window, calls, toasts } = await boot(s);
  await showModels(window);
  await reloadButton(window, "run").onclick();
  assert.deepEqual(sideEffects(calls), ["unload"], "no load after a refused unload");
  assert.ok(toasts.some((x) => x.isErr && /run is busy/.test(x.msg)), JSON.stringify(toasts));
});

test("Models tab: a failed load is toasted as an error and the button comes back", async () => {
  const s = { active: "run", loadFails: true,
              models: [LLM("run", { active: true, loaded: true })] };
  const { window, toasts } = await boot(s);
  await showModels(window);
  const btn = reloadButton(window, "run");
  await btn.onclick();
  assert.ok(toasts.some((x) => x.isErr && /Reloading run failed: boom/.test(x.msg)),
    JSON.stringify(toasts));
  assert.equal(btn.disabled, false);
  assert.equal(btn.textContent, "reload");
});

test("Models tab: a loaded model whose adapter change is pending offers reload", async () => {
  const adapter = { name: "lora-a", file: "a.gguf", scale: 1 };
  const s = { active: "other", models: [
    LLM("other", { active: true, loaded: true }),
    LLM("base", { loaded: true, adapters: [adapter], applied_adapters: [] }),
    LLM("plain", { loaded: true }),
  ] };
  const { window } = await boot(s);
  await showModels(window);
  assert.ok(rowButtons(window, "base").includes("reload"), "pending adapter: reload offered");
  assert.ok(!rowButtons(window, "plain").includes("reload"),
    "a loaded, non-active, unchanged model gets no reload");
});

async function renderSettings(s) {
  const ctx = await boot(s);
  runScript(ctx.window, "refreshSettingsPage();");
  assert.ok(await waitFor(() => ctx.window.document.querySelector("#settings-sec-core-Engine")),
    "the Engine section renders");
  return ctx;
}

const saveReloadButton = (window) =>
  window.document.querySelector("#settings-sec-core-Engine .settings-section-save-reload");

test("Settings: plain Save in the Engine section never unloads or loads", async () => {
  const { window, calls } = await renderSettings({ active: "run", models: [LLM("run", { active: true, loaded: true })] });
  window.document.querySelector("#settings-sec-core-Engine .settings-section-save").click();
  assert.ok(await waitFor(() => calls.some((c) => c.method === "PATCH")), "saved");
  await settle(150);
  assert.deepEqual(sideEffects(calls), ["patch"]);
});

test("Settings: Save and reload model saves, then unloads and loads the running model", async () => {
  const { window, calls } = await renderSettings({ active: "run", models: [LLM("run", { active: true, loaded: true })] });
  const btn = saveReloadButton(window);
  assert.ok(btn && !btn.hidden, "offered while a model is loaded");
  btn.click();
  assert.ok(await waitFor(() => sideEffects(calls).length >= 3), "patch, unload, load ran");
  assert.deepEqual(sideEffects(calls), ["patch", "unload", "load"]);
  assert.equal(JSON.parse(calls.find((c) => c.u.includes("/api/models/unload")).body).model, "run");
});

test("Settings: Save and reload model with a failed save does not reload", async () => {
  const s = { active: "run", models: [LLM("run", { active: true, loaded: true })] };
  const { window, calls } = await renderSettings(s);
  const base = window.fetch;
  window.fetch = async (url, opts = {}) => {
    if (String(url).endsWith("/v1/config") && (opts.method || "GET") === "PATCH") {
      calls.push({ u: String(url), method: "PATCH", body: opts.body });
      return { ok: false, status: 400, json: async () => ({ detail: "bad value" }) };
    }
    return base(url, opts);
  };
  runScript(window, "fetch = window.fetch;");
  saveReloadButton(window).click();
  assert.ok(await waitFor(() => calls.some((c) => c.method === "PATCH")));
  await settle(150);
  assert.deepEqual(sideEffects(calls), ["patch"], "no unload or load after a refused save");
});

test("Settings: no reload is offered while no model is loaded", async () => {
  const { window } = await renderSettings({ active: "", models: [LLM("idle")] });
  assert.ok(saveReloadButton(window).hidden, "Save and reload model is hidden");
  assert.ok(window.document.getElementById("perf-reload").hidden, "Reload model is hidden");
});

test("Settings: the Live tuning Reload model button reloads without saving", async () => {
  const s = { active: "run", models: [LLM("run", { active: true, loaded: true })] };
  const { window, calls } = await boot(s);
  const btn = window.document.getElementById("perf-reload");
  assert.ok(await waitFor(() => !btn.hidden), "shown once a model is known to be loaded");
  btn.click();
  assert.ok(await waitFor(() => sideEffects(calls).length >= 2));
  assert.deepEqual(sideEffects(calls), ["unload", "load"], "no PATCH: it saves nothing");
});
