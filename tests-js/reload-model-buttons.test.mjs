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
    if (u.includes("/api/models/unload") && method === "POST") {
      if (s.hangModelsAfterUnload) s.hangModels = true;
      if (s.stateful && s.active === JSON.parse(opts.body).model) {
        s.resumable = s.active;
        s.active = "";
      }
      s.unloaded = true;
      return json({ status: s.unloadStatus || "unloaded" });
    }
    if (u.includes("/api/models/load") && method === "POST") {
      if (s.loadFails) return json({ detail: "boom" }, 500);
      const req = JSON.parse(opts.body);
      if (s.stateful && req.activate !== false) { s.active = req.model; s.resumable = ""; }
      return json({ status: "loaded", model: req.model });
    }
    if (u.endsWith("/v1/config") && method === "PATCH") return json({});
    if (u.endsWith("/v1/config")) return json({ n_ctx: 4096, n_gpu_layers: 99 });
    if (u.startsWith("/api/models") && s.hangModels) return new Promise(() => {});
    if (u.startsWith("/api/models")) {
      const models = s.stateful
        ? s.models.map((m) => ({ ...m, active: m.name === s.active })) : s.models;
      return json({ models, active: s.active || "", resumable: s.resumable || "",
                    peer_routes: s.peerRoutes || {} });
    }
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
  assert.ok(toasts.some((x) => /Reloaded run/.test(x.msg)), JSON.stringify(toasts));
  assert.ok(!toasts.some((x) => /switched to/.test(x.msg)), JSON.stringify(toasts));
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

const activeModel = (window) => {
  runScript(window, "window.__active = modelCache.active; window.__resumable = modelCache.resumable;");
  return { active: window.__active, resumable: window.__resumable };
};
const loadBody = (calls) => JSON.parse(calls.find((c) => c.u.includes("/api/models/load")
  && c.method === "POST").body);
const PENDING = () => ({ active: "other", models: [
  LLM("other", { active: true, loaded: true }),
  LLM("base", { loaded: true, adapters: [{ name: "lora-a", file: "a.gguf", scale: 1 }],
                applied_adapters: [] }),
] });

test("Models tab: reloading the active model asks for it to stay active", async () => {
  const s = { active: "run", models: [LLM("run", { active: true, loaded: true })] };
  const { window, calls } = await boot(s);
  await showModels(window);
  await reloadButton(window, "run").onclick();
  assert.equal("activate" in loadBody(calls), false, JSON.stringify(loadBody(calls)));
  assert.equal(activeModel(window).active, "run");
});

test("Models tab: reloading a loaded model that is not active keeps the active model", async () => {
  const { window, calls, toasts } = await boot(PENDING());
  await showModels(window);
  await reloadButton(window, "base").onclick();
  assert.deepEqual(sideEffects(calls), ["unload", "load"]);
  assert.equal(loadBody(calls).model, "base");
  assert.equal(loadBody(calls).activate, false);
  assert.equal(activeModel(window).active, "other");
  assert.equal(window.document.getElementById("model-select").value, "other");
  assert.ok(toasts.some((x) => /Reloaded base/.test(x.msg)), JSON.stringify(toasts));
  assert.ok(!toasts.some((x) => /switched to base/.test(x.msg)), JSON.stringify(toasts));
});

test("Models tab: a failed reload of a model that is not active keeps the active model", async () => {
  const s = { ...PENDING(), loadFails: true };
  const { window, toasts } = await boot(s);
  await showModels(window);
  s.hangModelsAfterUnload = true;
  await reloadButton(window, "base").onclick();
  assert.ok(toasts.some((x) => x.isErr && /Reloading base failed: boom/.test(x.msg)),
    JSON.stringify(toasts));
  const after = activeModel(window);
  assert.equal(after.active, "other");
  assert.notEqual(after.resumable, "base");
});

test("Models tab: reloading the server's active model keeps this tab's peer-route selection", async () => {
  const s = { active: "run", peerRoutes: { peer: { host: "10.0.0.2", port: 8642 } },
              models: [LLM("run", { active: true, loaded: true })] };
  const { window, calls } = await boot(s);
  runScript(window, "_peerSelected = 'peer';");
  await window.refreshModels();
  assert.equal(activeModel(window).active, "peer", "precondition: the peer route is selected");
  await showModels(window);
  await reloadButton(window, "run").onclick();
  assert.equal("activate" in loadBody(calls), false, JSON.stringify(loadBody(calls)));
  runScript(window, "window.__peer = _peerSelected;");
  assert.equal(window.__peer, "peer");
  assert.equal(activeModel(window).active, "peer");
});

test("Models tab: a failed reload of the active model leaves it to reload on the next message", async () => {
  const s = { active: "run", loadFails: true, models: [LLM("run", { active: true, loaded: true })] };
  const { window } = await boot(s);
  await showModels(window);
  const before = window.fetch;
  window.fetch = async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models?") && s.unloaded) {
      return { ok: true, status: 200, json: async () => ({ models: s.models, active: "", resumable: "run" }),
               text: async () => "" };
    }
    return before(url, opts);
  };
  runScript(window, "fetch = window.fetch;");
  await reloadButton(window, "run").onclick();
  const after = activeModel(window);
  assert.equal(after.active, "");
  assert.equal(after.resumable, "run");
});

// A sidebar switch updates modelCache.active only; the rows stay as last polled.
async function switchedAtoB() {
  const s = { stateful: true, active: "a", models: [LLM("a", { loaded: true }), LLM("b", { loaded: true })] };
  const ctx = await boot(s);
  runScript(ctx.window, 'window.__p = switchModel("b");');
  await ctx.window.__p;
  assert.equal(s.active, "b", "precondition: the server made b active");
  runScript(ctx.window, 'window.__row = modelCache.models.find((m) => m.name === "a").active;');
  assert.equal(ctx.window.__row, true, "precondition: the rows are stale, no poll ran");
  ctx.calls.length = 0;
  return { ...ctx, s };
}

test("reload right after a sidebar switch keeps the switched-to model active", async () => {
  const { window, calls, s } = await switchedAtoB();
  runScript(window, 'window.__p = reloadModel("b");');
  await window.__p;
  assert.equal("activate" in loadBody(calls), false, JSON.stringify(loadBody(calls)));
  assert.equal(s.active, "b");
  assert.equal(activeModel(window).active, "b");
});

test("reload of the old model's stale row after a switch does not make it active again", async () => {
  const { window, calls, s } = await switchedAtoB();
  runScript(window, 'window.__p = reloadModel("a");');
  await window.__p;
  assert.equal(loadBody(calls).model, "a");
  assert.equal(loadBody(calls).activate, false);
  assert.equal(s.active, "b");
  assert.equal(activeModel(window).active, "b");
});

test("with a peer route selected no reload is offered and a save does not load a local copy", async () => {
  const s = { active: "run", peerRoutes: { peer: { host: "10.0.0.2", port: 8642 } },
              models: [LLM("run", { active: true, loaded: true }), LLM("peer")] };
  const { window, calls, toasts } = await boot(s);
  runScript(window, "_peerSelected = 'peer';");
  await window.refreshModels();
  await window.refreshPerfEstimate();
  assert.equal(window.document.getElementById("perf-reload").hidden, true);
  runScript(window, "window.__p = reloadActiveModelAfterSave();");
  await window.__p;
  assert.deepEqual(sideEffects(calls), [], "nothing unloaded or loaded");
  assert.ok(toasts.some((x) => /applies on the next model load/.test(x.msg)), JSON.stringify(toasts));
  runScript(window, "window.__peer = _peerSelected;");
  assert.equal(window.__peer, "peer", "the tab keeps its selection");
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
