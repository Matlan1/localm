// SPDX-License-Identifier: AGPL-3.0-or-later
// refreshModels() inserts a disabled placeholder option in the sidebar MODEL
// dropdown and selects it whenever no model is active. The placeholder is
// inert: it never loads and never unloads. A separate sidebar Unload button
// targets the active model.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

// A stateful fetch stub: tracks which model is active server-side and lets
// /api/models/load and /api/models/unload move that state, so a test can drive
// a full refresh -> click -> refresh cycle.
function makeFetch({ models, active = "" }, calls) {
  let currentActive = active;
  return async (url, opts = {}) => {
    const u = String(url);
    const body = opts.body ? JSON.parse(opts.body) : {};
    if (u.startsWith("/api/models/unload")) {
      calls.push({ url: u, body });
      const target = body.model || currentActive;
      if (!body.model || target === currentActive) currentActive = "";
      return {
        ok: true, status: 200,
        json: async () => ({ status: "unloaded", unloaded_models: target ? [target] : [] }),
        text: async () => "",
      };
    }
    if (u.startsWith("/api/models/load")) {
      calls.push({ url: u, body });
      currentActive = body.model;
      return {
        ok: true, status: 200,
        json: async () => ({ status: "loaded", model: currentActive }),
        text: async () => "",
      };
    }
    if (u.startsWith("/api/models?type=llm")) {
      return {
        ok: true, status: 200,
        json: async () => ({
          models: models.map((m) => ({ ...m, active: m.name === currentActive })),
          active: currentActive,
        }),
        text: async () => "",
      };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

async function render(win) {
  runScript(win, "refreshModels();");
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
}

function placeholderOf(select) {
  return [...select.options].find((o) => o.textContent === "No model loaded");
}

test("(a) no active model: the placeholder is selected and its value is not any real model name", async () => {
  const models = [{ name: "model-a", size_bytes: 1000 }, { name: "model-b", size_bytes: 2000 }];
  const { window: win } = loadApp({ fetchImpl: makeFetch({ models, active: "" }, []) });
  await render(win);

  const select = win.document.getElementById("model-select");
  assert.equal(select.value, "", "the select's value is the placeholder sentinel, not a real model");
  assert.ok(!models.some((m) => m.name === select.value), "selected value must not match any real model");
  const placeholder = placeholderOf(select);
  assert.ok(placeholder, "the placeholder option exists");
  assert.equal(placeholder.selected, true, "the placeholder is the one actually shown as selected");
});

test("(b) an active model: that option is selected, not the placeholder", async () => {
  const models = [{ name: "model-a", size_bytes: 1000 }, { name: "model-b", size_bytes: 2000 }];
  const { window: win } = loadApp({ fetchImpl: makeFetch({ models, active: "model-b" }, []) });
  await render(win);

  const select = win.document.getElementById("model-select");
  assert.equal(select.value, "model-b");
  const placeholder = placeholderOf(select);
  assert.ok(placeholder, "the placeholder option still exists in the list");
  assert.equal(placeholder.selected, false, "the placeholder is not the selected one while a model is active");
});

test("(c) the placeholder is inert: disabled, and forcing its value neither loads nor unloads anything", async () => {
  const models = [{ name: "model-a", size_bytes: 1000 }];
  const calls = [];
  const { window: win } = loadApp({ fetchImpl: makeFetch({ models, active: "model-a" }, calls) });
  await render(win);

  const select = win.document.getElementById("model-select");
  const placeholder = placeholderOf(select);
  assert.equal(placeholder.disabled, true, "the placeholder cannot be picked as an action in a real browser");

  // Forcing the select back to the placeholder's value past the `disabled`
  // attribute: the change handler treats it as a no-op.
  select.value = "";
  runScript(win, "window.__onchange = modelSelect.onchange;");
  await win.__onchange();
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(calls.length, 0,
    "an empty/placeholder selection must never reach /api/models/load or /api/models/unload");
});

test("(d) the sidebar Unload button POSTs /api/models/unload exactly once, targets the active model, and the dropdown reverts to the placeholder", async () => {
  const models = [{ name: "model-a", size_bytes: 1000 }];
  const calls = [];
  const { window: win } = loadApp({ fetchImpl: makeFetch({ models, active: "model-a" }, calls) });
  await render(win);

  const select = win.document.getElementById("model-select");
  const btn = win.document.getElementById("sidebar-unload-btn");
  assert.equal(select.value, "model-a", "precondition: model-a is shown active");
  assert.equal(btn.hidden, false, "the unload button is visible while a model is active");

  btn.click();
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));

  const unloadCalls = calls.filter((c) => c.url.startsWith("/api/models/unload"));
  assert.equal(unloadCalls.length, 1, "clicking Unload posts exactly once");
  assert.deepEqual(unloadCalls[0].body, { model: "model-a" },
    "the sidebar button targets only the active model, not an unload-everything call");

  assert.equal(select.value, "", "after unload the dropdown reverts to the placeholder");
  assert.equal(placeholderOf(select).selected, true);
  assert.equal(btn.hidden, true, "the unload button hides once there is nothing left to unload");
});

test("the sidebar Unload button is hidden when nothing is active", async () => {
  const models = [{ name: "model-a", size_bytes: 1000 }];
  const { window: win } = loadApp({ fetchImpl: makeFetch({ models, active: "" }, []) });
  await render(win);

  const btn = win.document.getElementById("sidebar-unload-btn");
  assert.equal(btn.hidden, true);
});

test("an in-use engine (HTTP 200, status 'in_use') is NOT reported as unloaded", async () => {
  // unload_one_model() (http_server.py) answers HTTP 200 with status "in_use"
  // for a model that is mid-generation, so r.ok alone does not mean unloaded.
  const models = [{ name: "model-a", size_bytes: 1000 }];
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models/unload")) {
      calls.push({ url: u, body: opts.body ? JSON.parse(opts.body) : {} });
      return {
        ok: true, status: 200,
        json: async () => ({ status: "in_use", model: "model-a", vram_freed: 0 }),
        text: async () => "",
      };
    }
    if (u.startsWith("/api/models?type=llm")) {
      return {
        ok: true, status: 200,
        json: async () => ({ models: models.map((m) => ({ ...m, active: true })), active: "model-a" }),
        text: async () => "",
      };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window: win } = loadApp({ fetchImpl });
  await render(win);

  const select = win.document.getElementById("model-select");
  const btn = win.document.getElementById("sidebar-unload-btn");
  assert.equal(select.value, "model-a", "precondition: model-a is active");

  btn.click();
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));

  assert.equal(calls.length, 1, "exactly one unload POST was attempted");
  // The toast text is what the click handler itself reported, before the
  // reconciling refresh.
  const toastText = win.document.getElementById("toast").textContent;
  assert.match(toastText, /still generating/,
    "the toast must say the model is still in use, not claim it was unloaded");
  assert.doesNotMatch(toastText, /^Unloaded/,
    "must never claim success for an unload that did not happen");
  assert.equal(select.value, "model-a",
    "the dropdown must NOT revert to the placeholder - the model is still loaded");
  assert.equal(btn.hidden, false, "the unload button stays visible - there is still something to unload");
  assert.notEqual(win.document.getElementById("status-text").textContent, "unloading model-a…",
    "the status line must not stay stuck on the busy message");
});
