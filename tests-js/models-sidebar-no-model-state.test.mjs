// SPDX-License-Identifier: AGPL-3.0-or-later
// NEW-MODEL-DROPDOWN-SHOWS-A-MODEL-THAT-IS-NOT-LOADED: the sidebar MODEL
// dropdown used to represent "nothing is active" by marking no <option>
// selected at all, so the browser fell back to displaying option INDEX 0 (an
// arbitrary registered model) as if it were loaded - while the status line
// directly beneath correctly said "no model". refreshModels() now inserts an
// explicit, disabled placeholder option and selects it whenever nothing is
// active, so the dropdown can represent that state honestly. A dedicated
// sidebar Unload button (targeting the active model specifically) is the
// actual action for "free this model" - the placeholder itself is inert in
// both directions (never loads, never unloads).
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

// A stateful fetch stub: tracks which model is "active" server-side and lets
// /api/models/load and /api/models/unload actually move that state, so a
// test can drive a full refresh -> click -> refresh cycle and observe the
// dropdown/button reflect the real result, not a canned response.
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

  // Defense in depth: even if something bypasses the `disabled` attribute and
  // forces the select back to the placeholder's value, the change handler
  // itself must treat that as a no-op rather than trying to load "".
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
