// SPDX-License-Identifier: AGPL-3.0-or-later
// After a declined or failed sidebar model switch, the model dropdown shows
// the model the server reports as serving (active, else resumable), or the
// "No model loaded" placeholder, never the model that was picked.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const MODELS = [{ name: "q05", size_bytes: 400 }, { name: "q7", size_bytes: 7000 },
                { name: "gptoss", size_bytes: 11300 }];

// *after* is the /api/models state once the switch was answered; *load* is
// the /api/models/load answer without force.
function makeFetch({ before, after, load }, calls) {
  let state = before;
  return async (url, opts = {}) => {
    const u = String(url);
    const body = opts.body ? JSON.parse(opts.body) : {};
    if (u.startsWith("/api/models/load")) {
      calls.push(body);
      state = after;
      if (load.ok === false) {
        return { ok: false, status: 503, statusText: "Service Unavailable",
                 json: async () => ({ detail: "load failed" }), text: async () => "" };
      }
      return { ok: true, status: 200, json: async () => load, text: async () => "" };
    }
    if (u.startsWith("/api/models?type=llm")) {
      const out = { models: MODELS.map((m) => ({ ...m, active: m.name === state.active })),
                    active: state.active };
      if (state.resumable) out.resumable = state.resumable;
      return { ok: true, status: 200, json: async () => out, text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

async function settle() {
  for (let i = 0; i < 12; i++) await new Promise((r) => setTimeout(r, 0));
}

const CONFIRM = { status: "confirm_required", model: "gptoss",
                  detail: "'gptoss' does not fit the estimated free VRAM" };

async function pickAndDecline(win, select) {
  runScript(win, "confirmDangerAsync = async () => false;");
  select.focus();
  select.value = "gptoss";
  runScript(win, "window.__onchange = modelSelect.onchange;");
  await win.__onchange();
  await settle();
}

test("declining keeps the dropdown on the model that is still loaded", async () => {
  const calls = [];
  const { window: win } = loadApp({ fetchImpl: makeFetch({
    before: { active: "q7" }, after: { active: "q7" }, load: CONFIRM }, calls) });
  runScript(win, "refreshModels();");
  await settle();
  const select = win.document.getElementById("model-select");
  assert.equal(select.value, "q7", "precondition: q7 is shown loaded");

  await pickAndDecline(win, select);

  assert.equal(calls.length, 1, "declining never retries with force");
  assert.equal(select.value, "q7", "the dropdown returns to the loaded model");
});

test("declining after the server freed the model shows the model that serves next", async () => {
  const { window: win } = loadApp({ fetchImpl: makeFetch({
    before: { active: "q7" }, after: { active: "", resumable: "q7" }, load: CONFIRM }, []) });
  runScript(win, "refreshModels();");
  await settle();
  const select = win.document.getElementById("model-select");

  await pickAndDecline(win, select);

  assert.equal(select.value, "q7");
});

test("declining with nothing loaded or resumable shows the placeholder", async () => {
  const { window: win } = loadApp({ fetchImpl: makeFetch({
    before: { active: "q7" }, after: { active: "" }, load: CONFIRM }, []) });
  runScript(win, "refreshModels();");
  await settle();
  const select = win.document.getElementById("model-select");

  await pickAndDecline(win, select);

  assert.equal(select.value, "", "the 'No model loaded' placeholder is shown");
  assert.equal(select.selectedOptions[0].textContent, "No model loaded");
});

test("a failed switch shows the model that is still loaded, not the pick", async () => {
  const { window: win } = loadApp({ fetchImpl: makeFetch({
    before: { active: "q7" }, after: { active: "q7" }, load: { ok: false } }, []) });
  runScript(win, "refreshModels();");
  await settle();
  const select = win.document.getElementById("model-select");
  select.focus();
  select.value = "gptoss";
  runScript(win, "window.__onchange = modelSelect.onchange;");

  await win.__onchange();
  await settle();

  assert.equal(select.value, "q7");
});
