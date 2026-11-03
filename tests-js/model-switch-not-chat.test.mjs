// SPDX-License-Identifier: AGPL-3.0-or-later
// Picking an embedding model, reranker or text encoder in the sidebar asks
// "Not a chat model" first; "Load anyway" retries with as_chat (not force),
// declining never retries.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const MODELS = [{ name: "q7", size_bytes: 7000 }, { name: "bge", size_bytes: 300 }];
const PROMPT = { status: "confirm_required", reason: "not_chat_model", model: "bge",
                 detail: "\"bge\" is an embedding model, which turns text into numbers." };

function makeFetch(answers, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models/load")) {
      calls.push(JSON.parse(opts.body));
      const answer = answers.shift();
      return { ok: true, status: 200, json: async () => answer, text: async () => "" };
    }
    if (u.startsWith("/api/models?type=llm")) {
      const out = { models: MODELS.map((m) => ({ ...m, active: m.name === "q7" })), active: "q7" };
      return { ok: true, status: 200, json: async () => out, text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

async function settle() {
  for (let i = 0; i < 12; i++) await new Promise((r) => setTimeout(r, 0));
}

async function pick(win, answer) {
  runScript(win, `window.__asked = null; confirmDangerAsync = async (title, text, label) => {
    window.__asked = { title, text, label }; return ${answer}; };`);
  runScript(win, "refreshModels();");
  await settle();
  const select = win.document.getElementById("model-select");
  select.focus();
  select.value = "bge";
  runScript(win, "window.__onchange = modelSelect.onchange;");
  await win.__onchange();
  await settle();
  return select;
}

test("Load anyway retries with as_chat, not force, under the Not a chat model title", async () => {
  const calls = [];
  const { window: win } = loadApp({ fetchImpl: makeFetch(
    [PROMPT, { status: "loaded", model: "bge" }], calls) });
  await pick(win, "true");

  assert.deepEqual(calls, [{ model: "bge" }, { model: "bge", as_chat: true }]);
  assert.equal(win.__asked.title, "Not a chat model");
  assert.equal(win.__asked.label, "Load anyway");
  assert.match(win.__asked.text, /embedding model/);
});

test("declining the Not a chat model box never retries", async () => {
  const calls = [];
  const { window: win } = loadApp({ fetchImpl: makeFetch([PROMPT], calls) });
  const select = await pick(win, "false");

  assert.deepEqual(calls, [{ model: "bge" }]);
  assert.equal(select.value, "q7", "the dropdown returns to the loaded model");
});

const BUSY = { status: "confirm_required", model: "bge",
               detail: "loading 'bge' needs to free 'q7', which is writing a reply" };

async function pickTwice(win, answers) {
  runScript(win, `window.__asks = []; window.__answers = ${JSON.stringify(answers)};
    confirmDangerAsync = async (title, text, label) => {
      window.__asks.push({ title, text, label }); return window.__answers.shift(); };
    window.__switched = [];
    window.addEventListener("localm:model-switched", (e) => window.__switched.push(e.detail.model));`);
  runScript(win, "refreshModels();");
  await settle();
  const select = win.document.getElementById("model-select");
  select.focus();
  select.value = "bge";
  runScript(win, "window.__onchange = modelSelect.onchange;");
  await win.__onchange();
  await settle();
  return select;
}

test("a second confirm after Load anyway asks the switch question and retries with force and as_chat", async () => {
  const calls = [];
  const { window: win } = loadApp({ fetchImpl: makeFetch(
    [PROMPT, BUSY, { status: "loaded", model: "bge" }], calls) });
  await pickTwice(win, [true, true]);

  assert.deepEqual(calls, [
    { model: "bge" },
    { model: "bge", as_chat: true },
    { model: "bge", force: true, as_chat: true },
  ]);
  assert.deepEqual(win.__asks.map((a) => a.title), ["Not a chat model", "Confirm model switch"]);
  assert.equal(win.__asks[1].label, "Switch anyway");
  assert.match(win.__asks[1].text, /needs to free 'q7'/);
  assert.deepEqual(win.__switched, ["bge"]);
});

test("declining the second confirm after Load anyway claims no load", async () => {
  const calls = [];
  const { window: win } = loadApp({ fetchImpl: makeFetch([PROMPT, BUSY], calls) });
  const select = await pickTwice(win, [true, false]);

  assert.deepEqual(calls, [{ model: "bge" }, { model: "bge", as_chat: true }]);
  assert.deepEqual(win.__switched, [], "no model-switched event for a load that never happened");
  runScript(win, "window.__active = modelCache.active;");
  assert.equal(win.__active, "q7");
  assert.equal(select.value, "q7", "the dropdown returns to the loaded model");
});

test("a confirm repeated after its flag was sent is a failed load, not a loaded model", async () => {
  const calls = [];
  const { window: win } = loadApp({ fetchImpl: makeFetch([PROMPT, BUSY, BUSY], calls) });
  await pickTwice(win, [true, true]);

  assert.equal(calls.length, 3);
  assert.equal(win.__asks.length, 2, "the same question is not asked twice");
  assert.deepEqual(win.__switched, []);
  runScript(win, "window.__active = modelCache.active;");
  assert.equal(win.__active, "q7");
});
