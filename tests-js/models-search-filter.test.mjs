// SPDX-License-Identifier: AGPL-3.0-or-later
// The Registered-models search box: filters the table as the user types, combines
// with the type tabs and their counts, shows a no-match row, Escape and Clear empty it.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(models) {
  return async (url) => {
    const u = String(url);
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models, active: null }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

const wait = (ms = 0) => new Promise((r) => setTimeout(r, ms));

const REGISTRY = [
  { name: "qwen-7b", model_type: "llm", size_bytes: 30, source: "hf:Qwen/Qwen-7B-GGUF" },
  { name: "qwen-embed", model_type: "embedding", size_bytes: 10 },
  { name: "aardvark", model_type: "llm", size_bytes: 10, architecture: "llama" },
  { name: "bge-small", model_type: "embedding", size_bytes: 10 },
  { name: "sdxl-unet", model_type: "diffusion-unet", size_bytes: 10 },
];

function rowNames(window) {
  return [...window.document.querySelectorAll("#models-table tbody tr .name")]
    .map((n) => n.textContent).sort();
}
function countOn(window, type) {
  return window.document
    .querySelector(`#models-tab-nav .tab-btn[data-type="${type}"] .tab-count`).textContent;
}
function clickTab(window, type) {
  window.document.querySelector(`#models-tab-nav .tab-btn[data-type="${type}"]`).click();
}
async function type(window, text) {
  const input = window.document.getElementById("models-search");
  input.value = text;
  input.dispatchEvent(new window.Event("input"));
  await wait(250);
}

async function load() {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(REGISTRY) });
  await window.refreshModelsPage();
  await wait();
  return window;
}

test("search: filters rows by name, case-insensitively", async () => {
  const window = await load();
  assert.equal(rowNames(window).length, 5);
  await type(window, "QWEN");
  assert.deepEqual(rowNames(window), ["qwen-7b", "qwen-embed"]);
});

test("search: also matches source, type and architecture", async () => {
  const window = await load();
  await type(window, "Qwen/Qwen-7B");
  assert.deepEqual(rowNames(window), ["qwen-7b"], "source");
  await type(window, "llama");
  assert.deepEqual(rowNames(window), ["aardvark"], "architecture");
  await type(window, "diffusion");
  assert.deepEqual(rowNames(window), ["sdxl-unet"], "type");
});

test("search: combines with the active type tab", async () => {
  const window = await load();
  clickTab(window, "embedding");
  await wait();
  await type(window, "qwen");
  assert.deepEqual(rowNames(window), ["qwen-embed"], "qwen-7b is an LLM and stays out");
  clickTab(window, "llm");
  await wait();
  assert.deepEqual(rowNames(window), ["qwen-7b"], "the query survives a tab switch");
});

test("search: tab counts follow the matches", async () => {
  const window = await load();
  assert.equal(countOn(window, "all"), "5");
  await type(window, "qwen");
  assert.equal(countOn(window, "all"), "2");
  assert.equal(countOn(window, "llm"), "1");
  assert.equal(countOn(window, "embedding"), "1");
  assert.equal(countOn(window, "diffusion-unet"), "", "a tab with no match shows no number");
});

test("search: no match shows a status row naming the query, not the empty state", async () => {
  const window = await load();
  await type(window, "zzz");
  const box = window.document.getElementById("models-table");
  const row = box.querySelector(".models-no-match");
  assert.ok(row, "a no-match row is shown");
  assert.equal(row.getAttribute("role"), "status");
  assert.match(row.textContent, /No models match "zzz"/);
  assert.ok(!/No models yet/.test(box.textContent), "the registry is not empty");
  assert.equal(box.querySelectorAll("table").length, 0);
});

test("search: Escape clears the box and restores every row", async () => {
  const window = await load();
  await type(window, "qwen");
  const input = window.document.getElementById("models-search");
  input.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  await wait(50);
  assert.equal(input.value, "");
  assert.equal(rowNames(window).length, 5);
  assert.equal(countOn(window, "all"), "5");
});

test("search: the Clear button appears with text, empties the box and hides itself", async () => {
  const window = await load();
  const clear = window.document.getElementById("models-search-clear");
  assert.equal(clear.hidden, true, "nothing to clear yet");
  await type(window, "qwen");
  assert.equal(clear.hidden, false);
  clear.click();
  await wait(50);
  assert.equal(window.document.getElementById("models-search").value, "");
  assert.equal(clear.hidden, true);
  assert.equal(rowNames(window).length, 5);
});

test("search: the box has an associated label", async () => {
  const window = await load();
  const input = window.document.getElementById("models-search");
  const label = window.document.querySelector('label[for="models-search"]');
  assert.ok(label && label.textContent.trim(), "a real <label for>");
  assert.equal(input.type, "search");
});

test("search: a blank (whitespace) query filters nothing", async () => {
  const window = await load();
  await type(window, "   ");
  assert.equal(rowNames(window).length, 5);
});
