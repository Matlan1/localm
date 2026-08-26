// SPDX-License-Identifier: AGPL-3.0-or-later
// Per-type counts on the Registered-models tabs, a single-select strip showing
// one type at a time. The page narrows to the active tab in the browser rather
// than through ?type=. "Other" holds every type with no tab of its own; "All"
// leaves those out until the merge toggle asks for them.

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

const tick = () => new Promise((r) => setTimeout(r, 0));

const REGISTRY = [
  { name: "qwen-7b", model_type: "llm", size_bytes: 10 },
  { name: "mistral", model_type: "llm", size_bytes: 10 },
  // What the route emits for a registry entry with no model_type: the "llm"
  // default it keeps for chat-picker candidacy, plus the flag marking that
  // type as a guess.
  { name: "legacy-untyped", model_type: "llm", model_type_recorded: false, size_bytes: 10 },
  { name: "bge-small", model_type: "embedding", size_bytes: 10 },
  { name: "ae-vae", model_type: "vae", size_bytes: 10 },
  { name: "llava-mmproj", model_type: "mmproj", size_bytes: 10 },   // no tab of its own
];

function countOn(window, type) {
  return window.document
    .querySelector(`#models-tab-nav .tab-btn[data-type="${type}"] .tab-count`).textContent;
}
function rowNames(window) {
  return [...window.document.querySelectorAll("#models-table tbody tr .name")]
    .map((n) => n.textContent).sort();
}

test("tab-counts: each tab shows how many models it holds, and All shows what All lists", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(REGISTRY) });
  await window.refreshModelsPage();
  await tick();

  // 6 registered. The mmproj has no tab of its own and the legacy entry has no
  // recorded type at all, so Other holds 2 and All lists the remaining 4.
  assert.equal(countOn(window, "all"), "4", "All counts the rows All actually lists");
  assert.equal(countOn(window, "other"), "2",
    "the projector and the never-classified entry both live here");
  assert.equal(countOn(window, "llm"), "2",
    "ONLY the two models actually recorded as llm - the route's default is kept for "
    + "chat-picker candidacy, but it is not a classification this tab may claim");
  assert.equal(countOn(window, "embedding"), "1");
  assert.equal(countOn(window, "vae"), "1");
});

test("tab-counts: a type with nothing registered shows no number rather than a 0", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(REGISTRY) });
  await window.refreshModelsPage();
  await tick();
  assert.equal(countOn(window, "lora"), "", "an empty type stays quiet");
  assert.equal(countOn(window, "diffusion-unet"), "");
  // CSS :empty hides the node, so an empty count is "" rather than " " or "0".
  const slot = window.document.querySelector('.tab-btn[data-type="lora"] .tab-count');
  assert.equal(slot.textContent.length, 0, "the node is empty so the CSS :empty rule hides it");
});

test("tab-counts: the number rides beside the label without disturbing it", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(REGISTRY) });
  await window.refreshModelsPage();
  await tick();
  const btn = window.document.querySelector('.tab-btn[data-type="llm"]');
  assert.equal(btn.firstChild.textContent.trim(), "LLMs", "the tab's own label is untouched");
  assert.equal(btn.querySelectorAll(".tab-count").length, 1,
    "exactly one count node, so repeated refreshes do not stack them up");
});

test("tab-counts: a repeated refresh updates the number instead of appending another", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(REGISTRY) });
  await window.refreshModelsPage();
  await tick();
  await window.refreshModelsPage();
  await tick();
  await window.refreshModelsPage();
  await tick();
  const btn = window.document.querySelector('.tab-btn[data-type="llm"]');
  assert.equal(btn.querySelectorAll(".tab-count").length, 1, "still one count node after 3 refreshes");
  assert.equal(countOn(window, "llm"), "2", "and it still reads correctly");
});

test("tab-counts: counts survive switching to a narrowed tab", async () => {
  // The counts describe the registry, not the current view.
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(REGISTRY) });
  await window.refreshModelsPage();
  await tick();
  window.document.querySelector('.tab-btn[data-type="vae"]').click();
  await tick();

  assert.equal(countOn(window, "llm"), "2", "the LLM count is still right while viewing VAEs");
  assert.equal(countOn(window, "all"), "4", "and All still reads what All would list");
  assert.deepEqual(rowNames(window), ["ae-vae"], "while the table shows only the VAE");
});

test("tab-counts: every named-type tab shows exactly the rows recorded as that type", async () => {
  // A named-type tab holds an exact match on a RECORDED model_type, so the
  // llm tab excludes the entry whose type is only the route's default
  // (model_type_recorded: false). The "all" and "other" multi-type views are
  // covered in models-other-tab.test.mjs.
  const expected = {
    llm: ["mistral", "qwen-7b"],
    embedding: ["bge-small"],
    vae: ["ae-vae"],
    lora: [],
  };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(REGISTRY) });
  await window.refreshModelsPage();
  await tick();

  for (const [type, want] of Object.entries(expected)) {
    window.document.querySelector(`.tab-btn[data-type="${type}"]`).click();
    await tick();
    assert.deepEqual(rowNames(window), want, `the "${type}" tab shows exactly its route rows`);
  }
});
