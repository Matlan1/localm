// SPDX-License-Identifier: AGPL-3.0-or-later
// Per-type counts on the Registered-models tabs. The tabs stay exactly what they
// were - a single-select strip, one type at a time - and only gain a number.
//
// The load-bearing test here is the LAST one: the page narrows to the active tab
// in the browser instead of through ?type=, so every NAMED-TYPE tab must still
// show precisely the rows the route would have returned, including the route's
// own "llm" default for an entry with no recorded type.
//
// UPDATED: that equivalence deliberately no longer covers the two multi-type
// views. It was written to pin "moving the filter from the route into the
// browser changed nothing", which it did not - it was never a decision that
// Other must mean the single type "unknown" forever. Other now means every type
// with no tab of its own (so mmproj has a home instead of living inside All
// alone), and All leaves those out until the merge toggle asks for them. The
// per-type tabs are untouched and still pinned against the route below.

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
  { name: "legacy-untyped", size_bytes: 10 },              // no model_type key at all
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

  // 6 registered, of which the mmproj has no tab of its own, so All lists 5 and
  // Other holds the 6th. The two partition the registry exactly.
  assert.equal(countOn(window, "all"), "5", "All counts the rows All actually lists");
  assert.equal(countOn(window, "other"), "1", "and the one with no tab of its own is on Other");
  assert.equal(countOn(window, "llm"), "3",
    "two typed LLMs plus the entry with no recorded type, matching the route's own default");
  assert.equal(countOn(window, "embedding"), "1");
  assert.equal(countOn(window, "vae"), "1");
});

test("tab-counts: a type with nothing registered shows no number rather than a 0", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(REGISTRY) });
  await window.refreshModelsPage();
  await tick();
  assert.equal(countOn(window, "lora"), "", "an empty type stays quiet");
  assert.equal(countOn(window, "diffusion-unet"), "");
  // :empty hides it in CSS, so an empty count must be genuinely empty, not " " or "0".
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
  assert.equal(countOn(window, "llm"), "3", "and it still reads correctly");
});

test("tab-counts: counts survive switching to a narrowed tab", async () => {
  // The point of the whole change: the counts describe the registry, not the
  // current view, so they must not collapse to the active tab's own rows.
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(REGISTRY) });
  await window.refreshModelsPage();
  await tick();
  window.document.querySelector('.tab-btn[data-type="vae"]').click();
  await tick();

  assert.equal(countOn(window, "llm"), "3", "the LLM count is still right while viewing VAEs");
  assert.equal(countOn(window, "all"), "5", "and All still reads what All would list");
  assert.deepEqual(rowNames(window), ["ae-vae"], "while the table shows only the VAE");
});

test("tab-counts: every named-type tab still shows EXACTLY the rows the ?type= route returned", async () => {
  // Filtering moved into the browser, so this pins the equivalence for every tab
  // that names a type. The expected sets are what
  // localm/plugins/gui/routes/models.py produces for each ?type=: an exact match
  // on model_type, with a missing model_type defaulting to llm.
  //
  // "all" and "other" are the two multi-type views and are deliberately NOT the
  // route's answer any more - they have their own test file
  // (models-other-tab.test.mjs). Listing them here with the route's sets would
  // re-pin behaviour that was changed on purpose.
  const expected = {
    llm: ["legacy-untyped", "mistral", "qwen-7b"],
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
