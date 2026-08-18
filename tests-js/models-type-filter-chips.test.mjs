// SPDX-License-Identifier: AGPL-3.0-or-later
// The Registered-models TYPE filter, after it stopped being a single-select tab
// strip and became the same multi-select .filter-chip component the HuggingFace
// search row uses. What these cover:
//
//   - multi-select actually narrows the rendered table (the tab strip could
//     only ever express one type at a time);
//   - a registry row whose type has NO chip of its own ("mmproj", and whatever
//     MODEL_TYPES gains next) still renders under Other, so client-side
//     filtering cannot make a registered model disappear;
//   - a row with NO model_type key at all still renders under LLMs, matching the
//     server's own default (routes/models.py) and the Role badge;
//   - the three empty states stay distinct: no models, no types ticked, and a
//     selection matching no row (AGENTS.md rule 5);
//   - per-chip counts and the dimmed zero-count chip;
//   - reset;
//   - the selection does NOT persist across a reload, unlike the search row's;
//   - the two rows share a widget but NOT a state, in both directions.
//
// Same page/harness/fetch-mock style as models-discover-type-scoped.test.mjs.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(models, calls = []) {
  return async (url, opts = {}) => {
    const u = String(url);
    calls.push(u);
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return {
        ok: true, status: 200,
        json: async () => ({ models, active: null }),
      };
    }
    if (u.startsWith("/api/discover/search")) {
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ query: "", vram: {}, hf_backend_available: true, results: [] }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

function rowNames(window) {
  return [...window.document.querySelectorAll("#models-table tbody tr .name")]
    .map((n) => n.textContent);
}

// Tick exactly `values`, firing a real change event for each box (the `.on`
// class and the re-render both hang off that event, not off `.checked`).
async function selectTypes(window, values) {
  for (const b of window.document.querySelectorAll(".reg-type")) {
    b.checked = values.includes(b.value);
    b.dispatchEvent(new window.Event("change"));
  }
  await tick();
}

function chip(window, type) {
  return window.document.querySelector(`.reg-chip[data-type="${type}"]`);
}

const MIXED = [
  { name: "qwen-7b", model_type: "llm", size_bytes: 10 },
  { name: "bge-small", model_type: "embedding", size_bytes: 10 },
  { name: "flux-unet", model_type: "diffusion-unet", size_bytes: 10 },
  { name: "clip-l", model_type: "text-encoder", size_bytes: 10 },
  { name: "ae-vae", model_type: "vae", size_bytes: 10 },
  { name: "detail-lora", model_type: "lora", size_bytes: 10 },
];

// --------------------------------------------------------------------------- //
//  The widget: shared component, multi-select                                 //
// --------------------------------------------------------------------------- //

test("reg-type-chips: the row is the same .filter-chip component as the search row, one chip per type", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  const row = window.document.getElementById("models-type-filter");
  assert.ok(row, "the Registered-models type filter row exists");
  assert.equal(window.document.getElementById("models-tab-nav"), null,
    "the old single-select tab strip is gone, not left alongside");

  const vals = [...window.document.querySelectorAll(".reg-type")].map((b) => b.value);
  assert.deepEqual(vals.sort(),
    ["diffusion-unet", "embedding", "llm", "lora", "text-encoder", "unknown", "vae"],
    "one chip per model type the table can show");
  for (const b of window.document.querySelectorAll(".reg-type")) {
    assert.ok(b.checked, `type ${b.value} defaults to ticked (show everything)`);
    assert.ok(b.closest(".filter-chip"),
      `${b.value} uses the shared .filter-chip component, same as the search row`);
  }
  // The search row must use that same component, or "unified" is not true.
  for (const b of window.document.querySelectorAll(".disc-type")) {
    assert.ok(b.closest(".filter-chip"), `search chip ${b.value} is the same component`);
  }
});

test("reg-type-chips: every chip's .on class stays in sync with its checkbox", async () => {
  // Same contract as the search chips: style.css colours `.filter-chip.on span`,
  // not a CSS :checked selector, so the class IS the visual state. _syncChip
  // matches .filter-chip; had it kept matching only .disc-chip these chips would
  // render grey while ticked.
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  for (const c of window.document.querySelectorAll(".reg-chip")) {
    assert.ok(c.classList.contains("on"),
      `${c.dataset.type} starts coloured, matching its ticked box`);
  }
  await selectTypes(window, ["vae"]);
  assert.ok(chip(window, "vae").classList.contains("on"), "the ticked chip is coloured");
  assert.ok(!chip(window, "llm").classList.contains("on"), "an unticked chip is not");
});

test("reg-type-chips: several types can be ticked at once - the tab strip could not do this", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(MIXED) });
  await window.refreshModelsPage();
  await tick();
  assert.equal(rowNames(window).length, 6, "all six render by default");

  // The grouping that motivated multi-select: every ComfyUI asset type at once,
  // without the LLM. Unreachable from a single-select strip.
  await selectTypes(window, ["diffusion-unet", "vae", "text-encoder", "lora"]);
  assert.deepEqual(rowNames(window).sort(), ["ae-vae", "clip-l", "detail-lora", "flux-unet"],
    "four types shown together, and the LLM and embedding rows excluded");
});

// --------------------------------------------------------------------------- //
//  No registered model may fall through the filter                            //
// --------------------------------------------------------------------------- //

test("reg-type-chips: a type with no chip of its own (mmproj) still shows, bucketed under Other", async () => {
  // registry.py MODEL_TYPES carries 'mmproj', which never had a tab and has no
  // chip. Client-side filtering that matched types literally would drop it from
  // the DEFAULT all-ticked view - a registered model silently gone.
  const models = [
    { name: "qwen-7b", model_type: "llm", size_bytes: 10 },
    { name: "llava-mmproj", model_type: "mmproj", size_bytes: 10 },
  ];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models) });
  await window.refreshModelsPage();
  await tick();
  assert.ok(rowNames(window).includes("llava-mmproj"),
    "an mmproj model is visible with every chip ticked");

  await selectTypes(window, ["unknown"]);
  assert.deepEqual(rowNames(window), ["llava-mmproj"],
    "and it is reachable via Other, which the old tab strip could not do");
});

test("reg-type-chips: a row with no model_type key at all still shows under LLMs", async () => {
  // The server defaults a model_type-less registry entry to 'llm' and so does
  // the Role badge. A fixture whose rows all carry an explicit model_type cannot
  // catch a filter that forgets that default, so this one omits the key.
  const models = [
    { name: "legacy-untyped", size_bytes: 10 },
    { name: "bge-small", model_type: "embedding", size_bytes: 10 },
  ];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models) });
  await window.refreshModelsPage();
  await tick();
  assert.ok(rowNames(window).includes("legacy-untyped"),
    "an untyped legacy entry is visible by default");

  await selectTypes(window, ["llm"]);
  assert.deepEqual(rowNames(window), ["legacy-untyped"],
    "it filters as an LLM, matching the server's own default");

  await selectTypes(window, ["embedding"]);
  assert.deepEqual(rowNames(window), ["bge-small"],
    "and it is correctly excluded from a type it is not");
});

// --------------------------------------------------------------------------- //
//  Three empty states, kept distinct                                          //
// --------------------------------------------------------------------------- //

test("reg-type-chips: an empty registry still says 'No models yet', not a filter message", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await window.refreshModelsPage();
  await tick();
  const text = window.document.getElementById("models-table").textContent;
  assert.match(text, /No models yet/, "an empty registry keeps its own empty state");
});

test("reg-type-chips: zero types ticked says so, instead of an empty table or a silent 'show all'", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(MIXED) });
  await window.refreshModelsPage();
  await tick();
  await selectTypes(window, []);

  const box = window.document.getElementById("models-table");
  assert.equal(box.querySelectorAll("tbody tr").length, 0, "no rows render");
  assert.match(box.textContent, /No types selected/i,
    "the reason is stated - an empty table would read as 'you own no models'");
  assert.doesNotMatch(box.textContent, /No models yet/,
    "and it is NOT the empty-registry message, which would be a lie");
});

test("reg-type-chips: a selection matching no row says how many are hidden, not 'No models yet'", async () => {
  const models = [{ name: "qwen-7b", model_type: "llm", size_bytes: 10 }];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models) });
  await window.refreshModelsPage();
  await tick();
  await selectTypes(window, ["vae"]);

  const box = window.document.getElementById("models-table");
  assert.equal(box.querySelectorAll("tbody tr").length, 0, "no rows match");
  assert.doesNotMatch(box.textContent, /No models yet/,
    "a filtered-out registry must never claim the registry is empty");
  assert.match(box.textContent, /hidden by this filter/i,
    "the message says the rows are hidden, not absent");
  assert.match(box.textContent, /1 registered model/,
    "and says how many, so the user knows the filter is the cause");
});

// --------------------------------------------------------------------------- //
//  Counts, dimming, reset                                                     //
// --------------------------------------------------------------------------- //

test("reg-type-chips: each chip carries its own count, and a zero-count type is dimmed", async () => {
  const models = [
    { name: "qwen-7b", model_type: "llm", size_bytes: 10 },
    { name: "mistral", model_type: "llm", size_bytes: 10 },
    { name: "bge-small", model_type: "embedding", size_bytes: 10 },
  ];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models) });
  await window.refreshModelsPage();
  await tick();

  const count = (t) => chip(window, t).querySelector(".chip-count").textContent;
  assert.equal(count("llm"), "2", "two LLMs registered");
  assert.equal(count("embedding"), "1", "one embedding model");
  assert.equal(count("vae"), "", "a type with none registered shows no number");
  assert.ok(chip(window, "vae").classList.contains("chip-empty"),
    "and is dimmed rather than removed, so the filter's vocabulary stays visible");
  assert.ok(!chip(window, "llm").classList.contains("chip-empty"), "a populated chip is not dimmed");

  // The count describes the WHOLE registry, so it keeps telling you what
  // ticking a chip would reveal even while that chip is unticked.
  await selectTypes(window, ["embedding"]);
  assert.equal(count("llm"), "2", "the LLM count survives the LLM chip being unticked");
});

test("reg-type-chips: reset appears only when the selection is narrowed, and restores every type", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(MIXED) });
  await window.refreshModelsPage();
  await tick();

  const reset = window.document.getElementById("models-type-reset");
  assert.ok(reset, "the reset affordance exists");
  assert.ok(reset.hidden, "hidden while everything is ticked - nothing to reset");

  await selectTypes(window, ["vae"]);
  assert.ok(!reset.hidden, "shown once the view is narrowed");

  reset.click();
  await tick();
  for (const b of window.document.querySelectorAll(".reg-type")) {
    assert.ok(b.checked, `${b.value} is ticked again after reset`);
  }
  for (const c of window.document.querySelectorAll(".reg-chip")) {
    assert.ok(c.classList.contains("on"), `${c.dataset.type} is recoloured after reset`);
  }
  assert.ok(reset.hidden, "and it hides itself again");
  assert.equal(rowNames(window).length, 6, "every row is back");
});

// --------------------------------------------------------------------------- //
//  Persistence, and independence from the search row                          //
// --------------------------------------------------------------------------- //

test("reg-type-chips: the selection does NOT persist across a reload", async () => {
  // Deliberate asymmetry with the search chips, which do persist. This filter is
  // a view over what you own: a narrowed state restored on a later visit makes
  // the app look like it lost your models.
  const first = loadAppWithPages({ fetchImpl: makeFetch(MIXED) });
  await first.window.refreshModelsPage();
  await tick();
  await selectTypes(first.window, ["vae"]);
  assert.deepEqual(rowNames(first.window), ["ae-vae"], "the first page is narrowed to VAEs");

  const second = loadAppWithPages({ fetchImpl: makeFetch(MIXED) });
  await second.window.refreshModelsPage();
  await tick();
  for (const b of second.window.document.querySelectorAll(".reg-type")) {
    assert.ok(b.checked, `${b.value} is ticked again on a fresh load`);
  }
  assert.equal(rowNames(second.window).length, 6, "a fresh load shows the whole library");
  const keys = Object.keys(second.window.localStorage).filter((k) => /reg.?type/i.test(k));
  assert.deepEqual(keys, [], "nothing about this filter was written to localStorage");
});

test("reg-type-chips: the two rows share a widget but not a state, in both directions", async () => {
  // The tested contract this whole change had to preserve: an earlier design
  // scoped the HuggingFace search by the active table tab, invisibly. Sharing a
  // chip component must not re-couple them.
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(MIXED, calls) });
  await window.refreshModelsPage();
  await tick();

  // Narrowing the table leaves the search scope alone.
  await selectTypes(window, ["vae"]);
  assert.deepEqual([...window.document.querySelectorAll(".disc-type")]
      .filter((b) => b.checked).map((b) => b.value).sort(),
    ["diffusion-unet", "embedding", "llm", "lora", "text-encoder", "unknown", "vae"],
    "every search type is still ticked after the table was narrowed to VAEs");
  assert.equal(calls.filter((u) => u.startsWith("/api/discover/search")).length, 0,
    "narrowing the table issued no search");

  // Narrowing the search leaves the table alone.
  for (const b of window.document.querySelectorAll(".disc-type")) {
    b.checked = (b.value === "lora");
    b.dispatchEvent(new window.Event("change"));
  }
  await tick();
  assert.deepEqual(rowNames(window), ["ae-vae"],
    "the table still shows exactly what ITS own chips select, not the search's");
});
