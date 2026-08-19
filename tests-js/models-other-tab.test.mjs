// SPDX-License-Identifier: AGPL-3.0-or-later
// The Registered-models "Other" tab, the merge-into-All toggle, and group-by-type.
//
// WHY THESE EXIST. "Other" used to carry data-type="unknown", so it meant one
// particular MODEL_TYPES member rather than the thing it is named for. A model
// whose type had no tab at all - mmproj today, whatever the registry grows
// tomorrow - was reachable from the All tab and nowhere else, which is a home
// only in the sense that a haystack is a home for a needle. Other now collects
// every type the strip does not name, All leaves those out until asked, and the
// leaving-out is said out loud rather than being silent (AGENTS.md rule 5).
//
// The fixture deliberately carries a type NOBODY has a tab for and that is not
// in MODEL_TYPES either ("tts-voice"): the whole point of deriving Other from the
// tab strip is that a type added upstream lands there without a second edit, and
// a fixture that only ever held today's known types could not fail on that.

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
  { name: "qwen-7b", model_type: "llm", size_bytes: 30 },
  { name: "aardvark", model_type: "llm", size_bytes: 10 },
  { name: "legacy-untyped", size_bytes: 20 },                      // no model_type key
  { name: "bge-small", model_type: "embedding", size_bytes: 10 },
  { name: "llava-mmproj", model_type: "mmproj", size_bytes: 10 },  // no tab of its own
  { name: "scratchpad", model_type: "unknown", size_bytes: 10 },   // no tab of its own
  { name: "kokoro", model_type: "tts-voice", size_bytes: 10 },     // not even in MODEL_TYPES
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
function noteText(window) {
  const n = window.document.querySelector("#models-table .models-other-note");
  return n ? n.textContent : null;
}
function groupHeads(window) {
  return [...window.document.querySelectorAll("#models-table tbody tr.group-head")]
    .map((tr) => ({
      label: tr.querySelector(".group-head-label").textContent,
      count: tr.querySelector(".group-head-count").textContent,
    }));
}
// Row order as rendered, headings included, so a test can see that a model sits
// UNDER the heading it belongs to rather than merely that both exist.
function renderedOrder(window) {
  return [...window.document.querySelectorAll("#models-table tbody tr")].map((tr) =>
    tr.classList.contains("group-head")
      ? "## " + tr.querySelector(".group-head-label").textContent
      : tr.querySelector(".name").textContent);
}

async function load(opts = {}) {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(REGISTRY), ...opts });
  await window.refreshModelsPage();
  await tick();
  return window;
}

/* ----------------------------- the Other tab ----------------------------- */

test("other-tab: collects EVERY type with no tab of its own, not just 'unknown'", async () => {
  const window = await load();
  clickTab(window, "other");
  await tick();
  assert.deepEqual(rowNames(window), ["kokoro", "llava-mmproj", "scratchpad"],
    "mmproj and an entirely unknown-to-the-strip type land here beside 'unknown'");
});

test("other-tab: a type the strip does not name needs no second edit to land here", async () => {
  // The forward-looking half. "tts-voice" is in no tab, is no MODEL_TYPES member
  // and appears in no list inside models.js - it is on Other purely because Other
  // is derived from the strip. A hardcoded {unknown, mmproj} would pass every
  // other test in this file and fail this one.
  const window = await load();
  clickTab(window, "other");
  await tick();
  assert.ok(rowNames(window).includes("kokoro"),
    "a registry type nothing in the GUI names is still reachable");
});

test("other-tab: a type WITH a tab is never swept into Other", async () => {
  const window = await load();
  clickTab(window, "other");
  await tick();
  const names = rowNames(window);
  for (const n of ["qwen-7b", "aardvark", "legacy-untyped", "bge-small"]) {
    assert.ok(!names.includes(n), `${n} has a tab of its own and stays on it`);
  }
});

/* --------------------------- the merge-in toggle -------------------------- */

test("merge-toggle: All leaves the no-tab types out by default", async () => {
  const window = await load();
  assert.deepEqual(rowNames(window),
    ["aardvark", "bge-small", "legacy-untyped", "qwen-7b"],
    "the four models whose type has a tab, and nothing else");
  assert.equal(countOn(window, "all"), "4", "and All's own count agrees with its list");
  assert.equal(countOn(window, "other"), "3");
});

test("merge-toggle: ticking it merges them back into All, on demand", async () => {
  const window = await load();
  window.localStorage.setItem("localm.showOtherModelsInAll", "true");
  await window.refreshModelsPage();
  await tick();
  assert.deepEqual(rowNames(window),
    ["aardvark", "bge-small", "kokoro", "legacy-untyped", "llava-mmproj", "qwen-7b", "scratchpad"],
    "every registered model");
  assert.equal(countOn(window, "all"), "7", "All now counts the whole registry");
  assert.equal(noteText(window), null, "and nothing is being left out, so nothing is said");
});

test("merge-toggle: the checkbox writes the flag and re-renders", async () => {
  const window = await load();
  const box = window.document.getElementById("models-show-other");
  assert.equal(box.checked, false, "off unless the browser already stored otherwise");
  assert.ok(noteText(window), "before: All is leaving rows out and saying so");

  box.checked = true;
  box.dispatchEvent(new window.Event("change"));
  await tick();

  assert.equal(window.localStorage.getItem("localm.showOtherModelsInAll"), "true");
  assert.ok(rowNames(window).includes("llava-mmproj"), "and the list refreshed itself");
  // The note going away is what only a real re-render THROUGH the new filter can
  // produce. "the row is present" alone is satisfied by never filtering at all,
  // so on its own it could not fail if the merge-out were reverted.
  assert.equal(noteText(window), null, "after: nothing is left out, so the note is gone");
});

test("merge-toggle: a value stored by a previous visit is reflected on load", async () => {
  const window = await load({ seedLocalStorage: { "localm.showOtherModelsInAll": "true" } });
  assert.equal(window.document.getElementById("models-show-other").checked, true);
  assert.ok(rowNames(window).includes("kokoro"), "and it applies to the first render");
});

/* ---------------------- rule 5: never hide it silently -------------------- */

test("hidden-note: All says how many it is leaving out and where they are", async () => {
  const window = await load();
  const note = noteText(window);
  assert.ok(note, "a note is shown at all");
  assert.match(note, /^3 models of a type with no tab of its own are not listed here/);
  assert.match(note, /Other tab/, "and it names where they went");
});

test("hidden-note: one hidden model reads as singular", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([
      { name: "qwen-7b", model_type: "llm", size_bytes: 10 },
      { name: "llava-mmproj", model_type: "mmproj", size_bytes: 10 },
    ]),
  });
  await window.refreshModelsPage();
  await tick();
  assert.match(noteText(window), /^1 model of a type with no tab of its own is not listed here/);
});

test("hidden-note: no note when All is leaving nothing out", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([{ name: "qwen-7b", model_type: "llm", size_bytes: 10 }]),
  });
  await window.refreshModelsPage();
  await tick();
  assert.equal(noteText(window), null);
});

test("hidden-note: it replaces the empty state when EVERY model is a hidden one", async () => {
  // The dangerous case. "No models yet - pull your first one" would be a false
  // statement about a registry that holds two models, and that empty state is
  // exactly what an unguarded filter produces here.
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([
      { name: "llava-mmproj", model_type: "mmproj", size_bytes: 10 },
      { name: "kokoro", model_type: "tts-voice", size_bytes: 10 },
    ]),
  });
  await window.refreshModelsPage();
  await tick();
  const box = window.document.getElementById("models-table");
  assert.match(noteText(window), /^2 models of a type with no tab of its own/);
  assert.ok(!/No models yet/.test(box.textContent),
    "the registry is not empty, so it must not say it is");
});

/* ----------------------------- group by type ----------------------------- */

test("group-by-type: off by default, so the list stays flat", async () => {
  const window = await load();
  assert.equal(groupHeads(window).length, 0);
});

test("group-by-type: one heading per present type, in the tab strip's own order", async () => {
  const window = await load({
    seedLocalStorage: {
      "localm.modelsGroupByType": "true",
      "localm.showOtherModelsInAll": "true",
    },
  });
  assert.deepEqual(groupHeads(window), [
    { label: "LLMs", count: "3" },
    { label: "Embedding", count: "1" },
    // No tab, so no label to borrow: the raw registry value, which is what the
    // row's own Role pill reads too.
    { label: "mmproj", count: "1" },
    { label: "unknown", count: "1" },
    // Last, and after "unknown", because it is a type neither the strip nor
    // MODEL_TYPE_OPTIONS names: known kinds keep their canonical order and
    // anything the registry grew afterwards sorts in behind them.
    { label: "tts-voice", count: "1" },
  ], "tabbed types first in strip order, then the remaining known ones, then the rest");
});

test("group-by-type: rows sit under their own heading, in the active sort order", async () => {
  const window = await load({ seedLocalStorage: { "localm.modelsGroupByType": "true" } });
  assert.deepEqual(renderedOrder(window), [
    "## LLMs", "aardvark", "legacy-untyped", "qwen-7b",
    "## Embedding", "bge-small",
  ], "name-ascending inside each section, not registry order");
});

test("group-by-type: a column sort re-orders WITHIN each group, never across them", async () => {
  const window = await load({
    seedLocalStorage: {
      "localm.modelsGroupByType": "true",
      "localm.modelsSortKey": "size_bytes",
      "localm.modelsSortDir": "desc",
    },
  });
  assert.deepEqual(renderedOrder(window), [
    "## LLMs", "qwen-7b", "legacy-untyped", "aardvark",
    "## Embedding", "bge-small",
  ], "30, 20, 10 inside the LLM section");
});

test("group-by-type: still exactly ONE table, so the overlap guard keeps working", async () => {
  // One table per type would have re-measured column widths per section AND
  // broken models-refresh-race's tableCount === 1, which is how a duplicated
  // render is caught. Headings are rows, not tables.
  const window = await load({ seedLocalStorage: { "localm.modelsGroupByType": "true" } });
  assert.equal(
    window.document.querySelectorAll("#models-table table.data-table").length, 1);
});

test("group-by-type: applies to Other too, the other view that holds many types", async () => {
  const window = await load({ seedLocalStorage: { "localm.modelsGroupByType": "true" } });
  clickTab(window, "other");
  await tick();
  assert.deepEqual(groupHeads(window).map((g) => g.label),
    ["mmproj", "unknown", "tts-voice"], "same ordering rule as All");
});

test("group-by-type: does nothing on a single-type tab", async () => {
  const window = await load({ seedLocalStorage: { "localm.modelsGroupByType": "true" } });
  clickTab(window, "llm");
  await tick();
  assert.equal(groupHeads(window).length, 0,
    "one section holding every row is the flat list with extra furniture");
  assert.deepEqual(rowNames(window), ["aardvark", "legacy-untyped", "qwen-7b"]);
});

test("group-by-type: the checkbox writes the flag and re-renders", async () => {
  const window = await load();
  const box = window.document.getElementById("models-group-by-type");
  box.checked = true;
  box.dispatchEvent(new window.Event("change"));
  await tick();
  assert.equal(window.localStorage.getItem("localm.modelsGroupByType"), "true");
  assert.ok(groupHeads(window).length > 0, "and the list refreshed itself");
});

/* -------------------- each control shows where it applies ----------------- */

test("view-opts: a control is hidden on a tab it cannot affect", async () => {
  const window = await load();
  const showOther = window.document.getElementById("models-show-other-wrap");
  const group = window.document.getElementById("models-group-wrap");
  assert.equal(showOther.hidden, false, "All: both apply");
  assert.equal(group.hidden, false);

  clickTab(window, "other");
  await tick();
  assert.equal(showOther.hidden, true, "Other: merging into All is not a thing here");
  assert.equal(group.hidden, false, "but Other holds several types, so grouping is");

  clickTab(window, "llm");
  await tick();
  assert.equal(showOther.hidden, true, "a single-type tab: neither applies");
  assert.equal(group.hidden, true);
});
