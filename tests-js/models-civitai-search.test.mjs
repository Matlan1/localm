// SPDX-License-Identifier: AGPL-3.0-or-later
// Models page CivitAI search: the source selector, CivitAI-only type/NSFW/
// legacy-format filters, CivitAI result rendering (license flags shown as
// CivitAI's own permission system, never mapped onto HF's license string),
// and the file list's pull-spec prefill (civitai:<version>:<file>).

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(searchPayload, filesPayload, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.includes("/api/discover/search")) {
      calls.push(u);
      return { ok: true, status: 200, json: async () => searchPayload, text: async () => "" };
    }
    if (u.includes("/api/discover/files")) {
      calls.push(u);
      return { ok: true, status: 200, json: async () => filesPayload, text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

const ONE_CIVITAI_RESULT = {
  query: "", source: "civitai",
  results: [
    { name: "Detail Tweaker", type: "LORA", allowCommercialUse: ["Image"],
      allowDerivatives: true, allowNoCredit: false, allowDifferentLicense: false,
      nsfw: false, modelVersions: [{ id: 135867 }], stats: { downloadCount: 4200 } },
  ],
};

const ONE_NSFW_RESULT = {
  query: "", source: "civitai",
  results: [
    { name: "Mature Style", type: "Checkpoint", nsfw: true, nsfwLevel: 4,
      modelVersions: [{ id: 999 }], stats: { downloadCount: 10 } },
  ],
};

function rowFor(win, name) {
  for (const row of win.document.querySelectorAll("#disc-results .disc-repo")) {
    if (row.querySelector(".name")?.textContent === name) return row;
  }
  return null;
}

function selectCivitai(win) {
  const sel = win.document.getElementById("disc-source");
  sel.value = "civitai";
  sel.dispatchEvent(new win.Event("change"));
}

test("selecting CivitAI hides the HF filter bar and shows the CivitAI one", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ results: [] }, {}, []) });
  selectCivitai(window);
  assert.equal(window.document.getElementById("disc-hf-filters").style.display, "none");
  assert.notEqual(window.document.getElementById("disc-civitai-filters").style.display, "none");
});

test("selecting CivitAI updates the search placeholder", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ results: [] }, {}, []) });
  selectCivitai(window);
  assert.match(window.document.getElementById("disc-query").placeholder, /CivitAI/);
});

test("switching back to HuggingFace restores the HF filter bar", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ results: [] }, {}, []) });
  selectCivitai(window);
  const sel = window.document.getElementById("disc-source");
  sel.value = "hf";
  sel.dispatchEvent(new window.Event("change"));
  assert.notEqual(window.document.getElementById("disc-hf-filters").style.display, "none");
  assert.equal(window.document.getElementById("disc-civitai-filters").style.display, "none");
});

test("civitai search sends source=civitai, the selected types, and nsfw=false by default", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(ONE_CIVITAI_RESULT, {}, calls) });
  selectCivitai(window);
  await window.discoverSearch();

  assert.equal(calls.length, 1);
  const u = decodeURIComponent(calls[0]);
  assert.ok(u.includes("source=civitai"), `source is sent (got ${u})`);
  assert.ok(u.includes("nsfw=false"), `nsfw defaults false (got ${u})`);
  assert.ok(u.includes("Checkpoint") && u.includes("Upscaler"),
    `all default-checked CivitAI types are sent (got ${u})`);
});

test("checking the NSFW toggle re-runs the search with nsfw=true once results are showing", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(ONE_CIVITAI_RESULT, {}, calls) });
  selectCivitai(window);
  await window.discoverSearch();
  assert.equal(calls.length, 1, "the initial search fired");

  const nsfw = window.document.getElementById("disc-civitai-nsfw");
  nsfw.checked = true;
  nsfw.dispatchEvent(new window.Event("change"));
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(calls.length, 2, "toggling NSFW with results already visible re-runs the search");
  assert.ok(decodeURIComponent(calls[1]).includes("nsfw=true"));
});

test("checking the NSFW toggle before any search has run does not fire one", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(ONE_CIVITAI_RESULT, {}, calls) });
  selectCivitai(window);
  const nsfw = window.document.getElementById("disc-civitai-nsfw");
  nsfw.checked = true;
  nsfw.dispatchEvent(new window.Event("change"));

  assert.equal(calls.length, 0, "no results are showing yet, so nothing re-runs");
});

test("unchecking every CivitAI type prompts and issues no search", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(ONE_CIVITAI_RESULT, {}, calls) });
  selectCivitai(window);
  for (const box of window.document.querySelectorAll(".disc-civitai-type")) box.checked = false;
  await window.discoverSearch();

  assert.equal(calls.length, 0, "no search with zero types selected");
  assert.match(window.document.getElementById("disc-results").textContent, /at least one/i);
});

test("a CivitAI result shows its own license flags, never HF's license field", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(ONE_CIVITAI_RESULT, {}, []) });
  selectCivitai(window);
  await window.discoverSearch();

  const row = rowFor(window, "Detail Tweaker");
  assert.ok(row, "the result row renders");
  // allowCommercialUse is CivitAI's own array of specific permitted uses, not
  // a boolean - shown as the actual values, never collapsed to yes/no.
  assert.match(row.textContent, /commercial use: Image/i);
});

test("an EMPTY allowCommercialUse array is shown as 'no', not 'unknown'", async () => {
  const payload = { query: "", source: "civitai",
    results: [{ ...ONE_CIVITAI_RESULT.results[0], allowCommercialUse: [] }] };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(payload, {}, []) });
  selectCivitai(window);
  await window.discoverSearch();
  const row = rowFor(window, "Detail Tweaker");
  assert.match(row.textContent, /commercial use: no/i,
    "an empty array is a real 'no commercial use', not the same as a missing field");
});

test("a MISSING allowCommercialUse field is shown as 'unknown', not 'no'", async () => {
  const item = { ...ONE_CIVITAI_RESULT.results[0] };
  delete item.allowCommercialUse;
  const payload = { query: "", source: "civitai", results: [item] };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(payload, {}, []) });
  selectCivitai(window);
  await window.discoverSearch();
  const row = rowFor(window, "Detail Tweaker");
  assert.match(row.textContent, /commercial use: unknown/i,
    "an absent field is genuinely unknown, distinct from an empty (no) array");
  assert.match(row.textContent, /derivatives: yes/i);
  assert.match(row.textContent, /different license: no/i);
});

test("NSFW badge shows only when the result is actually flagged nsfw", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(ONE_NSFW_RESULT, {}, []) });
  selectCivitai(window);
  await window.discoverSearch();
  const row = rowFor(window, "Mature Style");
  assert.match(row.textContent, /NSFW/);
});

test("a non-nsfw result shows no NSFW badge", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(ONE_CIVITAI_RESULT, {}, []) });
  selectCivitai(window);
  await window.discoverSearch();
  const row = rowFor(window, "Detail Tweaker");
  assert.doesNotMatch(row.textContent, /NSFW/);
});

test("expanding files fetches source=civitai and shows the safety-scan status, never a fit badge", async () => {
  const calls = [];
  const filesPayload = {
    repo: "135867", source: "civitai",
    files: [{ id: 99264, name: "detailTweaker.safetensors", sizeKB: 123456,
              metadata: { format: "SafeTensor" }, virusScanResult: "Success" }],
  };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(ONE_CIVITAI_RESULT, filesPayload, calls) });
  selectCivitai(window);
  await window.discoverSearch();
  const row = rowFor(window, "Detail Tweaker");
  const btn = [...row.querySelectorAll("button")].find((b) => b.textContent === "files");
  assert.ok(btn, "a files button renders when a version id is known");
  btn.click();
  await new Promise((r) => setTimeout(r, 0));

  const filesCall = calls.find((u) => u.includes("/api/discover/files"));
  assert.ok(filesCall, "a files request fired");
  assert.ok(decodeURIComponent(filesCall).includes("source=civitai"));
  assert.match(row.textContent, /scan: Success/);
  assert.ok(!row.querySelector(".fit"), "no HF-shaped VRAM fit badge on a CivitAI file");
});

test("pulling a CivitAI file prefills civitai:<version>:<file> into the Add box", async () => {
  const filesPayload = {
    repo: "135867", source: "civitai",
    files: [{ id: 99264, name: "detailTweaker.safetensors", sizeKB: 123456,
              metadata: { format: "SafeTensor" }, virusScanResult: "Success" }],
  };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(ONE_CIVITAI_RESULT, filesPayload, []) });
  selectCivitai(window);
  await window.discoverSearch();
  const row = rowFor(window, "Detail Tweaker");
  const filesBtn = [...row.querySelectorAll("button")].find((b) => b.textContent === "files");
  filesBtn.click();
  await new Promise((r) => setTimeout(r, 0));
  const pullBtn = [...row.querySelectorAll("button")].find((b) => b.textContent === "pull");
  pullBtn.click();

  assert.equal(window.document.getElementById("pull-spec").value, "civitai:135867:99264");
});

test("legacy-format files show their own scan status, not a false-confident clean badge", async () => {
  const filesPayload = {
    repo: "1", source: "civitai",
    files: [{ id: 1, name: "old.ckpt", sizeKB: 100,
              metadata: { format: "PickleTensor" }, virusScanResult: "Pending" }],
  };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(ONE_CIVITAI_RESULT, filesPayload, []) });
  selectCivitai(window);
  await window.discoverSearch();
  const row = rowFor(window, "Detail Tweaker");
  const btn = [...row.querySelectorAll("button")].find((b) => b.textContent === "files");
  btn.click();
  await new Promise((r) => setTimeout(r, 0));

  const scanBadge = row.querySelector(".moe-badge");
  assert.ok(scanBadge, "the scan status renders");
  assert.match(scanBadge.textContent, /Pending/);
  assert.ok(scanBadge.classList.contains("moe-likely"),
    "a non-Success scan borrows the dashed 'inferred, not confirmed' treatment, never the solid clean one");
});

test("legacy-formats toggle is sent as legacy_formats=true on the files request", async () => {
  const calls = [];
  const filesPayload = { repo: "1", source: "civitai", files: [] };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(ONE_CIVITAI_RESULT, filesPayload, calls) });
  selectCivitai(window);
  await window.discoverSearch();
  window.document.getElementById("disc-civitai-legacy").checked = true;
  const row = rowFor(window, "Detail Tweaker");
  const btn = [...row.querySelectorAll("button")].find((b) => b.textContent === "files");
  btn.click();
  await new Promise((r) => setTimeout(r, 0));

  const filesCall = calls.find((u) => u.includes("/api/discover/files"));
  assert.ok(decodeURIComponent(filesCall).includes("legacy_formats=true"));
});
