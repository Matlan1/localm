// SPDX-License-Identifier: AGPL-3.0-or-later
// HuggingFace discovery has always-visible filters: a row of model TYPE
// checkboxes (LLMs/Embedding/Diffusion/Encoders/VAEs/LoRAs/Other) and a FORMAT
// row (GGUF/Safetensors), both independent of the Registered-models tabs. Covers
// the filter rows, the types=/formats= search params, the empty-selection guards,
// result type badges, and the pull type-hint (detected type, or the single
// narrowed-to type).

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const ALL_TYPES = ["llm", "embedding", "diffusion-unet", "text-encoder", "vae", "lora", "unknown"];

function makeFetch({ discoverPayload = null, filesPayload = null, calls = [] } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, body: opts.body ? JSON.parse(opts.body) : null });
    if (u.startsWith("/api/discover/search")) {
      return { ok: true, status: 200, text: async () => "",
        json: async () => discoverPayload || { query: "", vram: {}, hf_backend_available: true, results: [] } };
    }
    if (u.startsWith("/api/discover/files")) {
      return { ok: true, status: 200, text: async () => "",
        json: async () => filesPayload || { files: [], mmprojs: [], vram: {} } };
    }
    if (u === "/api/models/pull") {
      return { ok: true, status: 200, json: async () => ({ job_id: "j1" }), text: async () => "" };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models: [], active: null }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

async function tick(n = 1) { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); }

function setTypes(window, values) {
  for (const b of window.document.querySelectorAll(".disc-type")) b.checked = values.includes(b.value);
}
function setFormats(window, { gguf, hf }) {
  window.document.getElementById("disc-fmt-gguf").checked = gguf;
  window.document.getElementById("disc-fmt-hf").checked = hf;
}
function stubStreamJobDone(window) { runScript(window, `streamJob = async () => ({ status: "done" });`); }

// --------------------------------------------------------------------------- //
//  The filters exist, are visible, and cover every model type                 //
// --------------------------------------------------------------------------- //

test("discover-filters: the Type + Format filter rows are present, visible, all default-on", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch() });
  const typeRow = window.document.getElementById("disc-type-filter");
  const fmtRow = window.document.getElementById("disc-formats");
  assert.ok(typeRow, "the Types filter row exists");
  assert.ok(fmtRow, "the Format filter row exists");
  assert.equal(window.document.getElementById("disc-placeholder"), null,
    "no 'coming soon' placeholder element remains");

  const typeVals = [...window.document.querySelectorAll(".disc-type")].map((b) => b.value);
  assert.deepEqual(typeVals.sort(), [...ALL_TYPES].sort(),
    "there is one Type checkbox per searchable model type");
  for (const b of window.document.querySelectorAll(".disc-type")) {
    assert.ok(b.checked, `type ${b.value} defaults to checked (search everything)`);
  }
  assert.ok(window.document.getElementById("disc-fmt-gguf").checked, "GGUF default on");
  assert.ok(window.document.getElementById("disc-fmt-hf").checked, "Safetensors default on");
});

test("discover-filters: each chip's active .on class stays in sync with its checkbox", async () => {
  // style.css colours `.disc-chip.on span`, not a CSS :checked selector, so the
  // `.on` class must track the checkbox on load and after every change.
  const { window } = loadAppWithPages({ fetchImpl: makeFetch() });
  const chips = [...window.document.querySelectorAll(".disc-chip")];
  assert.ok(chips.length >= 9, "every type + format toggle is a chip");
  for (const chip of chips) {
    const box = chip.querySelector("input");
    assert.equal(chip.classList.contains("on"), box.checked,
      `chip for "${box.value || box.id}" starts in sync (both ${box.checked})`);
  }
  // Toggle one off via a real change event and confirm the class follows.
  const vae = window.document.querySelector('.disc-chip[data-type="vae"]');
  const vaeBox = vae.querySelector("input");
  vaeBox.checked = false;
  vaeBox.dispatchEvent(new window.Event("change"));
  assert.equal(vae.classList.contains("on"), false, "unchecking removes the .on class (chip greys out)");
  vaeBox.checked = true;
  vaeBox.dispatchEvent(new window.Event("change"));
  assert.equal(vae.classList.contains("on"), true, "re-checking restores the .on class (chip recolours)");
});

test("discover-filters: filters stay visible regardless of the active Registered-models tab", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch() });
  await window.refreshModelsPage();
  await tick();
  for (const type of ["all", ...ALL_TYPES]) {
    const btn = window.document.querySelector(`#models-tab-nav .tab-btn[data-type="${type}"]`);
    if (!btn) continue;
    btn.click();
    await tick();
    assert.notEqual(window.document.getElementById("disc-type-filter").style.display, "none",
      `Types filter visible on the "${type}" tab`);
    assert.notEqual(window.document.getElementById("disc-formats").style.display, "none",
      `Format filter visible on the "${type}" tab`);
  }
});

// --------------------------------------------------------------------------- //
//  The search sends the checked types + formats                               //
// --------------------------------------------------------------------------- //

test("discover-filters: a full-default search sends every type and both formats", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ calls }) });
  await window.discoverSearch();
  const u = decodeURIComponent(calls.find((c) => c.url.startsWith("/api/discover/search")).url);
  assert.match(u, /formats=gguf,hf/, "both formats sent");
  const typesParam = u.match(/types=([^&]*)/)[1].split(",").sort();
  assert.deepEqual(typesParam, [...ALL_TYPES].sort(), "all seven types sent");
});

test("discover-filters: narrowing to VAEs + Safetensors sends exactly that", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ calls }) });
  setTypes(window, ["vae"]);
  setFormats(window, { gguf: false, hf: true });
  await window.discoverSearch();
  const u = decodeURIComponent(calls.find((c) => c.url.startsWith("/api/discover/search")).url);
  assert.match(u, /types=vae(&|$)/, "only vae in the types param");
  assert.match(u, /formats=hf(&|$)/, "only the safetensors format sent");
});

test("discover-filters: no types selected shows a guard and issues no search", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ calls }) });
  setTypes(window, []);
  await window.discoverSearch();
  assert.equal(calls.filter((c) => c.url.startsWith("/api/discover/search")).length, 0,
    "no search fired with zero types checked");
  assert.match(window.document.getElementById("disc-results").textContent, /at least one model type/i);
});

test("discover-filters: no formats selected shows a guard and issues no search", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ calls }) });
  setFormats(window, { gguf: false, hf: false });
  await window.discoverSearch();
  assert.equal(calls.filter((c) => c.url.startsWith("/api/discover/search")).length, 0,
    "no search fired with zero formats checked");
  assert.match(window.document.getElementById("disc-results").textContent, /at least one format/i);
});

test("discover-filters: toggling a type checkbox re-runs the search live when results are showing", async () => {
  const payload = { query: "", vram: {}, hf_backend_available: true,
    results: [{ id: "org/x", downloads: 1, likes: 0, updated: "", formats: ["hf"], detected_type: "llm" }] };
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload, calls }) });
  await window.discoverSearch();
  const before = calls.filter((c) => c.url.startsWith("/api/discover/search")).length;
  const vaeBox = window.document.querySelector('.disc-type[value="vae"]');
  vaeBox.checked = false;
  vaeBox.dispatchEvent(new window.Event("change"));
  await tick(2);
  const after = calls.filter((c) => c.url.startsWith("/api/discover/search")).length;
  assert.ok(after > before, "changing a type checkbox re-issued the search while results were showing");
});

// --------------------------------------------------------------------------- //
//  Result badges + the pull type hint                                         //
// --------------------------------------------------------------------------- //

test("discover-filters: a result renders the badge for its detected type", async () => {
  const payload = { query: "", vram: {}, hf_backend_available: true, results: [
    { id: "stabilityai/sd-vae-ft-mse", downloads: 900, likes: 10, updated: "", formats: ["hf"], detected_type: "unknown" },
    { id: "BAAI/bge-m3", downloads: 20, likes: 5, updated: "", formats: ["hf"], detected_type: "embedding" },
  ] };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload }) });
  await window.discoverSearch();
  const rows = [...window.document.querySelectorAll("#disc-results .disc-repo")];
  const vae = rows.find((r) => r.querySelector(".name").textContent === "stabilityai/sd-vae-ft-mse");
  const emb = rows.find((r) => r.querySelector(".name").textContent === "BAAI/bge-m3");
  assert.ok(vae.querySelector(".type-badge.type-unknown"), "an honestly-unclassified VAE still shows an 'unknown' badge");
  assert.ok(emb.querySelector(".type-badge.type-embedding"), "a classified result shows its detected type badge");
});

test("discover-filters: adding a result badged with a known type hints THAT type", async () => {
  const payload = { query: "", vram: {}, hf_backend_available: true,
    results: [{ id: "black-forest-labs/FLUX.1-dev", downloads: 500, likes: 9, updated: "",
                formats: ["hf"], detected_type: "diffusion-unet" }] };
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload, calls }) });
  stubStreamJobDone(window);
  setTypes(window, ALL_TYPES);   // many types checked
  await window.discoverSearch();
  const row = window.document.querySelector("#disc-results .disc-repo");
  [...row.querySelectorAll("button")].find((b) => b.textContent === "add full repo").click();
  window.document.getElementById("pull-start").click();
  await tick(3);
  const pull = calls.find((c) => c.url === "/api/models/pull");
  assert.equal(pull.body.model_type, "diffusion-unet",
    "what you see badged is what it registers as - the detected type is the hint");
});

test("discover-filters: an 'unknown'-badged result hints the single narrowed-to type", async () => {
  const payload = { query: "", vram: {}, hf_backend_available: true,
    results: [{ id: "stabilityai/sd-vae-ft-mse", downloads: 900, likes: 10, updated: "",
                formats: ["hf"], detected_type: "unknown" }] };
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload, calls }) });
  stubStreamJobDone(window);
  setTypes(window, ["vae"]);      // narrowed to exactly one type
  await window.discoverSearch();
  const row = window.document.querySelector("#disc-results .disc-repo");
  [...row.querySelectorAll("button")].find((b) => b.textContent === "add full repo").click();
  window.document.getElementById("pull-start").click();
  await tick(3);
  const pull = calls.find((c) => c.url === "/api/models/pull");
  assert.equal(pull.body.model_type, "vae",
    "a metadata-less VAE found while searching only VAEs registers as vae, not a guess");
});

test("discover-filters: an 'unknown'-badged result with MANY types checked forces no hint", async () => {
  const payload = { query: "", vram: {}, hf_backend_available: true,
    results: [{ id: "madebyollin/taesd", downloads: 50, likes: 2, updated: "",
                formats: ["hf"], detected_type: "unknown" }] };
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload, calls }) });
  stubStreamJobDone(window);
  setTypes(window, ["vae", "text-encoder", "unknown"]);   // more than one -> ambiguous
  await window.discoverSearch();
  const row = window.document.querySelector("#disc-results .disc-repo");
  [...row.querySelectorAll("button")].find((b) => b.textContent === "add full repo").click();
  window.document.getElementById("pull-start").click();
  await tick(3);
  const pull = calls.find((c) => c.url === "/api/models/pull");
  assert.equal(pull.body.model_type, undefined,
    "ambiguous (unknown badge, multiple types) falls back to auto-detect, never a wrong guess");
});

test("discover-filters: hand-editing the spec after prefill drops the stale hint", async () => {
  const payload = { query: "", vram: {}, hf_backend_available: true,
    results: [{ id: "some/vae", downloads: 1, likes: 0, updated: "", formats: ["hf"], detected_type: "vae" }] };
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload, calls }) });
  stubStreamJobDone(window);
  await window.discoverSearch();
  const row = window.document.querySelector("#disc-results .disc-repo");
  [...row.querySelectorAll("button")].find((b) => b.textContent === "add full repo").click();
  window.document.getElementById("pull-spec").value = "someone/else-entirely";   // user edits it
  window.document.getElementById("pull-start").click();
  await tick(3);
  const pull = calls.find((c) => c.url === "/api/models/pull");
  assert.equal(pull.body.model_type, undefined,
    "a hand-edited spec never carries the hint meant for the original result");
});

test("discover-filters: a per-quant GGUF file also carries the type hint", async () => {
  const discoverPayload = { query: "", vram: {}, hf_backend_available: true,
    results: [{ id: "unsloth/bge-small-en-v1.5-GGUF", downloads: 9, likes: 1, updated: "",
                formats: ["gguf"], detected_type: "embedding" }] };
  const filesPayload = { vram: {}, mmprojs: [],
    files: [{ file: "bge-small-en-v1.5-f16.gguf", quant: "F16", size_bytes: 1e8, n_parts: 1 }] };
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload, filesPayload, calls }) });
  stubStreamJobDone(window);
  await window.discoverSearch();
  const row = window.document.querySelector("#disc-results .disc-repo");
  [...row.querySelectorAll("button")].find((b) => b.textContent === "files").click();
  await tick();
  const fileRow = row.querySelector(".disc-file");
  [...fileRow.querySelectorAll("button")].find((b) => b.textContent === "pull").click();
  assert.equal(window.document.getElementById("pull-spec").value,
    "unsloth/bge-small-en-v1.5-GGUF:bge-small-en-v1.5-f16.gguf");
  window.document.getElementById("pull-start").click();
  await tick(3);
  const pull = calls.find((c) => c.url === "/api/models/pull");
  assert.equal(pull.body.model_type, "embedding",
    "the per-quant GGUF pull path (not just 'add full repo') also carries the hint");
});

test("discover-filters: a vision-projector (mmproj) file never inherits a type hint", async () => {
  const discoverPayload = { query: "", vram: {}, hf_backend_available: true,
    results: [{ id: "org/vision-model", downloads: 9, likes: 1, updated: "",
                formats: ["gguf"], detected_type: "llm" }] };
  const filesPayload = { vram: {},
    files: [{ file: "model-Q4_K_M.gguf", quant: "Q4_K_M", size_bytes: 1e8, n_parts: 1 }],
    mmprojs: [{ file: "mmproj-model-f16.gguf", quant: "F16", size_bytes: 5e7, n_parts: 1 }] };
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload, filesPayload, calls }) });
  stubStreamJobDone(window);
  window.localStorage.setItem("localm.showMmprojFiles", "true");
  await window.discoverSearch();
  const row = window.document.querySelector("#disc-results .disc-repo");
  [...row.querySelectorAll("button")].find((b) => b.textContent === "files").click();
  await tick();
  const mmprojRow = [...row.querySelectorAll(".disc-file")]
    .find((r) => r.querySelector(".fname").textContent === "mmproj-model-f16.gguf");
  assert.ok(mmprojRow, "the mmproj file row renders (show-mmproj-files is on)");
  [...mmprojRow.querySelectorAll("button")].find((b) => b.textContent === "pull").click();
  window.document.getElementById("pull-start").click();
  await tick(3);
  const pull = calls.find((c) => c.url === "/api/models/pull");
  assert.equal(pull.body.model_type, undefined,
    "an mmproj file is never forced to the searched-for type - there is no mmproj tab/checkbox");
});
