// SPDX-License-Identifier: AGPL-3.0-or-later
// HuggingFace discovery for every "Find models" tab, not just All/LLMs. Before
// this, every non-all/llm tab showed a static "coming soon" placeholder instead
// of a working search box (models.js's refreshModelsPage hardcoded the gate to
// currentTypeFilter === "all" || "llm"). This is the regression test for that
// bug across all 6 previously-broken tabs, plus the type-hint threading that
// makes a type-scoped find register with the right model_type explicitly
// rather than relying on HF's own (proven-unreliable-for-vae/text-encoder)
// metadata. Mirrors models-discover-formats.test.mjs and
// models-embedding-tab.test.mjs (same page, same harness/fetch-mock style).

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const PREVIOUSLY_BROKEN_TABS =
  ["embedding", "diffusion-unet", "text-encoder", "vae", "lora", "unknown"];
const FORMAT_HIDDEN_TABS = new Set(["lora", "vae", "text-encoder", "unknown"]);

function makeFetch({ discoverPayload = null, filesPayload = null, calls = [], pullResponds = true } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, body: opts.body ? JSON.parse(opts.body) : null });
    if (u.startsWith("/api/discover/search")) {
      return {
        ok: true, status: 200,
        json: async () => discoverPayload || { query: "", vram: {}, hf_backend_available: true, results: [] },
        text: async () => "",
      };
    }
    if (u.startsWith("/api/discover/files")) {
      return {
        ok: true, status: 200,
        json: async () => filesPayload || { files: [], mmprojs: [], vram: {} },
        text: async () => "",
      };
    }
    if (u === "/api/models/pull") {
      return pullResponds
        ? { ok: true, status: 200, json: async () => ({ job_id: "j1" }), text: async () => "" }
        : { ok: false, status: 400, json: async () => ({ detail: "bad" }), text: async () => "" };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models: [], active: null }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

function clickTab(window, type) {
  const btn = window.document.querySelector(`#models-tab-nav .tab-btn[data-type="${type}"]`);
  assert.ok(btn, `tab button for ${type} exists`);
  btn.click();
}

async function tick(n = 1) {
  for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0));
}

// --------------------------------------------------------------------------- //
//  The literal regression test: every tab gets a real search box, never the   //
//  static placeholder.                                                        //
// --------------------------------------------------------------------------- //

for (const type of PREVIOUSLY_BROKEN_TABS) {
  test(`discover-type-scoped: the "${type}" tab shows the real search box, not a placeholder`, async () => {
    const { window } = loadAppWithPages({ fetchImpl: makeFetch() });
    await window.refreshModelsPage();
    await tick();

    clickTab(window, type);
    await tick();

    assert.equal(window.document.getElementById("disc-placeholder"), null,
      "the 'coming soon' placeholder element no longer exists at all");
    const searchBox = window.document.getElementById("disc-search-box");
    assert.ok(searchBox, "the search box exists");
    assert.notEqual(searchBox.style.display, "none",
      `the search box is visible on the "${type}" tab`);
    assert.doesNotMatch(window.document.body.textContent, /coming soon/i,
      "no 'coming soon' text remains anywhere on the page");
  });
}

test("discover-type-scoped: format toggles are hidden for lora/vae/text-encoder/unknown, shown elsewhere", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch() });
  await window.refreshModelsPage();
  await tick();

  for (const type of ["all", "llm", "embedding", "diffusion-unet", "lora", "vae", "text-encoder", "unknown"]) {
    clickTab(window, type);
    await tick();
    const formatsBox = window.document.getElementById("disc-formats");
    if (FORMAT_HIDDEN_TABS.has(type)) {
      assert.equal(formatsBox.style.display, "none", `formats hidden on "${type}"`);
    } else {
      assert.notEqual(formatsBox.style.display, "none", `formats shown on "${type}"`);
    }
  }
});

// --------------------------------------------------------------------------- //
//  discoverSearch() sends the current tab as `type=`                          //
// --------------------------------------------------------------------------- //

test("discover-type-scoped: discoverSearch() sends type=<tab> in the query string", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ calls }) });
  await window.refreshModelsPage();
  await tick();
  clickTab(window, "vae");
  await tick();

  await window.discoverSearch();

  const discoverCalls = calls.filter((c) => c.url.startsWith("/api/discover/search"));
  assert.equal(discoverCalls.length, 1);
  assert.match(decodeURIComponent(discoverCalls[0].url), /type=vae/,
    "the search request carries the active tab as type=vae");
});

// --------------------------------------------------------------------------- //
//  A result's detected_type renders the matching color-coded badge            //
// --------------------------------------------------------------------------- //

test("discover-type-scoped: a result with detected_type renders the matching .type-badge", async () => {
  const payload = {
    query: "", vram: {}, hf_backend_available: true,
    results: [
      { id: "stabilityai/sd-vae-ft-mse", downloads: 900, likes: 10, updated: "",
        formats: ["hf"], detected_type: "unknown" },
      { id: "BAAI/bge-m3", downloads: 20, likes: 5, updated: "",
        formats: ["hf"], detected_type: "embedding" },
    ],
  };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload }) });
  await window.refreshModelsPage();
  await tick();
  clickTab(window, "vae");
  await tick();
  await window.discoverSearch();

  const rows = [...window.document.querySelectorAll("#disc-results .disc-repo")];
  const vaeRow = rows.find((r) => r.querySelector(".name")?.textContent === "stabilityai/sd-vae-ft-mse");
  const embRow = rows.find((r) => r.querySelector(".name")?.textContent === "BAAI/bge-m3");
  assert.ok(vaeRow.querySelector(".type-badge.type-unknown"),
    "an honestly-unclassified VAE result still shows an 'unknown' badge, not excluded or guessed");
  assert.ok(embRow.querySelector(".type-badge.type-embedding"),
    "a classified result shows its detected type badge");
});

test("discover-type-scoped: a result with no detected_type (all/llm tabs) renders no badge", async () => {
  const payload = {
    query: "", vram: {}, hf_backend_available: true,
    results: [{ id: "org/plain", downloads: 1, likes: 0, updated: "", formats: ["gguf"] }],
  };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload }) });
  await window.discoverSearch();   // default tab is "all"
  const row = window.document.querySelector("#disc-results .disc-repo");
  assert.equal(row.querySelector(".type-badge"), null,
    "no badge is rendered when the server sent no detected_type");
});

// --------------------------------------------------------------------------- //
//  Type-hint threading: a type-scoped find registers with that explicit type  //
// --------------------------------------------------------------------------- //

function stubStreamJobDone(window) {
  runScript(window, `streamJob = async () => ({ status: "done" });`);
}

test("discover-type-scoped: adding a VAE-tab result POSTs model_type: 'vae'", async () => {
  const payload = {
    query: "", vram: {}, hf_backend_available: true,
    results: [{ id: "stabilityai/sd-vae-ft-mse", downloads: 900, likes: 10, updated: "",
                formats: ["hf"], detected_type: "unknown" }],
  };
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload, calls }) });
  stubStreamJobDone(window);
  await window.refreshModelsPage();
  await tick();
  clickTab(window, "vae");
  await tick();
  await window.discoverSearch();

  const row = window.document.querySelector("#disc-results .disc-repo");
  const addBtn = [...row.querySelectorAll("button")].find((b) => b.textContent === "add full repo");
  assert.ok(addBtn, "the VAE result offers 'add full repo'");
  addBtn.click();

  assert.equal(window.document.getElementById("pull-spec").value, "stabilityai/sd-vae-ft-mse");

  window.document.getElementById("pull-start").click();
  await tick(3);

  const pullCall = calls.find((c) => c.url === "/api/models/pull");
  assert.ok(pullCall, "the pull request was sent");
  assert.equal(pullCall.body.model_type, "vae",
    "the explicit type hint from the VAE tab reaches the pull request, " +
    "bypassing HF's own (proven-unreliable-for-vae) auto-guess");
});

test("discover-type-scoped: hand-editing the spec after prefill drops the stale type hint", async () => {
  const payload = {
    query: "", vram: {}, hf_backend_available: true,
    results: [{ id: "stabilityai/sd-vae-ft-mse", downloads: 900, likes: 10, updated: "",
                formats: ["hf"], detected_type: "unknown" }],
  };
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload, calls }) });
  stubStreamJobDone(window);
  await window.refreshModelsPage();
  await tick();
  clickTab(window, "vae");
  await tick();
  await window.discoverSearch();

  const row = window.document.querySelector("#disc-results .disc-repo");
  const addBtn = [...row.querySelectorAll("button")].find((b) => b.textContent === "add full repo");
  addBtn.click();

  // The user changes their mind and edits the prefilled spec before hitting Add.
  window.document.getElementById("pull-spec").value = "someone/else-entirely";

  window.document.getElementById("pull-start").click();
  await tick(3);

  const pullCall = calls.find((c) => c.url === "/api/models/pull");
  assert.ok(pullCall, "the pull request was sent");
  assert.equal(pullCall.body.model_type, undefined,
    "a hand-edited spec silently falls back to auto-detect, never sends a hint for a different model");
});

test("discover-type-scoped: adding an 'Other'/unknown-tab result never forces model_type", async () => {
  const payload = {
    query: "", vram: {}, hf_backend_available: true,
    results: [{ id: "madebyollin/taesd", downloads: 50, likes: 2, updated: "",
                formats: ["hf"], detected_type: "unknown" }],
  };
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload, calls }) });
  stubStreamJobDone(window);
  await window.refreshModelsPage();
  await tick();
  clickTab(window, "unknown");
  await tick();
  await window.discoverSearch();

  const row = window.document.querySelector("#disc-results .disc-repo");
  const addBtn = [...row.querySelectorAll("button")].find((b) => b.textContent === "add full repo");
  addBtn.click();
  window.document.getElementById("pull-start").click();
  await tick(3);

  const pullCall = calls.find((c) => c.url === "/api/models/pull");
  assert.ok(pullCall, "the pull request was sent");
  assert.equal(pullCall.body.model_type, undefined,
    "the 'Other' tab never locks a find to model_type=unknown - a real guess still gets a chance");
});

// --------------------------------------------------------------------------- //
//  The OTHER pull affordance: a per-quant GGUF file (discoverFiles), not just  //
//  the whole-repo "add full repo" button. Real repo/file from a live search    //
//  this feature was verified against (unsloth/bge-small-en-v1.5-GGUF).         //
// --------------------------------------------------------------------------- //

test("discover-type-scoped: picking a per-quant GGUF file from a type-scoped tab also carries the type hint", async () => {
  const discoverPayload = {
    query: "", vram: {}, hf_backend_available: true,
    results: [{ id: "unsloth/bge-small-en-v1.5-GGUF", downloads: 9, likes: 1, updated: "",
                formats: ["gguf"], detected_type: "embedding" }],
  };
  const filesPayload = {
    vram: {}, mmprojs: [],
    files: [{ file: "bge-small-en-v1.5-f16.gguf", quant: "F16", size_bytes: 1e8, n_parts: 1 }],
  };
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({ discoverPayload, filesPayload, calls }) });
  stubStreamJobDone(window);
  await window.refreshModelsPage();
  await tick();
  clickTab(window, "embedding");
  await tick();
  await window.discoverSearch();

  const row = window.document.querySelector("#disc-results .disc-repo");
  const filesBtn = [...row.querySelectorAll("button")].find((b) => b.textContent === "files");
  assert.ok(filesBtn, "the GGUF result offers the per-quant file list");
  filesBtn.click();
  await tick();

  const fileRow = row.querySelector(".disc-file");
  assert.ok(fileRow, "the file row renders");
  const pullBtn = [...fileRow.querySelectorAll("button")].find((b) => b.textContent === "pull");
  pullBtn.click();

  assert.equal(window.document.getElementById("pull-spec").value,
    "unsloth/bge-small-en-v1.5-GGUF:bge-small-en-v1.5-f16.gguf",
    "the spec is repo:file, matching what the real HF API's tree listing produces");

  window.document.getElementById("pull-start").click();
  await tick(3);

  const pullCall = calls.find((c) => c.url === "/api/models/pull");
  assert.ok(pullCall, "the pull request was sent");
  assert.equal(pullCall.body.model_type, "embedding",
    "the per-quant GGUF pull path (not just 'add full repo') also carries the type hint");
});

// --------------------------------------------------------------------------- //
//  A vision-projector companion file never inherits the tab's type hint.       //
//  There is no "mmproj" tab, and mmproj pairing is filename-keyed, not         //
//  model_type-keyed - forcing e.g. model_type="vae" onto it would mislabel     //
//  its Role column / tab filtering for no reason.                              //
// --------------------------------------------------------------------------- //

test("discover-type-scoped: a vision-projector (mmproj) file never inherits the tab's type hint", async () => {
  const discoverPayload = {
    query: "", vram: {}, hf_backend_available: true,
    results: [{ id: "org/vision-model", downloads: 9, likes: 1, updated: "",
                formats: ["gguf"], detected_type: "vae" }],
  };
  const filesPayload = {
    vram: {},
    files: [{ file: "model-Q4_K_M.gguf", quant: "Q4_K_M", size_bytes: 1e8, n_parts: 1 }],
    mmprojs: [{ file: "mmproj-model-f16.gguf", quant: "F16", size_bytes: 5e7, n_parts: 1 }],
  };
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({ discoverPayload, filesPayload, calls }) });
  stubStreamJobDone(window);
  window.localStorage.setItem("localm.showMmprojFiles", "true");
  await window.refreshModelsPage();
  await tick();
  clickTab(window, "vae");
  await tick();
  await window.discoverSearch();

  const row = window.document.querySelector("#disc-results .disc-repo");
  const filesBtn = [...row.querySelectorAll("button")].find((b) => b.textContent === "files");
  filesBtn.click();
  await tick();

  const fileRows = [...row.querySelectorAll(".disc-file")];
  const mmprojRow = fileRows.find((r) => r.querySelector(".fname")?.textContent === "mmproj-model-f16.gguf");
  assert.ok(mmprojRow, "the mmproj file row renders (show-mmproj-files is on)");
  const pullBtn = [...mmprojRow.querySelectorAll("button")].find((b) => b.textContent === "pull");
  pullBtn.click();

  window.document.getElementById("pull-start").click();
  await tick(3);

  const pullCall = calls.find((c) => c.url === "/api/models/pull");
  assert.ok(pullCall, "the pull request was sent");
  assert.equal(pullCall.body.model_type, undefined,
    "an mmproj file is never forced to the tab's type - there is no mmproj tab");
});

// --------------------------------------------------------------------------- //
//  Stale results don't survive a tab switch (they'd carry the WRONG tab's      //
//  type hint if pulled after switching).                                       //
// --------------------------------------------------------------------------- //

test("discover-type-scoped: switching tabs clears stale search results", async () => {
  const discoverPayload = {
    query: "", vram: {}, hf_backend_available: true,
    results: [{ id: "org/found-under-vae", downloads: 1, likes: 0, updated: "",
                formats: ["hf"], detected_type: "vae" }],
  };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload }) });
  await window.refreshModelsPage();
  await tick();
  clickTab(window, "vae");
  await tick();
  await window.discoverSearch();

  assert.ok(window.document.querySelector("#disc-results .disc-repo"),
    "the VAE search result is showing");

  clickTab(window, "lora");
  await tick();

  assert.equal(window.document.querySelectorAll("#disc-results .disc-repo").length, 0,
    "switching tabs discards the previous tab's stale, wrongly-hinted results");
});
