// SPDX-License-Identifier: AGPL-3.0-or-later
// Covers discRepoRow's rendering of the architecture, MoE and param-count fields.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch({ discoverPayload = null } = {}) {
  return async (url) => {
    const u = String(url);
    if (u.startsWith("/api/discover/search")) {
      return { ok: true, status: 200, text: async () => "",
        json: async () => discoverPayload || { query: "", vram: {}, hf_backend_available: true, results: [] } };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

test("discover-arch-moe: a confirmed-MoE result shows architecture + a solid MoE badge", async () => {
  const payload = { query: "", vram: {}, hf_backend_available: true, results: [
    { id: "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", downloads: 100, likes: 5, updated: "",
      formats: ["gguf"], detected_type: "llm",
      architecture: "qwen3moe", moe: "confirmed", param_count: 30532122624 },
  ] };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload }) });
  await window.discoverSearch();
  const row = window.document.querySelector("#disc-results .disc-repo");
  const arch = row.querySelector(".arch-badge");
  assert.ok(arch, "architecture badge renders");
  assert.equal(arch.textContent, "qwen3moe");
  const moe = row.querySelector(".moe-badge");
  assert.ok(moe, "MoE badge renders");
  assert.ok(moe.classList.contains("moe-confirmed"), "confirmed MoE gets the confirmed class");
  assert.equal(moe.textContent, "MoE", "a confirmed MoE badge reads plainly, no '?'");
  assert.equal(moe.title, "", "a confirmed badge carries no 'inferred' caveat tooltip");
  const params = row.querySelector(".param-count");
  assert.ok(params, "param count renders");
  assert.equal(params.textContent, "30.5B params");
});

test("discover-arch-moe: a name-inferred MoE result is visibly weaker than a confirmed one", async () => {
  // an architecture header of 'llama' with an MoE repo name: the server marks
  // this 'likely', never 'confirmed'
  const payload = { query: "", vram: {}, hf_backend_available: true, results: [
    { id: "TheBloke/Mixtral-8x7B-v0.1-GGUF", downloads: 900, likes: 50, updated: "",
      formats: ["gguf"], detected_type: "llm",
      architecture: "llama", moe: "likely", param_count: null },
  ] };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload }) });
  await window.discoverSearch();
  const row = window.document.querySelector("#disc-results .disc-repo");
  const moe = row.querySelector(".moe-badge");
  assert.ok(moe, "MoE badge still renders for a name-only signal");
  assert.ok(moe.classList.contains("moe-likely"), "name-inferred MoE gets the likely class, not confirmed");
  assert.ok(!moe.classList.contains("moe-confirmed"), "never both classes at once");
  assert.equal(moe.textContent, "MoE?", "the label itself marks it as uncertain");
  assert.match(moe.title, /inferred/i, "a tooltip explains the guess is name-based, not header-confirmed");
  assert.equal(row.querySelector(".param-count"), null,
    "no param-count element renders when the server had no count to give");
});

test("discover-arch-moe: a result with no MoE evidence shows no MoE badge (never a false 'dense' claim)", async () => {
  const payload = { query: "", vram: {}, hf_backend_available: true, results: [
    { id: "meta-llama/Llama-3.1-8B-Instruct-GGUF", downloads: 1, likes: 0, updated: "",
      formats: ["gguf"], detected_type: "llm",
      architecture: "llama", moe: null, param_count: 8030261248 },
  ] };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload }) });
  await window.discoverSearch();
  const row = window.document.querySelector("#disc-results .disc-repo");
  assert.equal(row.querySelector(".moe-badge"), null,
    "absence of MoE evidence renders no badge at all, not a false negative claim");
  assert.ok(row.querySelector(".arch-badge"), "architecture still shows on its own");
  assert.equal(row.querySelector(".param-count").textContent, "8.0B params");
});

test("discover-arch-moe: a legacy/untyped result (no architecture/moe/param_count keys) renders none of the new badges", async () => {
  // model_types=None attaches none of these fields
  const payload = { query: "", vram: {}, hf_backend_available: true, results: [
    { id: "org/legacy-result", downloads: 1, likes: 0, updated: "", formats: ["gguf"] },
  ] };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload }) });
  await window.discoverSearch();
  const row = window.document.querySelector("#disc-results .disc-repo");
  assert.equal(row.querySelector(".arch-badge"), null);
  assert.equal(row.querySelector(".moe-badge"), null);
  assert.equal(row.querySelector(".param-count"), null);
});

test("discover-arch-moe: fmtParamCount formats B/M/plain thresholds and empty input", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch() });
  assert.equal(window.fmtParamCount(30532122624), "30.5B params");
  assert.equal(window.fmtParamCount(494_000_000), "494.0M params");
  assert.equal(window.fmtParamCount(500), "500 params");
  assert.equal(window.fmtParamCount(0), "");
  assert.equal(window.fmtParamCount(null), "");
});

test("discover-arch-moe: a legend explains the 'likely' MoE badge WITHOUT relying on hover (touch has no hover)", async () => {
  const payload = { query: "", vram: {}, hf_backend_available: true, results: [
    { id: "TheBloke/Mixtral-8x7B-v0.1-GGUF", downloads: 1, likes: 0, updated: "",
      formats: ["gguf"], detected_type: "llm", architecture: "llama", moe: "likely" },
  ] };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload }) });
  await window.discoverSearch();
  const legend = window.document.querySelector("#disc-results .moe-legend");
  assert.ok(legend, "a persistent (non-tooltip) legend renders");
  assert.match(legend.textContent, /inferred from the model's name/i);
});

test("discover-arch-moe: no legend when nothing on screen needs it", async () => {
  const payload = { query: "", vram: {}, hf_backend_available: true, results: [
    { id: "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", downloads: 1, likes: 0, updated: "",
      formats: ["gguf"], detected_type: "llm", architecture: "qwen3moe", moe: "confirmed" },
    { id: "org/no-moe-evidence", downloads: 1, likes: 0, updated: "",
      formats: ["gguf"], detected_type: "llm", architecture: "llama", moe: null },
  ] };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload }) });
  await window.discoverSearch();
  assert.equal(window.document.querySelector("#disc-results .moe-legend"), null,
    "no results carry an inferred (not confirmed) MoE signal, so no legend clutters the page");
});

test("discover-arch-moe: these badges are purely additive - every result still renders regardless", async () => {
  // two results with different metadata completeness
  const payload = { query: "", vram: {}, hf_backend_available: true, results: [
    { id: "org/rich-metadata", downloads: 1, likes: 0, updated: "", formats: ["gguf"],
      detected_type: "llm", architecture: "mixtral", moe: "confirmed", param_count: 46_700_000_000 },
    { id: "org/bare-metadata", downloads: 1, likes: 0, updated: "", formats: ["gguf"],
      detected_type: "unknown" },
  ] };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ discoverPayload: payload }) });
  await window.discoverSearch();
  const rows = [...window.document.querySelectorAll("#disc-results .disc-repo")];
  assert.equal(rows.length, 2, "both results render regardless of how much display metadata they carry");
});
