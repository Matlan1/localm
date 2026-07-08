// SPDX-License-Identifier: AGPL-3.0-or-later
// Models page HuggingFace search: the GGUF / HF-transformers format toggles.
// The search must send the toggled formats, badge each result by format, offer
// the right pull affordance (GGUF -> per-quant file list; HF -> whole-repo add
// prefilling a BARE owner/repo), and show a NON-BLOCKING hint when HF is searched
// but no transformers runtime is installed (the download still works).

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(payload, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.includes("/api/discover/search")) {
      calls.push(u);
      return { ok: true, status: 200, json: async () => payload, text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

const TWO_RESULTS = {
  query: "",
  vram: {},
  hf_backend_available: true,
  results: [
    { id: "org/gguf-only", downloads: 5, likes: 1, updated: "", formats: ["gguf"] },
    { id: "org/hf-only", downloads: 9, likes: 2, updated: "", formats: ["hf"] },
  ],
};

function rowFor(win, repoId) {
  for (const row of win.document.querySelectorAll("#disc-results .disc-repo")) {
    if (row.querySelector(".name")?.textContent === repoId) return row;
  }
  return null;
}
function buttonByText(row, text) {
  return [...row.querySelectorAll("button")].find((b) => b.textContent === text) || null;
}

test("discover: sends the toggled formats and badges each result by format", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(TWO_RESULTS, calls) });
  await window.discoverSearch();

  assert.equal(calls.length, 1, "one discover search fired");
  const u = decodeURIComponent(calls[0]);
  assert.ok(u.includes("formats=gguf,hf"),
    `both default-on toggles are sent as the formats CSV (got ${u})`);

  const gg = rowFor(window, "org/gguf-only");
  const hf = rowFor(window, "org/hf-only");
  assert.ok(gg && hf, "both result rows render");
  assert.ok(gg.querySelector(".fmt-badge.fmt-gguf"), "the GGUF result carries a GGUF badge");
  assert.ok(hf.querySelector(".fmt-badge.fmt-hf"), "the HF result carries an HF badge");
  assert.ok(buttonByText(gg, "files"), "the GGUF result offers the per-quant file list");
  assert.ok(buttonByText(hf, "add full repo"), "the HF result offers a whole-repo add");
});

test("discover: HF add prefills a BARE owner/repo (no :file) into the Add box", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(TWO_RESULTS, []) });
  await window.discoverSearch();

  const hf = rowFor(window, "org/hf-only");
  buttonByText(hf, "add full repo").click();

  assert.equal(window.document.getElementById("pull-spec").value, "org/hf-only",
    "the pull spec is the bare repo id (a bare owner/repo pulls the full HF model)");
  assert.ok(!window.document.getElementById("pull-spec").value.includes(":"),
    "no :file.gguf suffix, so `localm pull` fetches the whole transformers repo");
  assert.equal(window.document.getElementById("pull-name").value, "hf-only",
    "the suggested alias is the repo's last path segment");
});

test("discover: shows the non-blocking HF-runtime hint when HF is on but no backend", async () => {
  const payload = { ...TWO_RESULTS, hf_backend_available: false };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(payload, []) });
  await window.discoverSearch();

  const hint = window.document.getElementById("disc-hf-hint");
  assert.notEqual(hint.style.display, "none", "the hint is shown");
  assert.match(hint.textContent, /transformers|gpu/i, "the hint names the missing runtime");
  // Non-blocking: the HF row is still fully pullable.
  const hf = rowFor(window, "org/hf-only");
  assert.ok(buttonByText(hf, "add full repo"), "the HF model is still addable despite no runtime");
});

test("discover: hint stays hidden when the HF runtime IS available", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(TWO_RESULTS, []) });
  await window.discoverSearch();
  assert.equal(window.document.getElementById("disc-hf-hint").style.display, "none",
    "no hint when torch + transformers are present");
});

test("discover: unchecking HF sends only gguf", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(TWO_RESULTS, calls) });
  window.document.getElementById("disc-fmt-hf").checked = false;
  await window.discoverSearch();

  const u = decodeURIComponent(calls[0]);
  assert.ok(u.includes("formats=gguf") && !u.includes("hf"),
    `only gguf is requested once HF is toggled off (got ${u})`);
});

test("discover: no type selected prompts and does not query", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(TWO_RESULTS, calls) });
  window.document.getElementById("disc-fmt-gguf").checked = false;
  window.document.getElementById("disc-fmt-hf").checked = false;
  await window.discoverSearch();

  assert.equal(calls.length, 0, "with no format selected, no search is issued");
  assert.match(window.document.getElementById("disc-results").textContent, /at least one/i,
    "the user is told to pick a type");
});
