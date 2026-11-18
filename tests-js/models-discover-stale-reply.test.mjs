// SPDX-License-Identifier: AGPL-3.0-or-later
// Models page search: a reply to an older search never overwrites the page.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function deferredFetch(pending) {
  return (url, opts = {}) => {
    const u = String(url);
    if (!u.includes("/api/discover/search")) {
      return Promise.resolve({ ok: true, status: 200, json: async () => ({}), text: async () => "" });
    }
    return new Promise((resolve, reject) => {
      const entry = { url: decodeURIComponent(u), aborted: false, resolve, reject };
      if (opts.signal) {
        opts.signal.addEventListener("abort", () => {
          entry.aborted = true;
          const err = new Error("aborted");
          err.name = "AbortError";
          reject(err);
        });
      }
      pending.push(entry);
    });
  };
}

function hfReply(id) {
  return {
    ok: true, status: 200,
    json: async () => ({
      query: "", source: "hf", vram: { total: 0 }, hf_backend_available: true,
      results: [{ id, downloads: 1, likes: 0, formats: ["gguf"] }],
    }),
    text: async () => "",
  };
}

const names = (win) => [...win.document.querySelectorAll("#disc-results .disc-repo .name")]
  .map((n) => n.textContent);

test("a late reply of an older search cannot replace the newest results", async () => {
  const pending = [];
  const { window } = loadAppWithPages({ fetchImpl: deferredFetch(pending) });
  const first = window.discoverSearch();
  const second = window.discoverSearch();
  assert.equal(pending.length, 2);
  assert.equal(pending[0].aborted, true, "starting a search aborts the previous request");
  pending[1].resolve(hfReply("new/repo"));
  await second;
  pending[0].resolve(hfReply("old/repo"));
  await first;
  assert.deepEqual(names(window), ["new/repo"]);
  assert.equal(window.document.getElementById("disc-search").disabled, false);
});

test("an older reply that arrives while the newest is pending leaves the page and the button alone", async () => {
  const pending = [];
  const { window } = loadAppWithPages({ fetchImpl: deferredFetch(pending) });
  const first = window.discoverSearch();
  const second = window.discoverSearch();
  pending[0].resolve(hfReply("old/repo"));
  await first;
  assert.deepEqual(names(window), []);
  assert.equal(window.document.getElementById("disc-search").disabled, true,
    "Search stays disabled until the newest search answers");
  pending[1].resolve(hfReply("new/repo"));
  await second;
  assert.deepEqual(names(window), ["new/repo"]);
});

test("changing the source drops the running HF search", async () => {
  const pending = [];
  const { window } = loadAppWithPages({ fetchImpl: deferredFetch(pending) });
  const running = window.discoverSearch();
  const sel = window.document.getElementById("disc-source");
  sel.value = "civitai";
  sel.dispatchEvent(new window.Event("change"));
  pending[0].resolve(hfReply("hf/repo"));
  await running;
  assert.deepEqual(names(window), []);
  assert.equal(window.document.getElementById("disc-search").disabled, false);
});

test("ticking no format cancels the running search and clears the VRAM caption", async () => {
  const pending = [];
  const { window } = loadAppWithPages({ fetchImpl: deferredFetch(pending) });
  window.document.getElementById("disc-vram").textContent = "stale caption";
  const running = window.discoverSearch();
  for (const id of ["disc-fmt-gguf", "disc-fmt-hf"]) window.document.getElementById(id).checked = false;
  await window.discoverSearch();
  pending[0].resolve(hfReply("old/repo"));
  await running;
  assert.deepEqual(names(window), []);
  assert.equal(window.document.getElementById("disc-vram").textContent, "");
  assert.equal(window.document.getElementById("disc-search").disabled, false);
});
