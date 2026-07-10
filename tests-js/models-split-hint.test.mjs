// SPDX-License-Identifier: AGPL-3.0-or-later
// Models page multi-GPU split hint: splitFitHint(sizeBytes, gpus) in
// pages/models.js is a pure, non-blocking client-side estimate shown next to a
// model that would not fit on the single largest detected GPU but would fit
// spread across all of them (see Settings > "Split across GPUs"). It never
// auto-enables anything - it only surfaces a hint pointing at that control.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const GIB = 1024 ** 3;
const okFetch = async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => "" });

// splitFitHint is a pure function (no DOM/fetch touched), but it is shipped as
// an ES module that imports sibling app/*.js modules referencing `window` at
// load time, so it can only be evaluated inside the jsdom realm - not via a
// bare Node import. Load it once via the standard harness and call it directly
// off window, same as every other pages/*.js export under test elsewhere.
function hint() {
  const { window } = loadAppWithPages({ fetchImpl: okFetch });
  return window.splitFitHint;
}

test("splitFitHint: already fits the single largest GPU (with overhead) -> no hint", () => {
  const splitFitHint = hint();
  const gpus = [
    { index: 0, name: "GPU A", total: 24 * GIB, free: 20 * GIB },
    { index: 1, name: "GPU B", total: 12 * GIB, free: 10 * GIB },
  ];
  // 10 GB needs ~12.5 GB (weights*1.10 + 1.5 GB overhead) - still under the 20 GB
  // free on the largest GPU, so no split hint.
  assert.equal(splitFitHint(10 * GIB, gpus), "", "10 GB (need ~12.5 GB) fits the 20 GB-free GPU alone");
  // The hint uses the loader's need-math, not a raw byte compare: a 20 GB model
  // needs ~23.5 GB with overhead, which does NOT fit the 20 GB GPU alone but fits
  // the 30 GB combined - so it DOES hint (a raw size<=free compare would have
  // wrongly said it fits one GPU).
  const msg = splitFitHint(20 * GIB, gpus);
  assert.notEqual(msg, "", "20 GB needs ~23.5 GB with overhead - does not fit one 20 GB GPU");
  assert.match(msg, /may fit split/, "the split claim is hedged, not a promise");
});

test("splitFitHint: exceeds the largest GPU but fits the combined free VRAM -> hedged hint", () => {
  const splitFitHint = hint();
  const gpus = [
    { index: 0, name: "GPU A", total: 24 * GIB, free: 20 * GIB },
    { index: 1, name: "GPU B", total: 12 * GIB, free: 10 * GIB },
  ];
  // 25 GB needs ~29 GB (with overhead) > 20 GB largest, but <= 30 GB combined.
  const msg = splitFitHint(25 * GIB, gpus);
  assert.notEqual(msg, "", "a hint is produced when it only fits split");
  assert.match(msg, /may fit split across your 2 GPUs/, "hedged, and names the GPU count");
  assert.match(msg, /Split across GPUs/, "points at the split control that enables a split");
  assert.doesNotMatch(msg, /Main GPU/, "must NOT point at the single-device Main GPU control");
});

test("splitFitHint: exceeds even the combined total across every GPU -> no hint", () => {
  const splitFitHint = hint();
  const gpus = [
    { index: 0, name: "GPU A", total: 24 * GIB, free: 20 * GIB },
    { index: 1, name: "GPU B", total: 12 * GIB, free: 10 * GIB },
  ];
  // 40 GB exceeds the 30 GB combined free across both GPUs.
  assert.equal(splitFitHint(40 * GIB, gpus), "", "nothing fits even split, so no false hope");
});

test("splitFitHint: fewer than 2 GPUs -> no hint", () => {
  const splitFitHint = hint();
  const gpus = [{ index: 0, name: "Solo GPU", total: 16 * GIB, free: 12 * GIB }];
  assert.equal(splitFitHint(20 * GIB, gpus), "", "a single GPU has nothing to split across");
  assert.equal(splitFitHint(20 * GIB, []), "", "an empty GPU list also yields no hint");
});

test("splitFitHint: falsy sizeBytes -> no hint", () => {
  const splitFitHint = hint();
  const gpus = [
    { index: 0, name: "GPU A", total: 24 * GIB, free: 20 * GIB },
    { index: 1, name: "GPU B", total: 12 * GIB, free: 10 * GIB },
  ];
  assert.equal(splitFitHint(0, gpus), "", "size 0 (unknown) is treated as falsy");
  assert.equal(splitFitHint(undefined, gpus), "", "an undefined size yields no hint");
  assert.equal(splitFitHint(null, gpus), "", "a null size yields no hint");
});

test("splitFitHint: GPUs reporting only 'total' (no 'free') fall back to total", () => {
  const splitFitHint = hint();
  // No `free` field at all - the function falls back to `total` per GPU.
  const gpus = [
    { index: 0, name: "GPU A", total: 24 * GIB },
    { index: 1, name: "GPU B", total: 12 * GIB },
  ];
  // Fits the largest total (24 GB) alone -> no hint.
  assert.equal(splitFitHint(20 * GIB, gpus), "", "fits the largest GPU's total capacity alone");
  // Exceeds the largest (24 GB) but fits the combined total (36 GB) -> a hint.
  const msg = splitFitHint(30 * GIB, gpus);
  assert.notEqual(msg, "", "falls back to total-VRAM sums when free is absent");
  assert.match(msg, /split across your 2 GPUs/);
  // Exceeds even the combined total (36 GB) -> no hint.
  assert.equal(splitFitHint(40 * GIB, gpus), "", "exceeds the combined total capacity too");
});
