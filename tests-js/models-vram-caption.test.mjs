// SPDX-License-Identifier: AGPL-3.0-or-later
// Models search-page VRAM caption: vramBasisCaption(totalBytes, gpuInfo) in
// pages/models.js names what the fit-badge VRAM number is - a single GPU, the
// main GPU, or a combined split.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const GIB = 1024 ** 3;
const okFetch = async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => "" });

function caption() {
  const { window } = loadAppWithPages({ fetchImpl: okFetch });
  return window.vramBasisCaption;
}

test("vramBasisCaption: single GPU -> plain 'your N GB VRAM', never 'total'", () => {
  const cap = caption();
  const msg = cap(16 * GIB, { gpus: [{ index: 0, total: 16 * GIB, free: 14 * GIB }], gpu_split_indices: [] });
  assert.match(msg, /your 16 GB VRAM/);
  assert.doesNotMatch(msg, /total VRAM/, "a single-GPU ceiling must not be called the machine total");
  assert.doesNotMatch(msg, /main GPU/, "with one GPU there is no 'main' to distinguish");
});

test("vramBasisCaption: multi-GPU, NO split configured -> names the main GPU, not the machine total", () => {
  const cap = caption();
  // With no split, vram_capacity() returns the single main GPU's 16 GB.
  const msg = cap(16 * GIB, {
    gpus: [{ index: 0, total: 16 * GIB, free: 14 * GIB }, { index: 1, total: 16 * GIB, free: 15 * GIB }],
    gpu_split_indices: [],
  });
  assert.match(msg, /your main GPU's 16 GB/);
  assert.match(msg, /use all 2/, "invites configuring a split to combine both cards");
  assert.doesNotMatch(msg, /total VRAM/);
});

test("vramBasisCaption: 2-GPU split configured -> 'combined across 2 GPUs'", () => {
  const cap = caption();
  // With a split, vram_capacity() returns the COMBINED total, so the caption
  // says combined - matching what the number actually is.
  const msg = cap(32 * GIB, {
    gpus: [{ index: 0, total: 16 * GIB, free: 14 * GIB }, { index: 1, total: 16 * GIB, free: 15 * GIB }],
    gpu_split_indices: [0, 1],
  });
  assert.match(msg, /your 32 GB VRAM combined across 2 GPUs/);
});

test("vramBasisCaption: split indices that don't map to detected GPUs fall back to main GPU", () => {
  const cap = caption();
  // A stale split (indices point at devices no longer present): vram_capacity()
  // falls back to the single main GPU, so the caption must not claim 'combined'.
  const msg = cap(16 * GIB, {
    gpus: [{ index: 0, total: 16 * GIB, free: 14 * GIB }, { index: 1, total: 16 * GIB, free: 15 * GIB }],
    gpu_split_indices: [5, 6],
  });
  assert.doesNotMatch(msg, /combined/, "an unresolvable split is not a combined total");
  assert.match(msg, /main GPU's 16 GB/);
});

test("vramBasisCaption: duplicate split indices dedup like resolve_gpu_split -> not 'combined'", () => {
  const cap = caption();
  // A hand-edited [0, 0] is not a 2-device split: vram_capacity() dedups via
  // resolve_gpu_split and returns the single main GPU, so the caption must too.
  const msg = cap(16 * GIB, {
    gpus: [{ index: 0, total: 16 * GIB }, { index: 1, total: 16 * GIB }],
    gpu_split_indices: [0, 0],
  });
  assert.doesNotMatch(msg, /combined/, "a duplicated index is one device, not a combined split");
  assert.match(msg, /main GPU's 16 GB/);
});

test("vramBasisCaption: empty/failed gpu info -> plain caption, never crashes", () => {
  const cap = caption();
  assert.match(cap(8 * GIB, { gpus: [], gpu_split_indices: [] }), /your 8 GB VRAM/);
  assert.match(cap(8 * GIB, undefined), /your 8 GB VRAM/, "tolerates missing gpu info");
});
