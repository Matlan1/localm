// SPDX-License-Identifier: AGPL-3.0-or-later
// The model list and the detail modal render a "vision" pill from
// registry.model_vision_capability. The server sends true, false, or no key at
// all, since the answer is measured from the model's own files on every request
// and an unreachable path yields no evidence. Both false and absent render
// nothing. Covers the rendering only.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeListFetch(models) {
  return async (url) => {
    const u = String(url);
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return {
        ok: true, status: 200,
        json: async () => ({ models, active: models.find((m) => m.active)?.name || null }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

function makeDetailFetch(detail) {
  return async (url) => {
    if (String(url).startsWith("/v1/models/")) {
      return { ok: true, status: 200, json: async () => detail, text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

// A detail payload carrying every field showModelDetail reads, so a test can
// vary only the field under test.
function detailFor(extra) {
  return Object.assign({
    id: "m1", object: "model", owned_by: "localm", path: "m1.gguf",
    source: "local", sha256: null, size_bytes: 10, aliases: [],
    active: false, loaded: false, model_type: "llm",
  }, extra);
}

async function drain() {
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
}

/* ------------------------------- the list ------------------------------- */

test("vision-pill list: vision:true renders exactly one .cap-badge.cap-vision", async () => {
  const models = [{ name: "gemma-multimodal", active: false, loaded: false,
    model_type: "llm", vision: true }];
  const { window } = loadAppWithPages({ fetchImpl: makeListFetch(models) });
  await window.refreshModelsPage();
  const row = window.document.querySelector("#models-table tbody tr");
  const pills = row.querySelectorAll(".cap-badge.cap-vision");
  assert.equal(pills.length, 1, "exactly one vision pill, not zero and not two");
  assert.equal(pills[0].textContent, "vision");
});

test("vision-pill list: vision:false renders no pill (checked, and it cannot)", async () => {
  const models = [{ name: "plain-chat", active: false, loaded: false,
    model_type: "llm", vision: false }];
  const { window } = loadAppWithPages({ fetchImpl: makeListFetch(models) });
  await window.refreshModelsPage();
  const row = window.document.querySelector("#models-table tbody tr");
  assert.equal(row.querySelector(".cap-badge"), null, "a confirmed non-vision model gets no pill");
});

test("vision-pill list: the key ABSENT renders no pill AND no negative text", async () => {
  // The unreachable-path case: renders the same as vision:false, and carries no
  // "no vision" or "text-only" label.
  const models = [{ name: "unreachable-entry", active: false, loaded: false,
    model_type: "llm" }];
  const { window } = loadAppWithPages({ fetchImpl: makeListFetch(models) });
  await window.refreshModelsPage();
  const row = window.document.querySelector("#models-table tbody tr");
  assert.equal(row.querySelector(".cap-badge"), null, "unknown renders no pill, never a guess");
  assert.ok(!/vision/i.test(row.innerHTML),
    "unknown emits no vision markup or text at all - not even a negative one");
  assert.equal(row.querySelector(".name").textContent, "unreachable-entry",
    "the row itself still renders fully - the pill is purely additive");
});

test("vision-pill list: an explicit null (a JSON round-trip of a missing key) also renders nothing", async () => {
  const models = [{ name: "explicit-null", active: false, loaded: false,
    model_type: "llm", vision: null }];
  const { window } = loadAppWithPages({ fetchImpl: makeListFetch(models) });
  await window.refreshModelsPage();
  const row = window.document.querySelector("#models-table tbody tr");
  assert.equal(row.querySelector(".cap-badge"), null);
  assert.ok(!/vision/i.test(row.innerHTML));
});

test("vision-pill list: the pill does not displace the row's other badges or controls", async () => {
  const models = [{ name: "active-vlm", active: true, loaded: true, model_type: "llm",
    architecture: "qwen2vl", expert_count: 8, vision: true }];
  const { window } = loadAppWithPages({ fetchImpl: makeListFetch(models) });
  await window.refreshModelsPage();
  const row = window.document.querySelector("#models-table tbody tr");
  assert.ok(row.querySelector(".cap-badge.cap-vision"), "vision pill present");
  assert.ok(row.querySelector(".arch-badge"), "architecture badge still renders");
  assert.ok(row.querySelector(".moe-badge"), "MoE badge still renders");
  assert.ok(row.querySelector(".active-tag"), "active tag still renders");
  assert.ok(row.querySelector("select.model-type-select"), "the type control still renders");
});

/* --------------------------- the detail modal --------------------------- */

test("vision-pill detail: vision:true renders exactly one .cap-badge.cap-vision", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeDetailFetch(detailFor({ vision: true })) });
  await window.showModelDetail("m1");
  await drain();
  const body = window.document.getElementById("modal-body");
  const pills = body.querySelectorAll(".cap-badge.cap-vision");
  assert.equal(pills.length, 1, "exactly one vision pill in the modal");
  assert.equal(pills[0].textContent, "vision");
});

test("vision-pill detail: vision:false renders no pill", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeDetailFetch(detailFor({ vision: false })) });
  await window.showModelDetail("m1");
  await drain();
  const body = window.document.getElementById("modal-body");
  assert.equal(body.querySelector(".cap-badge"), null);
});

test("vision-pill detail: the key ABSENT renders no pill AND no negative text", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeDetailFetch(detailFor({})) });
  await window.showModelDetail("m1");
  await drain();
  const body = window.document.getElementById("modal-body");
  assert.equal(body.querySelector(".cap-badge"), null, "unknown renders no pill");
  assert.ok(!/vision/i.test(body.innerHTML),
    "unknown emits no vision markup or text at all in the modal either");
  assert.ok(/Aliases/.test(body.textContent),
    "the modal itself still renders fully - the pill is purely additive");
});
