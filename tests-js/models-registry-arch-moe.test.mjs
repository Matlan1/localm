// SPDX-License-Identifier: AGPL-3.0-or-later
// The local (registered) model list renders architecture and MoE badges from
// the file-header metadata captured at registration or pull time. Local
// metadata is always confirmed, so there is no "likely" tier here. Covers
// refreshModelsPage's rendering only, not the server-side fields.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(models) {
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

test("registry-arch-moe: a confirmed MoE model shows architecture + a solid MoE badge", async () => {
  const models = [{ name: "big-moe", active: false, loaded: false, model_type: "llm",
    architecture: "qwen3moe", expert_count: 8 }];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models) });
  await window.refreshModelsPage();
  const row = window.document.querySelector("#models-table tbody tr");
  const arch = row.querySelector(".arch-badge");
  assert.ok(arch, "architecture badge renders");
  assert.equal(arch.textContent, "qwen3moe");
  const moe = row.querySelector(".moe-badge");
  assert.ok(moe, "MoE badge renders");
  assert.ok(moe.classList.contains("moe-confirmed"), "local data is always hard metadata - confirmed, never likely");
  assert.equal(moe.textContent, "MoE", "a confirmed MoE badge reads plainly, no '?'");
});

test("registry-arch-moe: a confirmed DENSE model (expert_count: 0) shows architecture but no MoE badge", async () => {
  // expert_count 0 is a confirmed answer and renders as no badge, the same as
  // no evidence, never a positive "Dense" claim.
  const models = [{ name: "plain-llama", active: false, loaded: false, model_type: "llm",
    architecture: "llama", expert_count: 0 }];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models) });
  await window.refreshModelsPage();
  const row = window.document.querySelector("#models-table tbody tr");
  assert.ok(row.querySelector(".arch-badge"), "architecture still shows on its own");
  assert.equal(row.querySelector(".moe-badge"), null, "expert_count: 0 shows no MoE badge, not a false claim");
});

test("registry-arch-moe: a legacy entry (no architecture/expert_count keys at all) shows neither badge", async () => {
  // An entry registered before these keys existed carries neither, and the API
  // sends that through as null or absent rather than a default 0 or "".
  const models = [{ name: "old-model", active: false, loaded: false, model_type: "llm" }];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models) });
  await window.refreshModelsPage();
  const row = window.document.querySelector("#models-table tbody tr");
  assert.equal(row.querySelector(".arch-badge"), null, "unknown architecture renders no badge, not a guess");
  assert.equal(row.querySelector(".moe-badge"), null, "unknown expert_count renders no badge, never 'not MoE'");
  assert.ok(row.querySelector(".name").textContent === "old-model",
    "the row itself still renders fully - these badges are purely additive");
});

test("registry-arch-moe: an explicit null architecture/expert_count (JSON round-trip of a missing key) also shows neither badge", async () => {
  const models = [{ name: "explicit-null", active: false, loaded: false, model_type: "llm",
    architecture: null, expert_count: null }];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models) });
  await window.refreshModelsPage();
  const row = window.document.querySelector("#models-table tbody tr");
  assert.equal(row.querySelector(".arch-badge"), null);
  assert.equal(row.querySelector(".moe-badge"), null);
});

test("registry-arch-moe: badges never block the row's other controls (active/loaded tags, actions)", async () => {
  const models = [{ name: "active-moe", active: true, loaded: true, model_type: "llm",
    architecture: "mixtral", expert_count: 8 }];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(models) });
  await window.refreshModelsPage();
  const row = window.document.querySelector("#models-table tbody tr");
  assert.ok(row.querySelector(".arch-badge"), "arch badge present");
  assert.ok(row.querySelector(".moe-badge"), "moe badge present");
  assert.ok(row.querySelector(".active-tag"), "active tag still renders alongside the new badges");
  assert.ok(row.querySelector("select.model-type-select"), "the type control still renders");
});
