// SPDX-License-Identifier: AGPL-3.0-or-later
// A text encoder and the parts of a music model show what they are on the Models
// page, and the Knowledge page's embedding picker never offers a text encoder.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function modelsFetch(models, embedding) {
  return async (url) => {
    const u = String(url);
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models, active: null }) };
    }
    if (u === "/api/rag/embedding") {
      return { ok: true, status: 200, json: async () => embedding };
    }
    return { ok: true, status: 200, json: async () => ({ collections: [] }), text: async () => "" };
  };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

const PARTS = [
  { name: "umt5-xxl-encoder-Q4_K_M", model_type: "text-encoder", architecture: "t5encoder" },
  { name: "music_vocoder", model_type: "vocoder", architecture: "ADaMoSHiFiGANV1", part_of: "ACE-Step" },
];

test("a text encoder row says what it is used for", async () => {
  const { window } = loadAppWithPages({ fetchImpl: modelsFetch(PARTS) });
  await window.refreshModelsPage();
  const rows = [...window.document.querySelectorAll("#models-table tbody tr")];
  const row = rows.find((r) => r.querySelector(".name")?.textContent === "umt5-xxl-encoder-Q4_K_M");
  assert.ok(row, "the encoder row renders");
  assert.equal(row.querySelector(".role-hint").textContent,
    "Text encoder (used by image and video models)");
  assert.ok(row.querySelector("select.type-text-encoder"), "wears the text-encoder type");
  assert.equal(row.querySelector(".arch-badge").textContent, "t5encoder");
});

test("a music part shows its role and the model it belongs to", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: modelsFetch(PARTS),
    seedLocalStorage: { "localm.showOtherModelsInAll": "true" },
  });
  await window.refreshModelsPage();
  await tick();
  const rows = [...window.document.querySelectorAll("#models-table tbody tr")];
  const row = rows.find((r) => r.querySelector(".name")?.textContent === "music_vocoder");
  assert.ok(row, "the vocoder row renders: "
    + rows.map((r) => r.querySelector(".name")?.textContent).join(","));
  assert.ok(row.querySelector("select.type-vocoder"));
  const badges = [...row.querySelectorAll(".arch-badge")].map((b) => b.textContent);
  assert.deepEqual(badges, ["ADaMoSHiFiGANV1", "part of ACE-Step"]);
});

test("the embedding picker lists embedding models and chat models but never a text encoder", async () => {
  const models = [
    { name: "bge-small", model_type: "embedding" },
    { name: "umt5-xxl-encoder-Q4_K_M", model_type: "text-encoder" },
  ];
  const embedding = { status: "ready", model: "bge-small-en-v1.5", dim: 384,
                      internal: ["bge-small-en-v1.5"], error: null };
  const { window } = loadAppWithPages({ fetchImpl: modelsFetch(models, embedding) });
  await window.refreshEmbeddingPanel();
  const values = [...window.document.getElementById("kb-embed-model").options].map((o) => o.value);
  assert.ok(values.includes("bge-small"), "an embedding model is offered");
  assert.ok(!values.includes("umt5-xxl-encoder-Q4_K_M"), "a text encoder is not");
});
