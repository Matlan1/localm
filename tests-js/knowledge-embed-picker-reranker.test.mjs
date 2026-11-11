// SPDX-License-Identifier: AGPL-3.0-or-later
// The Knowledge page's embedding picker offers registered embedding models but
// never a reranker, which /api/models marks with reranker: true.

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

test("the embedding picker lists an embedding model and not a reranker", async () => {
  const models = [
    { name: "embed-bge", model_type: "embedding" },
    { name: "rerank-q3", model_type: "embedding", reranker: true },
  ];
  const embedding = { status: "ready", model: "bge-small-en-v1.5", dim: 384,
                      internal: ["bge-small-en-v1.5"], error: null };
  const { window } = loadAppWithPages({ fetchImpl: modelsFetch(models, embedding) });
  await window.refreshEmbeddingPanel();
  const values = [...window.document.getElementById("kb-embed-model").options].map((o) => o.value);
  assert.ok(values.includes("embed-bge"), "the embedding model is offered: " + values.join(","));
  assert.ok(!values.includes("rerank-q3"), "the reranker is not offered: " + values.join(","));
});
