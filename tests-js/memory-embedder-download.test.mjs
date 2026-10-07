// SPDX-License-Identifier: AGPL-3.0-or-later
// The Memory dialog's one-time embedding-model download shows each line the
// download job reports (how much has arrived) on its button while it runs.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

test("the download button shows the job's progress lines", async () => {
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u.includes("/api/rag/embedding/download"))
      return { ok: true, status: 200, json: async () => ({ job_id: "j1", model: "bge" }) };
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl });
  runScript(window, `
    globalThis.__seen = [];
    globalThis.__btn = document.createElement("button");
    refreshMemory = async () => {};
    streamJob = async (id, onLine) => {
      for (const line of ["Downloading embedding model 'bge' (one-time)...",
                          "Downloading the embedding model bge: 12 of 24 MB (50%)...",
                          "   "]) {
        onLine(line);
        globalThis.__seen.push(globalThis.__btn.textContent);
      }
      return { status: "done" };
    };
  `);
  const ok = await window.downloadMemoryEmbedder(window.__btn);
  assert.equal(ok, true);
  assert.deepEqual(Array.from(window.__seen), [
    "Downloading embedding model 'bge' (one-time)...",
    "Downloading the embedding model bge: 12 of 24 MB (50%)...",
    "Downloading the embedding model bge: 12 of 24 MB (50%)...",
  ]);
});
