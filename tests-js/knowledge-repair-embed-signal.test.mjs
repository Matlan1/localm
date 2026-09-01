// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// kbRepairCollection() must send an accurate `embed` value reflecting real
// embedder availability rather than always relying on the server's `embed=True`
// default, and the "Repair anyway" confirm dialog must actually be reachable
// when the server reports needs_confirm.

const tick = () => new Promise((r) => setTimeout(r, 0));

function embeddingStatus(status) {
  return { status, model: "bge-small-en-v1.5", dim: null, internal: [], error: null };
}

function setup({ embeddingStatusBody, repairResponses }) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    if (u.includes("/api/rag/embedding") && (opts.method || "GET") === "GET")
      return { ok: true, status: 200, text: async () => "", json: async () => embeddingStatusBody };
    if (/\/repair$/.test(u)) {
      const body = opts.body ? JSON.parse(opts.body) : {};
      const resp = body.confirm ? repairResponses.confirmed : repairResponses.initial;
      return { ok: true, status: 200, text: async () => "", json: async () => resp };
    }
    return { ok: true, status: 200, text: async () => "", json: async () => ({ collections: [] }) };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  runScript(window, `streamJob = () => Promise.resolve({ status: "done" });`);
  return { window, calls };
}

const repairCalls = (calls) => calls.filter((c) => /\/repair$/.test(c.url));

test("embedder ready: repair sends embed:true", async () => {
  const { window, calls } = setup({
    embeddingStatusBody: embeddingStatus("ready"),
    repairResponses: { initial: { job_id: "j1" } },
  });
  runScript(window, "kbRepairCollection('kb');");
  for (let i = 0; i < 8; i++) await tick();

  const hits = repairCalls(calls);
  assert.equal(hits.length, 1, "exactly one /repair POST - not needed here");
  assert.equal(JSON.parse(hits[0].opts.body).embed, true);
});

test("embedder status unknown (withheld for a non-owner key): repair still sends embed:true", async () => {
  // A false claim here would force the server to skip embedding even when it
  // is actually available server-side, since the server cannot overrule an
  // explicit embed:false from the caller.
  const { window, calls } = setup({
    embeddingStatusBody: embeddingStatus("unknown"),
    repairResponses: { initial: { job_id: "j1" } },
  });
  runScript(window, "kbRepairCollection('kb');");
  for (let i = 0; i < 8; i++) await tick();

  const hits = repairCalls(calls);
  assert.equal(hits.length, 1);
  assert.equal(JSON.parse(hits[0].opts.body).embed, true);
});

test("embedder status fetch fails: repair still sends embed:true", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    if (u.includes("/api/rag/embedding") && (opts.method || "GET") === "GET")
      throw new Error("network down");
    if (/\/repair$/.test(u))
      return { ok: true, status: 200, text: async () => "", json: async () => ({ job_id: "j1" }) };
    return { ok: true, status: 200, text: async () => "", json: async () => ({ collections: [] }) };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  runScript(window, `streamJob = () => Promise.resolve({ status: "done" });`);
  runScript(window, "kbRepairCollection('kb');");
  for (let i = 0; i < 8; i++) await tick();

  const hits = repairCalls(calls);
  assert.equal(hits.length, 1);
  assert.equal(JSON.parse(hits[0].opts.body).embed, true);
});

test("embedder confirmed not_installed: repair sends embed:false on both the initial and confirmed POST", async () => {
  const { window, calls } = setup({
    embeddingStatusBody: embeddingStatus("not_installed"),
    repairResponses: {
      initial: { needs_confirm: true, detail: "will remove existing embeddings" },
      confirmed: { job_id: "j1" },
    },
  });
  runScript(window, `kbConfirmRepair = () => Promise.resolve(true);`);
  runScript(window, "kbRepairCollection('kb');");
  for (let i = 0; i < 8; i++) await tick();

  const hits = repairCalls(calls);
  assert.equal(hits.length, 2, "the dry-run POST, then the confirmed POST");
  assert.equal(JSON.parse(hits[0].opts.body).embed, false);
  assert.equal(JSON.parse(hits[0].opts.body).confirm, undefined);
  assert.equal(JSON.parse(hits[1].opts.body).embed, false);
  assert.equal(JSON.parse(hits[1].opts.body).confirm, true);
});

test("needs_confirm actually reaches the user: the 'Repair anyway' dialog renders and blocks the repair", async () => {
  // This is the GUI-reachability bug itself: before the fix, the server never
  // returned needs_confirm to the GUI, so this dialog was dead code.
  const { window, calls } = setup({
    embeddingStatusBody: embeddingStatus("not_installed"),
    repairResponses: {
      initial: { needs_confirm: true,
                 detail: "'kb' currently has semantic (hybrid) search. Repairing "
                       + "without an embedding model available will remove the "
                       + "existing embeddings." },
      confirmed: { job_id: "j1" },
    },
  });
  runScript(window, "kbRepairCollection('kb');");
  await tick(); await tick(); await tick();

  const modal = window.document.getElementById("modal");
  assert.notEqual(modal.style.display, "none", "the confirm dialog is actually shown");
  assert.match(modal.textContent, /without an embedding model available/);
  assert.equal(repairCalls(calls).length, 1, "no confirmed POST before the user answers");

  const ok = [...modal.querySelectorAll("button")].find((b) => b.textContent === "Repair anyway");
  assert.ok(ok, "a 'Repair anyway' button is present");
  ok.onclick();
  for (let i = 0; i < 8; i++) await tick();

  const hits = repairCalls(calls);
  assert.equal(hits.length, 2, "confirming sends the confirmed POST and the repair proceeds");
  assert.equal(JSON.parse(hits[1].opts.body).confirm, true);
});

test("cancelling the 'Repair anyway' dialog sends no confirmed POST", async () => {
  const { window, calls } = setup({
    embeddingStatusBody: embeddingStatus("not_installed"),
    repairResponses: {
      initial: { needs_confirm: true, detail: "will remove existing embeddings" },
      confirmed: { job_id: "j1" },
    },
  });
  runScript(window, "kbRepairCollection('kb');");
  await tick(); await tick(); await tick();

  const modal = window.document.getElementById("modal");
  const cancel = [...modal.querySelectorAll("button")].find((b) => b.textContent === "Cancel");
  assert.ok(cancel, "a 'Cancel' button is present");
  cancel.onclick();
  for (let i = 0; i < 8; i++) await tick();

  assert.equal(repairCalls(calls).length, 1, "cancelling never sends the confirmed POST");
  const log = window.document.getElementById("kb-log");
  assert.match(log.textContent, /Cancelled/);
});
