// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(calls, state) {
  // state.pending is the mutable pending-corrections list; GET reflects it and a
  // resolve POST removes the addressed item
  return async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    calls.push({ u, method });
    if (u === "/api/memory" && method === "GET") {
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ text: "- User lives in Berlin", writable: true,
          items: [], corrections: state.pending.slice() }) };
    }
    const m = u.match(/^\/api\/memory\/corrections\/([^/]+)\/(accept|reject)$/);
    if (m && method === "POST") {
      state.pending = state.pending.filter((c) => c.id !== m[1]);
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ status: m[2] === "accept" ? "updated" : "rejected" }) };
    }
    return { ok: true, status: 200, text: async () => "", json: async () => ({}) };
  };
}

async function flush() {
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
}

function modalButtons(window) {
  const body = window.document.getElementById("modal-body");
  return [...body.querySelectorAll("button")];
}

test("F12b: modal renders a pending correction and Apply POSTs accept", async () => {
  const calls = [];
  const state = { pending: [{ id: "abc123", action: "update",
    target_text: "User lives in Berlin", proposed_text: "User moved to Munich",
    confidence: 0.9, target_updated: Date.now() / 1000 - 86400 * 5 }] };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(calls, state) });
  await window.refreshMemory();
  window.openMemoryModal();

  const body = window.document.getElementById("modal-body");
  assert.match(body.textContent, /User lives in Berlin/);   // was
  assert.match(body.textContent, /User moved to Munich/);   // now
  assert.match(body.textContent, /last confirmed/);         // staleness affordance

  const apply = modalButtons(window).find((b) => b.textContent === "Apply");
  assert.ok(apply, "Apply button present");
  await apply.onclick();
  await flush();
  assert.ok(calls.find((c) =>
    c.u === "/api/memory/corrections/abc123/accept" && c.method === "POST"),
    "posted accept");
  // after applying, the suggestion is gone from the re-rendered list
  assert.doesNotMatch(
    window.document.getElementById("modal-body").textContent, /Suggested corrections/);
});

test("F12b: Keep as is POSTs reject", async () => {
  const calls = [];
  const state = { pending: [{ id: "xy9", action: "update",
    target_text: "User lives in Berlin", proposed_text: "User moved to Munich",
    confidence: 0.9, target_updated: 0 }] };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(calls, state) });
  await window.refreshMemory();
  window.openMemoryModal();

  const reject = modalButtons(window).find((b) => b.textContent === "Keep as is");
  assert.ok(reject, "Reject button present");
  await reject.onclick();
  await flush();
  assert.ok(calls.find((c) =>
    c.u === "/api/memory/corrections/xy9/reject" && c.method === "POST"),
    "posted reject");
});

test("F12b: a delete proposal renders as Forget it", async () => {
  const state = { pending: [{ id: "del1", action: "delete",
    target_text: "User lives in Berlin", proposed_text: "", confidence: 0.9,
    target_updated: 0 }] };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([], state) });
  await window.refreshMemory();
  window.openMemoryModal();

  const body = window.document.getElementById("modal-body");
  assert.match(body.textContent, /suggest forgetting it/);
  assert.ok(modalButtons(window).some((b) => b.textContent === "Forget it"),
    "Forget it button present");
});

test("F12b: no corrections -> no suggestions section", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([], { pending: [] }) });
  await window.refreshMemory();
  window.openMemoryModal();
  const body = window.document.getElementById("modal-body");
  assert.doesNotMatch(body.textContent, /Suggested corrections/);
});
