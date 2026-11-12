// SPDX-License-Identifier: AGPL-3.0-or-later
// The memory dialog opens on a fresh copy of memory and its Save sends the
// text it showed as the base, so the server keeps facts added meanwhile.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(calls, state) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    let body = null;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = opts.body; }
    calls.push({ u, method, body });
    if (u === "/api/memory" && method === "GET") {
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ text: state.text, writable: true, items: [], corrections: [] }) };
    }
    if (u === "/api/memory" && method === "PUT") {
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ status: "saved", count: 2, kept_newer: state.keptNewer }) };
    }
    return { ok: true, status: 200, text: async () => "", json: async () => ({}) };
  };
}

const flush = async () => { for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0)); };

test("the dialog shows memory as it is when opened, not the page-load copy", async () => {
  const calls = [];
  const state = { text: "- old fact", keptNewer: 0 };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(calls, state) });
  await window.refreshMemory();
  state.text = "- old fact\n- fact the background pass added";
  await window.openMemoryModal();
  const ta = window.document.querySelector("#modal-body textarea");
  assert.equal(ta.value, "- old fact\n- fact the background pass added");
});

test("Save sends the text the dialog showed as its base", async () => {
  const calls = [];
  const state = { text: "- old fact", keptNewer: 1 };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(calls, state) });
  await window.openMemoryModal();
  const ta = window.document.querySelector("#modal-body textarea");
  ta.value = "- old fact\n- typed fact";
  state.text = "- old fact\n- newer fact";      // changes on the server after opening
  const save = [...window.document.querySelectorAll("#modal-body button")]
    .find((b) => b.className.includes("btn-primary"));
  save.click();
  await flush();
  const put = calls.find((c) => c.method === "PUT" && c.u === "/api/memory");
  assert.ok(put, "Save sent a PUT");
  assert.deepEqual(put.body, { text: "- old fact\n- typed fact", base: "- old fact" });
  assert.match(window.document.getElementById("toast").textContent, /1 fact\(s\) added since/);
});
