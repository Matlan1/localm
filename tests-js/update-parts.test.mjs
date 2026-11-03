// SPDX-License-Identifier: AGPL-3.0-or-later
// The Settings > Updates "Installed parts" sub-block: the GUI form of
// `localm update --repair`, and the results of the last runtime and component updates.
//
// Revealed only when a part differs or such a result exists. Repair is offered only
// when a part differs. The mock tells the read-only probe from the repair by its url.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(parts, repair, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    let body = null;
    if (u.includes("/api/update/parts")) body = parts;
    if (u.includes("/api/update/repair")) body = repair;
    if (body !== null) {
      calls.push(`${method} ${u}`);
      const status = body.__status || 200;
      return { ok: status < 400, status, text: async () => "", json: async () => body };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

const CONSISTENT = { consistent: true, problems: [], last_steps: [] };
const DIFFERS = { consistent: false, last_steps: [],
  problems: ["the vulkan runtime is b1, but localm 0.3.3 uses b2"] };

async function flush() {
  for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0));
}

test("Installed parts: hidden when every part matches and nothing ran", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(CONSISTENT, {}, calls) });
  await window.__localmPartsCheck();
  assert.equal(window.document.getElementById("app-parts-block").hidden, true);
  assert.equal(calls.some((c) => c.includes("/repair")), false,
    "reading the parts must never repair");
});

test("Installed parts: a part that differs is listed with Repair", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(DIFFERS, {}, []) });
  const doc = window.document;
  await window.__localmPartsCheck();
  assert.equal(doc.getElementById("app-parts-block").hidden, false);
  assert.equal(doc.getElementById("update-repair-actions").hidden, false);
  assert.equal(doc.getElementById("update-parts-list").textContent, DIFFERS.problems[0]);
});

test("Installed parts: a last result alone shows without Repair", async () => {
  const parts = { consistent: true, problems: [], last_steps: [
    { part: "runtime", state: "updated", note: "re-provisioned the cpu runtime: b1 -> b2" }] };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(parts, {}, []) });
  const doc = window.document;
  await window.__localmPartsCheck();
  assert.equal(doc.getElementById("app-parts-block").hidden, false);
  assert.equal(doc.getElementById("update-repair-actions").hidden, true,
    "nothing differs: nothing to repair");
  assert.equal(doc.getElementById("update-parts-steps-head").hidden, false);
  assert.match(doc.getElementById("update-parts-steps").textContent, /b1 -> b2/);
});

test("Installed parts: a problem line is text, never markup", async () => {
  const parts = { consistent: false, last_steps: [], problems: ["<img src=x onerror=1>"] };
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(parts, {}, []) });
  const doc = window.document;
  await window.__localmPartsCheck();
  assert.equal(doc.querySelector("#update-parts-list img"), null);
  assert.equal(doc.getElementById("update-parts-list").textContent, parts.problems[0]);
});

test("Repair: a restart hides the button and says so", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch(DIFFERS, { repaired: true, notes: [], restarting: true }, calls) });
  const doc = window.document;
  await window.__localmPartsCheck();
  doc.getElementById("update-repair").click();
  await flush();
  assert.ok(calls.some((c) => c.startsWith("POST") && c.includes("/api/update/repair")));
  assert.equal(doc.getElementById("update-repair").hidden, true);
  assert.match(doc.getElementById("update-repair-status").textContent, /Restarting/);
});

test("Repair: a failed step is shown and the button stays usable", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(DIFFERS,
    { repaired: false, error: "uv exited 3" }, []) });
  const doc = window.document;
  await window.__localmPartsCheck();
  doc.getElementById("update-repair").click();
  await flush();
  assert.match(doc.getElementById("update-repair-status").textContent, /uv exited 3/);
  assert.equal(doc.getElementById("update-repair").disabled, false);
});
