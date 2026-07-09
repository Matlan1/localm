// SPDX-License-Identifier: AGPL-3.0-or-later
// scan-reason (audit finding E): the Models page "Scan ComfyUI Models" button
// must surface WHY an empty scan found nothing instead of a bare "Added 0".
//
// The scanner returns a ScanResult whose `method` carries the reason on an empty
// scan ("none (comfy_workdir not configured)", "none (models folder not found
// under {path})"). The endpoint already exposes `method`; the old handler dropped
// it and always toasted "Added 0 models, skipped 0 existing", hiding a
// misconfiguration (AGENTS.md rule 5). This drives the real button + fetch + toast
// path and asserts the humanized reason shows, and that the internal
// folder-walk/hybrid method jargon never leaks into the toast.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

// Flush enough microtask/macrotask turns for the async click handler's
// fetch -> json -> toast -> refreshModelsPage chain to settle.
const flush = async () => {
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
};

function makeScanFetch(scanResult) {
  return async (url) => {
    const u = String(url);
    if (u.startsWith("/api/models/scan")) {
      return { ok: true, status: 200, json: async () => scanResult, text: async () => "" };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models: [], active: null }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

async function clickScan(scanResult) {
  const { window } = loadAppWithPages({ fetchImpl: makeScanFetch(scanResult) });
  const btn = window.document.getElementById("models-scan-btn");
  assert.ok(btn, "the Scan ComfyUI Models button exists");
  btn.click();
  await flush();
  return window.document.getElementById("toast").textContent;
}

test("scan-reason: an unconfigured ComfyUI workdir surfaces the config hint, not 'Added 0'", async () => {
  const msg = await clickScan({ added: 0, skipped: 0, method: "none (comfy_workdir not configured)" });
  // Negative case: the old bare "Added 0" text must be gone.
  assert.ok(!/Added 0/.test(msg), `must not show a bare "Added 0" (got: ${msg})`);
  // The humanized reason must point the user at where to fix it.
  assert.match(msg, /workdir/i, `must name the missing workdir (got: ${msg})`);
  assert.match(msg, /Settings/i, `must point the user to Settings (got: ${msg})`);
});

test("scan-reason: a missing models folder names the path it looked under", async () => {
  const path = "D:\\path\\to\\comfy";
  const msg = await clickScan({
    added: 0, skipped: 0,
    method: `none (models folder not found under ${path})`,
  });
  assert.ok(!/Added 0/.test(msg), `must not show a bare "Added 0" (got: ${msg})`);
  assert.match(msg, /models folder not found/i, `must say the folder was not found (got: ${msg})`);
  assert.ok(msg.includes(path), `must include the path it looked under (got: ${msg})`);
});

test("scan-reason: a successful scan keeps the clean 'Added X, skipped Y' and leaks no method jargon", async () => {
  const msg = await clickScan({ added: 3, skipped: 2, method: "hybrid (folder-walk + /object_info)" });
  assert.match(msg, /Added 3 models/, `success toast keeps the count (got: ${msg})`);
  assert.match(msg, /skipped 2 existing/, `success toast keeps the skipped count (got: ${msg})`);
  // The internal method jargon must never surface in the user-facing toast.
  assert.ok(!/folder-walk/i.test(msg), `must not leak "folder-walk" jargon (got: ${msg})`);
  assert.ok(!/hybrid/i.test(msg), `must not leak "hybrid" jargon (got: ${msg})`);
  assert.ok(!/object_info/i.test(msg), `must not leak "object_info" jargon (got: ${msg})`);
});

test("scan-reason: an empty-but-configured scan reads better than 'Added 0'", async () => {
  // Folders exist and were walked, but held no models: not a misconfig, but a
  // bare "Added 0 models, skipped 0 existing" is still poor. It should read as
  // "no models found", and must not leak the method jargon.
  const msg = await clickScan({ added: 0, skipped: 0, method: "folder-walk" });
  assert.ok(!/folder-walk/i.test(msg), `must not leak "folder-walk" jargon (got: ${msg})`);
  assert.match(msg, /no .*models found/i, `should read as no models found (got: ${msg})`);
});

test("scan-reason: scanResultMessage maps method reasons directly (unit)", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeScanFetch({}) });
  const fn = window.scanResultMessage;
  assert.equal(typeof fn, "function", "scanResultMessage is exported for reuse/testing");

  // The notfound and success mappings are already exercised end to end via the
  // click-driven tests above; this unit layer keeps one representative case.
  const unconfigured = fn({ added: 0, skipped: 0, method: "none (comfy_workdir not configured)" });
  assert.ok(!/Added 0/.test(unconfigured) && /workdir/i.test(unconfigured) && /Settings/i.test(unconfigured));
});
