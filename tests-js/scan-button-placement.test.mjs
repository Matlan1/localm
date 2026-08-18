// SPDX-License-Identifier: AGPL-3.0-or-later
// scan-button-placement: the ComfyUI re-scan button used to live in a
// tab-gated row (#models-scan-row, display:none by default) that only became
// visible after switching the Registered-models tabs to Diffusion/Encoders/
// VAEs/LoRAs/Other - it was invisible on page load (All tab) and on the LLMs
// tab, with no label/icon/copy hinting that a tab switch would reveal a new
// action. The button now lives inside the "Add a model" card, next to the
// separate "Import from ComfyUI…" guided-import flow, and is visible
// regardless of which Registered-models tab is active, since the scan itself
// is type-agnostic server-side. This drives the real DOM + tab-click handlers
// and asserts the button never disappears, that it still lives alongside the
// other add-a-model affordances (pull-spec/pull-start) rather than inside the
// Registered-models card, and that it stays clearly distinct from its sibling
// Import button (the two cover different cases - see docs/managed-comfyui.md).

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const okFetch = async (url) => {
  const u = String(url);
  if (u === "/api/models" || u.startsWith("/api/models?")) {
    return { ok: true, status: 200, json: async () => ({ models: [], active: null }) };
  }
  return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
};

function isVisible(el) {
  // jsdom does not compute layout, so "visible" here means "not explicitly
  // hidden via inline style or the hidden attribute" - the same signal the old
  // display:none/flex toggle used to control.
  return el && el.style.display !== "none" && !el.hidden;
}

test("scan-placement: the button was MOVED, not duplicated - exactly one #models-scan-btn/#models-scan-row in the whole page", async () => {
  const { window } = loadAppWithPages({ fetchImpl: okFetch });
  // getElementById silently returns the first document-order match, so it
  // alone can't tell "moved" from "a stale copy left behind elsewhere" (e.g. a
  // bad merge that re-adds the old tab-gated block without deleting it).
  // querySelectorAll on the id attribute surfaces every match.
  assert.equal(window.document.querySelectorAll('[id="models-scan-btn"]').length, 1,
    "exactly one scan button in the document - no leftover duplicate from the old tab-gated block");
  assert.equal(window.document.querySelectorAll('[id="models-scan-row"]').length, 1,
    "exactly one scan row in the document - no leftover duplicate from the old tab-gated block");
});

test("scan-placement: the Scan button sits inside the Add-a-model card, not the tab-gated Registered-models row", async () => {
  const { window } = loadAppWithPages({ fetchImpl: okFetch });
  const btn = window.document.getElementById("models-scan-btn");
  assert.ok(btn, "the scan button exists");

  const addCard = window.document.getElementById("pull-start").closest(".card");
  assert.ok(addCard, "the Add-a-model card is found via its Add button");
  assert.ok(addCard.contains(btn),
    "the scan button lives in the same card as the HuggingFace/URL/path add fields");

  const typeFilter = window.document.getElementById("models-type-filter");
  assert.ok(!typeFilter.closest(".card").contains(btn),
    "the scan button is no longer inside the Registered-models type-filtered card");
});

test("scan-placement: the Scan button is visible on initial load (all types ticked)", async () => {
  const { window } = loadAppWithPages({ fetchImpl: okFetch });
  await window.refreshModelsPage();
  const btn = window.document.getElementById("models-scan-btn");
  assert.ok(isVisible(btn), "visible by default, not hidden until a tab switch");
});

test("scan-placement: the Scan button stays visible across every Registered-models type filter", async () => {
  const { window } = loadAppWithPages({ fetchImpl: okFetch });
  const btn = window.document.getElementById("models-scan-btn");

  // Narrow to each single type in turn, then to none at all - the button lives
  // outside the filtered card, so no selection may take it away.
  for (const box of window.document.querySelectorAll(".reg-type")) {
    for (const b of window.document.querySelectorAll(".reg-type")) {
      b.checked = (b === box);
      b.dispatchEvent(new window.Event("change"));
    }
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(isVisible(btn),
      `scan button stays visible with only "${box.value}" ticked (it used to vanish on All/LLMs)`);
  }
  for (const b of window.document.querySelectorAll(".reg-type")) {
    b.checked = false;
    b.dispatchEvent(new window.Event("change"));
  }
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(isVisible(btn), "scan button stays visible with no type ticked at all");
});

test("scan-placement: the button carries an icon and copy that differentiates it from its sibling Import button", async () => {
  const { window } = loadAppWithPages({ fetchImpl: okFetch });
  const btn = window.document.getElementById("models-scan-btn");
  const row = btn.closest("#models-scan-row");
  assert.ok(row, "the scan button sits in its own row");
  assert.ok(row.querySelector("[data-icon], [data-icon-name]"),
    "the row carries an icon distinguishing it from a bare Browse/Unload-all button");

  const importBtn = window.document.getElementById("models-import-comfy-btn");
  assert.ok(importBtn, "the sibling guided-import button exists (this diff composes with it, not replaces it)");
  assert.notEqual(btn.textContent.trim(), importBtn.textContent.trim(),
    "the two ComfyUI-related buttons have distinct labels");

  const scanRowText = row.textContent;
  const importRowText = importBtn.closest(".row").textContent;
  assert.notEqual(scanRowText, importRowText,
    "the two rows' explanatory copy must not read as duplicates of each other");
  assert.match(scanRowText, /Settings/i,
    "the re-scan row's copy references the already-configured folder (its distinguishing behavior)");
});

// A real (non-dry-run) scan is job-based now (see gui_scan_models /
// scan_comfy_models's progress_cb): POST /api/models/scan returns {job_id}
// immediately and the result streams over GET /api/jobs/{id}/events, with the
// final progress event (phase "done") carrying the same added/skipped/method
// fields the old synchronous response body used to. Single-shot SSE playback,
// same shape as models-import-comfy.test.mjs's sseResponse.
function sseResponse(events) {
  const frames = events.map((ev) => `data: ${JSON.stringify(ev)}\n\n`);
  let idx = 0;
  const enc = new TextEncoder();
  return {
    ok: true, status: 200,
    body: {
      getReader() {
        return {
          async read() {
            if (idx < frames.length) {
              const chunk = enc.encode(frames[idx]);
              idx++;
              return { done: false, value: chunk };
            }
            return { done: true, value: undefined };
          },
          async cancel() {},
        };
      },
    },
  };
}

test("scan-placement: clicking the (now always-visible) button still POSTs the scan and toasts the result", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (/\/api\/jobs\/[^/]+\/events$/.test(u)) {
      return sseResponse([
        { type: "progress", phase: "done", done: 3, total: 3, added: 2, skipped: 1, method: "hybrid" },
        { type: "end", status: "done", returncode: 0 },
      ]);
    }
    if (u.startsWith("/api/models/scan")) {
      calls.push({ url: u, method: opts.method });
      return { ok: true, status: 200, json: async () => ({ job_id: "scan-placement-job" }), text: async () => "" };
    }
    return okFetch(u);
  };
  const { window } = loadAppWithPages({ fetchImpl });
  window.document.getElementById("models-scan-btn").click();
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));

  assert.equal(calls.length, 1, "the scan endpoint fired");
  assert.equal(calls[0].method, "POST");
  assert.match(window.document.getElementById("toast").textContent, /Added 2 models, skipped 1 existing/);
});
