// SPDX-License-Identifier: AGPL-3.0-or-later
// scan-button-placement: the ComfyUI folder scanner used to live in a
// tab-gated row (#models-scan-row, display:none by default) that only became
// visible after switching the Registered-models tabs to Diffusion/Encoders/
// VAEs/LoRAs/Other - it was invisible on page load (All tab) and on the LLMs
// tab, with no label/icon/copy hinting that a tab switch would reveal a new
// action. The button now lives inside the "Add a model" card as a labelled,
// icon-carrying callout that is visible regardless of which Registered-models
// tab is active, since the scan itself is type-agnostic server-side. This
// drives the real DOM + tab-click handlers and asserts the button never
// disappears, and that it still lives alongside the other add-a-model
// affordances (pull-spec/pull-start), not inside the Registered-models card.

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

  const tabNav = window.document.getElementById("models-tab-nav");
  assert.ok(!tabNav.closest(".card").contains(btn),
    "the scan button is no longer inside the Registered-models tab-gated card");
});

test("scan-placement: the Scan button is visible on initial load (the All tab)", async () => {
  const { window } = loadAppWithPages({ fetchImpl: okFetch });
  await window.refreshModelsPage();
  const btn = window.document.getElementById("models-scan-btn");
  assert.ok(isVisible(btn), "visible by default, not hidden until a tab switch");
});

test("scan-placement: the Scan button stays visible across every Registered-models tab", async () => {
  const { window } = loadAppWithPages({ fetchImpl: okFetch });
  const tabNav = window.document.getElementById("models-tab-nav");
  const btn = window.document.getElementById("models-scan-btn");

  for (const tabBtn of tabNav.querySelectorAll(".tab-btn")) {
    tabBtn.click();
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(isVisible(btn),
      `scan button stays visible on the "${tabBtn.dataset.type}" tab (it used to vanish on All/LLMs)`);
  }
});

test("scan-placement: the button carries an icon and explanatory copy, not a bare secondary button", async () => {
  const { window } = loadAppWithPages({ fetchImpl: okFetch });
  const btn = window.document.getElementById("models-scan-btn");
  const row = btn.closest(".comfy-import");
  assert.ok(row, "the scan button sits in its own labelled callout block");
  assert.ok(row.querySelector("[data-icon], [data-icon-name]"),
    "the callout carries an icon distinguishing it from Browse/Unload all");
  assert.match(row.textContent, /ComfyUI/i,
    "the callout names ComfyUI so the action is self-explanatory without a tab switch");
});

test("scan-placement: clicking the (now always-visible) button still POSTs the scan and toasts the result", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models/scan")) {
      calls.push({ url: u, method: opts.method });
      return { ok: true, status: 200, json: async () => ({ added: 2, skipped: 1, method: "hybrid" }), text: async () => "" };
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
