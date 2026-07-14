// SPDX-License-Identifier: AGPL-3.0-or-later
// import-comfy: the guided "Import from ComfyUI..." flow on the Models page
// (Add-a-model card). Folder pick -> dry-run preview (counts per category,
// nothing registered) -> confirm (real registration) -> toast + refresh.
// Separate from the existing "Scan ComfyUI Models" button/tests (scan-reason,
// models-add-disk): this flow always sends an explicit workdir + dry_run.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const flush = async () => {
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
};

function makeFetch({ managedStatus, scanImpl } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/comfy/managed-status")) {
      if (managedStatus === undefined) {
        return { ok: false, status: 403, json: async () => ({}) };
      }
      return { ok: true, status: 200, json: async () => managedStatus };
    }
    if (u.startsWith("/api/models/scan")) {
      const body = opts.body ? JSON.parse(opts.body) : {};
      return scanImpl(body);
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models: [], active: null }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

function openModalFor(window) {
  const btn = window.document.getElementById("models-import-comfy-btn");
  assert.ok(btn, "the Import from ComfyUI button exists on the Models page");
  btn.click();
}

test("import-comfy: the button exists and opens a modal with a path field, Preview, and Import", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ scanImpl: async () => ({ ok: true, json: async () => ({}) }) }) });
  openModalFor(window);
  await flush();
  const modal = window.document.getElementById("modal");
  assert.equal(modal.style.display, "flex", "the modal is shown");
  const body = window.document.getElementById("modal-body");
  const input = body.querySelector("input[type=text]");
  assert.ok(input, "a text input for the ComfyUI folder path is present");
  const buttons = [...body.querySelectorAll("button")].map((b) => b.textContent);
  assert.ok(buttons.some((t) => t.includes("Browse")), "a Browse button is present");
  assert.ok(buttons.some((t) => t.includes("Preview")), "a Preview button is present");
  assert.ok(buttons.some((t) => t.includes("Import")), "an Import button is present");
});

test("import-comfy: Browse fills the path field via pickDirectory", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({}) });
  window.pickDirectory = async () => "D:/comfy-install";
  openModalFor(window);
  await flush();
  const body = window.document.getElementById("modal-body");
  const input = body.querySelector("input[type=text]");
  const browseBtn = [...body.querySelectorAll("button")].find((b) => b.textContent.includes("Browse"));
  browseBtn.click();
  await flush();
  assert.equal(input.value, "D:/comfy-install");
});

test("import-comfy: a managed ComfyUI install shows the quick-fill row and it fills the path", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({ managedStatus: { installed: true, path: "D:/localm-home/comfyui" } }),
  });
  openModalFor(window);
  await flush();
  const body = window.document.getElementById("modal-body");
  const buttons = [...body.querySelectorAll("button")];
  const managedBtn = buttons.find((b) => b.textContent.includes("localm's own ComfyUI"));
  assert.ok(managedBtn, "the managed-ComfyUI quick-fill button exists");
  assert.equal(managedBtn.closest(".row").style.display, "flex", "its row becomes visible once installed resolves true");
  const input = body.querySelector("input[type=text]");
  managedBtn.click();
  assert.equal(input.value, "D:/localm-home/comfyui");
});

// The quick-fill button is always built (mirrors the hide-via-style pattern
// used elsewhere, e.g. disc-hf-hint / pull-mmproj) and only its containing row
// is toggled visible - so "not offered" means the row stays display:none, not
// that the button is absent from the DOM.
function managedRowOf(body) {
  const btn = [...body.querySelectorAll("button")].find((b) => b.textContent.includes("localm's own ComfyUI"));
  return btn ? btn.closest(".row") : null;
}

test("import-comfy: no managed ComfyUI (not installed) never shows the quick-fill", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({ managedStatus: { installed: false } }),
  });
  openModalFor(window);
  await flush();
  const body = window.document.getElementById("modal-body");
  const row = managedRowOf(body);
  assert.equal(row.style.display, "none", "the quick-fill row stays hidden when nothing is installed");
});

test("import-comfy: a 403/failed managed-status fetch degrades silently (no quick-fill, no crash)", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({}) });   // managedStatus undefined -> 403
  openModalFor(window);
  await flush();
  const body = window.document.getElementById("modal-body");
  const row = managedRowOf(body);
  assert.equal(row.style.display, "none", "a 403 on managed-status just hides the quick-fill, never breaks the modal");
});

test("import-comfy: clicking Preview with no folder chosen toasts and does not scan", async () => {
  let scanCalls = 0;
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({ scanImpl: async () => { scanCalls++; return { ok: true, json: async () => ({}) }; } }),
  });
  openModalFor(window);
  await flush();
  const body = window.document.getElementById("modal-body");
  const previewBtn = [...body.querySelectorAll("button")].find((b) => b.textContent === "Preview");
  previewBtn.click();
  await flush();
  assert.equal(scanCalls, 0, "no scan request is made without a chosen folder");
  assert.match(window.document.getElementById("toast").textContent, /choose a comfyui folder/i);
});

test("import-comfy: Preview renders per-category counts and enables Import", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({
      scanImpl: async (body) => {
        calls.push(body);
        return {
          ok: true,
          json: async () => ({
            dry_run: true, method: "folder-walk",
            counts: { "diffusion-unet": 2, lora: 3 },
            already_registered: 4, total_new: 5,
          }),
        };
      },
    }),
  });
  openModalFor(window);
  await flush();
  const body = window.document.getElementById("modal-body");
  const input = body.querySelector("input[type=text]");
  input.value = "D:/my-comfy";
  const previewBtn = [...body.querySelectorAll("button")].find((b) => b.textContent === "Preview");
  const importBtn = [...body.querySelectorAll("button")].find((b) => b.textContent === "Import");
  assert.equal(importBtn.disabled, true, "Import starts disabled");
  previewBtn.click();
  await flush();

  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0], { workdir: "D:/my-comfy", dry_run: true });

  const text = body.querySelector(".import-comfy-preview").textContent;
  assert.match(text, /found 5 new/i);
  assert.match(text, /Diffusion.*2/i);
  assert.match(text, /LoRAs.*3/i);
  assert.match(text, /4 already registered/i);
  assert.equal(importBtn.disabled, false, "a non-empty preview enables Import");
});

test("import-comfy: a 'none (...)' preview reuses the scanResultMessage wording and keeps Import disabled", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({
      scanImpl: async () => ({
        ok: true,
        json: async () => ({ dry_run: true, method: "none (models folder not found under D:\\nope)", counts: {}, already_registered: 0, total_new: 0 }),
      }),
    }),
  });
  openModalFor(window);
  await flush();
  const body = window.document.getElementById("modal-body");
  const input = body.querySelector("input[type=text]");
  input.value = "D:/nope";
  const previewBtn = [...body.querySelectorAll("button")].find((b) => b.textContent === "Preview");
  const importBtn = [...body.querySelectorAll("button")].find((b) => b.textContent === "Import");
  previewBtn.click();
  await flush();

  const text = body.querySelector(".import-comfy-preview").textContent;
  assert.match(text, /models folder not found/i);
  assert.ok(text.includes("D:\\nope"));
  assert.equal(importBtn.disabled, true, "an error preview never enables Import");
});

test("import-comfy: Import POSTs dry_run:false for the previewed folder, toasts, closes the modal, and refreshes", async () => {
  const calls = [];
  let modelsFetchCount = 0;
  const { window } = loadAppWithPages({
    fetchImpl: async (url, opts = {}) => {
      const u = String(url);
      if (u.startsWith("/api/models/scan")) {
        const b = opts.body ? JSON.parse(opts.body) : {};
        calls.push(b);
        if (b.dry_run) {
          return { ok: true, json: async () => ({ dry_run: true, method: "folder-walk", counts: { lora: 2 }, already_registered: 0, total_new: 2 }) };
        }
        return { ok: true, json: async () => ({ added: 2, skipped: 0, method: "folder-walk" }) };
      }
      if (u === "/api/models" || u.startsWith("/api/models?")) {
        modelsFetchCount++;
        return { ok: true, json: async () => ({ models: [], active: null }) };
      }
      return { ok: true, json: async () => ({}), text: async () => "" };
    },
  });
  openModalFor(window);
  await flush();
  const body = window.document.getElementById("modal-body");
  const input = body.querySelector("input[type=text]");
  input.value = "D:/my-comfy";
  const previewBtn = [...body.querySelectorAll("button")].find((b) => b.textContent === "Preview");
  previewBtn.click();
  await flush();
  const importBtn = [...body.querySelectorAll("button")].find((b) => b.textContent === "Import");
  assert.equal(importBtn.disabled, false);
  const fetchesBeforeImport = modelsFetchCount;
  importBtn.click();
  await flush();

  assert.equal(calls.length, 2);
  assert.deepEqual(calls[1], { workdir: "D:/my-comfy", dry_run: false });
  assert.match(window.document.getElementById("toast").textContent, /added 2 models/i);
  assert.equal(window.document.getElementById("modal").style.display, "none", "the modal closes on a successful import");
  assert.ok(modelsFetchCount > fetchesBeforeImport, "the models list is refreshed after import");
});

test("import-comfy: a failed Import surfaces the server error and re-enables the button", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({
      scanImpl: async (b) => {
        if (b.dry_run) {
          return { ok: true, json: async () => ({ dry_run: true, method: "folder-walk", counts: { lora: 1 }, already_registered: 0, total_new: 1 }) };
        }
        return { ok: false, status: 500, json: async () => ({ detail: "Scan failed: boom" }) };
      },
    }),
  });
  openModalFor(window);
  await flush();
  const body = window.document.getElementById("modal-body");
  body.querySelector("input[type=text]").value = "D:/my-comfy";
  const previewBtn = [...body.querySelectorAll("button")].find((b) => b.textContent === "Preview");
  previewBtn.click();
  await flush();
  const importBtn = [...body.querySelectorAll("button")].find((b) => b.textContent === "Import");
  importBtn.click();
  await flush();

  assert.match(window.document.getElementById("toast").textContent, /scan failed: boom/i);
  assert.equal(importBtn.disabled, false, "a failed import re-enables the button so the user can retry");
  assert.equal(window.document.getElementById("modal").style.display, "flex", "the modal stays open on failure");
});
