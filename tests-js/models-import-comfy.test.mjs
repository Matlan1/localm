// SPDX-License-Identifier: AGPL-3.0-or-later
// The guided "Import from ComfyUI..." flow on the Models page: folder pick ->
// dry-run preview -> confirm -> toast + refresh. Every request carries an
// explicit workdir and dry_run. A real (dry_run:false) scan is job-based: POST
// /api/models/scan returns {job_id} and progress streams over
// GET /api/jobs/{id}/events, which jobEvents below plays back.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const flush = async () => {
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
};

// A minimal filesystem for the picker's /api/fs/dirs + /api/fs/places, so the
// tests drive the real pickDirectory() end to end instead of stubbing it.
const FS = {
  "": { path: "", parent: null, entries: [
    { name: "comfy-root", is_dir: true, size: null, mtime: 1700000000 },
  ] },
  "comfy-root": {
    path: "comfy-root", parent: "", entries: [
      { name: "models", is_dir: true, size: null, mtime: 1700000000 },
    ],
  },
};

// A single-shot SSE playback of *events*: a fetch Response whose
// body.getReader() hands back one "data: <json>\n\n" frame per call, then EOF.
//
// delayMs>0 resolves each frame on a real timer instead of an already-resolved
// microtask, which is what lets a polling test (waitFor, below) observe an
// in-flight frame rather than only the fully settled end state.
function sseResponse(events, { delayMs = 0 } = {}) {
  const frames = events.map((ev) => `data: ${JSON.stringify(ev)}\n\n`);
  let idx = 0;
  const enc = new TextEncoder();
  return {
    ok: true, status: 200,
    body: {
      getReader() {
        return {
          read() {
            const emit = () => {
              if (idx < frames.length) {
                const chunk = enc.encode(frames[idx]);
                idx++;
                return { done: false, value: chunk };
              }
              return { done: true, value: undefined };
            };
            if (!delayMs) return Promise.resolve(emit());
            return new Promise((resolve) => setTimeout(() => resolve(emit()), delayMs));
          },
          async cancel() {},
        };
      },
    },
  };
}

// Poll *cond* until it returns true or *timeout* elapses.
async function waitFor(cond, { timeout = 2000, interval = 5 } = {}) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (cond()) return true;
    await new Promise((r) => setTimeout(r, interval));
  }
  return false;
}

function makeFetch({ managedStatus, scanImpl, jobEvents, jobEventsDelayMs } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/comfy/managed-status")) {
      if (managedStatus === undefined) {
        return { ok: false, status: 403, json: async () => ({}) };
      }
      return { ok: true, status: 200, json: async () => managedStatus };
    }
    if (/\/api\/jobs\/[^/]+\/events$/.test(u)) {
      return sseResponse(jobEvents || [], { delayMs: jobEventsDelayMs });
    }
    if (u.startsWith("/api/models/scan")) {
      const body = opts.body ? JSON.parse(opts.body) : {};
      return scanImpl(body);
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models: [], active: null }) };
    }
    if (u.includes("/api/fs/places")) {
      return { ok: true, status: 200, json: async () => ({
        places: [{ label: "Home", path: "/home/me", icon: "home" }],
        drives: [{ label: "/", path: "/", icon: "drive" }],
      }) };
    }
    if (u.includes("/api/fs/dirs")) {
      const m = u.match(/[?&]path=([^&]*)/);
      const p = decodeURIComponent(m ? m[1] : "");
      const d = FS[p] || FS[""];
      return { ok: true, status: 200, json: async () => d };
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

test("import-comfy: Browse fills the path field via the REAL pickDirectory, surviving the modal being rebuilt", async () => {
  // openModal() has no stack: a single shared #modal-body, replaced not
  // pushed, so pickDirectory()'s own openModal() call replaces this modal's
  // DOM. Every element reference has to be re-queried after the picker
  // resolves.
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({}) });
  openModalFor(window);
  await flush();
  let body = window.document.getElementById("modal-body");
  const browseBtn = [...body.querySelectorAll("button")].find((b) => b.textContent.includes("Browse"));
  browseBtn.click();
  await flush();
  // Navigate into the one folder the fake filesystem offers, then confirm.
  const row = [...window.document.querySelectorAll(".picker-row")].find(
    (r) => r.querySelector(".picker-name") && r.querySelector(".picker-name").textContent === "comfy-root");
  assert.ok(row, "the real folder picker opened and listed the fake folder (not a stub)");
  row.click();
  await flush();
  const okBtn = window.document.querySelector(".picker-foot .btn-primary");
  assert.equal(okBtn.textContent, "Use this folder");
  okBtn.click();
  await flush();
  // Re-query rather than reusing the now-detached input reference.
  const modal = window.document.getElementById("modal");
  assert.equal(modal.style.display, "flex", "the Import-from-ComfyUI dialog reopened, not left closed");
  body = window.document.getElementById("modal-body");
  const input = body.querySelector("input[type=text]");
  assert.ok(input, "a path input is present in the rebuilt modal");
  assert.equal(input.value, "comfy-root", "the picked folder was carried over");
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

// The quick-fill button is always built; only its containing row is toggled
// visible, so "not offered" means the row is display:none, not that the button
// is absent from the DOM.
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
      if (/\/api\/jobs\/[^/]+\/events$/.test(u)) {
        return sseResponse([
          { type: "progress", phase: "registering", done: 1, total: 2, name: "a.safetensors" },
          { type: "progress", phase: "registering", done: 2, total: 2, name: "b.safetensors" },
          { type: "progress", phase: "done", done: 2, total: 2, added: 2, skipped: 0, method: "folder-walk" },
          { type: "end", status: "done", returncode: 0 },
        ]);
      }
      if (u.startsWith("/api/models/scan")) {
        const b = opts.body ? JSON.parse(opts.body) : {};
        calls.push(b);
        if (b.dry_run) {
          return { ok: true, json: async () => ({ dry_run: true, method: "folder-walk", counts: { lora: 2 }, already_registered: 0, total_new: 2 }) };
        }
        return { ok: true, json: async () => ({ job_id: "scan-job-1" }) };
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
  // The final progress event's added/skipped/method feeds scanResultMessage().
  assert.match(window.document.getElementById("toast").textContent, /added 2 models/i);
  assert.equal(window.document.getElementById("modal").style.display, "none", "the modal closes on a successful import");
  assert.ok(modelsFetchCount > fetchesBeforeImport, "the models list is refreshed after import");
});

test("import-comfy: Import shows live 'registering model N of M' progress as the job runs", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({
      scanImpl: async (b) => {
        if (b.dry_run) {
          return { ok: true, json: async () => ({ dry_run: true, method: "folder-walk", counts: { lora: 3 }, already_registered: 0, total_new: 3 }) };
        }
        return { ok: true, json: async () => ({ job_id: "scan-job-2" }) };
      },
      // jobEventsDelayMs paces the frames on a real timer so the polling
      // below can observe an in-flight state.
      jobEvents: [
        { type: "progress", phase: "registering", done: 1, total: 3, name: "alpha.safetensors" },
        { type: "progress", phase: "registering", done: 2, total: 3, name: "beta.safetensors" },
        { type: "progress", phase: "registering", done: 3, total: 3, name: "gamma.safetensors" },
        { type: "progress", phase: "done", done: 3, total: 3, added: 3, skipped: 0, method: "folder-walk" },
        { type: "end", status: "done", returncode: 0 },
      ],
      jobEventsDelayMs: 30,
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
  const progressText = () => body.querySelector(".import-comfy-progress").textContent;
  // Match the FIRST tick specifically, not any registering text.
  const sawFirst = await waitFor(() => /registering model 1 of 3/i.test(progressText()));
  assert.ok(sawFirst, `never observed "1 of 3" while the job was in flight (last seen: ${progressText()})`);

  const toastShowsDone = await waitFor(
    () => /added 3 models/i.test(window.document.getElementById("toast").textContent));
  assert.ok(toastShowsDone, "the final toast still reads the same as before this unit (Added 3 models)");
});

test("import-comfy: a failed Import surfaces the job's own error and re-enables the button", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({
      scanImpl: async (b) => {
        if (b.dry_run) {
          return { ok: true, json: async () => ({ dry_run: true, method: "folder-walk", counts: { lora: 1 }, already_registered: 0, total_new: 1 }) };
        }
        // A real scan is job-based: the POST always starts the job (200 +
        // job_id) and a failure surfaces on the job's own stream.
        return { ok: true, json: async () => ({ job_id: "scan-job-3" }) };
      },
      jobEvents: [
        { type: "line", text: "Scanning ComfyUI model folders..." },
        { type: "line", text: "Scan failed: boom" },
        { type: "end", status: "failed", returncode: 1 },
      ],
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

test("import-comfy: a disconnected stream leaves the modal open with a 'connection lost' message, not a false failure", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch({
      scanImpl: async (b) => {
        if (b.dry_run) {
          return { ok: true, json: async () => ({ dry_run: true, method: "folder-walk", counts: { lora: 1 }, already_registered: 0, total_new: 1 }) };
        }
        return { ok: true, json: async () => ({ job_id: "scan-job-4" }) };
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
  // Override just the events route to a 404 for this one test.
  const origFetch = window.fetch;
  window.fetch = async (url, opts) => {
    if (/\/api\/jobs\/[^/]+\/events$/.test(String(url))) {
      return { ok: false, status: 404, statusText: "Not Found" };
    }
    return origFetch(url, opts);
  };
  const importBtn = [...body.querySelectorAll("button")].find((b) => b.textContent === "Import");
  importBtn.click();
  await flush();

  assert.match(window.document.getElementById("toast").textContent, /lost connection/i);
  assert.equal(importBtn.disabled, false, "a lost connection re-enables the button");
  assert.equal(window.document.getElementById("modal").style.display, "flex", "the modal stays open - the import may still be running");
  assert.match(body.querySelector(".import-comfy-progress").textContent, /connection lost/i);
});
