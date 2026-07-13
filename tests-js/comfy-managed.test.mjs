// SPDX-License-Identifier: AGPL-3.0-or-later
// S5 GUI-button slice: the managed-ComfyUI panel in Settings > Media - its own
// compact box at the TOP of the Media section, ahead of the three per-plugin
// (image/music/video) boxes. When no managed instance is installed it shows a
// "Set up localm's own ComfyUI" button that POSTs /api/comfy/setup (dispatched
// as a job, streamed). When one IS installed it shows "installed at <path>", the
// S1 coexistence controls (managed_comfy_enabled / comfy_target - core schema
// fields with group="Media", otherwise skipped from the flat form) with their
// own small Save, and a Remove button that POSTs /api/comfy/remove.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = { fields: [
  { key: "comfy_workdir", widget: "folder", label: "ComfyUI folder", help: "",
    group: "Media", owner: "image", default: "/shared" },
]};
// A schema that also carries the two S1 coexistence fields, for the tests that
// exercise them specifically (kept separate from SCHEMA above so the minimal
// no-toggle-fields case - Remove must still render even then - stays covered).
const SCHEMA_WITH_TOGGLES = { fields: [
  ...SCHEMA.fields,
  { key: "managed_comfy_enabled", widget: "toggle",
    label: "Use localm's own managed ComfyUI", help: "", group: "Media",
    owner: "image", default: false },
  { key: "comfy_target", widget: "select", label: "ComfyUI to use", help: "",
    group: "Media", owner: "image", options: ["own", "user"], default: "own" },
]};
const MEDIA = { plugins: [
  { plugin: "image", label: "Image", fields: [] },
]};

const INSTALLED = {
  installed: true, path: "/home/user/.localm/comfyui",
  models_dir: "/home/user/.localm/comfyui-models",
  api_url: "http://127.0.0.1:8189", enabled: false, target: "own",
  managed_active: false,
};
const NOT_INSTALLED = {
  installed: false, path: null, api_url: "http://127.0.0.1:8189",
  enabled: false, target: "own", managed_active: false,
};

function makeFetch(calls, { installed, schema = SCHEMA }) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    calls.push({ url: u, method, opts });
    if (u === "/v1/config/schema")
      return { ok: true, status: 200, json: async () => schema, text: async () => "" };
    if (u === "/v1/config" && method === "PATCH")
      return { ok: true, status: 200, json: async () => ({ ok: true }), text: async () => "" };
    if (u === "/v1/media/config" && method === "GET")
      return { ok: true, status: 200, json: async () => MEDIA, text: async () => "" };
    if (u === "/v1/comfy/status")
      return { ok: true, status: 200, json: async () => ({ alive: false, launched_by_localm: false }), text: async () => "" };
    if (u === "/api/comfy/managed-status")
      return { ok: true, status: 200, text: async () => "",
               json: async () => (installed ? INSTALLED : NOT_INSTALLED) };
    if (u === "/api/comfy/setup" && method === "POST")
      return { ok: true, status: 200, json: async () => ({ job_id: "job123" }), text: async () => "" };
    if (u === "/api/comfy/remove" && method === "POST")
      return { ok: true, status: 200, json: async () => ({ status: "removed", removed: [INSTALLED.path] }), text: async () => "" };
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function render(win) {
  // streamJob is stubbed so a setup click resolves without a real SSE job stream.
  runScript(win, "streamJob = () => Promise.resolve({ status: 'done' });");
  runScript(win, "refreshSettingsPage();");
  for (let i = 0; i < 16; i++) await new Promise((r) => setTimeout(r, 0));
}

test("not installed -> a Set-up button renders (and status was fetched)", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { installed: false }) });
  await render(win);
  const btn = win.document.querySelector(".comfy-managed-setup-btn");
  assert.ok(btn, "Set-up button rendered");
  assert.equal(btn.type, "button", "type=button so it never submits the settings form");
  assert.ok(calls.some((c) => c.url === "/api/comfy/managed-status"), "managed-status fetched");
  assert.ok(!win.document.querySelector(".comfy-managed-remove-btn"), "no Remove button when not installed");
});

test("clicking Set up POSTs /api/comfy/setup", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { installed: false }) });
  await render(win);
  win.document.querySelector(".comfy-managed-setup-btn").onclick();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  const post = calls.find((c) => c.url === "/api/comfy/setup" && c.method === "POST");
  assert.ok(post, "Set up POSTed /api/comfy/setup");
});

test("installed -> shows 'installed at <path>' + a Remove button, no Set-up button", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { installed: true }) });
  await render(win);
  const doc = win.document;
  assert.ok(doc.querySelector(".comfy-managed-remove-btn"), "Remove button rendered");
  assert.ok(!doc.querySelector(".comfy-managed-setup-btn"), "no Set-up button when installed");
  const panel = doc.querySelector(".media-comfy-box");
  assert.ok(panel, "media-comfy-box panel present");
  assert.match(panel.textContent, /installed at/i, "shows 'installed at'");
  assert.match(panel.textContent, /\.localm[/\\]comfyui/, "shows the install path");
});

test("clicking Remove POSTs /api/comfy/remove", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { installed: true }) });
  await render(win);
  // Remove is destructive: it goes through confirmDanger. Stub it to auto-confirm.
  runScript(win, "confirmDanger = (t, m, l, onConfirm) => onConfirm();");
  win.document.querySelector(".comfy-managed-remove-btn").onclick();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  const post = calls.find((c) => c.url === "/api/comfy/remove" && c.method === "POST");
  assert.ok(post, "Remove POSTed /api/comfy/remove");
});

test("installed + toggle fields in schema -> coexistence controls render inside the top box, ahead of the three-mode grid", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages(
    { fetchImpl: makeFetch(calls, { installed: true, schema: SCHEMA_WITH_TOGGLES }) });
  await render(win);
  const doc = win.document;
  const box = doc.querySelector(".media-comfy-box");
  assert.ok(box, "media-comfy-box present");

  const enabledCtrl = box.querySelector('input[data-key="managed_comfy_enabled"]');
  const targetCtrl = box.querySelector('select[data-key="comfy_target"]');
  assert.ok(enabledCtrl, "managed_comfy_enabled control renders inside the top box");
  assert.ok(targetCtrl, "comfy_target control renders inside the top box");
  assert.equal(enabledCtrl.type, "checkbox", "managed_comfy_enabled is the toggle widget");

  // "its own little thing on the top": the compact box precedes the three-mode
  // grid in DOM order, not after it.
  const grid = doc.querySelector("#settings-sec-media .media-grid");
  assert.ok(grid, "the three-mode grid exists");
  const pos = box.compareDocumentPosition(grid);
  assert.ok(pos & win.Node.DOCUMENT_POSITION_FOLLOWING,
    "the comfy box comes before the media grid in the DOM");

  // The top box also has its own Save button (distinct from the per-plugin
  // ".media-save" buttons and the Remove button).
  const saveButtons = [...box.querySelectorAll(".actions button")]
    .filter((b) => b.textContent === "Save");
  assert.equal(saveButtons.length, 1, "the top box has exactly one Save button");
});

test("toggling managed_comfy_enabled and clicking the top box's Save PATCHes /v1/config with just that key", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages(
    { fetchImpl: makeFetch(calls, { installed: true, schema: SCHEMA_WITH_TOGGLES }) });
  await render(win);
  const doc = win.document;
  const box = doc.querySelector(".media-comfy-box");
  const enabledCtrl = box.querySelector('input[data-key="managed_comfy_enabled"]');
  enabledCtrl.checked = true;
  enabledCtrl.dispatchEvent(new win.Event("change", { bubbles: true }));

  const save = [...box.querySelectorAll(".actions button")]
    .find((b) => b.textContent === "Save");
  assert.ok(save, "Save button present");
  save.onclick();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  const patch = calls.find((c) => c.url === "/v1/config" && c.method === "PATCH");
  assert.ok(patch, "Save PATCHed /v1/config");
  const body = JSON.parse(patch.opts.body);
  assert.equal(body.managed_comfy_enabled, true, "the changed toggle is in the PATCH body");
  // comfy_target is a SELECT widget: read() always returns its current value
  // (unlike a text/number field, which reports `undefined` when untouched), so
  // it rides along even though only the toggle was actually changed here - this
  // matches every other SELECT field's save behavior on this page, not
  // something specific to the managed-ComfyUI box.
  assert.equal(body.comfy_target, "own", "the untouched select's current value rides along too");
  assert.equal(Object.keys(body).length, 2, "exactly these two keys are sent, nothing else");
});
