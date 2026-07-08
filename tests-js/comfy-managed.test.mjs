// SPDX-License-Identifier: AGPL-3.0-or-later
// S5 GUI-button slice: the managed-ComfyUI panel in Settings > Media. When no
// managed instance is installed it shows a "Set up localm's own ComfyUI" button
// that POSTs /api/comfy/setup (dispatched as a job, streamed). When one IS
// installed it shows "installed at <path>" + a Remove button that POSTs
// /api/comfy/remove. The S1 coexistence toggle (comfy_target / managed_comfy_enabled)
// is a schema field rendered elsewhere and is NOT duplicated by this panel.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = { fields: [
  { key: "comfy_workdir", widget: "folder", label: "ComfyUI folder", help: "",
    group: "Media", owner: "image", default: "/shared" },
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

function makeFetch(calls, { installed }) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    calls.push({ url: u, method, opts });
    if (u === "/v1/config/schema")
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
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
  const panel = doc.querySelector(".managed-comfy");
  assert.ok(panel, "managed-comfy panel present");
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
