// SPDX-License-Identifier: AGPL-3.0-or-later
// The media subsection renders Stop and Restart controls; Stop POSTs
// /v1/comfy/stop.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = { fields: [
  { key: "comfy_workdir", widget: "folder", label: "ComfyUI folder", help: "",
    group: "Media", owner: "image", default: "/shared" },
]};
const MEDIA = { plugins: [
  { plugin: "image", label: "Image", fields: [
    { key: "workdir", widget: "folder", label: "ComfyUI folder", help: "",
      value: "/shared", is_override: false }] },
]};

function makeFetch(posts) {
  return async (url, opts = {}) => {
    const method = opts.method || "GET";
    if (url === "/v1/config/schema")
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    if (url === "/v1/media/config" && method === "GET")
      return { ok: true, status: 200, json: async () => MEDIA, text: async () => "" };
    if (url === "/v1/comfy/status")
      return { ok: true, status: 200, json: async () => ({ alive: true, launched_by_localm: true }), text: async () => "" };
    if (url === "/v1/comfy/stop" && method === "POST") {
      posts.push(url);
      return { ok: true, status: 200, json: async () => ({ ok: true, message: "stopped" }), text: async () => "" };
    }
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function render(win) {
  runScript(win, "refreshSettingsPage();");
  for (let i = 0; i < 12; i++) await new Promise((r) => setTimeout(r, 0));
}

test("media subsection renders Stop + Restart controls", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  const doc = win.document;
  assert.ok(doc.querySelector(".comfy-stop-btn"), "Stop button rendered");
  assert.ok(doc.querySelector(".comfy-restart-btn"), "Restart button rendered");
  // type=button, so they do not submit the settings form.
  assert.equal(doc.querySelector(".comfy-stop-btn").type, "button");
});

test("clicking Stop POSTs /v1/comfy/stop", async () => {
  const posts = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  await render(win);
  win.document.querySelector(".comfy-stop-btn").onclick();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(posts.includes("/v1/comfy/stop"), "Stop POSTed /v1/comfy/stop");
});
