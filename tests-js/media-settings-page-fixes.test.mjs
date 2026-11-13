// SPDX-License-Identifier: AGPL-3.0-or-later
// Settings > Media: a failed per-plugin read keeps the schema-drawn boxes, Copy
// from never copies a backend or native field or a choice the target box does
// not offer, a native backend hides the ComfyUI-only fields, an unset select
// shows a blank choice, "ComfyUI to use" is offered before anything is
// installed, and Remove with nothing installed says so.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = { fields: [
  { key: "comfy_target", widget: "select", label: "ComfyUI to use", help: "",
    group: "Media", owner: "image", default: "own", options: ["own", "user"] },
  { key: "comfy_launch_timeout", widget: "number", label: "Launch timeout", help: "",
    group: "Media", owner: "image", default: 300, shipped_default: 300 },
  { key: "comfy_gpu_placement", widget: "toggle", label: "GPU placement", help: "",
    group: "Media", owner: "image", default: false },
  { key: "embedding_pooling", widget: "select", label: "Embedding pooling", help: "",
    group: "Embeddings", owner: "core", default: null, shipped_default: null,
    options: ["mean", "auto", "cls", "last", "none"] },
] };

const field = (key, widget, value, extra = {}) =>
  ({ key, widget, label: key, help: "", value, is_override: true, ...extra });

const MEDIA = { plugins: [
  { plugin: "image", label: "Image", fields: [
    field("backend", "select", "native", { options: ["auto", "native", "comfy"] }),
    field("native_runtime", "select", "rocm", { options: ["auto", "vulkan", "rocm"] }),
    field("float_type", "select", null,
          { options: ["default", "fp8_e4m3fn"], is_override: false }),
    field("swap_policy", "select", "always", { options: ["auto", "always", "never"] }),
  ] },
  { plugin: "music", label: "Music", fields: [
    field("backend", "select", "comfy", { options: ["auto", "native", "comfy"] }),
    field("native_runtime", "select", "auto", { options: ["auto", "vulkan"] }),
    field("float_type", "select", "default", { options: ["default", "fp8_e4m3fn"] }),
    field("swap_policy", "select", "auto", { options: ["auto", "always", "never"] }),
  ] },
] };

function makeFetch({ mediaFails = false, managed = { installed: false, state: "absent" },
                     removeAnswer } = {}) {
  return async (url, opts = {}) => {
    const method = opts.method || "GET";
    const ok = (body) => ({ ok: true, status: 200, json: async () => body, text: async () => "" });
    if (url === "/v1/config/schema") return ok(structuredClone(SCHEMA));
    if (url === "/v1/media/config" && method === "GET") {
      if (mediaFails) return { ok: false, status: 500, statusText: "boom", json: async () => ({}) };
      return ok(structuredClone(MEDIA));
    }
    if (url === "/api/comfy/managed-status") return ok(managed);
    if (url === "/api/comfy/remove" && method === "POST") return ok(removeAnswer);
    return ok({ models: [], active: "", conversations: [], plugins: [] });
  };
}

async function settle(n = 14) {
  for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0));
}

async function open(opts) {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(opts) });
  await settle(1);
  runScript(win, "refreshSettingsPage();");
  await settle();
  return win;
}

const box = (win, plugin) =>
  win.document.querySelector(`.media-subsection[data-plugin="${plugin}"]`);
const ctl = (root, key) => root.querySelector(`[data-key="${key}"]`);

test("a failed per-plugin read keeps the Shared and Experimental boxes", async () => {
  const win = await open({ mediaFails: true });
  const media = win.document.getElementById("settings-sec-media");
  assert.ok(ctl(media, "comfy_launch_timeout"), "the Shared box is drawn");
  assert.ok(ctl(media, "comfy_gpu_placement"), "the Experimental box is drawn");
  assert.match(media.querySelector(".media-load-failed").textContent,
    /Could not load media settings/);
});

test("Copy from skips backend and native fields and a choice the target lacks", async () => {
  const win = await open();
  runScript(win, 'copyMediaFields("image", "music");');
  const music = box(win, "music");
  assert.equal(ctl(music, "backend").value, "comfy", "the backend choice is not copied");
  assert.equal(ctl(music, "native_runtime").value, "auto", "native settings are not copied");
  assert.equal(ctl(music, "swap_policy").value, "always", "a shared setting is copied");
});

test("a native backend hides the ComfyUI-only fields; comfy shows them", async () => {
  const win = await open();
  const image = box(win, "image");
  const dtype = ctl(image, "float_type").closest("[data-field-key]");
  assert.equal(dtype.hidden, true, "hidden while the backend is native");
  const backend = ctl(image, "backend");
  backend.value = "comfy";
  backend.dispatchEvent(new win.Event("change", { bubbles: true }));
  assert.equal(dtype.hidden, false, "shown for the comfy backend");
  assert.equal(ctl(box(win, "music"), "float_type").closest("[data-field-key]").hidden, false);
});

test("an unset select shows a blank choice instead of an empty box", async () => {
  const win = await open();
  const dtype = ctl(box(win, "image"), "float_type");
  assert.equal(dtype.value, "");
  assert.equal(dtype.selectedOptions[0].textContent, "(inherit)");
  const pooling = ctl(win.document, "embedding_pooling");
  assert.equal(pooling.value, "");
  assert.equal(pooling.selectedOptions[0].textContent, "(not set)");
});

test("ComfyUI to use is offered before localm's own ComfyUI is installed", async () => {
  const win = await open();
  const target = ctl(win.document.getElementById("settings-sec-media"), "comfy_target");
  assert.ok(target, "the control is drawn in the not-set-up state");
  assert.ok(win.document.querySelector(".comfy-managed-setup-btn"), "Set up is still offered");
});

test("Remove with nothing installed says nothing was removed", async () => {
  const win = await open({ managed: { installed: true, managed_active: true, path: "/c" },
                           removeAnswer: { status: "noop", message: "x", removed: [] } });
  runScript(win, "confirmDanger = (title, body, label, onOk) => onOk();");
  const remove = [...win.document.querySelectorAll("#settings-sec-media button")]
    .find((b) => b.textContent === "Remove");
  assert.ok(remove, "a Remove button is drawn for an installed ComfyUI");
  remove.click();
  await settle();
  assert.equal(win.document.getElementById("toast").textContent,
    "No managed ComfyUI is installed, so nothing was removed.");
});
