// SPDX-License-Identifier: AGPL-3.0-or-later
// The Media section must render EVERY visible group="Media" schema field
// somewhere. The flat settings loop deliberately skips group="Media" (those
// fields belong to the Media section), but buildMediaSection historically
// picked only two of them by name (comfy_target, comfy_gpu_placement) and the
// per-plugin boxes cover only the MEDIA_PLUGIN_FIELDS-mapped globals - which
// silently dropped comfy_launch_timeout, comfy_disable_auto_launch and
// comfy_func_shim from the GUI entirely (schema descriptions written to be
// seen, changeable only via CLI/API; found in the 2026-07-22 settings-
// exposure audit). The fix renders every un-mapped, un-special-cased Media
// field in a "Shared" box, keyed on the schema's media_per_plugin annotation
// (single source of truth: settings_schema.MEDIA_PLUGIN_FIELDS) - so a FUTURE
// Media field lands visible by default instead of vanishing.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = { fields: [
  // Per-plugin-mapped global (annotated): must NOT get a flat/shared control -
  // the per-plugin boxes own it.
  { key: "comfy_workdir", widget: "folder", label: "ComfyUI folder", help: "",
    group: "Media", owner: "image", default: "/shared", media_per_plugin: true },
  // The two explicitly-placed fields.
  { key: "comfy_target", widget: "select", label: "ComfyUI to use", help: "",
    group: "Media", owner: "image", options: ["auto", "managed", "external"],
    default: "auto", media_per_plugin: false },
  { key: "comfy_gpu_placement", widget: "toggle", label: "Per-component GPU placement",
    help: "", group: "Media", owner: "image", default: false, media_per_plugin: false },
  // The three orphans this fix makes visible.
  { key: "comfy_launch_timeout", widget: "number", label: "ComfyUI launch timeout (s)",
    help: "", group: "Media", owner: "image", default: 300, media_per_plugin: false },
  { key: "comfy_disable_auto_launch", widget: "toggle", label: "Keep ComfyUI headless",
    help: "", group: "Media", owner: "image", default: false, media_per_plugin: false },
  { key: "comfy_func_shim", widget: "toggle", label: "Fix ComfyUI ACE-Step crash",
    help: "", group: "Media", owner: "image", default: false, media_per_plugin: false },
  // Class guard: a field this test's harness invents. The renderer must not
  // have a hardcoded allowlist - anything un-mapped renders (fail-open to
  // VISIBLE), so the next Media knob cannot silently vanish like the three
  // above did.
  { key: "comfy_future_knob", widget: "toggle", label: "Future media knob",
    help: "", group: "Media", owner: "image", default: false, media_per_plugin: false },
]};

const MEDIA = { plugins: [
  { plugin: "image", label: "Image", fields: [
    { key: "workdir", widget: "folder", label: "ComfyUI folder", help: "",
      value: "/shared", is_override: false }] },
]};

function makeFetch() {
  return async (url, opts = {}) => {
    const method = opts.method || "GET";
    if (url === "/v1/config/schema")
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    if (url === "/v1/media/config" && method === "GET")
      return { ok: true, status: 200, json: async () => MEDIA, text: async () => "" };
    if (url === "/v1/comfy/status")
      return { ok: true, status: 200, json: async () => ({ alive: false, launched_by_localm: false }), text: async () => "" };
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function render(win) {
  runScript(win, "refreshSettingsPage();");
  for (let i = 0; i < 12; i++) await new Promise((r) => setTimeout(r, 0));
}

function mediaSection(win) {
  const sec = win.document.getElementById("settings-sec-media");
  assert.ok(sec, "the Media section rendered");
  return sec;
}

test("the previously-orphaned shared Media fields render in the Media section", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  const sec = mediaSection(win);
  for (const key of ["comfy_launch_timeout", "comfy_disable_auto_launch",
                     "comfy_func_shim"]) {
    assert.ok(sec.querySelector(`[data-field-key="${key}"]`),
      `${key} must render a control in the Media section (it was GUI-invisible)`);
  }
});

test("an unmapped future Media field renders too (fail-open, no allowlist)", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  const sec = mediaSection(win);
  assert.ok(sec.querySelector('[data-field-key="comfy_future_knob"]'),
    "a new group=Media field must be visible by default, never silently dropped");
});

test("a per-plugin-mapped global gets no duplicate shared control", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  const sec = mediaSection(win);
  // The per-plugin boxes render it under its per-plugin key ("workdir");
  // the shared box must not render the GLOBAL key alongside.
  assert.equal(sec.querySelector('[data-field-key="comfy_workdir"]'), null,
    "comfy_workdir is per-plugin-mapped; a shared duplicate would be confusing");
});

test("the shared box never duplicates the explicitly-placed fields", async () => {
  // comfy_target renders inside the managed-ComfyUI panel, whose visibility
  // depends on the managed-status fetch (it may legitimately render 0 times
  // in this harness), so the invariant is NOT "exactly once" - it is that the
  // Shared box excludes both explicitly-placed keys, and nothing renders
  // either of them twice.
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await render(win);
  const sec = mediaSection(win);
  const shared = [...sec.querySelectorAll(".media-comfy-box")]
    .find((b) => (b.querySelector("h4") || {}).textContent === "Shared");
  assert.ok(shared, "the Shared box rendered (it holds the orphaned fields)");
  for (const key of ["comfy_target", "comfy_gpu_placement"]) {
    assert.equal(shared.querySelector(`[data-field-key="${key}"]`), null,
      `${key} must not appear in the Shared box (it has a dedicated home)`);
    assert.ok(sec.querySelectorAll(`[data-field-key="${key}"]`).length <= 1,
      `${key} must never render twice in the Media section`);
  }
});
