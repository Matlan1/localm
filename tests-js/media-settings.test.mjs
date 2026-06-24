// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// The schema still carries the comfy_* media keys (group "Media"), but the
// settings form must SKIP them - they are edited in the Media section instead,
// per-plugin, via /v1/media/config.
const SCHEMA = {
  fields: [
    { key: "n_ctx", widget: "number", label: "Context window", help: "",
      group: "Engine", owner: "core", min: 512, step: 512, default: 4096 },
    { key: "comfy_workdir", widget: "folder", label: "ComfyUI folder", help: "",
      group: "Media", owner: "image", default: "/shared" },
    { key: "comfy_delete_outputs", widget: "toggle", label: "Remove copy",
      help: "", group: "Media", owner: "image", default: false },
  ],
};

// Resolved per-plugin media config: image has fast_dequant (Flux-only), the
// others do not; everything inherits the shared default here (is_override false).
const MEDIA = {
  plugins: [
    { plugin: "image", label: "Image", fields: [
      { key: "workdir", widget: "folder", label: "ComfyUI folder", help: "",
        value: "/shared", is_override: false },
      { key: "delete_outputs", widget: "toggle", label: "Remove ComfyUI copy",
        help: "", value: false, is_override: false },
      { key: "fast_dequant", widget: "toggle", label: "Fast GGUF dequant",
        help: "", value: true, is_override: false },
      { key: "swap_policy", widget: "select", label: "Media VRAM swap", help: "",
        value: "auto", is_override: false, options: ["auto", "always", "never"] },
    ] },
    { plugin: "music", label: "Music", fields: [
      { key: "workdir", widget: "folder", label: "ComfyUI folder", help: "",
        value: "/shared", is_override: false },
    ] },
    { plugin: "video", label: "Video", fields: [
      { key: "workdir", widget: "folder", label: "ComfyUI folder", help: "",
        value: "/shared", is_override: false },
    ] },
  ],
};

function makeFetch(posts) {
  return async (url, opts = {}) => {
    const method = opts.method || "GET";
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    }
    if (url === "/v1/media/config" && method === "GET") {
      return { ok: true, status: 200, json: async () => MEDIA, text: async () => "" };
    }
    if (url.startsWith("/v1/media/config/") && method === "POST") {
      const name = url.split("/").pop();
      posts.push({ name, body: JSON.parse(opts.body) });
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ plugin: name, fields: MEDIA.plugins[0].fields }) };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

async function render(win) {
  runScript(win, "refreshSettingsPage();");
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
}

test("media config renders one independent subsection per plugin", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  const doc = win.document;

  // The group=Media schema keys are NOT rendered in the core form.
  assert.equal(doc.querySelector('[data-key="comfy_workdir"]'), null,
    "comfy_* media keys are skipped from the schema form");

  const media = doc.querySelector("#settings-sec-media");
  assert.ok(media, "a Media section exists");
  const subs = media.querySelectorAll(".media-subsection");
  assert.equal(subs.length, 3, "image/music/video each get a subsection");
  const heads = [...subs].map((s) => s.querySelector(".media-sub-head").textContent);
  assert.deepEqual(heads, ["Image", "Music", "Video"], "in order, labelled");

  // fast_dequant (Flux-only) only on the image subsection.
  const imageSub = [...subs].find(
    (s) => s.querySelector(".media-sub-head").textContent === "Image");
  const musicSub = [...subs].find(
    (s) => s.querySelector(".media-sub-head").textContent === "Music");
  assert.ok(imageSub.querySelector('[data-key="fast_dequant"]'),
    "image has the fast_dequant control");
  assert.equal(musicSub.querySelector('[data-key="fast_dequant"]'), null,
    "music has no fast_dequant control");

  // Each subsection has its own Save, and the Media section is in the nav.
  assert.ok(imageSub.querySelector(".media-save"), "image subsection has a Save");
  const navLabels = [...doc.querySelectorAll("#settings-nav .settings-nav-link")]
    .map((l) => l.textContent);
  assert.ok(navLabels.includes("Media"), "Media appears in the settings nav");
});

test("saving a media plugin POSTs only the changed fields", async () => {
  const posts = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  await render(win);
  const doc = win.document;

  const imageSub = [...doc.querySelectorAll(".media-subsection")].find(
    (s) => s.querySelector(".media-sub-head").textContent === "Image");

  // Override the inherited workdir and toggle delete_outputs; leave fast_dequant
  // and swap_policy at their inherited values.
  imageSub.querySelector('input[data-key="workdir"]').value = "/img/own";
  imageSub.querySelector('input[data-key="delete_outputs"]').checked = true;

  imageSub.querySelector(".media-save").click();
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));

  assert.equal(posts.length, 1, "one POST to the image media config");
  assert.equal(posts[0].name, "image");
  // Compare by value (jsdom realm): only the changed fields are sent.
  assert.equal(JSON.stringify(posts[0].body),
    JSON.stringify({ workdir: "/img/own", delete_outputs: true }),
    "only the changed fields are sent (inherited untouched fields are not pinned)");
});

test("R12: saving one media subsection preserves unsaved edits in the others", async () => {
  // Each plugin's POST echoes ITS OWN fields back (so the R12 re-render is correct).
  const posts = [];
  const fetchImpl = async (url, opts = {}) => {
    const method = opts.method || "GET";
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    }
    if (url === "/v1/media/config" && method === "GET") {
      return { ok: true, status: 200, json: async () => MEDIA, text: async () => "" };
    }
    if (url.startsWith("/v1/media/config/") && method === "POST") {
      const name = url.split("/").pop();
      posts.push({ name, body: JSON.parse(opts.body) });
      const p = MEDIA.plugins.find((x) => x.plugin === name);
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ plugin: name, fields: p.fields }) };
    }
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await render(win);
  const doc = win.document;
  const sub = (name) => [...doc.querySelectorAll(".media-subsection")]
    .find((s) => s.dataset.plugin === name);

  // Edit the MUSIC subsection (unsaved), then SAVE the IMAGE subsection.
  sub("music").querySelector('input[data-key="workdir"]').value = "/music/edited";
  sub("image").querySelector('input[data-key="workdir"]').value = "/image/own";
  sub("image").querySelector(".media-save").click();
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));

  assert.equal(posts.length, 1, "only the image subsection was saved");
  assert.equal(posts[0].name, "image");
  // The music subsection's unsaved edit is still in the DOM (not wiped by a refresh).
  assert.equal(sub("music").querySelector('input[data-key="workdir"]').value,
    "/music/edited", "the other subsection's unsaved edit survives the save");
});

test("R14: in each media subsection the dropdown renders before the checkboxes", async () => {
  // Mirror the post-R14 server field order (folders/text, then SELECT, then toggles).
  const ORDERED = { plugins: [{ plugin: "image", label: "Image", fields: [
    { key: "workdir", widget: "folder", label: "Folder", help: "", value: "", is_override: false },
    { key: "swap_policy", widget: "select", label: "Swap", help: "", value: "auto",
      is_override: false, options: ["auto", "always", "never"] },
    { key: "delete_outputs", widget: "toggle", label: "Remove", help: "", value: false, is_override: false },
    { key: "reload_after", widget: "toggle", label: "Reload", help: "", value: false, is_override: false },
  ] }] };
  const fetchImpl = async (url, opts = {}) => {
    if (url === "/v1/config/schema") return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    if (url === "/v1/media/config" && (opts.method || "GET") === "GET")
      return { ok: true, status: 200, json: async () => ORDERED, text: async () => "" };
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await render(win);
  const imageSub = [...win.document.querySelectorAll(".media-subsection")]
    .find((s) => s.dataset.plugin === "image");
  const controls = [...imageSub.querySelectorAll(".settings-fields [data-key]")];
  const selIdx = controls.findIndex((c) => c.tagName === "SELECT");
  const firstToggle = controls.findIndex((c) => c.type === "checkbox");
  assert.ok(selIdx !== -1 && firstToggle !== -1, "both a dropdown and a checkbox render");
  assert.ok(selIdx < firstToggle, "the dropdown renders before any checkbox");
});
