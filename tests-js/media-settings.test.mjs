// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// The schema carries the comfy_* media keys in group "Media". The settings form
// skips the ones annotated media_per_plugin; those are edited per-plugin in the
// Media section via /v1/media/config.
const SCHEMA = {
  fields: [
    { key: "n_ctx", widget: "number", label: "Context window", help: "",
      group: "Engine", owner: "core", min: 512, step: 512, default: 4096 },
    { key: "comfy_workdir", widget: "folder", label: "ComfyUI folder", help: "",
      group: "Media", owner: "image", default: "/shared", media_per_plugin: true },
    { key: "comfy_delete_outputs", widget: "toggle", label: "Remove copy",
      help: "", group: "Media", owner: "image", default: false,
      media_per_plugin: true },
  ],
};

// Resolved per-plugin media config: only image has fast_dequant, and every field
// inherits the shared default (is_override false).
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
  // Lets the one-shot /api/capabilities fetch settle, then pins host access so
  // the host-path fields render.
  await new Promise((r) => setTimeout(r, 0));
  runScript(win, `caps.fsAccess = "host"; refreshSettingsPage();`);
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
}

// The subsection <h4> holds the plugin label as its first text node, followed by
// a ComfyUI status badge <span>, so only the first child node is read.
const headLabel = (s) => {
  const h = s.querySelector(".media-sub-head");
  return (h && h.firstChild ? h.firstChild.textContent : "").trim();
};

test("media config renders one independent subsection per plugin", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  const doc = win.document;

  // The group=Media schema keys are not rendered in the core form.
  assert.equal(doc.querySelector('[data-key="comfy_workdir"]'), null,
    "comfy_* media keys are skipped from the schema form");

  const media = doc.querySelector("#settings-sec-media");
  assert.ok(media, "a Media section exists");
  const subs = media.querySelectorAll(".media-subsection");
  assert.equal(subs.length, 3, "image/music/video each get a subsection");
  const heads = [...subs].map(headLabel);
  assert.deepEqual(heads, ["Image", "Music", "Video"], "in order, labelled");

  // fast_dequant appears only on the image subsection.
  const imageSub = [...subs].find((s) => headLabel(s) === "Image");
  const musicSub = [...subs].find((s) => headLabel(s) === "Music");
  assert.ok(imageSub.querySelector('[data-key="fast_dequant"]'),
    "image has the fast_dequant control");
  assert.equal(musicSub.querySelector('[data-key="fast_dequant"]'), null,
    "music has no fast_dequant control");

  // Each subsection has its own Save; the Media section is its own top-level group.
  assert.ok(imageSub.querySelector(".media-save"), "image subsection has a Save");
  assert.equal(doc.querySelector("#settings-sec-media").dataset.group, "media",
    "the Media section is its own top-level group");
  const navLabels = [...doc.querySelectorAll("#settings-nav .settings-nav-link")]
    .map((l) => l.textContent);
  assert.ok(navLabels.includes("Media"), "the Media group appears in the settings nav");

  // The three subsections sit in the responsive grid.
  const grid = media.querySelector(".media-grid");
  assert.ok(grid, "the three subsections sit inside a .media-grid container");
  assert.equal(grid.querySelectorAll(".media-subsection").length, 3,
    "all three subsections are inside the grid");
});

test("saving a media plugin POSTs only the changed fields", async () => {
  const posts = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  await render(win);
  const doc = win.document;

  const imageSub = [...doc.querySelectorAll(".media-subsection")].find(
    (s) => headLabel(s) === "Image");

  // Override the inherited workdir and toggle delete_outputs; leave fast_dequant
  // and swap_policy at their inherited values.
  imageSub.querySelector('input[data-key="workdir"]').value = "/img/own";
  imageSub.querySelector('input[data-key="delete_outputs"]').checked = true;

  imageSub.querySelector(".media-save").click();
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));

  assert.equal(posts.length, 1, "one POST to the image media config");
  assert.equal(posts[0].name, "image");
  // Compared by value: the parsed body carries the jsdom realm's prototype.
  assert.equal(JSON.stringify(posts[0].body),
    JSON.stringify({ workdir: "/img/own", delete_outputs: true }),
    "only the changed fields are sent (inherited untouched fields are not pinned)");
});

test("R12: saving one media subsection preserves unsaved edits in the others", async () => {
  // Each plugin's POST echoes its own fields back.
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

  // Edit the music subsection without saving, then save the image subsection.
  sub("music").querySelector('input[data-key="workdir"]').value = "/music/edited";
  sub("image").querySelector('input[data-key="workdir"]').value = "/image/own";
  sub("image").querySelector(".media-save").click();
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));

  assert.equal(posts.length, 1, "only the image subsection was saved");
  assert.equal(posts[0].name, "image");
  // The music subsection's unsaved edit is still in the DOM.
  assert.equal(sub("music").querySelector('input[data-key="workdir"]').value,
    "/music/edited", "the other subsection's unsaved edit survives the save");
});

test("EXPERIMENTAL comfy_gpu_placement toggle renders in Media and saves via /v1/config", async () => {
  // A schema carrying the experimental placement toggle: a core group=Media
  // toggle that renders in the Media section and saves through PATCH /v1/config.
  const SCHEMA_P = { fields: [
    ...SCHEMA.fields,
    { key: "comfy_gpu_placement", widget: "toggle",
      label: "Split media across GPUs (experimental)",
      help: "Off by default. Puts CLIP+VAE on a second card.",
      group: "Media", owner: "image", default: false, applies: "restart" },
  ] };
  const patches = [];
  const fetchImpl = async (url, opts = {}) => {
    const method = opts.method || "GET";
    if (url === "/v1/config/schema")
      return { ok: true, status: 200, json: async () => SCHEMA_P, text: async () => "" };
    if (url === "/v1/media/config" && method === "GET")
      return { ok: true, status: 200, json: async () => MEDIA, text: async () => "" };
    if (url === "/v1/config" && method === "PATCH") {
      patches.push(JSON.parse(opts.body));
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await render(win);
  const doc = win.document;

  // The toggle renders inside the Media section, not the flat core form.
  const media = doc.querySelector("#settings-sec-media");
  const toggle = media.querySelector('input[data-key="comfy_gpu_placement"]');
  assert.ok(toggle, "the experimental placement toggle renders in the Media section");
  assert.equal(toggle.type, "checkbox", "it is a checkbox (a TOGGLE widget)");
  assert.equal(toggle.checked, false, "default off");
  // It is a single global control, not part of a per-plugin subsection.
  const box = toggle.closest(".media-comfy-box");
  assert.ok(box, "the toggle sits in a media-comfy box, not a per-plugin subsection");
  assert.ok(/experimental/i.test(box.textContent),
    "the box is labelled experimental so the user knows it is unproven");

  // Turning it on and clicking its Save PATCHes /v1/config with just this key.
  toggle.checked = true;
  const save = box.querySelector("button.btn-primary");
  assert.ok(save, "the experimental box has its own Save");
  save.click();
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
  assert.equal(patches.length, 1, "one PATCH /v1/config");
  assert.equal(patches[0].comfy_gpu_placement, true,
    "the toggle value reaches the config PATCH");
});

test("R14: in each media subsection the dropdown renders before the checkboxes", async () => {
  // The server field order: folders and text, then selects, then toggles.
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
