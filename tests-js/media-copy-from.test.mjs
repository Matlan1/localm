// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = { fields: [
  { key: "n_ctx", widget: "number", label: "Context", help: "", group: "Engine",
    owner: "core", min: 512, step: 512, default: 4096 },
] };

const MEDIA = { plugins: [
  { plugin: "image", label: "Image", fields: [
    { key: "workdir", widget: "folder", label: "Folder", help: "", value: "/img", is_override: true },
    { key: "api_url", widget: "text", label: "API URL", help: "", value: "http://img", is_override: true },
    { key: "fast_dequant", widget: "toggle", label: "Fast", help: "", value: true, is_override: false },
  ] },
  { plugin: "music", label: "Music", fields: [
    { key: "workdir", widget: "folder", label: "Folder", help: "", value: "", is_override: false },
    { key: "api_url", widget: "text", label: "API URL", help: "", value: "", is_override: false },
  ] },
  { plugin: "video", label: "Video", fields: [
    { key: "workdir", widget: "folder", label: "Folder", help: "", value: "", is_override: false },
    { key: "api_url", widget: "text", label: "API URL", help: "", value: "", is_override: false },
  ] },
] };

function makeFetch(posts) {
  return async (url, opts = {}) => {
    const method = opts.method || "GET";
    if (url === "/v1/config/schema") return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    if (url === "/v1/media/config" && method === "GET") return { ok: true, status: 200, json: async () => MEDIA, text: async () => "" };
    if (url.startsWith("/v1/media/config/") && method === "POST") {
      posts.push({ name: url.split("/").pop(), body: JSON.parse(opts.body) });
      return { ok: true, status: 200, text: async () => "", json: async () => ({}) };
    }
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function render(win) {
  // let init.js's one-shot /api/capabilities fetch settle, then pin host access
  // so the host-path fields render
  await new Promise((r) => setTimeout(r, 0));
  runScript(win, `caps.fsAccess = "host"; refreshSettingsPage();`);
  for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0));
}
const sub = (doc, name) => [...doc.querySelectorAll(".media-subsection")]
  .find((s) => s.dataset.plugin === name);

test("R11: a subsection offers Copy-from for the OTHER plugins, never itself", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  const labels = [...sub(win.document, "music").querySelectorAll(".media-copy-from")]
    .map((b) => b.dataset.from);
  assert.deepEqual(labels.sort(), ["image", "video"], "Copy-from offers the other two only");
});

test("R11: Copy-from prefills the shared fields without any server call", async () => {
  const posts = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(posts) });
  await render(win);
  const doc = win.document;

  // copy Image's values into Music
  const btn = [...sub(doc, "music").querySelectorAll(".media-copy-from")]
    .find((b) => b.dataset.from === "image");
  btn.click();

  assert.equal(sub(doc, "music").querySelector('input[data-key="workdir"]').value, "/img",
    "the folder was prefilled from Image");
  assert.equal(sub(doc, "music").querySelector('input[data-key="api_url"]').value, "http://img",
    "the API URL was prefilled from Image");
  assert.equal(posts.length, 0, "prefill makes no server call - the user still Saves");
  // image-only fields (fast_dequant) are not shared and are not copied in
  assert.equal(sub(doc, "music").querySelector('[data-key="fast_dequant"]'), null);
});

test("R13: copy A->B then B->A does not loop or corrupt values (snapshot copy)", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  const doc = win.document;

  // Music starts blank. Copy Image -> Music, then Music -> Video.
  const copy = (into, from) => [...sub(doc, into).querySelectorAll(".media-copy-from")]
    .find((b) => b.dataset.from === from).click();
  copy("music", "image");           // music.workdir = "/img"
  copy("video", "music");           // video.workdir = "/img"
  // change image, then copy image -> music again
  sub(doc, "image").querySelector('input[data-key="workdir"]').value = "/img2";
  copy("music", "image");
  assert.equal(sub(doc, "music").querySelector('input[data-key="workdir"]').value, "/img2",
    "each copy is an independent snapshot of the source at click time");
  // video still holds the earlier snapshot
  assert.equal(sub(doc, "video").querySelector('input[data-key="workdir"]').value, "/img",
    "no live binding - an earlier copy is unaffected by later source edits");
});
