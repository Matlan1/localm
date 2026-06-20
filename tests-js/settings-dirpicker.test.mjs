// SPDX-License-Identifier: AGPL-3.0-or-later
// U10: FOLDER / PATH settings fields were plain text inputs with no way to
// browse for a directory, even though the GUI already ships a pickDirectory()
// modal (used by the coder cwd picker and image-move). buildSettingControl now
// renders a "Browse..." button next to folder/path fields that fills the input
// from pickDirectory(); other widget types get no button.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const SCHEMA = {
  fields: [
    { key: "binary_dir", widget: "folder", label: "Binary dir", help: "",
      group: "Engine", owner: "core", default: "/old/dir" },
    { key: "some_path", widget: "path", label: "A file", help: "",
      group: "Engine", owner: "core", default: "" },
    { key: "n_ctx", widget: "number", label: "Context window", help: "",
      group: "Engine", owner: "core", min: 512, step: 512, default: 4096 },
    { key: "mode", widget: "select", label: "Mode", help: "", group: "Privacy",
      owner: "core", options: ["privacy", "log", "full"], default: "log" },
  ],
};

function makeFetch(patches) {
  return async (url, opts = {}) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    }
    if (url === "/v1/config" && (opts.method || "GET") === "PATCH") {
      patches.push(JSON.parse(opts.body));
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

async function render(win) {
  runScript(win, "refreshSettingsPage();");
  await new Promise((r) => setTimeout(r, 0));
}

test("folder/path fields render a Browse button; other widgets do not", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  const form = win.document.getElementById("config-form");

  assert.ok(form.querySelector('button[data-browse="binary_dir"]'),
    "folder field has a Browse button");
  assert.ok(form.querySelector('button[data-browse="some_path"]'),
    "path field has a Browse button");
  // The browse buttons must not submit anything.
  assert.equal(form.querySelector('button[data-browse="binary_dir"]').type, "button");

  assert.equal(form.querySelector('button[data-browse="n_ctx"]'), null,
    "number field has no Browse button");
  assert.equal(form.querySelector('button[data-browse="mode"]'), null,
    "select field has no Browse button");
});

test("clicking Browse fills the input from pickDirectory and saves it", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(patches) });
  await render(win);
  const form = win.document.getElementById("config-form");

  // Stub the picker so no modal/fetchDirs machinery is needed; capture its args.
  runScript(win, `
    globalThis.__pickArgs = null;
    pickDirectory = (title, start) => { globalThis.__pickArgs = [title, start]; return Promise.resolve("/picked/dir"); };
  `);

  const input = form.querySelector('input[data-key="binary_dir"]');
  assert.equal(input.value, "/old/dir", "input prefilled with current value");

  form.querySelector('button[data-browse="binary_dir"]').click();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(input.value, "/picked/dir", "input updated to the picked path");
  // pickDirectory was started from the field's current value.
  assert.deepEqual(win.__pickArgs[1], "/old/dir");

  // The picked value persists through save.
  win.document.getElementById("config-save").click();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(patches.length, 1, "one PATCH sent");
  assert.equal(patches[0].binary_dir, "/picked/dir");
});

test("cancelling the picker (null) leaves the input unchanged", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  const form = win.document.getElementById("config-form");
  runScript(win, "pickDirectory = () => Promise.resolve(null);");

  const input = form.querySelector('input[data-key="binary_dir"]');
  form.querySelector('button[data-browse="binary_dir"]').click();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(input.value, "/old/dir", "cancel keeps the original value");
});
