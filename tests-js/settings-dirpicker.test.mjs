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

// Save the section that contains the given input (per-section Save button).
function saveSectionOf(win, key) {
  win.document.querySelector(`[data-key="${key}"]`)
    .closest(".settings-section").querySelector(".settings-section-save").click();
}

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

async function render(win, fsAccess = "host") {
  // init.js calls refreshPluginCommands() at load, whose /api/capabilities fetch
  // resolves a tick later and sets caps.fsAccess from the (stubbed) response.
  // Let that one-shot settle FIRST, then pin caps for this render - nothing
  // re-fetches capabilities during refreshSettingsPage, so it stays put.
  await new Promise((r) => setTimeout(r, 0));
  // Folder/path fields only render for a HOST-access caller; default the tests
  // to host so the Browse-button behaviour is exercised.
  runScript(win, `caps.fsAccess = ${JSON.stringify(fsAccess)}; refreshSettingsPage();`);
  await new Promise((r) => setTimeout(r, 0));
}

test("folder/path fields render a Browse button; other widgets do not", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  const doc = win.document;

  assert.ok(doc.querySelector('button[data-browse="binary_dir"]'),
    "folder field has a Browse button");
  assert.ok(doc.querySelector('button[data-browse="some_path"]'),
    "path field has a Browse button");
  // The browse buttons must not submit anything.
  assert.equal(doc.querySelector('button[data-browse="binary_dir"]').type, "button");

  assert.equal(doc.querySelector('button[data-browse="n_ctx"]'), null,
    "number field has no Browse button");
  assert.equal(doc.querySelector('button[data-browse="mode"]'), null,
    "select field has no Browse button");
});

test("without host access, host-path fields are hidden entirely", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win, "none");
  const doc = win.document;
  // The folder + path fields (server-side host config) are not rendered at all.
  assert.equal(doc.querySelector('[data-key="binary_dir"]'), null,
    "folder field hidden for a non-host caller");
  assert.equal(doc.querySelector('[data-key="some_path"]'), null,
    "path field hidden for a non-host caller");
  // Non-path fields still render normally.
  assert.ok(doc.querySelector('[data-key="n_ctx"]'), "number field still shown");
  assert.ok(doc.querySelector('[data-key="mode"]'), "select field still shown");
});

test("clicking Browse fills the input from pickDirectory and saves it", async () => {
  const patches = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(patches) });
  await render(win);
  const doc = win.document;

  // Stub the picker so no modal/fetchDirs machinery is needed; capture its args.
  runScript(win, `
    globalThis.__pickArgs = null;
    pickDirectory = (title, start) => { globalThis.__pickArgs = [title, start]; return Promise.resolve("/picked/dir"); };
  `);

  const input = doc.querySelector('input[data-key="binary_dir"]');
  assert.equal(input.value, "/old/dir", "input prefilled with current value");

  doc.querySelector('button[data-browse="binary_dir"]').click();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(input.value, "/picked/dir", "input updated to the picked path");
  // pickDirectory was started from the field's current value.
  assert.deepEqual(win.__pickArgs[1], "/old/dir");

  // The picked value persists through the section's own Save.
  saveSectionOf(win, "binary_dir");
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(patches.length, 1, "one PATCH sent");
  assert.equal(patches[0].binary_dir, "/picked/dir");
});

test("cancelling the picker (null) leaves the input unchanged", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await render(win);
  const doc = win.document;
  runScript(win, "pickDirectory = () => Promise.resolve(null);");

  const input = doc.querySelector('input[data-key="binary_dir"]');
  doc.querySelector('button[data-browse="binary_dir"]').click();
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(input.value, "/old/dir", "cancel keeps the original value");
});

// "Blank = auto-detect" must SHOW the resolved path (filled, greyed), not an
// empty box - but saving an unchanged auto value keeps it dynamic (sends null).
test("blank auto-detect field shows the resolved path and stays dynamic on save", async () => {
  const patches = [];
  const AUTO_SCHEMA = {
    fields: [
      { key: "binary_dir", widget: "folder", label: "Binary dir",
        help: "Blank auto-detects", group: "Engine", owner: "core",
        default: null, auto: "/runtime/lib" },
    ],
  };
  const fetchImpl = async (url, opts = {}) => {
    if (url === "/v1/config/schema")
      return { ok: true, status: 200, json: async () => AUTO_SCHEMA, text: async () => "" };
    if (url === "/v1/config" && (opts.method || "GET") === "PATCH") {
      patches.push(JSON.parse(opts.body));
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await render(win);
  const input = win.document.querySelector('input[data-key="binary_dir"]');

  assert.equal(input.value, "/runtime/lib", "auto-detected path is shown, not empty");
  assert.ok(input.classList.contains("auto-detected"), "shown as auto-detected (greyed)");

  // Save without touching it -> nothing changed (binary_dir stays blank/dynamic).
  saveSectionOf(win, "binary_dir");
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(patches.length, 0, "unchanged auto value does not pin the path");

  // Override it with a real path -> that IS sent.
  input.value = "/my/custom/build";
  input.dispatchEvent(new win.Event("input"));
  saveSectionOf(win, "binary_dir");
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(patches.length, 1);
  assert.equal(patches[0].binary_dir, "/my/custom/build");
});
