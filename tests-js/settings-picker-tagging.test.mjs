// SPDX-License-Identifier: AGPL-3.0-or-later
// Two regressions, one root cause (2026-08-13).
//
// buildSettingControl decided "is this a path field" partly by SUBSTRING-MATCHING
// the human label ("file"/"folder"/"dir"/"cmd") and the key spelling
// (_path/_file/_dir). Six fields that are not paths at all matched: a number
// labelled "Folder import depth", a toggle labelled "...missing files", a select
// labelled "Indexing folder rule", and two number caps whose key ends "_file".
//
// That produced TWO user-visible faults:
//   1. each got a "Browse..." button it has no use for; and
//   2. because a path field is hidden until /api/capabilities resolves, each
//      VANISHED from a cold render - taking the entire Knowledge section with it,
//      so RAG indexing could not be configured from the GUI at all.
//
// Note what the pre-existing tests could not catch: settings-dirpicker's render()
// helper PINS caps.fsAccess before calling refreshSettingsPage, so no test could
// ever observe the unresolved state that fault 2 lives in. These deliberately do
// not pin it.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// Every field here is named the way the OLD heuristic would have mis-flagged it:
// the label carries "folder"/"file", and coder_grep_max_per_file also ends "_file".
const SCHEMA = {
  fields: [
    { key: "binary_dir", widget: "folder", label: "llama.cpp binary folder",
      help: "", group: "Engine", owner: "core", default: "" },
    { key: "import_max_depth", widget: "number", label: "Folder import depth",
      help: "", group: "Models", owner: "core", default: 3 },
    { key: "autoprune_missing_models", widget: "toggle",
      label: "Auto-remove entries for missing files", help: "", group: "Models",
      owner: "core", default: false },
    { key: "coder_grep_max_per_file", widget: "number",
      label: "Coder grep matches per file", help: "", group: "Coder",
      owner: "coder", default: 20 },
    { key: "rag_indexing_mode", widget: "select", label: "Indexing folder rule",
      help: "", group: "Knowledge", owner: "rag",
      options: ["whitelist", "blacklist"], default: "whitelist" },
    { key: "rag_classify_unknown_files", widget: "toggle",
      label: "Classify unknown files with AI", help: "", group: "Knowledge",
      owner: "rag", default: false },
  ],
};

/** *capsDelay* ticks before /api/capabilities resolves, so a test can render
 *  while the answer is still in flight - the cold-load case. */
function makeFetch({ capsDelay = 0, fsAccess = "host" } = {}) {
  return async (url) => {
    if (url === "/v1/config/schema") {
      return { ok: true, status: 200, json: async () => SCHEMA, text: async () => "" };
    }
    if (String(url).startsWith("/api/capabilities")) {
      for (let i = 0; i < capsDelay; i++) await new Promise((r) => setTimeout(r, 0));
      return {
        ok: true, status: 200, text: async () => "",
        json: async () => ({ plugins: [], fs_access: fsAccess }),
      };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

async function settle(win, ticks = 8) {
  for (let i = 0; i < ticks; i++) await new Promise((r) => setTimeout(r, 0));
  return win;
}

test("a picker is attached by WIDGET TAG only, never by label or key spelling", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  await settle(win);
  runScript(win, "refreshSettingsPage();");
  await settle(win);
  const doc = win.document;

  // The one real path field keeps its Browse button.
  assert.ok(doc.querySelector('button[data-browse="binary_dir"]'),
    "folder widget still gets a Browse button");

  // None of these is a path, however it is spelled.
  for (const key of ["import_max_depth", "autoprune_missing_models",
                     "coder_grep_max_per_file", "rag_indexing_mode",
                     "rag_classify_unknown_files"]) {
    assert.equal(doc.querySelector(`button[data-browse="${key}"]`), null,
      `${key} must not get a Browse button (it is not a path field)`);
  }
});

test("a cold render, with capabilities still in flight, still shows every field", async () => {
  // The regression: refreshSettingsPage used to read caps.fsAccess while it was
  // still the safe default and drop every path-ish field. caps is deliberately
  // NOT pinned here - that is the whole point.
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch({ capsDelay: 4 }) });
  runScript(win, "refreshSettingsPage();");
  await settle(win, 20);
  const doc = win.document;

  assert.ok(doc.querySelector('[data-field-key="binary_dir"]'),
    "binary_dir must render once capabilities resolve to host, not be dropped "
    + "because the answer had not arrived when the render started");

  for (const key of ["import_max_depth", "autoprune_missing_models",
                     "coder_grep_max_per_file", "rag_indexing_mode",
                     "rag_classify_unknown_files"]) {
    assert.ok(doc.querySelector(`[data-field-key="${key}"]`),
      `${key} must render on a cold load`);
  }

  // The section that used to disappear wholesale.
  const knowledge = [...doc.querySelectorAll("section.settings-section")]
    .some((s) => (s.dataset.secLabel || "").toLowerCase().includes("knowledge"));
  assert.ok(knowledge, "the Knowledge section must exist on a cold render");
});

test("a genuine non-host caller still has host-path fields hidden", async () => {
  // The gate must still DO its job once the answer is known: this is what stops
  // the fix from becoming 'render host paths to everyone'.
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch({ fsAccess: "none" }),
  });
  await settle(win);
  runScript(win, "refreshSettingsPage();");
  await settle(win);
  const doc = win.document;

  assert.equal(doc.querySelector('[data-field-key="binary_dir"]'), null,
    "a folder field stays hidden for a caller without host filesystem access");
  // ...but a number/toggle/select is not a path field and must still render.
  assert.ok(doc.querySelector('[data-field-key="import_max_depth"]'),
    "a number field must render regardless of filesystem access");
});
