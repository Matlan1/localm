// SPDX-License-Identifier: AGPL-3.0-or-later
// S2: six raw window.prompt() call sites (model alias/rename, image/video/music
// rename, save-key-preset, save-persona, move-conversation-to-folder) went
// silent - no error, no toast - whenever a browser suppresses window.prompt()
// (the same mobile/PWA class helpers.js's confirmDanger() already documents
// and guards against for window.confirm()). promptText() (helpers.js) replaces
// all six with an in-page modal that never calls window.prompt() at all, so a
// browser suppressing it can no longer affect the feature.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));

function modalButton(win, label) {
  return [...win.document.querySelectorAll("#modal-body button")]
    .find((b) => b.textContent === label);
}

const MODELS = [
  { name: "gemma3-12b", active: false, loaded: false, model_type: "llm", size_bytes: 100 },
];

function makeModelsFetch(calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models/alias")) {
      calls.push(opts.body ? JSON.parse(opts.body) : {});
      return { ok: true, status: 200,
        json: async () => ({ status: "aliased", model: "gemma3-12b", alias: "daily-driver" }),
        text: async () => "" };
    }
    if (u === "/api/models" || u.startsWith("/api/models?")) {
      return { ok: true, status: 200, json: async () => ({ models: MODELS, active: null }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

test("models-alias: the feature still works when window.prompt() is suppressed", async () => {
  const calls = [];
  const toasts = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeModelsFetch(calls) });
  win.toast = (msg) => toasts.push(String(msg));
  // A suppressing browser's window.prompt() returns null (MDN) and, before this
  // fix, that was indistinguishable from Cancel - the alias request never fired
  // and nothing told the user why. Track whether it is even called any more.
  win.__promptCalled = false;
  win.prompt = () => { win.__promptCalled = true; return null; };

  await win.refreshModelsPage();
  await tick();

  const aliasBtn = [...win.document.querySelectorAll("#models-table tbody tr button")]
    .find((b) => b.textContent === "alias");
  assert.ok(aliasBtn, "the row exposes an alias control");
  aliasBtn.onclick();   // not awaited: promptText() suspends here until the modal answers
  await tick();

  const input = win.document.querySelector("#modal-body input[type=text]");
  assert.ok(input, "an in-page modal - not window.prompt() - collects the alias");
  input.value = "daily driver";
  const ok = modalButton(win, "OK");
  assert.ok(ok, "the modal has an OK button");
  ok.click();
  await tick();

  assert.equal(win.__promptCalled, false,
    "window.prompt() is never called - the fix does not depend on it at all");
  assert.deepEqual(calls, [{ model: "gemma3-12b", alias: "daily driver" }],
    "the alias request fired even though window.prompt() would have been suppressed");
  assert.ok(toasts.some((t) => t.includes("daily-driver")),
    `expected a success toast, got: ${JSON.stringify(toasts)}`);
});

test("models-alias: Cancel sends no request and calls window.prompt() zero times", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeModelsFetch(calls) });
  win.__promptCalled = false;
  win.prompt = () => { win.__promptCalled = true; return null; };

  await win.refreshModelsPage();
  await tick();

  const aliasBtn = [...win.document.querySelectorAll("#models-table tbody tr button")]
    .find((b) => b.textContent === "alias");
  aliasBtn.onclick();
  await tick();

  const cancel = modalButton(win, "Cancel");
  assert.ok(cancel, "the modal has a Cancel button");
  cancel.click();
  await tick();

  assert.equal(win.__promptCalled, false, "window.prompt() is never called");
  assert.deepEqual(calls, [], "cancelling must not fire a request");
});

test("chat conversation folder: Cancel leaves the folder untouched; an empty submit clears it", async () => {
  const { window: win } = loadApp({});
  const conv = { id: "c1", title: "Test", messages: [], folder: "work" };
  runScript(win, `
    window.__conv = ${JSON.stringify(conv)};
    window.__item = buildConvItem(window.__conv, null);
    document.body.appendChild(window.__item);
  `);
  const item = win.__item;
  const foldBtn = [...item.querySelectorAll("button.del")]
    .find((b) => b.title && b.title.startsWith("Folder:"));
  assert.ok(foldBtn, "the folder button shows the existing folder in its title");

  // --- Cancel: the folder must be left exactly as it was. ---
  foldBtn.onclick({ stopPropagation() {} });
  await tick();
  const cancelBtn = modalButton(win, "Cancel");
  assert.ok(cancelBtn, "the folder-name modal shows a Cancel button");
  cancelBtn.click();
  await tick();
  assert.equal(win.__conv.folder, "work", "cancelling must not touch the existing folder");

  // --- Submit empty: distinct from cancelling, this clears the folder. ---
  foldBtn.onclick({ stopPropagation() {} });
  await tick();
  const input = win.document.querySelector("#modal-body input[type=text]");
  assert.ok(input, "the folder-name modal shows a text input");
  input.value = "";
  const okBtn = modalButton(win, "OK");
  okBtn.click();
  await tick();
  assert.equal(win.__conv.folder, undefined, "submitting an emptied field clears the folder");
});

test("music rename: cancelling the nested prompt reopens the item detail view, not a bare closed modal",
  async () => {
    const calls = [];
    const fetchImpl = async (url, opts = {}) => {
      const u = String(url);
      calls.push({ url: u, opts });
      if (u.includes("/api/music/history")) {
        return { ok: true, status: 200, text: async () => "",
          json: async () => ({ tracks: [
            { name: "song.flac", path: "/g/song.flac", size_bytes: 100, meta: {} },
          ] }) };
      }
      return { ok: true, status: 200, text: async () => "", json: async () => ({}) };
    };
    const { window: win } = loadAppWithPages({ fetchImpl });
    runScript(win, "refreshMusicHistory();");
    await tick();

    win.document.getElementById("music-history").querySelector(".thumb").click();
    await tick();

    const renameBtn = [...win.document.getElementById("modal-body").querySelectorAll("button")]
      .find((b) => b.textContent.startsWith("rename"));
    assert.ok(renameBtn, "the item-detail modal exposes a rename control");
    renameBtn.click();
    await tick();

    const cancelBtn = modalButton(win, "Cancel");
    assert.ok(cancelBtn, "the rename prompt shows a Cancel button");
    cancelBtn.click();
    await tick();

    assert.notEqual(win.document.getElementById("modal").style.display, "none",
      "cancelling the rename must not close the item-detail view underneath it");
    assert.ok([...win.document.getElementById("modal-body").querySelectorAll("button")]
      .some((b) => b.textContent.startsWith("rename")),
      "the item-detail view (with its rename control) is restored after Cancel");
    assert.equal(calls.filter((c) => c.url.includes("/rename")).length, 0,
      "cancelling must not fire a rename request");
  });
