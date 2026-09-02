// SPDX-License-Identifier: AGPL-3.0-or-later
// Tests for the shared file/folder picker (app/picker.js): pickPath() plus the
// pickDirectory()/pickFile() back-compat wrappers. Drives the real module in
// jsdom against a fake /api/fs/dirs + /api/fs/places filesystem.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { loadApp, runScript } from "./harness.mjs";

const DE = JSON.parse(readFileSync(
  join(dirname(fileURLToPath(import.meta.url)), "..",
    "localm", "plugins", "gui", "static", "i18n", "de.json"),
  "utf-8"));

const FS = {
  "/root": {
    path: "/root", parent: "/", entries: [
      { name: "sub", is_dir: true, size: null, mtime: 1700000000 },
      { name: ".git", is_dir: true, size: null, mtime: 1700000000 },
      { name: "apple.md", is_dir: false, size: 1234, mtime: 1700000000 },
      { name: "photo.png", is_dir: false, size: 5000, mtime: 1700000000 },
    ],
  },
  "/root/sub": {
    path: "/root/sub", parent: "/root", entries: [
      { name: "note.txt", is_dir: false, size: 10, mtime: 1700000000 },
    ],
  },
  "/root/.git": {
    path: "/root/.git", parent: "/root", entries: [
      { name: "HEAD", is_dir: false, size: 21, mtime: 1700000000 },
    ],
  },
  "/root/Fresh": { path: "/root/Fresh", parent: "/root", entries: [] },
  "": { path: "", parent: null, entries: [
    { name: "/", is_dir: true, size: null, mtime: null },
  ] },
};

function json(obj) {
  return { ok: true, status: 200, json: async () => obj, text: async () => "" };
}
function fetchImpl(url) {
  const u = String(url);
  if (u.includes("/api/fs/places")) {
    return Promise.resolve(json({
      places: [{ label: "Home", path: "/home/me", icon: "home" }],
      drives: [{ label: "/", path: "/", icon: "drive" }],
    }));
  }
  if (u.includes("/api/fs/dirs")) {
    const m = u.match(/[?&]path=([^&]*)/);
    const p = decodeURIComponent(m ? m[1] : "");
    const d = FS[p] || FS["/root"];
    return Promise.resolve(json(d));
  }
  return Promise.resolve(json({}));
}

const ticks = async (n = 4) => { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); };
const body = (win) => win.document.getElementById("modal-body");
const rows = (win) => [...body(win).querySelectorAll(".picker-row")];
const rowNamed = (win, name) => rows(win).find(
  (r) => r.querySelector(".picker-name") && r.querySelector(".picker-name").textContent === name);
const okBtn = (win) => body(win).querySelector(".picker-foot .btn-primary");
const cancelBtn = (win) => body(win).querySelector(".picker-foot .btn-secondary");

function start(win, exprOpts) {
  runScript(win, `
    globalThis.__res = undefined; globalThis.__done = false;
    pickPath(${exprOpts}).then((v) => { globalThis.__res = v; globalThis.__done = true; });
  `);
}

test("dir mode lists a folder and 'Use this folder' resolves the current path", async () => {
  const { window: win } = loadApp({ fetchImpl });
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();
  assert.ok(rowNamed(win, "sub"), "folder row shown");
  assert.ok(rowNamed(win, "apple.md"), "file row shown for context");
  const ok = okBtn(win);
  assert.equal(ok.textContent, "Use this folder");
  assert.equal(ok.disabled, false, "enabled once a directory is loaded");
  ok.click();
  await ticks();
  assert.equal(win.__res, "/root", "resolves the browsed directory");
});

test("clicking a folder navigates into it", async () => {
  const { window: win } = loadApp({ fetchImpl });
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();
  rowNamed(win, "sub").click();
  await ticks();
  assert.ok(rowNamed(win, "note.txt"), "now showing the subfolder contents");
  okBtn(win).click();
  await ticks();
  assert.equal(win.__res, "/root/sub");
});

test("file mode resolves the clicked file", async () => {
  const { window: win } = loadApp({ fetchImpl });
  start(win, `{ mode: "file", startPath: "/root" }`);
  await ticks();
  // No confirm button in file mode - you click a file.
  assert.equal(okBtn(win).style.display, "none");
  rowNamed(win, "apple.md").click();
  await ticks();
  assert.equal(win.__res, "/root/apple.md");
});

test("multi mode: exts gate selection, checked files+folders resolve as an array", async () => {
  const { window: win } = loadApp({ fetchImpl });
  start(win, `{ mode: "multi", startPath: "/root", exts: [".md", ".txt", ".pdf"] }`);
  await ticks();
  // photo.png is unsupported -> greyed, no checkbox.
  const png = rowNamed(win, "photo.png");
  assert.ok(png.classList.contains("unselectable"), "unsupported file is not selectable");
  assert.equal(png.querySelector(".picker-cb"), null, "unsupported file has no checkbox");
  // apple.md (supported) and sub (a folder, selectable in multi) both have checkboxes.
  rowNamed(win, "apple.md").querySelector(".picker-cb").click();
  rowNamed(win, "sub").querySelector(".picker-cb").click();
  await ticks(1);
  const ok = okBtn(win);
  assert.equal(ok.textContent, "Add 2 items");
  ok.click();
  await ticks();
  assert.deepEqual([...win.__res].sort(), ["/root/apple.md", "/root/sub"].sort());
});

test("filter narrows the visible rows", async () => {
  const { window: win } = loadApp({ fetchImpl });
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();
  const filter = body(win).querySelector(".picker-filter input");
  filter.value = "apple";
  filter.dispatchEvent(new win.Event("input"));
  await ticks(1);
  const names = rows(win).map((r) => r.querySelector(".picker-name").textContent);
  assert.deepEqual(names, ["apple.md"], "only the matching row remains");
});

test("issue #1220: dot-directories are hidden by default and shown after toggling", async () => {
  const { window: win } = loadApp({ fetchImpl });
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();
  assert.equal(rowNamed(win, ".git"), undefined, "dot-directory hidden by default");
  const toggle = body(win).querySelector(".picker-hidden-toggle input");
  assert.ok(toggle, "the Show-hidden toggle is rendered");
  toggle.checked = true;
  toggle.dispatchEvent(new win.Event("change"));
  await ticks(1);
  const gitRow = rowNamed(win, ".git");
  assert.ok(gitRow, "dot-directory appears once the toggle is on");
  assert.ok(gitRow.classList.contains("is-dir"), "still recognized as a directory");
  // Browsing into a dot-directory works exactly like any other folder.
  gitRow.click();
  await ticks();
  assert.ok(rowNamed(win, "HEAD"), "navigated into the dot-directory's own contents");
  toggle.checked = false;
  toggle.dispatchEvent(new win.Event("change"));
  await ticks(1);
  assert.equal(rowNamed(win, ".git"), undefined, "hidden again after toggling off");
});

test("issue #1220: reaching a dot-directory directly turns the toggle on for its siblings", async () => {
  // navigate() cannot tell a typed/pasted path from startPath - both funnel
  // through the same function - so startPath exercises the identical code
  // path a user's pasted "~/.git"-style path would.
  const { window: win } = loadApp({ fetchImpl });
  start(win, `{ mode: "dir", startPath: "/root/.git" }`);
  await ticks();
  const toggle = body(win).querySelector(".picker-hidden-toggle input");
  assert.equal(toggle.checked, true, "toggle auto-enables on landing in a dot-directory");
  assert.ok(rowNamed(win, "HEAD"), "shows the dot-directory's own (non-dot) contents");
  const upBtn = body(win).querySelector('button[title="Up one level"]');
  upBtn.click();
  await ticks(1);
  assert.ok(rowNamed(win, ".git"), "back at the parent, the sibling dot-dir is now visible unprompted");
});

test("issue #1220: navigate always asks the server for hidden entries", async () => {
  // Both #1220 tests above use the module-level fetchImpl, which returns the
  // SAME entries[] regardless of query string - the server's own
  // include_hidden gate (tests/test_fs_picker.py) is only exercised there, in
  // isolation. Neither suite binds the CLIENT to actually requesting hidden
  // entries: if navigate() ever stopped passing includeHidden=true to
  // fetchDirs(), every test above would stay green (the fake still returns
  // .git regardless), while the real server would simply omit dot-entries
  // from every response and the "Hidden" toggle would have nothing left to
  // reveal - the exact #1220 regression, invisible to the rest of the suite.
  // This test is the one thing that would catch it: it inspects the actual
  // request URL(s) navigate() sends.
  const seenDirsUrls = [];
  const recordingFetch = (url, opts) => {
    const u = String(url);
    if (u.includes("/api/fs/dirs")) seenDirsUrls.push(u);
    return fetchImpl(url, opts);
  };
  const { window: win } = loadApp({ fetchImpl: recordingFetch });
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();
  rowNamed(win, "sub").click();   // a second navigate(), into a subfolder
  await ticks();

  assert.ok(seenDirsUrls.length >= 2,
    "expected at least the initial nav plus the click-into-folder nav");
  for (const u of seenDirsUrls) {
    assert.ok(u.includes("include_hidden=true"),
      `navigate() must always ask the server for hidden entries (the picker `
      + `hides them client-side via the "Hidden" toggle, not by omitting the `
      + `request) - got ${u}`);
  }
});

test("dismissing the modal resolves null", async () => {
  const { window: win } = loadApp({ fetchImpl });
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();
  cancelBtn(win).click();
  await ticks();
  assert.equal(win.__res, null);
});

test("pickDirectory / pickFile keep their string|null contract", async () => {
  const { window: win } = loadApp({ fetchImpl });
  runScript(win, `
    globalThis.__d = undefined;
    pickDirectory("Pick", "/root").then((v) => { globalThis.__d = v; });
  `);
  await ticks();
  okBtn(win).click();
  await ticks();
  assert.equal(win.__d, "/root", "pickDirectory resolves a path string");

  runScript(win, `
    globalThis.__f = undefined;
    pickFile("Pick a file", "/root").then((v) => { globalThis.__f = v; });
  `);
  await ticks();
  rowNamed(win, "apple.md").click();
  await ticks();
  assert.equal(win.__f, "/root/apple.md", "pickFile resolves the clicked file");
});

// --------------------------------------------------------------------------- //
//  New Folder                                                                  //
// --------------------------------------------------------------------------- //

test("New Folder button is disabled at the drive-list root, enabled inside a real folder", async () => {
  const { window: win } = loadApp({ fetchImpl });
  start(win, `{ mode: "dir", startPath: "" }`);
  await ticks();
  const btn = body(win).querySelector('button[title="New folder"]');
  assert.ok(btn, "New folder button renders");
  assert.equal(btn.disabled, true, "disabled at the drive-list root");

  rowNamed(win, "/").click();
  await ticks();
  assert.equal(btn.disabled, false, "enabled once a real directory is loaded");
});

test("New Folder creates a folder with the typed name and navigates into it", async () => {
  const mkdirCalls = [];
  const customFetch = (url, opts = {}) => {
    const u = String(url);
    if (u.includes("/api/fs/mkdir")) {
      const b = JSON.parse(opts.body);
      mkdirCalls.push(b);
      return Promise.resolve(json({ path: "/root/" + b.name, parent: "/root", name: b.name }));
    }
    return fetchImpl(url, opts);
  };
  const { window: win } = loadApp({ fetchImpl: customFetch });
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();

  const btn = body(win).querySelector('button[title="New folder"]');
  btn.click();
  await ticks(1);
  const inp = body(win).querySelector(".picker-inline-input");
  assert.ok(inp, "an inline name input appears");
  assert.equal(inp.value, "New folder", "prefilled with a default name");
  inp.value = "Fresh";
  inp.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  await ticks(3);

  assert.deepEqual(mkdirCalls, [{ path: "/root", name: "Fresh" }],
    "sent the currently-browsed path and the typed name");
  assert.ok(body(win).querySelector(".picker-crumb-cur").textContent === "Fresh",
    "navigated into the newly created folder");
  const backBtn = body(win).querySelector('button[title="Back"]');
  assert.equal(backBtn.disabled, false, "Back is enabled - the prior folder was pushed to history");
});

test("New Folder surfaces the server's real refusal reason and reverts", async () => {
  const customFetch = (url, opts = {}) => {
    const u = String(url);
    if (u.includes("/api/fs/mkdir")) {
      return Promise.resolve({
        ok: false, status: 409,
        json: async () => ({ detail: "'sub' already exists" }),
        text: async () => "",
      });
    }
    return fetchImpl(url, opts);
  };
  const { window: win } = loadApp({ fetchImpl: customFetch });
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();

  body(win).querySelector('button[title="New folder"]').click();
  await ticks(1);
  const inp = body(win).querySelector(".picker-inline-input");
  inp.value = "sub";
  inp.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  await ticks(3);

  const toastEl = win.document.getElementById("toast");
  assert.ok(toastEl.textContent.includes("already exists"),
    "the real server refusal reason is surfaced, not a generic message");
  assert.ok(toastEl.className.includes("error"), "shown as an error toast");
  assert.equal(body(win).querySelector(".picker-inline-input"), null,
    "the inline editor is gone - reverted to the normal listing");
  assert.ok(rowNamed(win, "sub"), "back to the normal folder listing, nothing navigated");
});

test("Escape cancels New Folder without calling the server", async () => {
  let called = false;
  const customFetch = (url, opts = {}) => {
    if (String(url).includes("/api/fs/mkdir")) { called = true; return Promise.resolve(json({})); }
    return fetchImpl(url, opts);
  };
  const { window: win } = loadApp({ fetchImpl: customFetch });
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();

  body(win).querySelector('button[title="New folder"]').click();
  await ticks(1);
  const inp = body(win).querySelector(".picker-inline-input");
  inp.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  await ticks(1);

  assert.equal(body(win).querySelector(".picker-inline-input"), null, "editor closed");
  assert.equal(called, false, "no request was sent");
  assert.ok(rowNamed(win, "sub"), "back to the normal listing");
});

// --------------------------------------------------------------------------- //
//  Rename                                                                      //
// --------------------------------------------------------------------------- //

test("rename affordance is offered only for real entries, never the drive list", async () => {
  const { window: win } = loadApp({ fetchImpl });
  start(win, `{ mode: "dir", startPath: "" }`);
  await ticks();
  const driveRow = rowNamed(win, "/");
  assert.equal(driveRow.querySelector(".picker-row-rename"), null,
    "no rename affordance at the drive-list root");

  driveRow.click();
  await ticks();
  const subRow = rowNamed(win, "sub");
  assert.ok(subRow.querySelector(".picker-row-rename"), "rename affordance inside a real folder");
});

test("Rename commits a new name and refreshes the listing", async () => {
  const renameCalls = [];
  let dirsRefetches = 0;
  const customFetch = (url, opts = {}) => {
    const u = String(url);
    if (u.includes("/api/fs/rename")) {
      const b = JSON.parse(opts.body);
      renameCalls.push(b);
      return Promise.resolve(json({ path: "/root/" + b.new_name, parent: "/root", name: b.new_name }));
    }
    if (u.includes("/api/fs/dirs")) dirsRefetches++;
    return fetchImpl(url, opts);
  };
  const { window: win } = loadApp({ fetchImpl: customFetch });
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();
  const dirsBeforeRename = dirsRefetches;

  const row = rowNamed(win, "apple.md");
  const renameBtn = row.querySelector(".picker-row-rename");
  assert.ok(renameBtn, "rename affordance present on a file row");
  renameBtn.click();
  await ticks(1);
  const inp = row.querySelector(".picker-inline-input");
  assert.ok(inp, "row switches to an inline name input");
  assert.equal(inp.value, "apple.md", "prefilled with the current name");
  inp.value = "renamed.md";
  inp.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  await ticks(3);

  assert.deepEqual(renameCalls, [{ path: "/root/apple.md", new_name: "renamed.md" }]);
  assert.ok(dirsRefetches > dirsBeforeRename, "the listing was refetched after the rename");
});

test("Rename on a folder selects the whole name; on a file it selects the stem only", async () => {
  const { window: win } = loadApp({ fetchImpl });
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();

  const fileRow = rowNamed(win, "apple.md");
  fileRow.querySelector(".picker-row-rename").click();
  await ticks(1);
  const fileInp = fileRow.querySelector(".picker-inline-input");
  assert.equal(fileInp.selectionStart, 0);
  assert.equal(fileInp.selectionEnd, "apple".length, "stem selected, extension left alone");
  fileInp.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  await ticks(1);

  const dirRow = rowNamed(win, "sub");
  dirRow.querySelector(".picker-row-rename").click();
  await ticks(1);
  const dirInp = dirRow.querySelector(".picker-inline-input");
  assert.equal(dirInp.selectionStart, 0);
  assert.equal(dirInp.selectionEnd, "sub".length, "whole folder name selected");
});

test("Rename surfaces the server's real refusal reason and reverts", async () => {
  const customFetch = (url, opts = {}) => {
    const u = String(url);
    if (u.includes("/api/fs/rename")) {
      return Promise.resolve({
        ok: false, status: 409,
        json: async () => ({ detail: "'photo.png' already exists" }),
        text: async () => "",
      });
    }
    return fetchImpl(url, opts);
  };
  const { window: win } = loadApp({ fetchImpl: customFetch });
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();

  const row = rowNamed(win, "apple.md");
  row.querySelector(".picker-row-rename").click();
  await ticks(1);
  const inp = row.querySelector(".picker-inline-input");
  inp.value = "photo.png";
  inp.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  await ticks(3);

  const toastEl = win.document.getElementById("toast");
  assert.ok(toastEl.textContent.includes("already exists"),
    "the real server refusal reason is surfaced, not a generic message");
  assert.ok(toastEl.className.includes("error"), "shown as an error toast");
  assert.ok(rowNamed(win, "apple.md"), "reverted - the original name is back");
});

test("Escape cancels Rename without calling the server", async () => {
  let called = false;
  const customFetch = (url, opts = {}) => {
    if (String(url).includes("/api/fs/rename")) { called = true; return Promise.resolve(json({})); }
    return fetchImpl(url, opts);
  };
  const { window: win } = loadApp({ fetchImpl: customFetch });
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();

  const row = rowNamed(win, "apple.md");
  row.querySelector(".picker-row-rename").click();
  await ticks(1);
  const inp = row.querySelector(".picker-inline-input");
  inp.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  await ticks(1);

  assert.equal(called, false, "no request was sent");
  assert.ok(rowNamed(win, "apple.md"), "row reverted to its display name");
});

// --------------------------------------------------------------------------- //
//  Language                                                                    //
// --------------------------------------------------------------------------- //

function fetchWithGerman(url, opts) {
  if (String(url).includes("/i18n/de.json")) return Promise.resolve(json(DE));
  return fetchImpl(url, opts);
}

test("the open picker's toolbar, rail and footer redraw in German when the language changes", async () => {
  const { window: win } = loadApp({ fetchImpl: fetchWithGerman });
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();
  assert.equal(okBtn(win).textContent, "Use this folder", "starts in English");
  assert.equal(cancelBtn(win).textContent, "Cancel");

  runScript(win, `window.__lang = applyLanguage("de");`);
  await win.__lang;
  await ticks(4);

  assert.equal(okBtn(win).textContent, "Diesen Ordner verwenden",
    "the footer confirm button redraws without closing the modal");
  assert.equal(cancelBtn(win).textContent, "Abbrechen");
  assert.ok(body(win).querySelector('button[title="Zurück"]'), "Back retranslated");
  assert.ok(body(win).querySelector('button[title="Eine Ebene nach oben"]'), "Up one level retranslated");
  assert.ok(body(win).querySelector('button[title="Neuer Ordner"]'), "New folder retranslated");
  const railHead = body(win).querySelector(".picker-rail-head");
  assert.equal(railHead && railHead.textContent, "Orte",
    "the places rail heading is repainted (its own re-fetch runs after the sync part of the switch)");
});

test("closing the picker removes its language listener - a later language change never touches it again",
  async () => {
    const seenPlacesCalls = [];
    const countingFetch = (url, opts) => {
      if (String(url).includes("/api/fs/places")) seenPlacesCalls.push(url);
      return fetchWithGerman(url, opts);
    };
    const { window: win } = loadApp({ fetchImpl: countingFetch });
    start(win, `{ mode: "dir", startPath: "/root" }`);
    await ticks();
    const callsWhileOpen = seenPlacesCalls.length;
    assert.ok(callsWhileOpen >= 1, "the places rail loads once while the picker is open");

    cancelBtn(win).click();
    await ticks();
    assert.equal(win.__res, null, "the picker resolved and closed");

    runScript(win, `window.__lang = applyLanguage("de");`);
    await win.__lang;
    await ticks(4);

    assert.equal(seenPlacesCalls.length, callsWhileOpen,
      "a language change after the picker closed must not re-run its retranslate - "
      + "the listener was removed in cleanup, so nothing fires or touches stale DOM");
  });
