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

for (const startPath of ["/root", ""]) {
  test(`a listing that keeps failing is asked for a bounded number of times (start ${JSON.stringify(startPath)})`, async () => {
    const dirCalls = [];
    const refusing = (url) => {
      const u = String(url);
      if (u.includes("/api/fs/dirs")) {
        dirCalls.push(u);
        // Answers on a later task; after 50 calls it never answers.
        if (dirCalls.length > 50) return new Promise(() => {});
        return new Promise((r) => setTimeout(() => r({ ok: false, status: 403, statusText: "Forbidden",
          json: async () => ({ detail: "Key lacks host filesystem access" }), text: async () => "" }), 0));
      }
      return fetchImpl(url);
    };
    const { window: win } = loadApp({ fetchImpl: refusing });
    start(win, `{ mode: "dir", startPath: ${JSON.stringify(startPath)} }`);
    await ticks(40);
    assert.equal(dirCalls.length, startPath ? 2 : 1,
      `the start folder, then the drive list once, and no more: ${dirCalls.join(" ")}`);
    assert.match(win.document.getElementById("toast").textContent, /Key lacks host filesystem access/);
    cancelBtn(win).click();
    await ticks();
    assert.equal(win.__res, null);
  });
}

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

// --------------------------------------------------------------------------- //
//  Keyboard: Tab to a button in an open editor, then Enter or Space            //
// --------------------------------------------------------------------------- //

const pastBlurTimer = () => new Promise((r) => setTimeout(r, 130));
const keydown = (win, target, key) => {
  const e = new win.KeyboardEvent("keydown", { key, bubbles: true, cancelable: true });
  target.dispatchEvent(e);
  return e;
};
const mousedown = (win, target) => {
  const e = new win.MouseEvent("mousedown", { bubbles: true, cancelable: true });
  target.dispatchEvent(e);
  return e;
};

async function openPathEditor(win) {
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();
  body(win).querySelector(".picker-path-edit").click();
  return {
    inp: body(win).querySelector(".picker-path-input"),
    go: body(win).querySelector(".picker-path-go"),
  };
}

test("Tab to Go keeps the path editor open, and Enter on it navigates", async () => {
  const { window: win } = loadApp({ fetchImpl });
  const { inp, go } = await openPathEditor(win);
  inp.value = "/root/sub";
  inp.focus();
  go.focus();
  await pastBlurTimer();
  assert.ok(go.isConnected, "focus on Go does not close the editor");
  assert.equal(win.document.activeElement, go);

  const e = keydown(win, go, "Enter");
  assert.equal(e.defaultPrevented, false, "the browser turns Enter on a button into a click");
  go.click();
  await ticks();
  assert.ok(rowNamed(win, "note.txt"), "navigated to the typed path");
  assert.equal(win.__done, false, "the picker stays open");
  assert.ok(win.document.activeElement.classList.contains("picker-list"),
    "focus lands on the listing, not on nothing");
});

test("a mouse press on Go keeps focus in the input, and the click navigates once", async () => {
  const dirCalls = [];
  const customFetch = (url, opts = {}) => {
    if (String(url).includes("/api/fs/dirs")) dirCalls.push(String(url));
    return fetchImpl(url, opts);
  };
  const { window: win } = loadApp({ fetchImpl: customFetch });
  const { inp, go } = await openPathEditor(win);
  inp.value = "/root/sub";
  inp.focus();
  dirCalls.length = 0;

  const e = mousedown(win, go);
  assert.equal(e.defaultPrevented, true, "the press does not move focus to the button");
  assert.equal(win.document.activeElement, inp);
  await pastBlurTimer();
  assert.ok(go.isConnected);
  assert.deepEqual(dirCalls, [], "the press alone does not navigate");

  go.click();
  await ticks();
  assert.equal(dirCalls.length, 1, "one request for the press and click together");
  assert.ok(rowNamed(win, "note.txt"));
});

test("Tab past Go closes the path editor", async () => {
  const { window: win } = loadApp({ fetchImpl });
  const { inp, go } = await openPathEditor(win);
  inp.focus();
  go.focus();
  await pastBlurTimer();
  assert.ok(go.isConnected);
  body(win).querySelector(".picker-filter input").focus();
  await pastBlurTimer();
  assert.equal(body(win).querySelector(".picker-path-input"), null, "back to breadcrumbs");
  assert.equal(win.__done, false);
});

test("Tab back from Go to the input keeps the path editor open", async () => {
  const { window: win } = loadApp({ fetchImpl });
  const { inp, go } = await openPathEditor(win);
  go.focus();
  inp.focus();
  await pastBlurTimer();
  assert.ok(inp.isConnected);
});

test("Escape on Go closes the path editor, not the picker", async () => {
  const { window: win } = loadApp({ fetchImpl });
  const { go } = await openPathEditor(win);
  go.focus();
  keydown(win, go, "Escape");
  await ticks(1);
  assert.equal(body(win).querySelector(".picker-path-input"), null, "back to breadcrumbs");
  assert.equal(win.__done, false, "the picker stays open");
  assert.ok(win.document.activeElement.classList.contains("picker-list"));
});

test("Escape on the picker's other buttons still closes the picker", async () => {
  const { window: win } = loadApp({ fetchImpl });
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();
  const ok = okBtn(win);
  ok.focus();
  keydown(win, ok, "Escape");
  await ticks();
  assert.equal(win.__done, true);
  assert.equal(win.__res, null);
});

function mkdirFetch(mkdirCalls) {
  return (url, opts = {}) => {
    const u = String(url);
    if (u.includes("/api/fs/mkdir")) {
      const b = JSON.parse(opts.body);
      mkdirCalls.push(b);
      return Promise.resolve(json({ path: "/root/" + b.name, parent: "/root", name: b.name }));
    }
    return fetchImpl(url, opts);
  };
}

async function openNewFolder(win) {
  start(win, `{ mode: "dir", startPath: "/root" }`);
  await ticks();
  body(win).querySelector('button[title="New folder"]').click();
  await ticks(1);
  return {
    inp: body(win).querySelector(".picker-inline-input"),
    create: body(win).querySelector('.picker-inline-btn[title="Create"]'),
    cancel: body(win).querySelector('.picker-inline-btn[title="Cancel"]'),
  };
}

test("Tab to Create keeps the new folder row open, and Enter on it creates the folder", async () => {
  const mkdirCalls = [];
  const { window: win } = loadApp({ fetchImpl: mkdirFetch(mkdirCalls) });
  const { inp, create } = await openNewFolder(win);
  inp.value = "Fresh";
  create.focus();
  await pastBlurTimer();
  assert.ok(create.isConnected, "focus on Create does not close the row");

  const e = keydown(win, create, "Enter");
  assert.equal(e.defaultPrevented, false, "the browser turns Enter on a button into a click");
  assert.equal(win.__done, false, "Enter does not reach the listing, which would pick the folder");
  create.click();
  await ticks(3);
  assert.deepEqual(mkdirCalls, [{ path: "/root", name: "Fresh" }]);
  assert.equal(body(win).querySelector(".picker-crumb-cur").textContent, "Fresh");
  assert.ok(win.document.activeElement.classList.contains("picker-list"));
});

test("Tab to Cancel and Enter closes the new folder row without creating anything", async () => {
  const mkdirCalls = [];
  const { window: win } = loadApp({ fetchImpl: mkdirFetch(mkdirCalls) });
  const { inp, create, cancel } = await openNewFolder(win);
  create.focus();
  cancel.focus();
  await pastBlurTimer();
  assert.ok(cancel.isConnected, "moving between the row's buttons does not close it");
  assert.equal(inp.isConnected, true);

  keydown(win, cancel, "Enter");
  cancel.click();
  await ticks(1);
  assert.equal(body(win).querySelector(".picker-inline-input"), null, "row closed");
  assert.deepEqual(mkdirCalls, []);
  assert.ok(rowNamed(win, "sub"), "back to the normal listing");
  assert.equal(win.__done, false);
  assert.ok(win.document.activeElement.classList.contains("picker-list"));
});

test("a mouse press on Create keeps focus in the name, and the click creates once", async () => {
  const mkdirCalls = [];
  const { window: win } = loadApp({ fetchImpl: mkdirFetch(mkdirCalls) });
  const { inp, create } = await openNewFolder(win);
  inp.value = "Fresh";
  const e = mousedown(win, create);
  assert.equal(e.defaultPrevented, true);
  assert.equal(win.document.activeElement, inp);
  await pastBlurTimer();
  assert.deepEqual(mkdirCalls, [], "the press alone does not create");
  create.click();
  await ticks(3);
  assert.equal(mkdirCalls.length, 1);
});

test("a mouse press on the row's Cancel keeps it open until the click", async () => {
  const { window: win } = loadApp({ fetchImpl: mkdirFetch([]) });
  const { inp, cancel } = await openNewFolder(win);
  mousedown(win, cancel);
  await pastBlurTimer();
  assert.ok(inp.isConnected);
  cancel.click();
  await ticks(1);
  assert.equal(body(win).querySelector(".picker-inline-input"), null);
});

test("Tab past the new folder row closes it", async () => {
  const { window: win } = loadApp({ fetchImpl: mkdirFetch([]) });
  const { create } = await openNewFolder(win);
  create.focus();
  body(win).querySelector(".picker-filter input").focus();
  await pastBlurTimer();
  assert.equal(body(win).querySelector(".picker-inline-input"), null);
  assert.equal(win.__done, false);
});

test("Escape on a new folder button closes the row, not the picker", async () => {
  const { window: win } = loadApp({ fetchImpl: mkdirFetch([]) });
  const { create } = await openNewFolder(win);
  create.focus();
  keydown(win, create, "Escape");
  await ticks(1);
  assert.equal(body(win).querySelector(".picker-inline-input"), null);
  assert.equal(win.__done, false);
});

test("keys typed in the new folder name stay out of the listing's key handler", async () => {
  const mkdirCalls = [];
  const { window: win } = loadApp({ fetchImpl: mkdirFetch(mkdirCalls) });
  const { inp } = await openNewFolder(win);
  for (const key of ["Backspace", "ArrowLeft", "ArrowDown"]) {
    const e = keydown(win, inp, key);
    assert.equal(e.defaultPrevented, false, key + " edits the name");
    await ticks();
    assert.ok(inp.isConnected, key + " leaves the row open");
  }
  assert.equal(body(win).querySelector(".picker-crumb-cur").textContent, "root",
    "Backspace and the left arrow do not go up a folder");

  inp.value = "Fresh";
  keydown(win, inp, "Enter");
  await ticks(3);
  assert.equal(win.__done, false, "Enter does not also pick the current folder");
  assert.deepEqual(mkdirCalls, [{ path: "/root", name: "Fresh" }]);
});
