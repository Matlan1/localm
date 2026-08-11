// SPDX-License-Identifier: AGPL-3.0-or-later
// Tests for the shared file/folder picker (app/picker.js): pickPath() plus the
// pickDirectory()/pickFile() back-compat wrappers. Drives the real module in
// jsdom against a fake /api/fs/dirs + /api/fs/places filesystem.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

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
