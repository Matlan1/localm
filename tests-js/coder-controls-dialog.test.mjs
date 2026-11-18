// SPDX-License-Identifier: AGPL-3.0-or-later
// The coder Session controls dialog: "browse..." must hand the pick back to
// the still-open dialog (the one #modal is shared with the picker), and every
// call the dialog makes is bound to the session it was opened for, never to
// whichever session happens to be active when a button is pressed.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function reply(status, data) {
  return { ok: status < 400, status, json: async () => data, text: async () => "" };
}

const FS = {
  "/root": { path: "/root", parent: "/", entries: [
    { name: "sub", is_dir: true, size: null, mtime: 1700000000 } ] },
  "/root/sub": { path: "/root/sub", parent: "/root", entries: [] },
};

function makeFetch(calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    if (u.includes("/api/fs/places")) return reply(200, { places: [], drives: [] });
    if (u.includes("/api/fs/dirs")) {
      const m = u.match(/[?&]path=([^&]*)/);
      return reply(200, FS[decodeURIComponent(m ? m[1] : "")] || FS["/root"]);
    }
    if (/^\/api\/coder\/sessions\/[^/]+\/(settings|cwd|memory)$/.test(u) && method === "POST") {
      const id = u.split("/")[4];
      const body = JSON.parse(opts.body);
      calls.push({ url: u, id, body });
      return reply(200, { id, cwd: body.cwd || "/root", auto_approve: true });
    }
    if (/\/memory$/.test(u)) return reply(200, { exists: false, text: "", path: "x" });
    return reply(200, {});
  };
}

const ticks = async (n = 4) => { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); };
const body = (win) => win.document.getElementById("modal-body");
const modalOpen = (win) => win.document.getElementById("modal").style.display !== "none";

function seed(win) {
  runScript(win, `
    for (const id of ["s1", "s2"]) {
      coder.sessions.set(id, { info: { id, cwd: "/root", model: "m", total_tokens: 0, turns: 0,
                                       patch_mode: false, backend_info: { backend: "local" } },
                               busy: false, feedEl: document.createElement("div") });
    }
    activateSession("s1");
  `);
}

test("browse... in Session controls returns to the dialog with the picked folder in the field", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  seed(win);
  win.openSessionControls();
  const field = win.document.getElementById("ctl-cwd");
  field.nextElementSibling.click();
  await ticks();
  [...body(win).querySelectorAll(".picker-row")].find(
    (r) => r.querySelector(".picker-name")?.textContent === "sub").click();
  await ticks();
  body(win).querySelector(".picker-foot .btn-primary").click();
  await ticks();

  assert.ok(modalOpen(win), "the controls dialog is still open after the pick");
  assert.equal(win.document.getElementById("ctl-cwd"), field, "the same dialog, not a rebuilt one");
  assert.equal(field.value, "/root/sub", "the picked folder is in the field");
  assert.ok(!win.document.getElementById("modal").querySelector(".picker-modal"),
    "the picker's sizing class is gone");

  const move = [...body(win).querySelectorAll("button")].find(
    (b) => b.textContent === win.t("coder.controls.moveBtn"));
  await move.onclick();
  const posts = calls.filter((c) => c.url.endsWith("/cwd"));
  assert.equal(posts.length, 1);
  assert.equal(posts[0].id, "s1");
  assert.deepEqual(posts[0].body, { cwd: "/root/sub" });
});

test("cancelling the picker over Session controls also returns to the dialog", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  seed(win);
  win.openSessionControls();
  const field = win.document.getElementById("ctl-cwd");
  field.nextElementSibling.click();
  await ticks();
  body(win).querySelector(".picker-foot .btn-secondary").click();
  await ticks();
  assert.ok(modalOpen(win));
  assert.equal(win.document.getElementById("ctl-cwd"), field);
  assert.equal(field.value, "/root", "the field is untouched");
});

test("a picker opened with no dialog under it closes the modal as before", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  runScript(win, `pickDirectory("t", "/root");`);
  await ticks();
  body(win).querySelector(".picker-foot .btn-secondary").click();
  await ticks();
  assert.ok(!modalOpen(win));
});

test("the controls dialog acts on its own session after another becomes active", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  seed(win);
  win.openSessionControls();
  runScript(win, `activateSession("s2");`);
  const box = win.document.getElementById("ctl-auto-approve");
  box.checked = true;
  await box.onchange();
  const posts = calls.filter((c) => c.url.endsWith("/settings"));
  assert.equal(posts.length, 1);
  assert.equal(posts[0].id, "s1", "auto-approve goes to the session the dialog was opened for");
  assert.deepEqual(posts[0].body, { auto_approve: true });
});

test("a dialog opened for a session closes when that session is dropped", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  seed(win);
  win.openSessionControls();
  assert.ok(modalOpen(win));
  runScript(win, `dropCoderSession(coder.sessions.get("s1"));`);
  assert.ok(!modalOpen(win), "controls for the ended session are gone");

  seed(win);
  win.openSessionControls();
  runScript(win, `dropCoderSession(coder.sessions.get("s2"));`);
  assert.ok(modalOpen(win), "dropping a different session leaves the dialog alone");
});

test("a dialog of another kind is not closed by dropping a session", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  seed(win);
  win.openSessionControls();
  runScript(win, `openModal("Other", (b) => b.appendChild(document.createElement("p")));`);
  runScript(win, `dropCoderSession(coder.sessions.get("s1"));`);
  assert.ok(modalOpen(win));
});
