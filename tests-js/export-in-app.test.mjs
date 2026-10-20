// SPDX-License-Identifier: AGPL-3.0-or-later
// With host filesystem access, Export saves the conversation through the app's
// own folder picker and /api/fs/save-text, with no browser download. Without
// it, Export falls back to the browser download.
import { test } from "node:test";
import assert from "node:assert/strict";

import { loadAppWithPages, runScript } from "./harness.mjs";

function setup({ fsAccess, picked = "D:/exports", saveStatus = 200, privacy = false }) {
  const posts = [];
  const fetchImpl = async (url, opts = {}) => {
    if (String(url) === "/api/fs/save-text") {
      const body = JSON.parse(opts.body);
      posts.push(body);
      const ok = saveStatus === 200;
      return { ok, status: saveStatus, headers: { get: () => null }, text: async () => "",
               json: async () => (ok ? { path: `${body.path}/${body.name}`, parent: body.path, name: body.name }
                                     : { detail: "Permission denied: x.md" }) };
    }
    return { ok: true, status: 200, headers: { get: () => null },
             json: async () => ({}), text: async () => "" };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  let downloads = 0;
  window.URL.createObjectURL = () => "blob:x";
  window.URL.revokeObjectURL = () => {};
  const origCreate = window.document.createElement.bind(window.document);
  window.document.createElement = (tag) => {
    const el = origCreate(tag);
    if (tag === "a") el.click = () => { downloads++; };
    return el;
  };
  const pickerCalls = [];
  window.pickPath = async (opts) => { pickerCalls.push(opts); return picked; };
  runScript(window, `caps.fsAccess = ${JSON.stringify(fsAccess)};
    chat.modeKnown = true; chat.privacy = ${privacy}; chat.serverPrivacy = ${privacy};`);
  const conv = { id: "c1", title: "Derived request", messages: [
    { role: "user", content: "how does it work", id: "m1" },
    { role: "assistant", content: "It copies the scope.", id: "m2" },
  ] };
  runScript(window, `chat.conversations = [${JSON.stringify(conv)}]; chat.activeId = "c1";`);
  const toasts = [];
  window.toast = (msg, isErr) => toasts.push({ msg, isErr: !!isErr });
  return { window, posts, pickerCalls, toasts, downloads: () => downloads };
}

test("with host access, export saves in the picked folder and never downloads", async () => {
  const s = setup({ fsAccess: "host" });
  await s.window.exportConversation();
  assert.equal(s.downloads(), 0, "no browser download was started");
  assert.equal(s.pickerCalls.length, 1);
  assert.equal(s.pickerCalls[0].mode, "dir");
  assert.equal(s.posts.length, 1);
  assert.equal(s.posts[0].path, "D:/exports");
  assert.equal(s.posts[0].name, "Derived_request.md");
  assert.match(s.posts[0].content, /It copies the scope\./);
  assert.match(s.toasts.at(-1).msg, /D:\/exports\/Derived_request\.md/);
  assert.equal(s.toasts.at(-1).isErr, false);
});

test("the next export starts in the folder of the last one", async () => {
  const s = setup({ fsAccess: "host" });
  await s.window.exportConversation();
  await s.window.exportConversation();
  assert.equal(s.pickerCalls[1].startPath, "D:/exports");
});

test("in privacy mode the save folder is not remembered in browser storage", async () => {
  const s = setup({ fsAccess: "host", privacy: true });
  await s.window.exportConversation();
  assert.equal(s.posts.length, 1, "the export itself is saved");
  assert.equal(s.window.localStorage.getItem("localm.exportDir"), null);
  await s.window.exportConversation();
  assert.equal(s.pickerCalls[1].startPath, "");
});

test("the remembered save folder is on the instance wipe list", () => {
  const s = setup({ fsAccess: "host" });
  runScript(s.window, "window.__keys = INSTANCE_SCOPED_KEYS;");
  assert.ok(s.window.__keys.includes("localm.exportDir"));
});

test("a title that is a Windows device name gets a usable file name", async () => {
  const s = setup({ fsAccess: "host" });
  runScript(s.window, 'chat.conversations[0].title = "CON";');
  await s.window.exportConversation();
  assert.equal(s.posts[0].name, "CON_chat.md");
});

test("cancelling the picker saves nothing", async () => {
  const s = setup({ fsAccess: "host", picked: null });
  await s.window.exportConversation();
  assert.equal(s.posts.length, 0);
  assert.equal(s.downloads(), 0);
});

test("a refused save is reported, not shown as saved", async () => {
  const s = setup({ fsAccess: "host", saveStatus: 403 });
  await s.window.exportConversation();
  assert.equal(s.toasts.at(-1).isErr, true);
  assert.match(s.toasts.at(-1).msg, /Permission denied/);
});

test("saveTextFile (used by the coder exports) saves any text file in the app", async () => {
  const s = setup({ fsAccess: "host" });
  await s.window.saveTextFile("coder-result-s1.json", "{\"ok\": true}", "application/json");
  assert.equal(s.downloads(), 0);
  assert.deepEqual(s.posts[0], { path: "D:/exports", name: "coder-result-s1.json",
                                 content: "{\"ok\": true}" });
});

test("without host access, export falls back to the browser download", async () => {
  const s = setup({ fsAccess: "none" });
  await s.window.exportConversation();
  assert.equal(s.downloads(), 1);
  assert.equal(s.pickerCalls.length, 0);
  assert.equal(s.posts.length, 0);
});
