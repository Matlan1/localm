// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// rec#210: the music/video history "move..." button used window.prompt(), which
// mobile/PWA browsers suppress - so the button was dead there. It now uses the
// in-page pickDirectory() modal, exactly like the image page already does.

function setup() {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    if (u.includes("/api/video/history"))
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ videos: [{ name: "clip.mp4", size_bytes: 1048576, meta: {} }] }) };
    if (u.includes("/api/music/history"))
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ tracks: [{ name: "song.flac", size_bytes: 1048576, meta: {} }] }) };
    if (u.includes("/move"))
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ path: "/dest/folder" }) };
    return { ok: true, status: 200, text: async () => "", json: async () => ({}) };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  return { window, calls };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

function moveButton(win, boxId) {
  const box = win.document.getElementById(boxId);
  return [...box.querySelectorAll("button")].find(
    (b) => b.textContent.startsWith("move"));
}

const KINDS = [
  { name: "video", box: "video-history", refresh: "refreshVideoHistory",
    endpoint: "/api/video/file/clip.mp4/move" },
  { name: "music", box: "music-history", refresh: "refreshMusicHistory",
    endpoint: "/api/music/file/song.flac/move" },
];

for (const kind of KINDS) {
  test(`${kind.name} move uses pickDirectory (not prompt) and POSTs the chosen dir`, async () => {
    const { window: win, calls } = setup();
    // prompt() must never be reached (it is suppressed on mobile/PWA); the
    // picker is stubbed so no real modal/fetchDirs machinery is needed.
    runScript(win, `
      globalThis.__promptCalled = false;
      window.prompt = () => { globalThis.__promptCalled = true; return "/from/prompt"; };
      globalThis.__pickArgs = null;
      pickDirectory = (title, start) => { globalThis.__pickArgs = [title, start]; return Promise.resolve("/picked/dir"); };
    `);
    runScript(win, `${kind.refresh}();`);
    await tick();

    const btn = moveButton(win, kind.box);
    assert.ok(btn, "move button rendered in history");
    btn.click();
    await tick();
    await tick();

    assert.equal(win.__promptCalled, false, "window.prompt() is not used");
    assert.ok(win.__pickArgs, "pickDirectory was called");
    const moves = calls.filter((c) => c.url.includes(kind.endpoint));
    assert.equal(moves.length, 1, "exactly one move POST");
    assert.equal(moves[0].opts.method, "POST");
    assert.deepEqual(JSON.parse(moves[0].opts.body), { dest: "/picked/dir" });
  });

  test(`${kind.name} move: cancelling the picker sends no request`, async () => {
    const { window: win, calls } = setup();
    runScript(win, "pickDirectory = () => Promise.resolve(null);");
    runScript(win, `${kind.refresh}();`);
    await tick();
    moveButton(win, kind.box).click();
    await tick();
    await tick();
    assert.equal(calls.filter((c) => c.url.includes("/move")).length, 0,
      "no move POST when the picker is dismissed");
  });
}
