// SPDX-License-Identifier: AGPL-3.0-or-later
/* Studio parity: Music and Video get the library affordances Images had.
 *
 * Three properties, each written so it can only pass for the right reason:
 *
 *  1. The reload-chat-model toggle writes each backend's OWN per-plugin key.
 *     Asserted on the REQUEST BODY and the METHOD, not on a status code - the
 *     old global write and the new per-plugin write both return 200, so a
 *     status assertion cannot tell them apart. The negative half (no PATCH of
 *     /v1/config) is the half that actually pins the fix.
 *
 *  2. Bulk delete removes exactly the selected items. The fetch mock is gated
 *     on an explicit flag flipped after setup, never on call order or URL: the
 *     page fires its own history read on refresh, so counting by arrival order
 *     would attribute the page's own traffic to the test.
 *
 *  3. "Reuse settings" restores every form field. The fixture uses values that
 *     appear NOWHERE else - a field left at its default cannot accidentally
 *     satisfy the assertion, which is what makes this distinguish "restored"
 *     from "already looked like that".
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));
const settle = async (n = 4) => { for (let i = 0; i < n; i++) await tick(); };

/* Values chosen so NO default and no other fixture can produce them. If a page
   left a field untouched, the assertion fails - see item 19 in the review
   catalogue (a fixture that cannot express the failing case). */
const MUSIC_META = {
  tags: "ZZ-TAGS-4417-doomjazz",
  lyrics: "ZZ-LYRICS-8830",
  duration_seconds: 137,
  seed: 991177,
  steps: 43,
  cfg: 7.25,
};
const VIDEO_META = {
  prompt: "ZZ-PROMPT-5521-a marmot piloting a submarine",
  negative_prompt: "ZZ-NEG-6642",
  input_image: "ZZ-INPUT-7753.png",
  seconds: 9,
  fps: 17,
  width: 592,
  height: 336,
  seed: 224466,
  steps: 51,
  cfg: 3.75,
};

function setup({ tracks = [], videos = [], images = [] } = {}) {
  const calls = [];
  let recording = false;              // EXPLICIT gate - never call order
  let win0 = globalThis;              // set once the window exists (for Blob)
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (recording) calls.push({ url: u, opts });
    const ok = (json) => ({ ok: true, status: 200, text: async () => "",
                           json: async () => json,
                           blob: async () => new win0.Blob(["x"]) });
    if (u.includes("/api/music/history")) return ok({ tracks });
    if (u.includes("/api/video/history")) return ok({ videos });
    if (u.includes("/api/imagine/history")) return ok({ images });
    if (u.includes("/v1/media/config")) {
      return ok({ plugins: [
        { plugin: "image", fields: [{ key: "reload_after", value: true }] },
        { plugin: "music", fields: [{ key: "reload_after", value: true }] },
        { plugin: "video", fields: [{ key: "reload_after", value: true }] },
      ] });
    }
    return ok({});
  };
  const { window } = loadAppWithPages({ fetchImpl });
  win0 = window;
  if (!window.URL.createObjectURL) {
    window.URL.createObjectURL = (b) => "blob:mock/" + Math.random().toString(36).slice(2);
    window.URL.revokeObjectURL = () => {};
  }
  return { window, calls, startRecording: () => { recording = true; } };
}

const track = (name, meta = {}) =>
  ({ name, path: "/g/" + name, size_bytes: 1048576, meta });

/* ---------------------------------------------------------------- 1. toggle */

for (const kind of [
  { plugin: "music", box: "music-reload-llm", refresh: "refreshMusicHistory" },
  { plugin: "video", box: "video-reload-llm", refresh: "refreshVideoHistory" },
  { plugin: "image", box: "img-reload-llm", refresh: "refreshImageHistory" },
]) {
  test(`${kind.plugin}: the reload toggle writes its OWN per-plugin key, not the global one`,
    async () => {
      const { window: win, calls, startRecording } = setup();
      const box = win.document.getElementById(kind.box);
      assert.ok(box, `${kind.box} exists in the markup`);

      startRecording();
      box.checked = false;
      box.dispatchEvent(new win.Event("change"));
      await settle();

      const writes = calls.filter((c) => (c.opts.method || "GET") !== "GET");
      assert.equal(writes.length, 1, "exactly one write");
      assert.equal(writes[0].url, `/v1/media/config/${kind.plugin}`);
      assert.equal(writes[0].opts.method, "POST");
      assert.deepEqual(JSON.parse(writes[0].opts.body), { reload_after: false },
        "writes the per-plugin reload_after field");

      // THE NEGATIVE HALF - this is what pins the fix. The old behaviour was a
      // PATCH of /v1/config {reload_llm_after_imagine}, which is the LEGACY
      // GLOBAL fallback every media backend reads, so it silently changed all
      // three plugins at once.
      const globalWrites = calls.filter(
        (c) => (c.opts.method || "GET") !== "GET"
               && c.url.includes("/v1/config") && !c.url.includes("/v1/media/config"));
      assert.equal(globalWrites.length, 0,
        "must not touch the global reload_llm_after_imagine key");
    });
}

test("a failed toggle write reverts the checkbox instead of showing it as saved",
  async () => {
    const calls = [];
    const fetchImpl = async (url, opts = {}) => {
      const u = String(url);
      calls.push(u);
      if (u.includes("/v1/media/config/music") && opts.method === "POST") {
        return { ok: false, status: 403, text: async () => "",
                 json: async () => ({ detail: "nope" }) };
      }
      if (u.includes("/api/music/history"))
        return { ok: true, status: 200, text: async () => "", json: async () => ({ tracks: [] }) };
      return { ok: true, status: 200, text: async () => "", json: async () => ({ plugins: [] }) };
    };
    const { window: win } = loadAppWithPages({ fetchImpl });
    const box = win.document.getElementById("music-reload-llm");
    box.checked = false;
    box.dispatchEvent(new win.Event("change"));
    await settle(6);
    assert.equal(box.checked, true,
      "a rejected write must not leave the UI claiming the setting was saved");
  });

/* ------------------------------------------------------------ 2. bulk delete */

for (const kind of [
  { name: "music", listKey: "tracks", box: "music-history", refresh: "refreshMusicHistory",
    items: ["a.flac", "b.flac", "c.flac"], api: "/api/music/file/" },
  { name: "video", listKey: "videos", box: "video-history", refresh: "refreshVideoHistory",
    items: ["a.mp4", "b.mp4", "c.mp4"], api: "/api/video/file/" },
]) {
  test(`${kind.name}: bulk delete removes exactly the selected items`, async () => {
    const seed = { [kind.listKey]: kind.items.map((n) => track(n)) };
    const { window: win, calls, startRecording } = setup(seed);
    runScript(win, `${kind.refresh}();`);
    await settle();

    const cards = [...win.document.getElementById(kind.box).querySelectorAll(".thumb")];
    assert.equal(cards.length, 3, "three cards rendered");

    // select the FIRST and THIRD - picking a non-contiguous pair means an
    // off-by-one or a "delete everything" bug cannot pass by coincidence.
    cards[0].querySelector(".thumb-sel").click();
    cards[2].querySelector(".thumb-sel").click();
    await tick();

    const bar = win.document.getElementById(`${kind.name}-bulk`);
    assert.notEqual(bar.style.display, "none", "bulk bar is shown while a selection exists");
    assert.match(bar.textContent, /2 selected/);

    startRecording();
    [...bar.querySelectorAll("button")].find((b) => b.textContent === "delete").click();
    await tick();
    // the in-page confirm dialog, not window.confirm (suppressed on mobile/PWA)
    const confirmBtn = [...win.document.getElementById("modal-body")
      .querySelectorAll("button")].find((b) => b.textContent === "Delete");
    assert.ok(confirmBtn, "destructive action goes through the in-page confirm");
    confirmBtn.click();
    await settle(8);

    const deleted = calls
      .filter((c) => c.opts.method === "DELETE")
      .map((c) => c.url.replace(kind.api, ""));
    assert.deepEqual(deleted.sort(), [kind.items[0], kind.items[2]].sort(),
      "exactly the selected items were deleted");
  });
}

/* --------------------------------------------------------- 3. reuse settings */

test("music: reuse settings restores every field from the sidecar", async () => {
  const { window: win } = setup({ tracks: [track("song.flac", MUSIC_META)] });
  runScript(win, "refreshMusicHistory();");
  await settle();

  win.document.getElementById("music-history").querySelector(".thumb").click();
  const reuse = [...win.document.getElementById("modal-body").querySelectorAll("button")]
    .find((b) => b.textContent === "reuse settings");
  assert.ok(reuse, "reuse settings offered when a sidecar exists");
  reuse.click();
  await tick();

  const $v = (id) => win.document.getElementById(id).value;
  assert.equal($v("music-tags"), MUSIC_META.tags);
  assert.equal($v("music-lyrics"), MUSIC_META.lyrics);
  assert.equal($v("music-duration"), String(MUSIC_META.duration_seconds));
  assert.equal($v("music-seed"), String(MUSIC_META.seed));
  assert.equal($v("music-steps"), String(MUSIC_META.steps));
  assert.equal($v("music-cfg"), String(MUSIC_META.cfg));
});

test("video: reuse settings restores every field from the sidecar", async () => {
  const { window: win } = setup({ videos: [track("clip.mp4", VIDEO_META)] });
  runScript(win, "refreshVideoHistory();");
  await settle();

  win.document.getElementById("video-history").querySelector(".thumb").click();
  const reuse = [...win.document.getElementById("modal-body").querySelectorAll("button")]
    .find((b) => b.textContent === "reuse settings");
  assert.ok(reuse, "reuse settings offered when a sidecar exists");
  reuse.click();
  await tick();

  const $v = (id) => win.document.getElementById(id).value;
  assert.equal($v("video-prompt"), VIDEO_META.prompt);
  assert.equal($v("video-negative"), VIDEO_META.negative_prompt);
  assert.equal($v("video-image"), VIDEO_META.input_image);
  assert.equal($v("video-seconds"), String(VIDEO_META.seconds));
  assert.equal($v("video-fps"), String(VIDEO_META.fps));
  assert.equal($v("video-width"), String(VIDEO_META.width));
  assert.equal($v("video-height"), String(VIDEO_META.height));
  assert.equal($v("video-seed"), String(VIDEO_META.seed));
  assert.equal($v("video-steps"), String(VIDEO_META.steps));
  assert.equal($v("video-cfg"), String(VIDEO_META.cfg));
});

/* ------------------------------------------------------- 4. rename + previews */

test("music: rename POSTs the new name to the per-medium rename route", async () => {
  const { window: win, calls, startRecording } = setup({ tracks: [track("song.flac")] });
  runScript(win, "refreshMusicHistory();");
  await settle();
  runScript(win, 'window.prompt = () => "renamed.flac";');

  win.document.getElementById("music-history").querySelector(".thumb").click();
  startRecording();
  [...win.document.getElementById("modal-body").querySelectorAll("button")]
    .find((b) => b.textContent.startsWith("rename")).click();
  await settle(6);

  const renames = calls.filter((c) => c.url.includes("/rename"));
  assert.equal(renames.length, 1, "exactly one rename POST");
  assert.equal(renames[0].url, "/api/music/file/song.flac/rename");
  assert.deepEqual(JSON.parse(renames[0].opts.body), { new_name: "renamed.flac" });
});

test("video cards carry a real <video> preview; music cards carry no fake image",
  async () => {
    const { window: win } = setup({
      videos: [track("clip.mp4", { seconds: 5 })],
      tracks: [track("song.flac", { tags: "ZZ-TAGS-4417-doomjazz", duration_seconds: 137 })],
    });
    runScript(win, "refreshVideoHistory(); refreshMusicHistory();");
    await settle();

    const vcard = win.document.getElementById("video-history").querySelector(".thumb");
    assert.ok(vcard.querySelector("video"), "video card previews with a real <video> element");
    assert.equal(vcard.querySelector("video").getAttribute("preload"), "metadata",
      "metadata preload is what makes the frame paint without fetching the whole clip");

    const mcard = win.document.getElementById("music-history").querySelector(".thumb");
    assert.equal(mcard.querySelector("img"), null,
      "audio has no frame - a card must not pretend to show one");
    assert.match(mcard.textContent, /ZZ-TAGS-4417-doomjazz/,
      "the track's tags are what identify it, so they lead the card");
    assert.match(mcard.textContent, /2:17/, "duration is rendered as a badge");
  });

/* ------------------------------------------------- 5. failures stay visible */

/* A media element reports a refused source by firing an ERROR EVENT on itself,
   not by throwing - so nothing above it can catch this. Before, that left a
   silent black rectangle; the card must say the preview failed AND must stay,
   because a clip you cannot preview is exactly one you may want to delete. */
for (const kind of [
  { name: "video", listKey: "videos", box: "video-history", refresh: "refreshVideoHistory",
    file: "broken.mp4", sel: "video" },
  { name: "images", listKey: "images", box: "img-history", refresh: "refreshImageHistory",
    file: "broken.png", sel: "img" },
]) {
  test(`${kind.name}: an undecodable file says so and keeps its card`, async () => {
    const { window: win } = setup({ [kind.listKey]: [track(kind.file)] });
    runScript(win, `${kind.refresh}();`);
    await settle();

    const card = win.document.getElementById(kind.box).querySelector(".thumb");
    assert.ok(card, "card rendered");
    card.querySelector(kind.sel).dispatchEvent(new win.Event("error"));
    await tick();

    assert.ok(win.document.getElementById(kind.box).querySelector(".thumb"),
      "the card must SURVIVE a failed preview - it is how the file gets deleted");
    assert.match(card.textContent, /Preview unavailable/,
      "a failed preview must say so, not sit blank");
  });
}
