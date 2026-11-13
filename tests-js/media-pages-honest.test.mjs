// SPDX-License-Identifier: AGPL-3.0-or-later
// The Images, Music and Video pages: what they say and do on the paths a person
// meets (one-item toasts, a cancelled move, an out-of-range duration, reused
// settings, the workflow in use, a job started from the chat).
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));
const settle = async () => { for (let i = 0; i < 6; i++) await tick(); };
const ok = (body) => ({ ok: true, status: 200, text: async () => JSON.stringify(body),
                        json: async () => body });

function setup(routes = {}) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    for (const [part, body] of Object.entries(routes)) {
      if (u.includes(part)) return ok(typeof body === "function" ? body(u, opts) : body);
    }
    return ok({});
  };
  const { window } = loadAppWithPages({ fetchImpl });
  return { window, calls };
}

test("a bulk move of one image says 'image', not 'images'", async () => {
  const { window: win } = setup({
    "/api/imagine/history": { images: [{ name: "a.png", meta: {} }] },
    "/move": { path: "/dest/a.png" },
  });
  runScript(win, 'pickDirectory = () => Promise.resolve("/dest");');
  runScript(win, "refreshImageHistory();");
  await settle();
  win.document.querySelector("#img-history .thumb-sel").click();
  const move = [...win.document.querySelectorAll("#img-bulk button")]
    .find((b) => b.textContent === win.t("mediaGallery.moveToFolder"));
  move.click();
  await settle();
  assert.equal(win.document.getElementById("toast").textContent, "Moved 1 image");
});

test("cancelling the move picker re-opens the item's detail view", async () => {
  const { window: win, calls } = setup({
    "/api/music/history": { tracks: [{ name: "song.flac", meta: {} }] },
  });
  runScript(win, "pickDirectory = () => { closeModal(); return Promise.resolve(null); };");
  runScript(win, "refreshMusicHistory();");
  await settle();
  win.document.querySelector("#music-history .thumb").click();
  const move = [...win.document.querySelectorAll("#modal-body button")]
    .find((b) => b.textContent === win.t("mediaGallery.moveToFolder"));
  move.click();
  await settle();
  assert.equal(win.document.getElementById("modal").style.display, "flex");
  assert.equal(win.document.getElementById("modal-title").textContent, "song.flac");
  assert.equal(calls.filter((c) => c.url.includes("/move")).length, 0);
});

for (const value of ["0", "-5", "", "4000"]) {
  test(`a music Duration of '${value}' is refused with the range and starts no job`, async () => {
    const { window: win, calls } = setup({ "/api/music/backend": { active: "comfy" } });
    win.document.getElementById("music-tags").value = "lofi";
    win.document.getElementById("music-duration").value = value;
    win.document.getElementById("music-generate").click();
    await settle();
    assert.match(win.document.getElementById("toast").textContent,
      /Duration must be between 5 and 3600 seconds/);
    assert.equal(calls.filter((c) => c.url.endsWith("/api/music") && c.opts.method === "POST").length, 0);
  });
}

test("Reuse settings on a native track restores its Shift", async () => {
  const { window: win } = setup({
    "/api/music/history": { tracks: [{ name: "t.flac", meta: { tags: "jazz", shift: 2.5 } }] },
  });
  runScript(win, "refreshMusicHistory();");
  await settle();
  win.document.querySelector("#music-history .thumb").click();
  const reuse = [...win.document.querySelectorAll("#modal-body button")]
    .find((b) => b.textContent === win.t("mediaGallery.reuseSettings"));
  reuse.click();
  await settle();
  assert.equal(win.document.getElementById("music-shift").value, "2.5");
});

test("Reuse on a clip made with the recommended set leaves the model on its default", async () => {
  const { window: win } = setup({
    "/api/video/history": { videos: [{ name: "c.mp4", size_bytes: 1, meta: {
      backend: "native", model: "wan2.1-t2v-1.3b", prompt: "p", flow_shift: 3 } }] },
  });
  runScript(win, `
    globalThis.__set = null;
    videoModelPicker.matchModel = async () => "";
    videoModelPicker.setModels = (m) => { globalThis.__set = m; return Promise.resolve(); };
    videoModelPicker.setLoras = () => {};
    refreshVideoHistory();`);
  await settle();
  win.document.querySelector("#video-history .thumb").click();
  [...win.document.querySelectorAll("#modal-body button")]
    .find((b) => b.textContent === win.t("mediaGallery.reuseSettings")).click();
  await settle();
  assert.ok(win.__set, "setModels was called");
  assert.ok(!win.__set.model, "no unregistered set name is put in the picker");
  assert.equal(win.document.getElementById("video-flow-shift").value, "3");
});

test("the workflow in use has its Delete disabled; another one can be deleted", () => {
  const { window: win } = setup();
  runScript(win, `
    globalThis.__active = workflowRow("image", "a.json", "a.json", true, true);
    globalThis.__other = workflowRow("image", "b.json", "b.json", false, true);`);
  const activeDel = win.__active.querySelector(".workflow-del");
  const otherDel = win.__other.querySelector(".workflow-del");
  assert.equal(activeDel.disabled, true);
  assert.equal(activeDel.title, win.t("workflow.deleteActiveTitle"));
  assert.equal(otherDel.disabled, false);
});

test("an unreachable ComfyUI is described with the catalog text", async () => {
  const { window: win } = setup({
    "/comfy-models": { reachable: false, message: "server English text", slots: [] },
  });
  runScript(win, 'comfyModelPicker("image").then((w) => { globalThis.__wrap = w; });');
  await settle();
  const text = win.__wrap.textContent;
  assert.ok(text.includes(win.t("workflow.comfyUnreachable")), text);
  assert.ok(!text.includes("server English text"), text);
});

test("a music job started elsewhere is followed on the Music page", async () => {
  const { window: win } = setup({
    "/api/activity": { operations: [
      { id: "old", kind: "music", status: "done", created_at: 1 },
      { id: "j-run", kind: "music", status: "running", created_at: 2 },
      { id: "img", kind: "imagine", status: "running", created_at: 3 }] },
    "/api/music/history": { tracks: [] },
  });
  runScript(win, `
    globalThis.__followed = [];
    streamJob = async (id, onLine) => { globalThis.__followed.push(id); onLine("adopted line");
                                        return { status: "failed" }; };`);
  runScript(win, "refreshMusicHistory(); refreshMusicHistory();");
  await settle();
  const music = [...win.__followed].filter((id) => id !== "img");
  assert.deepEqual(music, ["j-run"], "the running music job is followed once, a finished one never");
  assert.match(win.document.getElementById("music-log").textContent, /adopted line/);
});

test("an image the browser cannot decode shows the reason in its detail view", async () => {
  const { window: win } = setup({
    "/api/imagine/history": { images: [{ name: "bad.png", meta: {} }] },
  });
  runScript(win, 'fetchImageURL = async () => "blob:x"; refreshImageHistory();');
  await settle();
  win.document.querySelector("#img-history .thumb").click();
  await settle();
  const img = win.document.querySelector("#modal-body img");
  assert.ok(img, "the detail view has its image");
  img.dispatchEvent(new win.Event("error"));
  assert.ok(win.document.getElementById("modal-body").textContent
    .includes(win.t("mediaGallery.previewUnavailable")));
});
