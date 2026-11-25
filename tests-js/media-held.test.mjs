// SPDX-License-Identifier: AGPL-3.0-or-later
// Media that privacy mode does not keep: listed under the page's result with its
// own actions, never in the History grid; named in the chat; chosen in Settings.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

const settle = async () => { for (let i = 0; i < 12; i++) await new Promise((r) => setTimeout(r, 0)); };
const ok = (body) => ({ ok: true, status: 200, text: async () => JSON.stringify(body),
                        json: async () => body, blob: async () => new Blob([]) });

for (const [page, refresh, key, api] of [
  ["img", "refreshImageHistory", "images", "imagine"],
  ["music", "refreshMusicHistory", "tracks", "music"],
  ["video", "refreshVideoHistory", "videos", "video"],
]) {
  test(`the ${api} page lists held results outside History and removes them on request`, async () => {
    let held = [{ name: "a.bin", meta: {}, path: "/p/a.bin", held: true },
                { name: "b.bin", meta: {}, path: "/p/b.bin", held: true }];
    const deleted = [];
    const fetchImpl = async (url, opts = {}) => {
      const u = String(url);
      if (u === `/api/${api}/history`) return ok({ [key]: [], held });
      if (opts.method === "DELETE") { deleted.push(u); held = held.filter((h) => !u.endsWith(h.name)); }
      return ok({});
    };
    const { window: win } = loadAppWithPages({ fetchImpl });
    runScript(win, `${refresh}();`);
    await settle();
    const box = win.document.getElementById(`${page}-result-held`);
    assert.equal(box.querySelectorAll(".thumb").length, 2);
    assert.ok(box.textContent.includes(win.t("mediaGallery.held.note")));
    assert.equal(box.querySelectorAll(".thumb-sel").length, 0);
    assert.equal(win.document.querySelectorAll(`#${page}-history .thumb`).length, 0);
    runScript(win, "confirmDanger = (title, body, label, go) => go();");
    [...box.querySelectorAll("button")]
      .find((b) => b.textContent === win.t("mediaGallery.held.discardAll")).click();
    await settle();
    await settle();
    assert.deepEqual(deleted.sort(), [`/api/${api}/file/a.bin`, `/api/${api}/file/b.bin`]);
    assert.equal(win.document.getElementById(`${page}-result-held`), null);
  });
}

test("a page with nothing held shows no such box", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: async () => ok({ images: [] }) });
  runScript(win, "refreshImageHistory();");
  await settle();
  assert.equal(win.document.getElementById("img-result-held"), null);
});

function stream(events) {
  const bytes = new TextEncoder().encode(events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join(""));
  let sent = false;
  return { ok: true, status: 200, body: { getReader: () => ({
    read: async () => (sent ? { done: true } : ((sent = true), { done: false, value: bytes })),
    cancel: async () => {}, releaseLock() {},
  }) } };
}

for (const [runner, route] of [["runImagineInChat", "/api/imagine"],
                               ["runMusicInChat", "/api/music"],
                               ["runVideoInChat", "/api/video"]]) {
  for (const isHeld of [true, false]) {
    test(`${runner} ${isHeld ? "says the result is not kept" : "says nothing about keeping"}`, async () => {
      const events = [...(isHeld ? [{ type: "held" }] : []), { type: "end", status: "done", result: "a.bin" }];
      const fetchImpl = async (url) => {
        const u = String(url);
        if (u === route) return ok({ job_id: "j1" });
        if (u.includes("/api/jobs/j1/events")) return stream(events);
        return ok({});
      };
      const { window: win } = loadApp({ fetchImpl });
      await win[runner]("a cat");
      const last = win.currentConv().messages.at(-1);
      const text = typeof last.content === "string" ? last.content : last.content[0].text;
      assert.equal(text.includes(win.t("slash.heldNote")), isHeld);
    });
  }
}

test("a list setting with options is drawn as one checkbox per option and saved as a list", async () => {
  const patches = [];
  const fetchImpl = async (url, opts = {}) => {
    if (url === "/v1/config/schema") {
      return ok({ fields: [{ key: "privacy_keep", widget: "list", label: "Keep in privacy mode",
        help: "h", group: "Privacy", owner: "core", options: ["media"], default: [] }] });
    }
    if (url === "/v1/config" && opts.method === "PATCH") { patches.push(JSON.parse(opts.body)); return ok({}); }
    return ok({ models: [], active: "", conversations: [], plugins: [] });
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  await settle();
  runScript(win, "refreshSettingsPage();");
  await settle();
  const boxes = [...win.document.querySelectorAll('input[data-key="privacy_keep"]')];
  assert.deepEqual(boxes.map((b) => [b.type, b.value, b.checked]), [["checkbox", "media", false]]);
  assert.ok(boxes[0].parentElement.textContent.includes(win.t("settings.field.privacyKeep.option.media")));
  boxes[0].click();
  [...win.document.querySelectorAll("button")].find((b) => /^Save /.test(b.textContent)
    && b.closest(".card, section, .settings-section")?.contains(boxes[0])).click();
  await settle();
  assert.deepEqual(patches.at(-1), { privacy_keep: ["media"] });
});
