// SPDX-License-Identifier: AGPL-3.0-or-later
// PWA "feel real" features: ingest images shared into localm from the phone,
// download coder-created files onto the device, and a one-tap camera button.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

test("ingestSharedFiles adds shared images as chat attachments and clears the inbox", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    calls.push({ url, method: opts.method || "GET", body: opts.body });
    if (url === "/api/share/pending") {
      return { ok: true, status: 200, text: async () => "", json: async () => ({
        items: [{ id: "a1", name: "photo.png", type: "image/png",
                  data_uri: "data:image/png;base64,AAAA" }] }) };
    }
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
  const { window: win } = loadApp({ fetchImpl });
  await win.ingestSharedFiles();
  await new Promise((r) => setTimeout(r, 0));

  const chips = win.document.getElementById("attach-chips");
  assert.ok(chips.querySelector("img"), "shared image rendered as an attachment chip");
  const clear = calls.find((c) => c.url === "/api/share/clear");
  assert.ok(clear && clear.method === "POST", "inbox cleared via POST after ingest");
  assert.deepEqual(JSON.parse(clear.body).ids, ["a1"]);
});

test("ingestSharedFiles is a no-op (no clear) when nothing is pending", async () => {
  let clearCalled = false;
  const fetchImpl = async (url) => {
    if (url === "/api/share/clear") clearCalled = true;
    if (url === "/api/share/pending") {
      return { ok: true, status: 200, text: async () => "", json: async () => ({ items: [] }) };
    }
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
  const { window: win } = loadApp({ fetchImpl });
  await win.ingestSharedFiles();
  assert.equal(clearCalled, false);
});

test("downloadCoderFile fetches the confined endpoint and saves with the basename", async () => {
  let fetched = null;
  const { window: win } = loadApp({
    fetchImpl: async (url) => {
      if (String(url).includes("/files/download")) fetched = url;   // ignore bg fetches
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
               blob: async () => new win.Blob(["data"]) };
    },
  });
  win.URL.createObjectURL = () => "blob:fake";
  win.URL.revokeObjectURL = () => {};
  let clicked = null;
  win.HTMLAnchorElement.prototype.click = function () { clicked = this; };

  await win.downloadCoderFile({ info: { id: "sess9" } }, "src/deep/file.py");
  assert.match(fetched,
    /\/api\/coder\/sessions\/sess9\/files\/download\?path=src%2Fdeep%2Ffile\.py/);
  assert.ok(clicked, "a download anchor was clicked");
  assert.equal(clicked.download, "file.py", "saves with the file's basename");
});

test("the camera button opens the dedicated capture input", () => {
  const { window: win } = loadApp();
  const cam = win.document.getElementById("chat-camera");
  const camFile = win.document.getElementById("chat-camera-file");
  assert.ok(cam && camFile, "camera button + capture input present");
  assert.equal(camFile.getAttribute("capture"), "environment", "opens the camera");
  assert.equal(camFile.getAttribute("accept"), "image/*");
  let clicked = false;
  camFile.click = () => { clicked = true; };
  cam.onclick();
  assert.ok(clicked, "clicking the camera button opens the capture input");
});
