// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for drag-and-drop file attachment on the chat view
// (addAttachedFiles + the #view-chat drop handler in app.js).
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));

// polls until `fn()` is truthy or the timeout expires; FileReader.onload is async
async function waitFor(fn, timeout = 800) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    if (fn()) return true;
    await settle(15);
  }
  return false;
}

function goodFetch() {
  return async (url) => {
    const u = String(url);
    if (u.includes("/api/models"))
      return { ok: true, status: 200, json: async () => ({ models: [], active: "" }) };
    if (u.includes("/api/plugins"))
      return { ok: true, status: 200, json: async () => ({ plugins: [] }) };
    return { ok: true, status: 200, json: async () => ({}) };
  };
}

function imageFile(win, name = "pic.png") {
  return new win.File([new Uint8Array([1, 2, 3])], name, { type: "image/png" });
}

test("addAttachedFiles puts an image attachment chip in #attach-chips", async () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  assert.equal(typeof window.addAttachedFiles, "function", "addAttachedFiles on window");
  window.addAttachedFiles([imageFile(window, "shot.png")]);
  const chips = window.document.getElementById("attach-chips");
  assert.ok(await waitFor(() => /shot\.png/.test(chips.textContent)),
    "the attached image shows as a chip");
});

test("dropping files on #view-chat attaches them and prevents default", async () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const zone = window.document.getElementById("view-chat");
  assert.ok(zone, "#view-chat exists");

  const ev = new window.Event("drop", { bubbles: true, cancelable: true });
  let prevented = false;
  ev.preventDefault = () => { prevented = true; };
  ev.dataTransfer = { files: [imageFile(window, "dropped.png")], types: ["Files"] };
  zone.dispatchEvent(ev);

  assert.equal(prevented, true, "drop calls preventDefault so the browser doesn't open the file");
  const chips = window.document.getElementById("attach-chips");
  assert.ok(await waitFor(() => /dropped\.png/.test(chips.textContent)),
    "the dropped image is attached");
});

test("a drop with no files is ignored (negative)", async () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const zone = window.document.getElementById("view-chat");
  const ev = new window.Event("drop", { bubbles: true, cancelable: true });
  let prevented = false;
  ev.preventDefault = () => { prevented = true; };
  ev.dataTransfer = { files: [], types: [] };
  zone.dispatchEvent(ev);
  await settle(10);
  assert.equal(prevented, false, "an empty drop is not intercepted");
  assert.equal(window.document.getElementById("attach-chips").textContent.trim(), "",
    "no chip is added for an empty drop");
});

test("dragover with files adds the .drag-over highlight class", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  const zone = window.document.getElementById("view-chat");
  const ev = new window.Event("dragover", { bubbles: true, cancelable: true });
  ev.preventDefault = () => {};
  ev.dataTransfer = { types: ["Files"], dropEffect: "" };
  zone.dispatchEvent(ev);
  assert.ok(zone.classList.contains("drag-over"), "drag-over class is applied while a file hovers");
});
