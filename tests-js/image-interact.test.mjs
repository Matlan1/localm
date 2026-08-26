// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 50));
const okFetch = async () => ({
  ok: true, status: 200,
  json: async () => ({ models: [], conversations: [], plugins: [] }),
  text: async () => "",
});

const DATA_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB" +
  "CAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

test("VIS-2: imageFilename derives a sane download name", async () => {
  const { window } = loadApp({ fetchImpl: okFetch });
  await tick();
  assert.equal(window.imageFilename("data:image/png;base64,AAAA"), "localm-image.png");
  assert.equal(window.imageFilename("data:image/jpeg;base64,AAAA"), "localm-image.jpg");
  assert.equal(window.imageFilename("/api/images/gui_123/out.png"), "out.png");
  assert.equal(window.imageFilename("/api/images/nope"), "localm-image.png");
});

test("VIS-2: clicking a chat image opens a lightbox with Save + Copy; it closes", async () => {
  const { window } = loadApp({ fetchImpl: okFetch });
  await tick();
  const doc = window.document;
  const container = doc.createElement("div");
  doc.body.appendChild(container);
  window.addMessageRow(container, "user", "", { images: [DATA_PNG] });
  const img = container.querySelector(".msg-img");
  assert.ok(img, "the image rendered");
  assert.equal(img.style.cursor, "zoom-in", "image signals it is clickable");
  img.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  const box = doc.querySelector(".img-lightbox");
  assert.ok(box, "a lightbox opened on click");
  assert.equal(box.querySelector(".img-lightbox-img").src, DATA_PNG, "shows the image");
  const labels = [...box.querySelectorAll("button")].map((b) => b.textContent);
  assert.ok(labels.includes("Save"), "a Save (download) control is present");
  assert.ok(labels.includes("Copy image"), "a Copy control is present");
  [...box.querySelectorAll("button")].find((b) => b.textContent === "Close").click();
  assert.equal(doc.querySelector(".img-lightbox"), null, "Close dismisses the lightbox");
});

test("VIS-2: Save triggers an <a download> with the derived filename", async () => {
  const { window } = loadApp({ fetchImpl: okFetch });
  await tick();
  const doc = window.document;
  let dl = null;
  const proto = window.HTMLAnchorElement.prototype;
  const orig = proto.click;
  proto.click = function () { dl = { href: this.href, name: this.getAttribute("download") }; };
  try {
    window.openImageLightbox(DATA_PNG, "shot.png");
    const box = doc.querySelector(".img-lightbox");
    [...box.querySelectorAll("button")].find((b) => b.textContent === "Save").click();
  } finally {
    proto.click = orig;
  }
  assert.ok(dl, "a download was triggered");
  assert.equal(dl.name, "shot.png", "with the given filename");
});

test("VIS-2: copy on an image-only message copies the IMAGE, not empty text", async () => {
  const { window } = loadApp({ fetchImpl: okFetch });
  await tick();
  const doc = window.document;
  const writes = [];
  Object.defineProperty(window.navigator, "clipboard", {
    configurable: true,
    value: { writeText: (t) => { writes.push(t); return Promise.resolve(); } },
  });
  const container = doc.createElement("div");
  doc.body.appendChild(container);
  window.addMessageRow(container, "user", "", { images: [DATA_PNG] });
  container.querySelector(".copy-btn").click();
  await tick();
  assert.equal(writes.length, 0, "the empty-text copy path is NOT taken for an image-only message");
});

test("VIS-2: copy on a text message still copies the text", async () => {
  const { window } = loadApp({ fetchImpl: okFetch });
  await tick();
  const doc = window.document;
  const writes = [];
  Object.defineProperty(window.navigator, "clipboard", {
    configurable: true,
    value: { writeText: (t) => { writes.push(t); return Promise.resolve(); } },
  });
  const container = doc.createElement("div");
  doc.body.appendChild(container);
  window.addMessageRow(container, "assistant", "hello world", {});
  container.querySelector(".copy-btn").click();
  await tick();
  assert.deepEqual(writes, ["hello world"], "plain text copy is unchanged");
});

test("copy on a text message toasts instead of claiming success when the clipboard write fails", async () => {
  const { window } = loadApp({ fetchImpl: okFetch });
  await tick();
  const doc = window.document;
  Object.defineProperty(window.navigator, "clipboard", {
    configurable: true,
    value: { writeText: () => Promise.reject(new window.DOMException("denied", "NotAllowedError")) },
  });
  const container = doc.createElement("div");
  doc.body.appendChild(container);
  window.addMessageRow(container, "assistant", "hello world", {});
  const btn = container.querySelector(".copy-btn");
  btn.click();
  await tick();
  assert.equal(btn.textContent, "copy",
    "a failed clipboard write must not leave the button reading 'copied'");
  const toastEl = doc.getElementById("toast");
  assert.match(toastEl.textContent, /[Cc]ould not copy/);
});
