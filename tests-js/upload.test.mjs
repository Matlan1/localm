// SPDX-License-Identifier: AGPL-3.0-or-later
// R37: the Settings "Upload files" section uploads files from this device/phone
// into <home>/uploads/ (a multipart POST to /api/upload), lists what's there
// (GET /api/uploads), and removes one (DELETE /api/uploads/{name}). The list is
// built with safe DOM nodes (textContent), never innerHTML.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(calls, uploads) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    calls.push({ url: u, method, headers: opts.headers || {}, body: opts.body });
    if (u === "/api/uploads" && method === "GET") {
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ items: uploads, dir: "/home/uploads" }) };
    }
    if (u === "/api/upload" && method === "POST") {
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ uploaded: [{ name: "note.txt", bytes: 5 }], dir: "/home/uploads" }) };
    }
    if (u.startsWith("/api/uploads/") && method === "DELETE") {
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ removed: decodeURIComponent(u.split("/").pop()) }) };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [], items: [] }) };
  };
}

test("R37: refreshUploadsList renders one li per uploaded file (safe textContent)", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch(calls, [
      { name: "a.txt", bytes: 10 },
      { name: "<img src=x>.png", bytes: 2048 },   // a hostile name must stay inert text
    ]),
  });
  await window.refreshUploadsList();
  const list = window.document.getElementById("upload-list");
  const items = list.querySelectorAll("li");
  assert.equal(items.length, 2, "one row per uploaded file");
  assert.ok(items[0].textContent.includes("a.txt"), "shows the file name");
  // The hostile name is rendered as TEXT, not parsed into an <img> element.
  assert.equal(list.querySelectorAll("img").length, 0, "no markup injected from a file name");
  assert.ok(items[1].textContent.includes("<img src=x>.png"), "hostile name shown literally");
});

test("R37: clicking a row's Remove button DELETEs that file then refreshes", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(calls, [{ name: "gone.bin", bytes: 4 }]) });
  await window.refreshUploadsList();
  const delBtn = window.document.querySelector("#upload-list .upload-del");
  assert.ok(delBtn, "each row has a Remove button");
  delBtn.click();
  await new Promise((r) => setTimeout(r, 0));
  const del = calls.find((c) => c.method === "DELETE" && c.url === "/api/uploads/gone.bin");
  assert.ok(del, "Remove issues DELETE /api/uploads/gone.bin");
});

test("R37: Upload posts the chosen files as multipart (no JSON Content-Type)", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(calls, []) });
  const input = window.document.getElementById("upload-input");
  const file = new window.File(["hello"], "note.txt", { type: "text/plain" });
  Object.defineProperty(input, "files", { value: [file], configurable: true });

  window.document.getElementById("upload-send").click();
  await new Promise((r) => setTimeout(r, 0));

  const post = calls.find((c) => c.url === "/api/upload" && c.method === "POST");
  assert.ok(post, "Upload POSTs to /api/upload");
  assert.ok(post.body instanceof window.FormData, "body is multipart FormData");
  // Crucial: no JSON Content-Type, so the browser sets multipart/form-data + boundary.
  const ctKey = Object.keys(post.headers).find((k) => k.toLowerCase() === "content-type");
  assert.ok(!ctKey, "the JSON Content-Type header is stripped for the multipart upload");
});

test("R37: Upload with no file chosen does not POST", async () => {
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch(calls, []) });
  // #upload-input has no files selected.
  window.document.getElementById("upload-send").click();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(calls.filter((c) => c.url === "/api/upload").length, 0,
    "nothing is uploaded without a chosen file");
});
