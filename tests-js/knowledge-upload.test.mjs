// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// Without host filesystem access, kbAddDocs routes to kbUploadDocs, which reads
// each device file to base64 and POSTs to /api/rag/collections/<name>/upload.

const tick = () => new Promise((r) => setTimeout(r, 0));

function setup(fetchImpl) {
  const { window } = loadAppWithPages({ fetchImpl });
  runScript(window, `
    caps.fsAccess = "none";                    // no host access -> device upload
    streamJob = () => Promise.resolve({ status: "done" });
  `);
  return window;
}

test("non-host add-docs uploads device files to /upload", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    if (u.includes("/upload"))
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ job_id: "j1" }) };
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ collections: [], fs_access: "none" }) };
  };
  const window = setup(fetchImpl);
  runScript(window, `
    pickDeviceFiles = () => Promise.resolve([{ name: "notes.md", size: 12 }]);
    fileToB64 = () => Promise.resolve("aGVsbG8=");
  `);
  runScript(window, "kbAddDocs('mycoll');");
  for (let i = 0; i < 8; i++) await tick();

  const up = calls.find((c) => c.url.includes("/upload"));
  assert.ok(up, "an /upload POST was sent");
  const body = JSON.parse(up.opts.body);
  assert.equal(body.files.length, 1);
  assert.equal(body.files[0].filename, "notes.md");
  assert.equal(body.files[0].content_b64, "aGVsbG8=");
  // a non-host caller does not hit the server-path /add
  assert.equal(calls.filter((c) => c.url.endsWith("/add")).length, 0,
    "no server-path /add for a non-host caller");
});

test("oversized device files are skipped, not uploaded", async () => {
  const calls = [];
  const fetchImpl = async (url) => {
    calls.push(String(url));
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ collections: [], fs_access: "none" }) };
  };
  const window = setup(fetchImpl);
  runScript(window, `
    pickDeviceFiles = () => Promise.resolve([{ name: "huge.pdf", size: 31 * 1024 * 1024 }]);
    fileToB64 = () => Promise.resolve("x");
  `);
  runScript(window, "kbAddDocs('mycoll');");
  for (let i = 0; i < 6; i++) await tick();
  assert.equal(calls.filter((c) => c.includes("/upload")).length, 0,
    "a file over the 30 MB cap is not uploaded");
});

test("cancelling the device picker sends no upload", async () => {
  const calls = [];
  const fetchImpl = async (url) => {
    calls.push(String(url));
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ collections: [], fs_access: "none" }) };
  };
  const window = setup(fetchImpl);
  runScript(window, `pickDeviceFiles = () => Promise.resolve([]);`);
  runScript(window, "kbAddDocs('mycoll');");
  for (let i = 0; i < 6; i++) await tick();
  assert.equal(calls.filter((c) => c.includes("/upload")).length, 0,
    "no upload when the picker is dismissed");
});
