// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// The Knowledge indexing UI's "Compute embeddings when indexing" checkbox
// (default on) feeds embed into POST /api/rag/collections/<name>/add.

function setup() {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    if (u.includes("/add"))
      return { ok: true, status: 200, text: async () => "", json: async () => ({ job_id: "j1" }) };
    return { ok: true, status: 200, text: async () => "", json: async () => ({ collections: [] }) };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  // pickPath is stubbed to resolve an array of paths, streamJob to resolve at
  // once so the add completes without a real job stream.
  runScript(window, `
    globalThis.__pickOpts = null;
    globalThis.__pickResult = ["/some/docs"];
    caps.fsAccess = "host";   // host access -> the picker path is offered
    window.prompt = () => { throw new Error("prompt() must not be used (suppressed on mobile/PWA)"); };
    pickPath = (opts) => { globalThis.__pickOpts = opts; return Promise.resolve(globalThis.__pickResult); };
    streamJob = () => Promise.resolve({ status: "done" });
  `);
  return { window, calls };
}

const tick = () => new Promise((r) => setTimeout(r, 0));
const addBody = (calls) => {
  const c = calls.find((x) => x.url.includes("/add"));
  return c ? JSON.parse(c.opts.body) : null;
};

test("the embed checkbox exists and is checked by default", () => {
  const { window } = setup();
  const box = window.document.getElementById("kb-embed");
  assert.ok(box, "kb-embed checkbox exists");
  assert.equal(box.type, "checkbox");
  assert.equal(box.checked, true, "defaults to on (server default preserved)");
});

test("checked (default) -> add sends embed:true", async () => {
  const { window, calls } = setup();
  runScript(window, "kbAddDocs('mycoll');");
  await tick();
  await tick();
  const body = addBody(calls);
  assert.ok(body, "an /add POST was sent");
  assert.deepEqual(body.paths, ["/some/docs"]);
  assert.equal(body.embed, true);
});

test("unchecked -> add sends embed:false (BM25-only)", async () => {
  const { window, calls } = setup();
  window.document.getElementById("kb-embed").checked = false;
  runScript(window, "kbAddDocs('mycoll');");
  await tick();
  await tick();
  const body = addBody(calls);
  assert.ok(body, "an /add POST was sent");
  assert.equal(body.embed, false);
});

test("multi-select -> every chosen path is sent in one /add job", async () => {
  const { window, calls } = setup();
  runScript(window, `globalThis.__pickResult = ["/a/one.md", "/a/sub", "/b/two.pdf"];`);
  runScript(window, "kbAddDocs('mycoll');");
  await tick();
  await tick();
  const body = addBody(calls);
  assert.ok(body, "an /add POST was sent");
  assert.deepEqual(body.paths, ["/a/one.md", "/a/sub", "/b/two.pdf"]);
  assert.equal(calls.filter((c) => c.url.includes("/add")).length, 1, "exactly one add job");
});

test("the picker is opened in multi mode with the RAG extension filter", async () => {
  const { window } = setup();
  runScript(window, "kbAddDocs('mycoll');");
  await tick();
  const opts = window.__pickOpts;
  assert.ok(opts, "pickPath was called");
  assert.equal(opts.mode, "multi");
  assert.ok(Array.isArray(opts.exts) && opts.exts.includes(".pdf") && opts.exts.includes(".md"),
    "supported extensions are passed so unsupported files grey out");
});

test("cancelling the picker (null) sends no /add request", async () => {
  const { window, calls } = setup();
  runScript(window, `globalThis.__pickResult = null;`);
  runScript(window, "kbAddDocs('mycoll');");
  await tick();
  await tick();
  assert.equal(calls.filter((c) => c.url.includes("/add")).length, 0,
    "no add when the picker is dismissed");
});

test("without host access, add-docs never opens the picker or POSTs", async () => {
  const { window, calls } = setup();
  // A non-host caller: add-docs short-circuits without opening the picker.
  runScript(window, `
    caps.fsAccess = "none";
    globalThis.__pickOpts = null;
  `);
  runScript(window, "kbAddDocs('mycoll');");
  await tick();
  await tick();
  assert.equal(window.__pickOpts, null, "the host picker is not opened");
  assert.equal(calls.filter((c) => c.url.includes("/add")).length, 0, "no /add POST");
});
