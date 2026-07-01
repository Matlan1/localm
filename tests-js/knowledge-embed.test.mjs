// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// rec#438: the Knowledge indexing UI always sent embed=true (the server default)
// with no way to choose BM25-only. A "Compute embeddings when indexing" checkbox
// (default on) now feeds embed into POST /api/rag/collections/<name>/add.

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
  // Path comes from prompt() (unchanged); streamJob is stubbed so the add
  // resolves without a real job stream.
  runScript(window, `
    window.prompt = () => "/some/docs";
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
