// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// Ask-to-add: when the owner adds a folder outside the whitelist, /add replies
// 409 needs_consent. The GUI confirms, appends the folder to rag_allowed_roots
// (owner-only config), and retries the add once.

const tick = () => new Promise((r) => setTimeout(r, 0));

function setup(fetchImpl) {
  const { window } = loadAppWithPages({ fetchImpl });
  runScript(window, `
    caps.fsAccess = "host";
    window.prompt = () => { throw new Error("prompt() must not be used"); };
    pickPath = (opts) => Promise.resolve(["/ext/docs"]);
    streamJob = () => Promise.resolve({ status: "done" });
  `);
  return window;
}

test("outside-whitelist add: 409 -> confirm -> widen allowed roots -> retry indexes",
  async () => {
    const calls = [];
    let addCount = 0;
    const fetchImpl = async (url, opts = {}) => {
      const u = String(url);
      calls.push({ url: u, opts });
      if (u.includes("/add")) {
        addCount += 1;
        if (addCount === 1) {   // first attempt: outside the whitelist
          return { ok: false, status: 409, text: async () => "",
            json: async () => ({ needs_consent: true, reason: "outside_allowed",
                                 addable: ["/ext/docs"] }) };
        }
        return { ok: true, status: 200, text: async () => "",
                 json: async () => ({ job_id: "j1" }) };   // retry succeeds
      }
      if (u === "/v1/config" && opts.method === "PATCH")
        return { ok: true, status: 200, text: async () => "", json: async () => ({}) };
      if (u === "/v1/config")   // current allowed roots (owner view)
        return { ok: true, status: 200, text: async () => "",
                 json: async () => ({ rag_allowed_roots: ["/existing"] }) };
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ collections: [], fs_access: "host" }) };
    };
    const window = setup(fetchImpl);
    // Auto-approve the in-page confirm modal.
    runScript(window, `kbConfirmAddRoots = () => Promise.resolve(true);`);
    runScript(window, "kbAddDocs('mycoll');");
    for (let i = 0; i < 10; i++) await tick();

    // The allow-list was widened by PATCH, MERGING the new folder with the existing.
    const patch = calls.find((c) => c.url === "/v1/config" && c.opts.method === "PATCH");
    assert.ok(patch, "a PATCH /v1/config widened the allowed roots");
    const body = JSON.parse(patch.opts.body);
    assert.deepEqual(body.rag_allowed_roots.slice().sort(),
      ["/existing", "/ext/docs"].sort(), "new folder merged with existing roots");
    // Two /add POSTs: the 409 and the successful retry.
    assert.equal(calls.filter((c) => c.url.includes("/add")).length, 2,
      "the add retried after consent");
  });

test("a non-consent 409 shows the server's real detail, not 'body stream already read'",
  async () => {
    // The stub emulates single-read Response semantics: json() succeeds once,
    // then throws "body stream already read".
    let addBodyReads = 0;
    const fetchImpl = async (url) => {
      if (String(url).includes("/add")) {
        return {
          ok: false, status: 409, statusText: "Conflict", text: async () => "",
          json: async () => {
            addBodyReads += 1;
            if (addBodyReads > 1) throw new TypeError("body stream already read");
            return { detail: "collection is busy" };   // a 409 that is NOT needs_consent
          },
        };
      }
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ collections: [], fs_access: "host" }) };
    };
    const window = setup(fetchImpl);
    runScript(window, `
      window.__log = document.createElement("div");
      kbRunAdd("mycoll", ["/some/path"], true, window.__log);
    `);
    for (let i = 0; i < 8; i++) await tick();
    const toast = window.document.getElementById("toast").textContent;
    const log = window.__log.textContent;
    assert.ok(/collection is busy/.test(toast),
      "the server's real 409 detail reaches the user: " + toast);
    assert.ok(!/already read/.test(toast + log),
      "the double-read bug is gone (no 'body stream already read')");
  });

test("declining the confirm does NOT widen roots and does not retry", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    if (u.includes("/add"))
      return { ok: false, status: 409, text: async () => "",
        json: async () => ({ needs_consent: true, addable: ["/ext/docs"] }) };
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ collections: [], fs_access: "host" }) };
  };
  const window = setup(fetchImpl);
  runScript(window, `kbConfirmAddRoots = () => Promise.resolve(false);`);
  runScript(window, "kbAddDocs('mycoll');");
  for (let i = 0; i < 8; i++) await tick();
  assert.equal(calls.filter((c) => c.url === "/v1/config" && c.opts.method === "PATCH").length,
    0, "no PATCH when the user declines");
  assert.equal(calls.filter((c) => c.url.includes("/add")).length, 1,
    "only the initial add attempt, no retry");
});
