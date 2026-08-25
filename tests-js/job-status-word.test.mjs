// SPDX-License-Identifier: AGPL-3.0-or-later
// jobStatusWord (app/helpers.js) turns a streamJob() end status into a word for
// "<Operation> " + jobStatusWord(status) messages: "disconnected" becomes
// "interrupted", every other status passes through unchanged. streamJob callers
// run their end.status through it instead of interpolating it directly.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

function call(win, expr) {
  runScript(win, `globalThis.__out = ${expr};`);
  return win.__out;
}

function stubFetch() {
  return async () => ({
    ok: true, status: 200, text: async () => "",
    json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
  });
}

// --------------------------------------------------------------------------- //
//  jobStatusWord(status) - pure function                                      //
// --------------------------------------------------------------------------- //

test("jobStatusWord: 'disconnected' becomes 'interrupted', not the raw jargon word", () => {
  const { window: win } = loadApp({ fetchImpl: stubFetch() });
  assert.equal(call(win, `jobStatusWord("disconnected")`), "interrupted");
});

test("jobStatusWord: cancelled/failed/anything else pass through unchanged", () => {
  const { window: win } = loadApp({ fetchImpl: stubFetch() });
  assert.equal(call(win, `jobStatusWord("cancelled")`), "cancelled");
  assert.equal(call(win, `jobStatusWord("failed")`), "failed");
  assert.equal(call(win, `jobStatusWord("something-new")`), "something-new");
});

// --------------------------------------------------------------------------- //
//  Wiring: a representative caller of each pattern uses it                    //
// --------------------------------------------------------------------------- //

const tick = () => new Promise((r) => setTimeout(r, 0));

function makeImageFetch() {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u === "/api/media/image/preflight") {
      return { ok: true, status: 200, json: async () => ({ missing: [] }) };
    }
    if (u === "/api/imagine" && (opts.method || "GET") === "POST") {
      return { ok: true, status: 200, json: async () => ({ job_id: "j1" }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

test("images.js toast (pattern: \"Generation \" + status): disconnected reads as 'interrupted', not raw jargon", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeImageFetch() });
  runScript(win, `streamJob = async () => ({ status: "disconnected" });`);
  win.document.getElementById("img-prompt").value = "a fox in snow";
  win.document.getElementById("img-generate").onclick();
  await tick(); await tick(); await tick();

  const toastEl = win.document.getElementById("toast");
  assert.match(toastEl.textContent, /Generation interrupted/);
  assert.doesNotMatch(toastEl.textContent, /disconnected/i,
    "the raw status word must not reach the user");
});

test("images.js toast: 'cancelled' still reads naturally (unchanged behavior)", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeImageFetch() });
  runScript(win, `streamJob = async () => ({ status: "cancelled" });`);
  win.document.getElementById("img-prompt").value = "a fox in snow";
  win.document.getElementById("img-generate").onclick();
  await tick(); await tick(); await tick();

  const toastEl = win.document.getElementById("toast");
  assert.match(toastEl.textContent, /Generation cancelled/);
});

function makeKnowledgeReembedFetch() {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/rag/") && u.endsWith("/reembed") && (opts.method || "GET") === "POST") {
      return { ok: true, status: 200, json: async () => ({ job_id: "j2" }) };
    }
    return { ok: true, status: 200, json: async () => ({ collections: [] }), text: async () => "" };
  };
}

test("knowledge.js re-embed toast (pattern: ternary + status): disconnected reads as 'interrupted'", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeKnowledgeReembedFetch() });
  runScript(win, `streamJob = async () => ({ status: "disconnected" });`);
  runScript(win, `kbConfirmReembed = () => Promise.resolve(true);`);
  await win.kbReembedCollection("manuals");
  await tick(); await tick(); await tick();

  const toastEl = win.document.getElementById("toast");
  assert.match(toastEl.textContent, /Re-embedding interrupted/);
  assert.doesNotMatch(toastEl.textContent, /disconnected/i);
});
