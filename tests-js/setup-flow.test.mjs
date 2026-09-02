// SPDX-License-Identifier: AGPL-3.0-or-later
// The Setup view (ADR-0001): a guided desktop flow for the llama.cpp runtime
// and the first model, reached only by a deliberate click. Covers the
// trigger gating (never auto-opens), the runtime status/provision card
// (reusing models.js's postRuntimeUpdate, not a fork of it), the model-count
// card, and the "Find a model" / "Go to chat" navigation.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function makeFetch(routes) {
  return async (url, opts = {}) => {
    const u = String(url);
    for (const [frag, resp] of routes) {
      if (u.includes(frag)) {
        const body = typeof resp === "function" ? resp(opts) : resp;
        return { ok: true, status: 200, text: async () => "", json: async () => body };
      }
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function flush() {
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
}

/** Stub the job stream so a provision completes without a real subprocess -
 *  same shape as runtime-update.test.mjs's stub, so both tests would be
 *  changed together if streamJob's contract ever changes. */
function stubJobStream(window, status = "done", lines = ["Fetching release b10361 ..."]) {
  runScript(window, `streamJob = (id, onLine) => {
    ${JSON.stringify(lines)}.forEach(onLine);
    return Promise.resolve({ status: ${JSON.stringify(status)} });
  };`);
}

function showSetup(window) {
  const doc = window.document;
  doc.getElementById("nav-setup").click();
  return doc;
}

// --------------------------------------------------------------------------- //
//  Trigger gating: never auto-opens                                           //
// --------------------------------------------------------------------------- //

test("Setup: a fresh boot lands on chat, not Setup, with no click", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  const doc = window.document;
  assert.equal(doc.getElementById("view-chat").classList.contains("active"), true);
  assert.equal(doc.getElementById("view-setup").classList.contains("active"), false);
});

test("Setup: visiting Models does not incidentally open Setup", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  const doc = window.document;
  doc.getElementById("nav-models").click();
  assert.equal(doc.getElementById("view-models").classList.contains("active"), true);
  assert.equal(doc.getElementById("view-setup").classList.contains("active"), false);
});

test("Setup: only the nav-setup click reaches it", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  const doc = showSetup(window);
  assert.equal(doc.getElementById("view-setup").classList.contains("active"), true);
});

// --------------------------------------------------------------------------- //
//  Runtime status card                                                        //
// --------------------------------------------------------------------------- //

test("Setup: nothing installed shows the recommendation and offers Install", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/backend", { installed: null, vendor: "nvidia", recommended: "cuda", warning: null }],
  ]) });
  const doc = showSetup(window);
  await flush();
  assert.match(doc.getElementById("onb-runtime-status").textContent, /cuda/);
  assert.equal(doc.getElementById("onb-runtime-install").hidden, false);
});

test("Setup: an installed backend hides the Install button and shows it by name", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/backend", { installed: "vulkan", vendor: "amd", recommended: "vulkan", warning: null }],
  ]) });
  const doc = showSetup(window);
  await flush();
  assert.match(doc.getElementById("onb-runtime-status").textContent, /vulkan/);
  assert.equal(doc.getElementById("onb-runtime-install").hidden, true);
});

test("Setup: a resolution warning surfaces independently of installed state", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/backend", { installed: null, vendor: null, recommended: null,
      warning: "LLAMA_CPP_LIB points at a file that does not exist" }],
  ]) });
  const doc = showSetup(window);
  await flush();
  const warn = doc.getElementById("onb-runtime-warning");
  assert.equal(warn.hidden, false);
  assert.match(warn.textContent, /LLAMA_CPP_LIB/);
});

test("Setup: Install now POSTs with no backend/tag, streams to done, and re-checks", async () => {
  // /api/backend is stateful: not-installed until the (mocked) provision
  // completes, so the post-success re-check is genuinely exercised rather
  // than just re-reading the same fixture.
  let installed = null;
  const calls = [];
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/backend", () => ({ installed, vendor: "amd", recommended: "vulkan", warning: null })],
    ["/api/runtime/update", (opts) => {
      calls.push(JSON.parse(opts.body));
      installed = "vulkan";
      return { job_id: "j1" };
    }],
  ]) });
  stubJobStream(window, "done", ["Provisioning vulkan ..."]);
  const doc = showSetup(window);
  await flush();
  assert.match(doc.getElementById("onb-runtime-status").textContent, /vulkan/);
  doc.getElementById("onb-runtime-install").click();
  await flush();
  assert.deepEqual(calls, [{}], "Setup must never send an explicit backend/tag - only Settings does");
  assert.match(doc.getElementById("onb-runtime-status").textContent, /Installed: vulkan/);
  assert.equal(doc.getElementById("onb-runtime-install").hidden, true);
});

test("Setup: a failed provision shows the job's own reason, not a generic failure", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/backend", { installed: null, vendor: null, recommended: "cpu", warning: null }],
    ["/api/runtime/update", { job_id: "j1" }],
  ]) });
  stubJobStream(window, "failed", ["error: could not fetch release listing"]);
  const doc = showSetup(window);
  await flush();
  doc.getElementById("onb-runtime-install").click();
  await flush();
  assert.match(doc.getElementById("onb-runtime-status").textContent, /could not fetch release listing/);
  assert.equal(doc.getElementById("onb-runtime-install").disabled, false,
    "a failed attempt must leave the button usable again, not stuck disabled");
});

// --------------------------------------------------------------------------- //
//  Model-count card                                                           //
// --------------------------------------------------------------------------- //

test("Setup: zero models shows the empty message", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/models", { models: [] }],
  ]) });
  const doc = showSetup(window);
  await flush();
  assert.match(doc.getElementById("onb-models-status").textContent, /No models yet/);
});

test("Setup: some models shows a count, singular and plural", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/models", { models: [{ name: "a" }] }],
  ]) });
  const doc = showSetup(window);
  await flush();
  assert.match(doc.getElementById("onb-models-status").textContent, /^1 model ready/);

  const { window: w2 } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/models", { models: [{ name: "a" }, { name: "b" }, { name: "c" }] }],
  ]) });
  const doc2 = showSetup(w2);
  await flush();
  assert.match(doc2.getElementById("onb-models-status").textContent, /^3 models ready/);
});

test("Setup: Find a model navigates to Models and focuses the search box", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  const doc = showSetup(window);
  await flush();
  doc.getElementById("onb-models-go").click();
  assert.equal(doc.getElementById("view-models").classList.contains("active"), true);
  assert.equal(doc.activeElement, doc.getElementById("disc-query"));
});

test("Setup: the Models page empty state links into Setup on a deliberate click", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([
    ["/api/models", { models: [] }],
  ]) });
  const doc = window.document;
  doc.getElementById("nav-models").click();
  await window.refreshModelsPage();
  const box = doc.getElementById("models-table");
  assert.match(box.textContent, /No models yet/);
  // Never opens on its own - only present, and only acts, on a click.
  assert.equal(doc.getElementById("view-setup").classList.contains("active"), false);
  const link = box.querySelector("a");
  assert.ok(link, "the empty state must carry a link into Setup");
  link.click();
  assert.equal(doc.getElementById("view-setup").classList.contains("active"), true);
});

test("Setup: Go to chat navigates to the chat view", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  const doc = showSetup(window);
  await flush();
  doc.getElementById("onb-go-chat").click();
  assert.equal(doc.getElementById("view-chat").classList.contains("active"), true);
});
