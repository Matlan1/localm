// SPDX-License-Identifier: AGPL-3.0-or-later
// The one-time, non-persistent network-policy override in the GUI. The server
// decides who may authorize the download (can_download); the client offers the
// action - a greyed mic, the Knowledge panel's download button - and POSTs only
// after an explicit confirm.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));

/* ---- voice: mic click offers the one-time download when permitted ------- */

function voiceFetch(statusBody, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    if (u === "/api/voice/status") {
      return { ok: true, status: 200, json: async () => statusBody };
    }
    if (u === "/api/voice/model/download") {
      return { ok: true, status: 200, json: async () => ({ job_id: "j1" }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

const BLOCKED = {
  available: false,
  reason: "The Whisper 'base' speech model is not downloaded yet, and "
        + "net_mode=ask does not download automatically.",
  model_cached: false, model: "base", can_download: true,
};

test("greyed mic + can_download: click confirms then POSTs the download", async () => {
  const calls = [];
  const { window: win } = loadApp({ fetchImpl: voiceFetch(BLOCKED, calls) });
  await tick(); await tick(); await tick();   // boot's auto refreshVoiceStatus
  runScript(win, `
    globalThis.__confirms = [];
    window.confirm = (msg) => { globalThis.__confirms.push(msg); return true; };
    streamJob = () => Promise.resolve({ status: "done" });
  `);
  const mic = win.document.getElementById("chat-mic");
  assert.ok(mic.classList.contains("unavailable"), "precondition: mic greyed");
  assert.match(mic.title, /download it now/i, "tooltip advertises the action");

  await win.toggleMic();
  await tick(); await tick();
  assert.equal(win.__confirms.length, 1, "exactly one consent dialog");
  assert.match(win.__confirms[0], /changes no settings/i,
    "the consent text states nothing is persisted");
  const posts = calls.filter((c) => c.url === "/api/voice/model/download");
  assert.equal(posts.length, 1, "one POST to the download route");
  assert.equal((posts[0].opts || {}).method, "POST");
});

test("declining the consent dialog sends nothing", async () => {
  const calls = [];
  const { window: win } = loadApp({ fetchImpl: voiceFetch(BLOCKED, calls) });
  await tick(); await tick(); await tick();
  runScript(win, `window.confirm = () => false;`);
  await win.toggleMic();
  await tick();
  assert.equal(calls.filter((c) => c.url === "/api/voice/model/download").length, 0,
    "no download without the explicit yes");
});

test("without can_download the click only reports the reason", async () => {
  const calls = [];
  const body = { ...BLOCKED, can_download: false };
  const { window: win } = loadApp({ fetchImpl: voiceFetch(body, calls) });
  await tick(); await tick(); await tick();
  runScript(win, `
    window.confirm = () => { throw new Error("consent dialog must not open"); };
  `);
  await win.toggleMic();
  await tick();
  assert.equal(calls.filter((c) => c.url === "/api/voice/model/download").length, 0,
    "a caller the server did not clear gets no download attempt");
  const mic = win.document.getElementById("chat-mic");
  assert.doesNotMatch(mic.title, /download it now/i,
    "the tooltip does not advertise an action the caller cannot take");
});

/* ---- Knowledge panel: the Download-now button ---------------------------- */

function kbFetch(embeddingBody, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    if (u === "/api/rag/embedding" && !(opts && opts.method)) {
      return { ok: true, status: 200, json: async () => embeddingBody };
    }
    if (u === "/api/rag/embedding/download") {
      return { ok: true, status: 200, json: async () => ({ job_id: "j2" }) };
    }
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], collections: [] }) };
  };
}

const EMB_BLOCKED = {
  model: "bge-small-en-v1.5", default: "bge-small-en-v1.5",
  internal: ["bge-small-en-v1.5", "nomic-embed-text-v1.5"],
  installed: false, dim: null, error: null, gpu_fallback_reason: null,
  status: "not_installed", can_download: true,
};

test("can_download shows the Download-now button, named after the CONFIGURED model", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: kbFetch(EMB_BLOCKED, calls) });
  await win.refreshEmbeddingPanel();
  const btn = win.document.getElementById("kb-embed-download");
  assert.ok(btn, "the button exists in the panel");
  assert.notEqual(btn.style.display, "none", "visible when the server offers it");
  assert.match(btn.textContent, /bge-small-en-v1\.5/,
    "labels the configured model, not the dropdown selection");
});

test("no can_download keeps the button hidden", async () => {
  const calls = [];
  const body = { ...EMB_BLOCKED, can_download: false };
  const { window: win } = loadAppWithPages({ fetchImpl: kbFetch(body, calls) });
  await win.refreshEmbeddingPanel();
  const btn = win.document.getElementById("kb-embed-download");
  assert.equal(btn.style.display, "none");
});

test("clicking Download-now POSTs the download route and streams the job", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: kbFetch(EMB_BLOCKED, calls) });
  runScript(win, `streamJob = () => Promise.resolve({ status: "done" });`);
  await win.refreshEmbeddingPanel();
  win.document.getElementById("kb-embed-download").click();
  await tick(); await tick(); await tick();
  const posts = calls.filter((c) => c.url === "/api/rag/embedding/download");
  assert.equal(posts.length, 1, "one POST to the download route");
  assert.equal((posts[0].opts || {}).method, "POST");
  // nothing goes to the model-switch route
  assert.equal(calls.filter((c) => c.url === "/api/rag/embedding"
                                   && c.opts && c.opts.method === "POST").length, 0,
    "the download action must never touch the model-switch route");
});

/* ---- Memory modal: the same Download-now hint, for a Memory-only user ---- */

function memFetch(memoryBody, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    if (u === "/api/memory" && !(opts && opts.method)) {
      return { ok: true, status: 200, json: async () => memoryBody };
    }
    if (u === "/api/rag/embedding/download") {
      return { ok: true, status: 200, json: async () => ({ job_id: "j3" }) };
    }
    return { ok: true, status: 200, text: async () => "", json: async () => ({}) };
  };
}

const MEM_BLOCKED = {
  text: "", writable: true, items: [], corrections: [],
  can_download_embedder: true, embedder_model: "bge-small-en-v1.5",
};

test("memory modal offers a Download-now hint when can_download_embedder is true", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: memFetch(MEM_BLOCKED, calls) });
  await win.refreshMemory();
  win.openMemoryModal();
  const body = win.document.getElementById("modal-body");
  assert.match(body.textContent, /Semantic recall is off/,
    "explains why recall is degraded");
  const btn = win.document.getElementById("mem-embed-download");
  assert.ok(btn, "the download button exists");
  assert.match(btn.textContent, /bge-small-en-v1\.5/, "labels the configured model");
});

test("no can_download_embedder renders neither hint nor button", async () => {
  const calls = [];
  const body = { ...MEM_BLOCKED, can_download_embedder: false, embedder_model: null };
  const { window: win } = loadAppWithPages({ fetchImpl: memFetch(body, calls) });
  await win.refreshMemory();
  win.openMemoryModal();
  assert.equal(win.document.getElementById("mem-embed-download"), null);
  assert.doesNotMatch(
    win.document.getElementById("modal-body").textContent, /Semantic recall is off/);
});

test("clicking Download-now POSTs the SAME route the Knowledge page uses, and clears on success",
  async () => {
    const calls = [];
    const { window: win } = loadAppWithPages({ fetchImpl: memFetch(MEM_BLOCKED, calls) });
    runScript(win, `streamJob = () => Promise.resolve({ status: "done" });`);
    await win.refreshMemory();
    win.openMemoryModal();
    win.document.getElementById("mem-embed-download").click();
    await tick(); await tick(); await tick();
    const posts = calls.filter((c) => c.url === "/api/rag/embedding/download");
    assert.equal(posts.length, 1, "one POST to the SAME download route Knowledge uses");
    assert.equal((posts[0].opts || {}).method, "POST");
    assert.equal(win.document.getElementById("mem-embed-download"), null,
      "the button is removed once the download completes");
    assert.doesNotMatch(
      win.document.getElementById("modal-body").textContent, /Semantic recall is off/,
      "the hint text is removed too");
  });
