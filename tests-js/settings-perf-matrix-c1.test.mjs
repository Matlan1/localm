// SPDX-License-Identifier: AGPL-3.0-or-later
// settings-perf.js outside the chat turn: the conversation export, the text
// read aloud, the memory dialog, the voice download and microphone, web
// lookup errors, the App update block and the startup update check, and the
// Live tuning labels.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const jsonResp = (obj, status = 200) => ({
  ok: status < 400, status, statusText: status >= 400 ? "Error" : "", headers: { get: () => null },
  json: async () => obj, text: async () => JSON.stringify(obj),
});

function appWith(routes = {}) {
  const calls = [];
  const impl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    for (const [prefix, fn] of Object.entries(routes)) {
      if (u.startsWith(prefix)) return fn(opts);
    }
    return jsonResp({});
  };
  const { window } = loadApp({ fetchImpl: impl });
  return { window, calls };
}

function captureExport(window) {
  let captured = null;
  const OrigBlob = window.Blob;
  window.Blob = function (parts, opts) {
    captured = (parts || []).join("");
    return new OrigBlob(parts, opts);
  };
  let name = null;
  const origCreate = window.document.createElement.bind(window.document);
  window.document.createElement = (tag) => {
    const el = origCreate(tag);
    if (tag === "a") el.click = () => { name = el.download; };
    return el;
  };
  return { text: () => captured, name: () => name };
}

function setActiveConv(window, conv) {
  runScript(window,
    `chat.conversations = [${JSON.stringify(conv)}]; chat.activeId = ${JSON.stringify(conv.id)};`);
}

const toastText = (window) => window.document.getElementById("toast").textContent;

// --------------------------------------------------------------------------
//  Export
// --------------------------------------------------------------------------

test("export: each reply is labelled with the model that wrote it, not the model loaded now", () => {
  const { window } = appWith();
  const out = captureExport(window);
  runScript(window, "modelCache.active = 'model-now';");
  setActiveConv(window, { id: "c1", title: "Switch", messages: [
    { role: "user", content: "q1" },
    { role: "assistant", content: "a1", model: "model-early" },
    { role: "user", content: "q2" },
    { role: "assistant", content: "a2", model: "model-now" },
    { role: "assistant", content: "a3" },
  ] });
  window.exportConversation();
  assert.match(out.text(), /\*\*model-early:\*\*\n\na1/);
  assert.match(out.text(), /\*\*model-now:\*\*\n\na2/);
  assert.match(out.text(), /\*\*Model:\*\*\n\na3/, "a reply with no recorded model is not credited to the loaded one");
});

test("export: reasoning is set apart as a quote, never written as raw think tags", () => {
  const { window } = appWith();
  const out = captureExport(window);
  setActiveConv(window, { id: "c1", title: "Think", messages: [
    { role: "user", content: "2+2?" },
    { role: "assistant", content: "<think>\nadd two\nand two\n</think>\nIt is 4.", model: "m" },
  ] });
  window.exportConversation();
  const text = out.text();
  assert.doesNotMatch(text, /<\/?think>/);
  assert.match(text, /\*\*m:\*\*\n\n\*Reasoning:\*\n\n> add two\n> and two\n\nIt is 4\./);
});

test("export: the other branches of the chat are written after the shown one", () => {
  const { window } = appWith();
  const out = captureExport(window);
  setActiveConv(window, { id: "c1", title: "Branches", messages: [
    { id: "u1", role: "user", content: "question" },
    { id: "r2", role: "assistant", content: "second reply", model: "m" },
  ], branches: [{ parent: "u1", current: 1, tails: [
    [{ id: "r1", role: "assistant", content: "first reply", model: "m" }], null] }] });
  window.exportConversation();
  const text = out.text();
  assert.match(text, /second reply/);
  assert.match(text, /## Other branches \(1\)[\s\S]*### Branch 1[\s\S]*first reply/);
});

test("export: the file name keeps letters of any script", () => {
  const { window } = appWith();
  assert.equal(window.exportFileStem("Über uns"), "Über_uns");
  assert.equal(window.exportFileStem("日本語のメモ"), "日本語のメモ");
  assert.equal(window.exportFileStem("a/b:c*?"), "abc");
  assert.equal(window.exportFileStem("???"), "conversation");
  assert.equal(window.exportFileStem("con"), "con_chat");
  const out = captureExport(window);
  setActiveConv(window, { id: "c1", title: "Über uns", messages: [{ role: "user", content: "x" }] });
  window.exportConversation();
  assert.equal(out.name(), "Über_uns.md");
});

// --------------------------------------------------------------------------
//  Read aloud
// --------------------------------------------------------------------------

test("speechText drops Markdown syntax and keeps snake_case and parentheses", () => {
  const { window } = appWith();
  const said = window.speechText(
    "# Title\nUse `snake_case` (see [the docs](http://x)) and **bold** or _it_.\n> quoted\n- item\n2 * 3 = f(a_b)");
  assert.equal(said, "Title\nUse snake_case (see the docs) and bold or it.\nquoted\nitem\n2 * 3 = f(a_b)");
});

test("speak reads the cleaned text", () => {
  const { window } = appWith();
  window.speak("call my_func(1) **now**");
  assert.equal(window.__spoken[0].text, "call my_func(1) now");
});

// --------------------------------------------------------------------------
//  Memory dialog
// --------------------------------------------------------------------------

async function memoryDialogText(status, body = {}) {
  const { window } = appWith({ "/api/memory": () => jsonResp(body, status) });
  await window.openMemoryModal();
  return { window, text: window.document.getElementById("modal").textContent };
}

test("memory dialog: a key without the memory scope is told so, not that privacy mode blocks writes", async () => {
  const { window, text } = await memoryDialogText(403, { detail: "missing scope: memory" });
  assert.ok(text.includes(window.t("memory.modal.noScope")));
  assert.ok(!text.includes(window.t("memory.modal.readOnly")));
});

test("memory dialog: a server error is shown as one, with its reason", async () => {
  const { window, text } = await memoryDialogText(500, { detail: "the facts file is damaged" });
  assert.match(text, /the facts file is damaged/);
  assert.ok(!text.includes(window.t("memory.modal.readOnly")));
});

test("memory dialog: a missing memory plugin is named", async () => {
  const { window, text } = await memoryDialogText(404, { detail: "Not Found" });
  assert.ok(text.includes(window.t("memory.modal.notActive")));
});

test("memory dialog: a read that works and is not writable still says read-only", async () => {
  const { window, text } = await memoryDialogText(200, { text: "- a fact", writable: false });
  assert.ok(text.includes(window.t("memory.modal.readOnly")));
});

test("Synthesize shows the run's eviction warning", async () => {
  const warning = "1 user-typed memory record(s) were evicted at the 256-record cap";
  const { window } = appWith({
    "/api/memory/consolidate": () => jsonResp({ status: "ok", added: 1, warning }),
    "/api/memory": () => jsonResp({ text: "- x", writable: true }),
  });
  const status = window.document.createElement("div");
  await window.synthesizeMemoryNow(status);
  assert.match(status.textContent, /evicted at the 256-record cap/);
  assert.match(toastText(window), /evicted/);
  assert.match(window.document.getElementById("toast").className, /error/);
});

test("a refused embedding model download names the reason in the toast", async () => {
  const { window } = appWith({
    "/api/rag/embedding/download": () => jsonResp({ job_id: "j1" }),
    "/api/jobs/": () => jsonResp({}),
  });
  window.readSSE = async (_r, onData) => {
    onData(JSON.stringify({ type: "line", text: "Downloading embedding model 'e' (one-time)..." }));
    onData(JSON.stringify({ type: "line", text: "error: huggingface.co is not on the allowed domain list" }));
    onData(JSON.stringify({ type: "end", status: "failed" }));
  };
  const btn = window.document.createElement("button");
  const ok = await window.downloadMemoryEmbedder(btn);
  assert.equal(ok, false);
  assert.match(toastText(window), /allowed domain list/);
  assert.doesNotMatch(btn.textContent, /^error:/, "the error line is not left as the button label");
});

// --------------------------------------------------------------------------
//  Voice
// --------------------------------------------------------------------------

test("downloadJobOutcome tells done, cancelled and failed apart, and never uses progress as a reason", () => {
  const { window } = appWith();
  const outcome = (...args) => JSON.parse(JSON.stringify(window.downloadJobOutcome(...args)));
  assert.deepEqual(outcome({ status: "done" }, "Ready", "x"), { state: "done" });
  assert.deepEqual(outcome({ status: "cancelled" }, "Downloading ... (one-time)...", "x"),
    { state: "cancelled" });
  assert.deepEqual(outcome({ status: "failed" }, "Downloading ... (one-time)...", "fallback"),
    { state: "failed", reason: "fallback" });
  assert.deepEqual(outcome({ status: "failed" }, "error: disk full", "fallback"),
    { state: "failed", reason: "disk full" });
  assert.deepEqual(outcome({ status: "done" }, "error: checksum", "fallback"),
    { state: "failed", reason: "checksum" });
});

test("a cancelled speech model download is reported as cancelled, not failed", async () => {
  const { window } = appWith({
    "/api/voice/model/download": () => jsonResp({ job_id: "j1" }),
    "/api/jobs/": () => jsonResp({}),
  });
  window.confirm = () => true;
  window.readSSE = async (_r, onData) => {
    onData(JSON.stringify({ type: "line", text: "Downloading Whisper model 'base' (one-time)..." }));
    onData(JSON.stringify({ type: "end", status: "cancelled" }));
  };
  await window.downloadVoiceModel();
  assert.equal(toastText(window), window.t("voice.downloadCancelledToast"));
});

test("the microphone on a plain-http address names the address, not the browser", async () => {
  const { window } = appWith();
  runScript(window, "voice.available = true; voice.modelCached = true;");
  Object.defineProperty(window, "isSecureContext", { value: false, configurable: true });
  await window.toggleMic();
  assert.equal(toastText(window), window.t("voice.needsSecureAddress"));
});

test("a denied microphone does not count as consent to the first-use download", async () => {
  const { window } = appWith();
  runScript(window, "voice.available = true; voice.modelCached = false;");
  let asked = 0;
  window.confirm = () => { asked++; return true; };
  window.MediaRecorder = class {};
  Object.defineProperty(window.navigator, "mediaDevices", { configurable: true,
    value: { getUserMedia: async () => { throw new Error("Permission denied"); } } });
  await window.toggleMic();
  runScript(window, "window.__cached = voice.modelCached;");
  assert.equal(window.__cached, false);
  await window.toggleMic();
  assert.equal(asked, 2, "the download is asked about again on the next click");
});

// --------------------------------------------------------------------------
//  Web lookup errors
// --------------------------------------------------------------------------

test("a failed search carries the bundle's own sentence with no stacked prefix", async () => {
  const sentence = "The search backend set in Settings > Server & network failed: timed out.";
  const { window } = appWith({
    "/api/web/retrieve": () => jsonResp({ query: "x", search_status: "failed", search_error: sentence,
                                           grounding: "failed", sources: [], chunks: [] }),
  });
  await assert.rejects(window.requestWebTool({ name: "web_search", args: { query: "x" } }),
    (e) => e.message === sentence);
});

// --------------------------------------------------------------------------
//  Updates
// --------------------------------------------------------------------------

test("with no update server the App update block is replaced by a note saying why", () => {
  const { window } = appWith();
  const doc = window.document;
  window.syncAppUpdateBlock(false);
  assert.equal(doc.getElementById("app-update-block").hidden, true);
  const note = doc.getElementById("app-update-off-note");
  assert.ok(note && !note.hidden);
  assert.equal(note.textContent, window.t("settings.update.notConfiguredNote"));
  window.syncAppUpdateBlock(true);
  assert.equal(doc.getElementById("app-update-block").hidden, false);
  assert.equal(note.hidden, true);
});

test("the startup update check says so outside Settings when a newer build is offered", async () => {
  const { window } = appWith();
  window.localStorage.removeItem("localm.updateCheckAt");
  window.__localmUpdateCheck = async () => { window.document.getElementById("update-apply").hidden = false; };
  await window.maybeAutoUpdateCheck();
  assert.equal(toastText(window), window.t("settings.update.foundToast"));
});

test("the startup update check stays quiet when nothing newer is offered", async () => {
  const { window } = appWith();
  window.localStorage.removeItem("localm.updateCheckAt");
  window.__localmUpdateCheck = async () => {};
  await window.maybeAutoUpdateCheck();
  assert.notEqual(toastText(window), window.t("settings.update.foundToast"));
});

// --------------------------------------------------------------------------
//  Live tuning labels
// --------------------------------------------------------------------------

test("the Live tuning controls do not claim to show the running model's values", () => {
  const { window } = appWith();
  for (const key of ["settings.perf.ctxLabel", "settings.perf.gpuLayersLabel"]) {
    assert.doesNotMatch(window.t(key), /running model/);
    assert.match(window.t(key), /saved/);
  }
});
