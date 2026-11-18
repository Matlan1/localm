// SPDX-License-Identifier: AGPL-3.0-or-later
// Chat composer fixes: Enter on a slash-menu pick only inserts the command,
// the stop square is labelled Stop, a running media generation keeps its
// bubble and its outcome in the conversation, a conversation title can be
// renamed by double-click, Copy image works for an attached picture, the
// export labels each reply with the model that wrote it, and the parameters
// drawer offers Knowledge only with the rag plugin.

import { test as nodeTest } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

const test = (name, fn) => nodeTest(name, { timeout: 15000 }, fn);
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function until(cond, what) {
  for (let i = 0; i < 200; i++) {
    if (cond()) return;
    await wait(10);
  }
  assert.fail(`timed out waiting for ${what}`);
}

function typeInto(win, text) {
  const ta = win.document.getElementById("chat-input");
  ta.value = text;
  ta.dispatchEvent(new win.Event("input", { bubbles: true }));
  return ta;
}

test("Enter on a slash-menu pick that takes an argument inserts the command and sends nothing", () => {
  const { window: win } = loadApp();
  const ta = typeInto(win, "/ren");
  ta.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true }));
  assert.equal(ta.value, "/rename ");
  assert.equal(win.document.getElementById("toast").textContent, "", "no usage toast");
});

test("composerEnterToSend leaves an Enter another handler already used", () => {
  const { window: win } = loadApp();
  const ta = win.document.getElementById("chat-input");
  let sent = 0;
  const ev = (defaultPrevented) => ({
    key: "Enter", isComposing: false, shiftKey: false, defaultPrevented, target: ta,
    preventDefault() {},
  });
  win.composerEnterToSend(ev(true), () => { sent++; });
  assert.equal(sent, 0);
  win.composerEnterToSend(ev(false), () => { sent++; });
  assert.equal(sent, 1);
});

test("the send button reads Stop while a reply runs and Send after", async () => {
  const { window: win } = loadApp();
  const btn = win.document.getElementById("chat-send");
  assert.equal(btn.title, "Send");
  let during = null;
  await win.runChatTurn(async () => { during = btn.title; return {}; });
  assert.equal(during, "Stop reply");
  assert.equal(btn.title, "Send");
});

test("dataUrlToBlob decodes base64 and percent-encoded data URLs without a fetch", async () => {
  const { window: win } = loadApp();
  const png = win.dataUrlToBlob("data:image/png;base64,AAEC");
  assert.equal(png.type, "image/png");
  assert.equal(png.size, 3);
  const txt = win.dataUrlToBlob("data:text/plain;charset=utf-8,hi%20x");
  assert.equal(txt.type, "text/plain");
  assert.equal(txt.size, 4);
  assert.equal(win.dataUrlToBlob("https://example.com/a.png"), null);
});

test("copyImageSrc copies a data: picture without fetching it", async () => {
  const fetched = [];
  const { window: win } = loadApp({ fetchImpl: async (url) => {
    fetched.push(String(url));
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  } });
  const written = [];
  win.ClipboardItem = function (obj) { this.obj = obj; };
  Object.defineProperty(win.navigator, "clipboard", { configurable: true,
    value: { write: async (items) => { written.push(items); } } });
  assert.equal(await win.copyImageSrc("data:image/png;base64,AAEC"), true);
  assert.equal(written.length, 1);
  assert.deepEqual(fetched.filter((u) => u.startsWith("data:")), [], "the data URL was decoded locally");
});

test("a double-click on the open conversation's title opens the rename field", async () => {
  const { window: win } = loadApp();
  const conv = { id: "c1", title: "Rivers", messages: [], updated_at: 1 };
  runScript(win, `chat.conversations = [${JSON.stringify(conv)}]; chat.activeId = "c1";
    window.__item = buildConvItem(chat.conversations[0], null);`);
  const item = win.__item;
  win.document.body.appendChild(item);
  const title = item.querySelector(".title");
  item.click();
  await wait(10);
  assert.equal(item.querySelector(".title"), title, "a click on the open chat keeps its title element");
  title.dispatchEvent(new win.MouseEvent("dblclick", { bubbles: true }));
  assert.ok(item.querySelector("input"), "the rename field is shown");
});

test("export labels each reply with the model that wrote it", () => {
  const { window: win } = loadAppWithPages();
  let captured = null;
  const OrigBlob = win.Blob;
  win.Blob = function (parts, opts) { captured = (parts || []).join(""); return new OrigBlob(parts, opts); };
  win.URL.createObjectURL = () => "blob:x";
  win.URL.revokeObjectURL = () => {};
  const origCreate = win.document.createElement.bind(win.document);
  win.document.createElement = (tag) => {
    const el = origCreate(tag);
    if (tag === "a") el.click = () => {};
    return el;
  };
  const conv = { id: "c1", title: "Two models", messages: [
    { role: "user", content: "q1", id: "m1" },
    { role: "assistant", content: "a1", model: "small-0.5B", id: "m2" },
    { role: "user", content: "q2", id: "m3" },
    { role: "assistant", content: "a2", model: "big-7B", id: "m4" },
  ] };
  runScript(win, `chat.conversations = [${JSON.stringify(conv)}]; chat.activeId = "c1";
    modelCache.active = "big-7B";`);
  win.exportConversation();
  assert.match(captured, /\*\*small-0\.5B:\*\*\n\na1/);
  assert.match(captured, /\*\*big-7B:\*\*\n\na2/);
});

test("the Knowledge selector shows in the drawer only with the rag plugin active", async () => {
  const { window: win } = loadApp();
  const row = win.document.getElementById("p-kb-row");
  const drawer = win.document.getElementById("params");
  assert.equal(row.hidden, true, "hidden before any plugin answer");
  runScript(win, `pluginState = [{ name: "rag", active: true }];`);
  drawer.classList.toggle("open");
  await wait(10);
  assert.equal(row.hidden, false);
  runScript(win, `pluginState = [];`);
  drawer.classList.toggle("open");
  await wait(10);
  assert.equal(row.hidden, true);
});

test("the pin label names the selected model, never nothing", () => {
  const { window: win } = loadApp();
  win.syncPinModelToggle(null);
  assert.equal(win.document.getElementById("p-pin-model-name").textContent, "the selected model");
});

function pendingJobFetch(release) {
  return async (url, opts = {}) => {
    const u = String(url);
    const ok = (body) => ({ ok: true, status: 200, json: async () => body, text: async () => "" });
    if (u === "/api/media/video/preflight") return ok({ status: "verified", missing: [], warning: null });
    if (u === "/api/video" && opts.method === "POST") return ok({ job_id: "job-1" });
    if (u === "/api/jobs/job-1/events") {
      let sent = false;
      return { ok: true, status: 200, body: { getReader: () => ({ async read() {
        await release.promise;
        if (sent) return { done: true, value: undefined };
        sent = true;
        return { done: false,
                 value: new TextEncoder().encode(`data: ${JSON.stringify({ type: "end", status: "failed" })}\n\n`) };
      } }) } };
    }
    return ok({});
  };
}

test("a running /generate-video keeps its bubble through a re-render, and a failed one stays in the chat", async () => {
  let resolve;
  const release = { promise: new Promise((r) => { resolve = r; }) };
  const { window: win } = loadApp({ fetchImpl: pendingJobFetch(release) });
  const run = win.runVideoInChat("a cat walking");
  const box = win.document.getElementById("chat-messages");
  await until(() => /Generating/.test(box.textContent), "the live bubble");
  win.renderChat();
  assert.match(box.textContent, /Generating/, "a re-render keeps the live bubble");
  resolve();
  await run;
  const conv = win.currentConv();
  const last = conv.messages[conv.messages.length - 1];
  assert.equal(last.role, "assistant");
  assert.match(String(last.content), /Not generated|failed|status/i);
  assert.match(box.textContent, new RegExp(String(last.content).slice(0, 12)));
  win.renderChat();
  assert.doesNotMatch(box.textContent, /Generating/, "the bubble is gone once the job ended");
});
