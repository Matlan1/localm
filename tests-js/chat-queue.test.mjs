// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function abortError() {
  const e = new Error("The operation was aborted");
  e.name = "AbortError";
  return e;
}

/** Resolves when *gate* resolves, rejects with an AbortError when *signal*
 *  aborts first (the way a real fetch or stream read does). */
function gated(gate, signal) {
  return new Promise((resolve, reject) => {
    if (signal && signal.aborted) { reject(abortError()); return; }
    if (signal) signal.addEventListener("abort", () => reject(abortError()), { once: true });
    gate.then(resolve);
  });
}

function makeGate() {
  let open;
  const promise = new Promise((r) => { open = r; });
  return { promise, open };
}

/** A jsdom app whose fetch records every call. Streamed completions reply
 *  "reply" and, while `holdStream` is set, wait for `streamGate`. The
 *  compaction request and the knowledge query wait for their own gates when
 *  set. Every held request honours its abort signal. */
function setupApp(options = {}) {
  const fetchCalls = [];
  const ctl = {
    holdStream: false,
    streamGate: makeGate(),
    compactGate: options.compactGate || null,
    kbGate: options.kbGate || null,
  };
  const impl = async (url, opts = {}) => {
    const u = String(url);
    let body;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = null; }
    fetchCalls.push({ url: u, opts, body });
    if (u === "/v1/chat/completions" && body && body.stream === false) {
      if (ctl.compactGate) await gated(ctl.compactGate.promise, opts.signal);
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ choices: [{ message: { content: "summary" }, finish_reason: "stop" }] }) };
    }
    if (u === "/v1/chat/completions") {
      return { ok: true, status: 200, body: null, headers: { get: () => null },
        json: async () => ({}), __signal: opts.signal,
        __hold: ctl.holdStream ? ctl.streamGate.promise : null };
    }
    if (u.startsWith("/api/rag/collections/")) {
      if (ctl.kbGate) await gated(ctl.kbGate.promise, opts.signal);
      return { ok: true, status: 200,
        json: async () => ({ hits: [{ source: "doc.txt", pos: 12, text: "excerpt text" }] }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl: impl });
  if (!options.realCompaction) window.maybeCompactConversation = async () => {};
  window.readSSE = async (r, onData) => {
    if (r.__hold) await gated(r.__hold, r.__signal);
    onData(JSON.stringify({ choices: [{ delta: { content: "reply" } }] }));
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }));
  };
  runScript(window, "modelCache.active = 'test-model';");
  return { window, fetchCalls, ctl };
}

function activateConv(window, conv) {
  window.__testConv = conv;
  runScript(window, "chat.conversations = [window.__testConv]; chat.activeId = window.__testConv.id;");
}

function getChatQueue(window) {
  runScript(window, "window.__testQueue = chat.queue;");
  return window.__testQueue;
}

function isBusy(window) {
  runScript(window, "window.__testBusy = chatBusy();");
  return window.__testBusy;
}

function streamedCalls(fetchCalls) {
  return fetchCalls.filter((c) => c.url === "/v1/chat/completions" && c.body && c.body.stream === true);
}

/** Text of the last user message in a streamed completion request. */
function lastUserText(call) {
  const users = call.body.messages.filter((m) => m.role === "user");
  const last = users[users.length - 1];
  return typeof last.content === "string" ? last.content : "";
}

async function waitFor(cond, what, ms = 3000) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (cond()) return;
    await new Promise((r) => setTimeout(r, 5));
  }
  assert.fail("timed out waiting for " + what);
}

/** Awaits *promise*, failing the test when it has not settled within *ms*. */
async function within(promise, what, ms = 3000) {
  let timer;
  const timeout = new Promise((_resolve, reject) => {
    timer = setTimeout(() => reject(new Error("timed out waiting for " + what)), ms);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

/** Waits until no turn is running and nothing queued for the active
 *  conversation is left to dispatch. */
async function settle(window) {
  await waitFor(() => !isBusy(window), "the chat to go idle");
  await new Promise((r) => setTimeout(r, 20));
  await waitFor(() => !isBusy(window), "the chat to stay idle");
}

async function send(window, text) {
  runScript(window, "modelCache.active = 'test-model';");
  window.document.getElementById("chat-input").value = text;
  return window.sendChat();
}

function clickStop(window) {
  window.document.getElementById("chat-send").click();
}

/** Action labels (title or text) on the transcript row at *index*. */
function rowActions(window, index) {
  const rows = [...window.document.querySelectorAll("#chat-messages .msg-row:not(.queued)")];
  const row = index < 0 ? rows[rows.length + index] : rows[index];
  return [...row.querySelectorAll(".msg-meta button.action")]
    .map((b) => b.title || b.textContent.trim());
}

test("sendChat queues message when chat is busy and displays queued indicator", async () => {
  const { window, fetchCalls } = setupApp();
  const conv = { id: "c1", title: "t", messages: [] };
  activateConv(window, conv);
  runScript(window, "chat.busy = true;");

  const doc = window.document;
  const input = doc.getElementById("chat-input");
  input.value = "follow up question";

  await window.sendChat();

  assert.equal(streamedCalls(fetchCalls).length, 0, "must not fire completion immediately while busy");

  const queue = getChatQueue(window);
  assert.equal(queue.length, 1, "message must be queued in chat.queue");
  assert.equal(queue[0].text, "follow up question");
  assert.equal(input.value, "", "input must be cleared");

  const queuedRow = doc.querySelector(".msg-row.queued");
  assert.ok(queuedRow, "queued message row must be rendered in chat messages");
  assert.ok(queuedRow.textContent.includes("follow up question"));
});

test("composerEnterToSend does not block enter key while streaming", () => {
  const { window } = setupApp();
  runScript(window, "chat.abort = new AbortController();");

  let sent = false;
  const fakeEvent = {
    key: "Enter",
    shiftKey: false,
    isComposing: false,
    target: window.document.getElementById("chat-input"),
    preventDefault: () => {},
  };

  window.composerEnterToSend(fakeEvent, () => {
    sent = true;
  });

  assert.equal(sent, true, "send function must be called on Enter even when abort is active");
});

test("a message queued during a reply is sent by itself when the reply finishes", async () => {
  const { window, fetchCalls, ctl } = setupApp();
  activateConv(window, { id: "c1", title: "t", messages: [] });
  ctl.holdStream = true;

  const first = send(window, "first");
  await waitFor(() => streamedCalls(fetchCalls).length === 1, "the first completion");
  await send(window, "second");
  assert.equal(getChatQueue(window).length, 1, "second is queued while first streams");

  ctl.holdStream = false;
  ctl.streamGate.open();
  await within(first, "the first turn to return");
  await waitFor(() => streamedCalls(fetchCalls).length === 2, "the queued completion");
  assert.equal(lastUserText(streamedCalls(fetchCalls)[1]), "second");
  await settle(window);
  assert.equal(getChatQueue(window).length, 0, "queue is empty after the drain");
});

test("canceling a queued message removes it from queue and DOM", () => {
  const { window } = setupApp();
  const conv = { id: "c1", title: "t", messages: [] };
  activateConv(window, conv);
  runScript(window, `chat.queue = [{ text: "remove me", attachments: [], docs: [], convId: "${conv.id}" }];`);
  window.renderQueuedIndicator();

  const doc = window.document;
  const cancelBtn = doc.querySelector(".msg-row.queued .msg-meta button.action");
  assert.ok(cancelBtn, "cancel button must exist in queued row meta");

  cancelBtn.click();

  const queue = getChatQueue(window);
  assert.equal(queue.length, 0, "queue must be empty after cancel click");
  assert.equal(doc.querySelectorAll(".msg-row.queued").length, 0, "queued row must be removed");
});

test("a send with a knowledge base shows the searching indicator and passes the turn's abort signal", async () => {
  const kbGate = makeGate();
  const { window, fetchCalls } = setupApp({ kbGate });
  const doc = window.document;
  const sel = doc.getElementById("p-kb");
  const opt = doc.createElement("option");
  opt.value = "test-kb";
  sel.appendChild(opt);
  sel.value = "test-kb";
  activateConv(window, { id: "c1", title: "t", messages: [] });

  const turn = send(window, "test query");
  const isKb = (c) => c.url === "/api/rag/collections/test-kb/query";
  await waitFor(() => fetchCalls.some(isKb), "the knowledge query");

  const indicator = doc.querySelector("#chat-messages .msg-status-indicator");
  assert.ok(indicator, "a status indicator is shown during knowledge retrieval");
  assert.ok(indicator.textContent.includes(window.t("chat.status.searchingKnowledge")),
    "the indicator says the knowledge base is being searched");

  const kbSignal = fetchCalls.find(isKb).opts.signal;
  runScript(window, "window.__testAbortSignal = chat.abort && chat.abort.signal;");
  assert.ok(kbSignal, "the knowledge query carries an abort signal");
  assert.equal(kbSignal, window.__testAbortSignal, "the signal is the turn's chat.abort signal");

  clickStop(window);
  assert.equal(kbSignal.aborted, true, "Stop aborts the knowledge query");
  await within(turn, "the turn to return");
  assert.equal(streamedCalls(fetchCalls).length, 0, "no completion after a stop during retrieval");
});

test("Stop during compaction cancels the turn before any reply streams", async () => {
  const compactGate = makeGate();
  const { window, fetchCalls } = setupApp({ compactGate, realCompaction: true });
  const messages = [];
  for (let i = 0; i < 10; i++) {
    messages.push({ role: i % 2 === 0 ? "user" : "assistant", content: ("old-" + i).padEnd(120, ".") });
  }
  activateConv(window, { id: "c1", title: "t", messages });
  runScript(window, "chat.ctxMax = 200;");

  const turn = send(window, "next question");
  const isCompact = (c) => c.url === "/v1/chat/completions" && c.body && c.body.stream === false;
  await waitFor(() => fetchCalls.some(isCompact), "the compaction request");

  clickStop(window);
  compactGate.open();
  await within(turn, "the turn to return");
  await settle(window);

  assert.equal(streamedCalls(fetchCalls).length, 0, "no streamed completion after Stop");
  const conv = window.__testConv;
  const last = conv.messages[conv.messages.length - 1];
  assert.equal(last.role, "user", "no assistant reply was saved");
  assert.equal(last.content, "next question");
});

test("Stop during a reply keeps the queued message and does not send it", async () => {
  const { window, fetchCalls, ctl } = setupApp();
  activateConv(window, { id: "c1", title: "t", messages: [] });
  ctl.holdStream = true;

  const first = send(window, "first");
  await waitFor(() => streamedCalls(fetchCalls).length === 1, "the first completion");
  await send(window, "second");
  clickStop(window);
  await within(first, "the first turn to return");

  assert.equal(getChatQueue(window).length, 1, "the queued message stays queued after Stop");
  await settle(window);
  assert.equal(getChatQueue(window).length, 1, "the queued message is not sent later either");
  assert.equal(streamedCalls(fetchCalls).length, 1, "Stop does not send the queued message");
  assert.ok(window.document.querySelector(".msg-row.queued"), "the queued message is still shown");
});

test("a message queued during Regenerate is sent when the regenerated reply finishes", async () => {
  const { window, fetchCalls, ctl } = setupApp();
  const conv = { id: "c1", title: "t", messages: [
    { role: "user", content: "hi" },
    { role: "assistant", content: "old reply" },
  ] };
  activateConv(window, conv);
  ctl.holdStream = true;

  window.regenerate(conv);
  await waitFor(() => streamedCalls(fetchCalls).length === 1, "the regenerate completion");
  await send(window, "second");
  assert.equal(getChatQueue(window).length, 1, "second is queued during Regenerate");

  ctl.holdStream = false;
  ctl.streamGate.open();
  await waitFor(() => streamedCalls(fetchCalls).length === 2, "the queued completion");
  assert.equal(lastUserText(streamedCalls(fetchCalls)[1]), "second");
  await settle(window);
});

test("an idle send goes after an older queued message of the same conversation", async () => {
  const { window, fetchCalls } = setupApp();
  const conv = { id: "c1", title: "t", messages: [] };
  activateConv(window, conv);
  runScript(window, `chat.queue = [{ text: "second", attachments: [], docs: [], convId: "${conv.id}" }];`);

  await within(send(window, "third"), "the idle send to return");
  await waitFor(() => streamedCalls(fetchCalls).length === 2, "both completions");
  await settle(window);

  assert.deepEqual(streamedCalls(fetchCalls).map(lastUserText), ["second", "third"]);
});

test("Stop during knowledge retrieval keeps the queue, and the next send keeps FIFO order", async () => {
  const kbGate = makeGate();
  const { window, fetchCalls, ctl } = setupApp({ kbGate });
  const doc = window.document;
  const sel = doc.getElementById("p-kb");
  const opt = doc.createElement("option");
  opt.value = "test-kb";
  sel.appendChild(opt);
  sel.value = "test-kb";
  activateConv(window, { id: "c1", title: "t", messages: [] });

  const first = send(window, "first");
  await waitFor(() => fetchCalls.some((c) => c.url.startsWith("/api/rag/")), "the knowledge query");
  await send(window, "second");
  clickStop(window);
  await within(first, "the first turn to return");

  assert.equal(getChatQueue(window).length, 1, "the queued message stays queued after Stop");
  await settle(window);
  assert.equal(getChatQueue(window).length, 1, "the queued message is not sent later either");
  assert.equal(streamedCalls(fetchCalls).length, 0, "Stop does not send anything");
  assert.ok(doc.querySelector(".msg-row.queued"), "the queued message is still shown");

  ctl.kbGate = null;
  await within(send(window, "third"), "the idle send to return");
  await waitFor(() => streamedCalls(fetchCalls).length === 2, "both completions");
  await settle(window);
  assert.deepEqual(streamedCalls(fetchCalls).map(lastUserText).map((s) => s.split("\n\n").pop()),
    ["second", "third"]);
});

test("after a send settles the transcript shows Edit, Revert and Regenerate", async () => {
  const { window } = setupApp();
  activateConv(window, { id: "c1", title: "t", messages: [] });

  await send(window, "hello");
  await settle(window);

  const user = rowActions(window, -2);
  const reply = rowActions(window, -1);
  assert.ok(user.includes("edit") && user.includes("revert"), "user row actions: " + user.join(", "));
  assert.ok(reply.includes("regenerate"), "reply row actions: " + reply.join(", "));
});
