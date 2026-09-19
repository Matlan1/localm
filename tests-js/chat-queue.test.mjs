// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function setupApp(options = {}) {
  const fetchCalls = [];
  const responses = options.responses || {};
  const impl = async (url, opts = {}) => {
    fetchCalls.push({ url: String(url), opts });
    if (responses[String(url)]) {
      return responses[String(url)](opts);
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({}),
      text: async () => "",
    };
  };
  const { window } = loadApp({ fetchImpl: impl });
  window.maybeCompactConversation = async () => {};
  return { window, fetchCalls };
}

function activateConv(window, conv) {
  window.__testConv = conv;
  runScript(window, "chat.conversations = [window.__testConv]; chat.activeId = window.__testConv.id;");
}

function getChatQueue(window) {
  runScript(window, "window.__testQueue = chat.queue;");
  return window.__testQueue;
}

test("sendChat queues message when chat is busy and displays queued indicator", async () => {
  const { window, fetchCalls } = setupApp();
  const conv = { id: "c1", title: "t", messages: [] };
  activateConv(window, conv);
  runScript(window, "modelCache.active = 'test-model';");
  runScript(window, "chat.busy = true;");

  const doc = window.document;
  const input = doc.getElementById("chat-input");
  input.value = "follow up question";

  await window.sendChat();

  const chatCalls = fetchCalls.filter((c) => c.url.includes("/chat/completions"));
  assert.equal(chatCalls.length, 0, "must not fire completion immediately while busy");

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

test("processChatQueue automatically runs next queued turn after completion", async () => {
  const { window, fetchCalls } = setupApp({
    responses: {
      "/v1/chat/completions": () => ({
        ok: true,
        status: 200,
        body: null,
        json: async () => ({}),
      }),
    },
  });
  runScript(window, "modelCache.active = 'test-model';");
  window.readSSE = async (_r, onData) => {
    onData(JSON.stringify({ choices: [{ delta: { content: "reply" } }] }));
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }));
  };

  const conv = { id: "c1", title: "t", messages: [] };
  activateConv(window, conv);
  runScript(window, `chat.queue = [{ text: "queued turn", attachments: [], docs: [], convId: "${conv.id}" }];`);

  assert.equal(fetchCalls.filter((c) => c.url.includes("/chat/completions")).length, 0);

  await window.processChatQueue();

  const chatCalls = fetchCalls.filter((c) => c.url.includes("/chat/completions"));
  assert.equal(chatCalls.length, 1, "completion must be called for dequeued turn");
  const queue = getChatQueue(window);
  assert.equal(queue.length, 0, "queue must be emptied after processing");
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

test("retrieveKnowledge mounts searching status indicator and supports abort signal", async () => {
  let signalReceived = null;
  const { window } = setupApp({
    responses: {
      "/api/rag/collections/test-kb/query": (opts) => {
        signalReceived = opts.signal;
        return {
          ok: true,
          status: 200,
          json: async () => ({
            hits: [{ source: "doc.txt", pos: 12, text: "excerpt text" }],
          }),
        };
      },
    },
  });

  const doc = window.document;
  const sel = doc.getElementById("p-kb");
  const opt = doc.createElement("option");
  opt.value = "test-kb";
  sel.appendChild(opt);
  sel.value = "test-kb";

  const conv = { id: "c1", title: "t", messages: [] };
  activateConv(window, conv);
  const controller = new AbortController();

  await window.retrieveKnowledge(conv, "test query", { signal: controller.signal });

  assert.ok(signalReceived, "abort signal must be passed to knowledge query fetch");
  assert.equal(conv.messages.length, 1, "knowledge excerpt message must be added");
  assert.equal(conv.messages[0].tag, "kb");
});
