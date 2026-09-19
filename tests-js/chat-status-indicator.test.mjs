// SPDX-License-Identifier: AGPL-3.0-or-later
// Unit tests for the live inference status indicator in the web chat UI.
// Covers timer formatting, DOM indicator creation, warning class toggling,
// and lifecycle cleanup.

import { test } from "node:test";
import assert from "node:assert/strict";

import { loadApp, runScript } from "./harness.mjs";

test("formatStatusElapsed formats seconds into m:ss correctly", () => {
  const { window } = loadApp();
  assert.equal(window.formatStatusElapsed(0), "0:00");
  assert.equal(window.formatStatusElapsed(5), "0:05");
  assert.equal(window.formatStatusElapsed(12), "0:12");
  assert.equal(window.formatStatusElapsed(59), "0:59");
  assert.equal(window.formatStatusElapsed(60), "1:00");
  assert.equal(window.formatStatusElapsed(65), "1:05");
  assert.equal(window.formatStatusElapsed(125), "2:05");
});

test("createStatusIndicator creates the status pill element hierarchy", () => {
  const { window } = loadApp();
  const ind = window.createStatusIndicator("Processing prompt...", false);
  assert.equal(ind.tagName.toLowerCase(), "div");
  assert.ok(ind.classList.contains("msg-status-indicator"));
  assert.ok(!ind.classList.contains("st-warn"));

  const pulse = ind.querySelector(".status-pulse");
  assert.ok(pulse);

  const text = ind.querySelector(".status-text");
  assert.ok(text);
  assert.equal(text.textContent, "Processing prompt...");

  const timer = ind.querySelector(".status-timer");
  assert.ok(timer);
  assert.equal(timer.textContent, "0:00");
});

test("createStatusIndicator applies st-warn when isWarn is true", () => {
  const { window } = loadApp();
  const ind = window.createStatusIndicator("GPU vision encode failed; retrying on CPU...", true);
  assert.ok(ind.classList.contains("msg-status-indicator"));
  assert.ok(ind.classList.contains("st-warn"));
});

test("mountStatusIndicator attaches indicator and sets up timer interval", () => {
  const { window } = loadApp();
  const doc = window.document;
  const body = doc.createElement("div");
  body.className = "msg-body";
  doc.body.appendChild(body);

  const ind = window.mountStatusIndicator(body, "Encoding image...");
  assert.ok(ind);
  assert.equal(body.querySelector(".msg-status-indicator"), ind);
  assert.ok(body._statusTimer);

  window.removeStatusIndicator(body);
  assert.equal(body.querySelector(".msg-status-indicator"), null);
  assert.equal(body._statusTimer, undefined);
  body.remove();
});

test("updateStatusIndicator updates text and adds st-warn on retry messages", () => {
  const { window } = loadApp();
  const doc = window.document;
  const body = doc.createElement("div");
  body.className = "msg-body";
  doc.body.appendChild(body);

  window.mountStatusIndicator(body, "Encoding image (GPU)...");
  const ind = body.querySelector(".msg-status-indicator");
  assert.ok(!ind.classList.contains("st-warn"));

  // Update with retry message
  window.updateStatusIndicator(body, "GPU vision encode failed; retrying on CPU (this may take longer)...");
  assert.ok(ind.classList.contains("st-warn"));
  assert.equal(ind.querySelector(".status-text").textContent,
    "GPU vision encode failed; retrying on CPU (this may take longer)...");

  // Update back to regular status
  window.updateStatusIndicator(body, "Generating response...");
  assert.ok(!ind.classList.contains("st-warn"));
  assert.equal(ind.querySelector(".status-text").textContent, "Generating response...");

  window.removeStatusIndicator(body);
  body.remove();
});

test("runCompletion mounts status indicator and removes it when tokens stream", async () => {
  const fetchImpl = async (url, _opts) => {
    if (String(url) === "/v1/chat/completions") {
      return {
        ok: true,
        status: 200,
        headers: { get: () => null },
        json: async () => ({}),
        text: async () => "",
      };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };

  const { window } = loadApp({ fetchImpl });
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;

  let streamDataCb = null;
  let finishStream = null;
  const streamDonePromise = new Promise((res) => { finishStream = res; });
  window.__saveStreamCb = (cb) => { streamDataCb = cb; };
  window.__streamWait = streamDonePromise;

  runScript(window, `
    modelCache.active = 'test-model';
    maybeCompactConversation = async () => {};
    readSSE = async (r, onData) => {
      window.__saveStreamCb(onData);
      await window.__streamWait;
    };
  `);

  const conv = { id: "c_test", title: "t", messages: [{ role: "user", content: "hello" }] };
  const runPromise = window.runCompletion(conv);

  // Allow setup to run up to readSSE waiting
  await new Promise((r) => setTimeout(r, 20));

  const messagesBox = doc.getElementById("chat-messages");
  const liveBody = messagesBox.querySelector(".msg-row.assistant .msg-body");
  assert.ok(liveBody, "assistant message row body should be added");

  const indicator = liveBody.querySelector(".msg-status-indicator");
  assert.ok(indicator, "status indicator should be mounted initially");

  // Simulate status chunk from SSE
  assert.ok(streamDataCb, "readSSE callback should be captured");
  streamDataCb(JSON.stringify({
    id: "chunk_1",
    choices: [{ delta: { status: "Generating response..." } }],
  }));

  assert.equal(indicator.querySelector(".status-text").textContent, "Generating response...");

  // Simulate first token chunk
  streamDataCb(JSON.stringify({
    id: "chunk_2",
    choices: [{ delta: { content: "Hi" } }],
  }));

  assert.equal(liveBody.querySelector(".msg-status-indicator"), null,
    "status indicator should be removed once tokens stream");

  // Complete stream
  streamDataCb("[DONE]");
  finishStream();
  await runPromise;
});

