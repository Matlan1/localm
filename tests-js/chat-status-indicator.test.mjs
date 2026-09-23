// SPDX-License-Identifier: AGPL-3.0-or-later
// Unit tests for the live inference status indicator in the web chat UI.
// Covers timer formatting, DOM indicator creation, warning class toggling,
// and lifecycle cleanup.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { loadApp, runScript } from "./harness.mjs";

const STATIC = join(dirname(fileURLToPath(import.meta.url)), "..",
  "localm", "plugins", "gui", "static");
const DE = JSON.parse(readFileSync(join(STATIC, "i18n", "de.json"), "utf-8"));

/** A window with German loaded from the real catalog. */
async function loadGerman() {
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u.includes("/i18n/de.json")) return { ok: true, status: 200, json: async () => DE };
    if (u.includes("/i18n/")) return { ok: false, status: 404, json: async () => ({}) };
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl });
  runScript(window, 'window.__p = applyLanguage("de");');
  await window.__p;
  return window;
}

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

test("updateStatusIndicator localizes the pill from status_code when German is active", async () => {
  const window = await loadGerman();
  const doc = window.document;
  const body = doc.createElement("div");
  body.className = "msg-body";
  doc.body.appendChild(body);

  window.updateStatusIndicator(body, "Encoding image (GPU)...", "encoding_image_gpu");
  const ind = body.querySelector(".msg-status-indicator");
  const label = () => ind.querySelector(".status-text").textContent;
  assert.equal(label(), "Bild wird kodiert (GPU)…");
  assert.ok(!label().includes("Encoding"), `expected no English leftover, got ${label()}`);

  window.removeStatusIndicator(body);
  body.remove();
});

test("updateStatusIndicator picks the warning style from status_code, not from the localized text",
  async () => {
    const window = await loadGerman();
    const doc = window.document;
    const body = doc.createElement("div");
    body.className = "msg-body";
    doc.body.appendChild(body);

    window.updateStatusIndicator(body, "Encoding image (GPU)...", "encoding_image_gpu");
    const ind = body.querySelector(".msg-status-indicator");
    assert.ok(!ind.classList.contains("st-warn"));

    window.updateStatusIndicator(
      body,
      "GPU vision encode failed; retrying on CPU (this may take longer)...",
      "vision_cpu_retry",
    );
    assert.ok(ind.classList.contains("st-warn"),
      "vision_cpu_retry must apply the warning style even though the German text carries none of the English warning words");
    const label = ind.querySelector(".status-text").textContent;
    assert.ok(!label.includes("failed") && !label.includes("retrying"),
      `expected the German label with no English warning words, got ${label}`);

    window.updateStatusIndicator(body, "Generating response...", "generating");
    assert.ok(!ind.classList.contains("st-warn"), "the warning style must clear once the code is no longer vision_cpu_retry");

    window.removeStatusIndicator(body);
    body.remove();
  });

test("updateStatusIndicator falls back to the raw English text for an unrecognized status_code",
  () => {
    const { window } = loadApp();
    const doc = window.document;
    const body = doc.createElement("div");
    body.className = "msg-body";
    doc.body.appendChild(body);

    window.updateStatusIndicator(body, "Doing something new...", "some_future_stage");
    const ind = body.querySelector(".msg-status-indicator");
    assert.equal(ind.querySelector(".status-text").textContent, "Doing something new...");

    window.removeStatusIndicator(body);
    body.remove();
  });

