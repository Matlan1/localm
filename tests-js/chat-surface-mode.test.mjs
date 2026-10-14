// SPDX-License-Identifier: AGPL-3.0-or-later
// The chat page follows the chat surface's session mode (/v1/config
// effective_chat_mode, which applies the Chat persistence override), and its
// chat requests name the chat surface with X-Client-Id: chat.
import { test } from "node:test";
import assert from "node:assert/strict";

import { loadApp, runScript } from "./harness.mjs";

const drain = async (n = 12) => {
  for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0));
};

const SAVED_CONV = {
  id: "saved-1", title: "Saved earlier", updated_at: 999, pinned: false,
  folder: null, branches: [], messages: [{ role: "user", content: "earlier words" }],
};

// Answers /v1/config with window.__cfg.
const CONFIG_FETCH = `
  window.fetch = async (url) => {
    const u = String(url);
    if (u === "/v1/config") {
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ n_ctx_max: 16384, instance_id: "backend-x",
                              ...window.__cfg }) };
    }
    if (u.startsWith("/api/conversations")) {
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ enabled: true, conversations: [] }) };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
  window.chatState = chat;
`;

function boot(cfg) {
  const { window } = loadApp({
    seedLocalStorage: {
      "localm.instanceId": "backend-x",
      "localm.conversations": JSON.stringify([SAVED_CONV]),
    },
  });
  runScript(window, `window.__cfg = ${JSON.stringify(cfg)};` + CONFIG_FETCH +
                    "refreshCtxLimit();");
  return window;
}

test("a Chat persistence override of privacy puts the chat page in privacy mode " +
     "while the server mode is full", async () => {
  const window = boot({ effective_mode: "full", effective_chat_mode: "privacy" });
  await drain();
  assert.equal(window.chatState.privacy, true);
  assert.ok(window.document.getElementById("privacy-hint"),
    "the privacy hint is shown");
  assert.equal(window.localStorage.getItem("localm.conversations"), null,
    "privacy mode leaves no conversation in localStorage");
});

test("a Chat persistence override of full keeps the chat page out of privacy " +
     "mode while the server mode is privacy", async () => {
  const window = boot({ effective_mode: "privacy", effective_chat_mode: "full" });
  await drain();
  assert.equal(window.chatState.privacy, false);
  assert.equal(window.document.getElementById("privacy-hint"), null);
  assert.ok(window.localStorage.getItem("localm.conversations")
              .includes("Saved earlier"),
    "the saved conversation is kept");
});

test("a server that does not report effective_chat_mode falls back to " +
     "effective_mode", async () => {
  const window = boot({ effective_mode: "privacy" });
  await drain();
  assert.equal(window.chatState.privacy, true);
});

test("a chat request names the chat surface", async () => {
  const calls = [];
  const { window } = loadApp({
    fetchImpl: async (url, opts = {}) => {
      calls.push({ url: String(url), headers: opts.headers || {} });
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    },
  });
  window.maybeCompactConversation = async () => {};
  window.readSSE = async () => {};
  const doc = window.document;
  doc.getElementById("p-speak").checked = false;
  doc.getElementById("p-memory").checked = false;
  doc.getElementById("p-web").checked = false;
  runScript(window, "modelCache.active = 'm';");

  await window.runCompletion(
    { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] });

  const posts = calls.filter((c) => c.url === "/v1/chat/completions");
  assert.equal(posts.length, 1, "the chat request fired");
  assert.equal(posts[0].headers["X-Client-Id"], "chat");
});

test("a Chat persistence override of full does not start saving non-chat keys " +
     "while the server mode is privacy", async () => {
  const { window } = loadApp({
    seedLocalStorage: {
      "localm.instanceId": "backend-x",
      "localm.coderCwd": "left-over-project",
    },
  });
  runScript(window, `window.__cfg = ${JSON.stringify(
    { effective_mode: "privacy", effective_chat_mode: "full" })};` + CONFIG_FETCH +
    "refreshCtxLimit();");
  await drain();
  assert.equal(window.chatState.privacy, false);
  assert.equal(window.localStorage.getItem("localm.coderCwd"), null,
    "a non-chat key is wiped under a privacy server mode");
  runScript(window, `lsSetScoped("localm.coderCwd", "new-project");
                     lsSetScoped("localm.webAccess", "1");`);
  assert.equal(window.localStorage.getItem("localm.coderCwd"), null,
    "a non-chat key is not written under a privacy server mode");
  assert.equal(window.localStorage.getItem("localm.webAccess"), "1",
    "a chat setting follows the chat mode");
});

test("a Chat persistence override of privacy keeps saving non-chat keys while " +
     "the server mode is full", async () => {
  const { window } = loadApp({
    seedLocalStorage: {
      "localm.instanceId": "backend-x",
      "localm.coderCwd": "kept-project",
    },
  });
  runScript(window, `window.__cfg = ${JSON.stringify(
    { effective_mode: "full", effective_chat_mode: "privacy" })};` + CONFIG_FETCH +
    "refreshCtxLimit();");
  await drain();
  assert.equal(window.chatState.privacy, true);
  assert.equal(window.localStorage.getItem("localm.coderCwd"), "kept-project");
  runScript(window, `lsSetScoped("localm.coderCwd", "new-project");
                     lsSetScoped("localm.webAccess", "1");`);
  assert.equal(window.localStorage.getItem("localm.coderCwd"), "new-project");
  assert.equal(window.localStorage.getItem("localm.webAccess"), null,
    "a chat setting is not written under a privacy chat mode");
});
