// SPDX-License-Identifier: AGPL-3.0-or-later
// The conversations sidebar fetches a lightweight paginated meta index and
// lazy-loads each conversation's messages on open.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const FULL = {
  c1: { id: "c1", title: "First", updated_at: 2, pinned: false, folder: null,
        branches: [], messages: [{ role: "user", content: "hi one" },
                                 { role: "assistant", content: "reply one" }] },
  c2: { id: "c2", title: "Second", updated_at: 1, pinned: false, folder: null,
        branches: [], messages: [{ role: "user", content: "hi two" }] },
};

function makeFetch(calls) {
  return async (url) => {
    const u = String(url);
    calls.push(u);
    if (u.startsWith("/api/conversations?")) {
      return { ok: true, status: 200, text: async () => "", json: async () => ({
        enabled: true, total: 2, conversations: [
          { id: "c1", title: "First", updated_at: 2, pinned: false, folder: null, n_messages: 2 },
          { id: "c2", title: "Second", updated_at: 1, pinned: false, folder: null, n_messages: 1 },
        ] }) };
    }
    if (u.startsWith("/api/conversations/")) {
      const id = u.split("/").pop();
      return { ok: true, status: 200, text: async () => "", json: async () => FULL[id] };
    }
    if (u === "/v1/config") {
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ effective_mode: "log", n_ctx_max: 16384 }) };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function drain(times = 12) {
  for (let i = 0; i < times; i++) await new Promise((r) => setTimeout(r, 0));
}

test("R40: the GUI fetches a meta index and lazy-loads bodies on open", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: makeFetch(calls) });
  runScript(window, "window.chatState = chat;");
  await drain();   // boot: refreshCtxLimit -> initServerConversations

  // The list request is the LIGHTWEIGHT index, not the full payload.
  assert.ok(calls.some((u) => u.includes("/api/conversations?meta=true")),
    "the sidebar list is fetched as a metadata-only index");
  assert.ok(!calls.some((u) => u === "/api/conversations"),
    "the GUI never fetches the heavy full-payload list");

  const convs = window.chatState.conversations;
  const c1 = convs.find((c) => c.id === "c1");
  const c2 = convs.find((c) => c.id === "c2");
  assert.ok(c1 && c2, "both conversations are in the sidebar");

  // The active conversation (c1, newest) was hydrated automatically on load.
  assert.ok(calls.includes("/api/conversations/c1"), "the active conversation's body was loaded");
  assert.equal(c1.messages.length, 2, "active conversation has its full messages");
  assert.ok(!c1._meta, "active conversation is no longer an index-only placeholder");

  // The non-active conversation is still an index-only placeholder (not downloaded).
  assert.equal(c2._meta, true, "an unopened conversation keeps its lightweight placeholder");
  assert.equal(c2.messages.length, 0, "its body was NOT downloaded for the list");

  // Opening it hydrates the body on demand.
  const items = [...window.document.querySelectorAll("#conv-list .conv-item")];
  const c2item = items.find((it) => /Second/.test(it.textContent));
  assert.ok(c2item, "the second conversation is listed");
  c2item.click();
  await drain();

  assert.ok(calls.includes("/api/conversations/c2"), "opening it fetched its full body");
  assert.equal(window.chatState.conversations.find((c) => c.id === "c2").messages.length, 1,
    "the opened conversation now has its messages");
});
