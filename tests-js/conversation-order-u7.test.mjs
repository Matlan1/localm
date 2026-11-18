// SPDX-License-Identifier: AGPL-3.0-or-later
// A reload must not re-stamp conversations (the list stays newest first), and the
// sidebar search must find text in chats whose bodies the browser has not loaded.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const full = (id, title, at, text) => ({
  id, title, updated_at: at, pinned: false, folder: null, branches: [],
  messages: [{ role: "user", content: text }],
});

function makeFetch({ index, puts, searches }) {
  return async (url, init) => {
    const u = String(url); 
    const ok = (body) => ({ ok: true, status: 200, text: async () => "", json: async () => body });
    if (u.startsWith("/api/conversations?")) {
      if (u.includes("&q=")) {
        searches.push(decodeURIComponent(u.split("&q=")[1]));
        return ok({ enabled: true, total: 1, conversations: [
          { id: "c3", title: "Fruit notes", updated_at: 50, n_messages: 1,
            snippet: "i like cherry pie" }] });
      }
      return ok({ enabled: true, total: index.length, conversations: index });
    }
    if (u.startsWith("/api/conversations/")) {
      if (init && init.method === "PUT") {
        puts.push({ id: u.split("/").pop(), updated_at: JSON.parse(init.body).updated_at });
        return ok({ status: "saved" });
      }
      return ok(full(u.split("/").pop(), "Cherries", 900, "i like cherry pie"));
    }
    if (u === "/v1/config") return ok({ effective_mode: "log", n_ctx_max: 16384, instance_id: "backend-x" });
    return ok({ models: [], active: "", conversations: [], plugins: [] });
  };
}

const drain = async (n = 12) => { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); };
const meta = (c) => ({ id: c.id, title: c.title, updated_at: c.updated_at, pinned: false,
                       folder: null, n_messages: 1 });

test("a reload keeps every cached conversation's own updated_at and the order", async () => {
  const older = full("c1", "First", 100, "one");
  const newer = full("c2", "Say hi", 200, "two");
  const puts = [];
  const { window } = loadApp({
    fetchImpl: makeFetch({ index: [meta(newer), meta(older)], puts, searches: [] }),
    seedLocalStorage: { "localm.instanceId": "backend-x", "localm.conversations": JSON.stringify([older, newer]) },
  });
  runScript(window, "window.chatState = chat;");
  await drain();
  await new Promise((r) => setTimeout(r, 900));   // past the 600 ms push debounce

  assert.deepEqual(Array.from(window.chatState.conversations, (c) => c.id), ["c2", "c1"]);
  assert.deepEqual(Array.from(window.chatState.conversations, (c) => c.updated_at), [200, 100]);
  assert.equal(puts.length, 0, "a tie with the server copy uploads nothing");
});

test("a local copy newer than the server is uploaded with its own stamp", async () => {
  const local = full("c1", "First", 500, "one");
  const puts = [];
  const { window } = loadApp({
    fetchImpl: makeFetch({ index: [meta({ ...local, updated_at: 100 })], puts, searches: [] }),
    seedLocalStorage: { "localm.instanceId": "backend-x", "localm.conversations": JSON.stringify([local]) },
  });
  runScript(window, "window.chatState = chat;");
  await drain();
  await new Promise((r) => setTimeout(r, 900));

  assert.equal(puts.length, 1);
  assert.equal(puts[0].id, "c1");
  assert.equal(puts[0].updated_at, 500);
  assert.equal(window.chatState.conversations[0].updated_at, 500);
});

test("the sidebar search finds text in a chat the browser holds only as an index row", async () => {
  const mine = full("c1", "First", 100, "one");
  const searches = [];
  const { window } = loadApp({
    fetchImpl: makeFetch({
      index: [{ id: "c3", title: "Fruit notes", updated_at: 50, pinned: false, folder: null, n_messages: 1 },
              meta(mine)],
      puts: [], searches }),
    seedLocalStorage: { "localm.instanceId": "backend-x", "localm.conversations": JSON.stringify([mine]) },
  });
  runScript(window, "window.chatState = chat;");
  await drain();

  const box = window.document.getElementById("conv-search");
  box.value = "cher";
  box.dispatchEvent(new window.Event("input", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 500));
  await drain();

  assert.deepEqual(searches, ["cher"]);
  const list = window.document.getElementById("conv-list").textContent;
  assert.match(list, /Fruit notes/);
  assert.match(list, /cherry pie/);
});
