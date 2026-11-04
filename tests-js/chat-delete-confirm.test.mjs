// SPDX-License-Identifier: AGPL-3.0-or-later
// The x on a sidebar chat asks first, names the chat, and deletes only on
// confirm; a pinned chat and a chat with no title go through the same path.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function makeFetch(calls) {
  return async (url, opts = {}) => {
    calls.push({ url: String(url), method: opts.method || "GET" });
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [], enabled: true, total: 0 }) };
  };
}

const drain = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); };

async function setup(extra) {
  extra.forEach((c, i) => { c.updated_at = 1000 - i; });
  const calls = [];
  const { window } = loadApp({ fetchImpl: makeFetch(calls) });
  runScript(window, "window.chatState = chat;");
  await drain();
  runScript(window, `
    chat.conversations.length = 0;
    chat.conversations.push(...${JSON.stringify(extra)});
    chat.activeId = chat.conversations[0].id;
    renderConvList();
  `);
  return { window, calls };
}

const modalOpen = (w) => w.document.getElementById("modal").style.display !== "none";
const modalTitle = (w) => w.document.getElementById("modal-title").textContent;
const delButton = (w, i) => [...w.document.querySelectorAll("#conv-list .conv-item")][i]
  .querySelector("button.del:last-of-type");
const ids = (w) => Array.from(w.chatState.conversations, (c) => c.id);

const conv = (id, title, extra = {}) => ({
  id, title, updated_at: 1, folder: null, branches: [],
  messages: [{ role: "user", content: "hi" }], ...extra,
});

test("chat delete: the x opens a confirm naming the chat and deletes nothing yet", async () => {
  const { window, calls } = await setup([conv("a", "Trip planning"), conv("b", "Other")]);
  const del = [...window.document.querySelectorAll("#conv-list .conv-item")]
    .find((i) => i.textContent.includes("Trip planning")).querySelector("button.del:last-of-type");
  del.click();
  assert.ok(modalOpen(window), "a confirm dialog opens");
  assert.ok(modalTitle(window).includes("Trip planning"), `title names the chat: ${modalTitle(window)}`);
  assert.deepEqual(ids(window).sort(), ["a", "b"], "nothing is deleted before the confirm");
  assert.ok(!calls.some((c) => c.method === "DELETE"), "no server delete before the confirm");
});

test("chat delete: cancel keeps the chat, confirm removes it and deletes it on the server", async () => {
  const { window, calls } = await setup([conv("a", "Trip planning"), conv("b", "Other")]);
  const first = () => [...window.document.querySelectorAll("#conv-list .conv-item")]
    .find((i) => i.textContent.includes("Trip planning")).querySelector("button.del:last-of-type");
  first().click();
  [...window.document.querySelectorAll("#modal-body button")].find((b) => b.classList.contains("btn-secondary") && !b.classList.contains("btn-danger")).click();
  await drain();
  assert.deepEqual(ids(window).sort(), ["a", "b"], "cancel keeps the chat");

  first().click();
  window.document.querySelector("#modal-body button.btn-danger").click();
  await drain();
  assert.deepEqual(ids(window), ["b"], "confirm removes it");
  assert.ok(calls.some((c) => c.method === "DELETE" && c.url.includes("/api/conversations/a")),
    "confirm deletes it on the server");
});

test("chat delete: a chat with no title is named by the default title", async () => {
  const { window } = await setup([conv("a", ""), conv("b", "New chat"), conv("c", "Keep")]);
  delButton(window, 0).click();
  assert.ok(modalOpen(window));
  assert.ok(!/undefined|""/.test(modalTitle(window)), `no raw empty name: ${modalTitle(window)}`);
  assert.ok(modalTitle(window).includes("New chat"), `default title is used: ${modalTitle(window)}`);
  window.document.querySelector("#modal-body button.btn-danger").click();
  await drain();
  assert.ok(!ids(window).includes("a"), "the untitled chat is deleted on confirm");
});

test("chat delete: a pinned chat asks the same way and is deleted on confirm", async () => {
  const { window } = await setup([conv("p", "Pinned one", { pinned: true }), conv("b", "Other")]);
  delButton(window, 0).click();
  assert.ok(modalOpen(window));
  assert.ok(modalTitle(window).includes("Pinned one"));
  assert.ok(ids(window).includes("p"), "still present before the confirm");
  window.document.querySelector("#modal-body button.btn-danger").click();
  await drain();
  assert.deepEqual(ids(window), ["b"]);
});
