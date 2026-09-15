// SPDX-License-Identifier: AGPL-3.0-or-later
// WEB-FUNC-003: an injected web/kb/doc note is stored with role:"user" (see
// chat.js's noteLabel() comment) so the model reads it as user-turn content,
// but on screen it must never inherit the real user turn's right-aligned
// bubble layout, and its Web/Doc/Sources label must stay visible. Both broke
// because addMessageRow put "user" and "web-note" on the same row, and
// style.css's .msg-row.user rules (flex/right-align, the bubble background,
// and .msg-role{display:none}) applied right alongside .msg-row.web-note's.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function setActiveConv(window, conv) {
  runScript(window,
    `chat.conversations = [${JSON.stringify(conv)}]; chat.activeId = ${JSON.stringify(conv.id)};`);
}

test("renderChat: a web-search-result row is not styled like a user bubble, and shows the Web label", () => {
  const { window } = loadApp();
  setActiveConv(window, {
    id: "c1", title: "t",
    messages: [
      { role: "user", content: "what's the weather", id: "m1" },
      { role: "user", web: true, id: "m2", content: "[web_search results] Cloudy, 18C." },
    ],
  });
  window.renderChat();
  const box = window.document.getElementById("chat-messages");
  const rows = [...box.querySelectorAll(".msg-row")];
  assert.equal(rows.length, 2);

  const [typed, note] = rows;
  assert.ok(typed.classList.contains("user"),
    "the genuinely typed message keeps the user row class");

  assert.ok(!note.classList.contains("user"),
    "a web-result row must not carry the user row class - that is what let " +
    ".msg-row.user's bubble/right-align rules apply to it (WEB-FUNC-003)");
  assert.ok(note.classList.contains("web-note"), "the note-styling class is present");
  const label = note.querySelector(".msg-role");
  assert.ok(label, "the row still has a role label element");
  assert.equal(label.textContent, "Web", "the Web label survives (not the You default)");
});

test("renderChat: a knowledge-base excerpt row is not styled like a user bubble, and shows the Sources label", () => {
  const { window } = loadApp();
  setActiveConv(window, {
    id: "c2", title: "t",
    messages: [
      { role: "user", tag: "kb", id: "m1", content: "[Excerpts from \"docs\"]\nInstall via pip." },
    ],
  });
  window.renderChat();
  const row = window.document.getElementById("chat-messages").querySelector(".msg-row");
  assert.ok(!row.classList.contains("user"));
  assert.ok(row.classList.contains("web-note"));
  assert.equal(row.querySelector(".msg-role").textContent, "Sources");
});

test("renderChat: an attached-document row is not styled like a user bubble, and shows the Doc label", () => {
  const { window } = loadApp();
  setActiveConv(window, {
    id: "c3", title: "t",
    messages: [
      { role: "user", tag: "doc", id: "m1", content: "[Attached document: notes.txt]\nMeeting at 3pm." },
    ],
  });
  window.renderChat();
  const row = window.document.getElementById("chat-messages").querySelector(".msg-row");
  assert.ok(!row.classList.contains("user"));
  assert.ok(row.classList.contains("web-note"));
  assert.equal(row.querySelector(".msg-role").textContent, "Doc");
});

test("addMessageRow: an ordinary row with no opts.cls keeps its role class (regression guard)", () => {
  const { window } = loadApp();
  const box = window.document.getElementById("chat-messages");
  window.addMessageRow(box, "user", "hi");
  window.addMessageRow(box, "assistant", "hello");
  assert.ok(box.querySelector(".msg-row.user"), "a plain user row keeps the user row class");
  assert.ok(box.querySelector(".msg-row.assistant"), "a plain assistant row keeps the assistant row class");
});
