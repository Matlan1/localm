// SPDX-License-Identifier: AGPL-3.0-or-later
// Compaction archives summarised-away alternative branches into
// conv.droppedBranches; exportConversation emits their content.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function setActiveConv(window, conv) {
  // `chat` is module-scoped, not a window property: assign into it from inside
  // the realm.
  runScript(window,
    `chat.conversations = [${JSON.stringify(conv)}]; chat.activeId = ${JSON.stringify(conv.id)};`);
}

function captureExport(window) {
  // exportConversation builds `new Blob([markdown], ...)` then downloads it.
  // jsdom's Blob has no .text(), so capture the markdown array directly.
  let captured = null;
  const OrigBlob = window.Blob;
  window.Blob = function (parts, opts) {
    captured = (parts || []).join("");
    return new OrigBlob(parts, opts);
  };
  window.URL.createObjectURL = () => "blob:x";
  window.URL.revokeObjectURL = () => {};
  const origCreate = window.document.createElement.bind(window.document);
  window.document.createElement = (tag) => {
    const el = origCreate(tag);
    if (tag === "a") el.click = () => {};       // no real navigation in jsdom
    return el;
  };
  return () => captured;
}

test("F5: exportConversation includes archived (droppedBranches) content", async () => {
  const { window } = loadAppWithPages();
  const getBlob = captureExport(window);
  setActiveConv(window, {
    id: "c1", title: "Greenhouse",
    messages: [
      { role: "user", content: "live question", id: "m1" },
      { role: "assistant", content: "live answer", id: "m2" },
    ],
    droppedBranches: [
      [{ role: "user", content: "an archived alternative I explored", id: "a1" },
       { role: "assistant", content: "the archived reply", id: "a2" }],
    ],
  });

  window.exportConversation();
  const text = getBlob();
  assert.ok(text, "export markdown was produced");
  assert.match(text, /live question/, "live messages are exported");
  assert.match(text, /Archived alternative branches \(1\)/,
    "the archived-branches section is present");
  assert.match(text, /an archived alternative I explored/,
    "the archived branch content is actually in the export (not a facade)");
  assert.match(text, /the archived reply/);
});

test("F5: export omits the archived section when there are no dropped branches", async () => {
  const { window } = loadAppWithPages();
  const getBlob = captureExport(window);
  setActiveConv(window, {
    id: "c2", title: "Plain",
    messages: [{ role: "user", content: "hello", id: "m1" }],
  });
  window.exportConversation();
  const text = getBlob();
  assert.match(text, /hello/);
  assert.doesNotMatch(text, /Archived alternative branches/);
});
