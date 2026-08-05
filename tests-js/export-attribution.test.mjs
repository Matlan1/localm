// SPDX-License-Identifier: AGPL-3.0-or-later
// NEW-WEBSEARCH-UX-EXPORT-ATTRIBUTES-TOOL-RESULTS-TO-USER: web-search results,
// retrieved knowledge-base excerpts, and attached-document text are pushed
// with role:"user" (settings-perf.js) so the MODEL reads them as
// conversation context - a deliberate choice, not a bug, and not something
// this fix touches. But exportConversation() used to label every message
// purely by role, so those messages rendered in the exported markdown as
// "**You:**", exactly as if the human had typed raw search-result text.
// This asserts the export now uses noteLabel() (chat.js), the same
// Web/Doc/Sources override renderChat() already applies on screen, so the
// exported transcript and the live UI can never disagree about who "said"
// an injected message.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function setActiveConv(window, conv) {
  runScript(window,
    `chat.conversations = [${JSON.stringify(conv)}]; chat.activeId = ${JSON.stringify(conv.id)};`);
}

function captureExport(window) {
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
    if (tag === "a") el.click = () => {};
    return el;
  };
  return () => captured;
}

test("export: a web-search-result message is labelled Web, not You", async () => {
  const { window } = loadAppWithPages();
  const getBlob = captureExport(window);
  setActiveConv(window, {
    id: "c1", title: "Web lookup",
    messages: [
      { role: "user", content: "what's the weather in Vienna", id: "m1" },
      {
        role: "user", web: true, id: "m2",
        content: "[web_search results for \"weather Vienna\"]\nCloudy, 18C.",
      },
      { role: "assistant", content: "It's cloudy and 18C in Vienna.", id: "m3" },
    ],
  });

  window.exportConversation();
  const text = getBlob();
  assert.match(text, /\*\*You:\*\*\n\nwhat's the weather in Vienna/,
    "the genuinely user-typed message still reads as You");
  assert.doesNotMatch(text, /\*\*You:\*\*\n\n\[web_search results/,
    "the web-search-result message must NOT be attributed to the user");
  assert.match(text, /\*\*Web:\*\*\n\n\[web_search results for "weather Vienna"\]/,
    "the web-search-result message is labelled Web");
});

test("export: a knowledge-base excerpt is labelled Sources, not You", async () => {
  const { window } = loadAppWithPages();
  const getBlob = captureExport(window);
  setActiveConv(window, {
    id: "c2", title: "KB lookup",
    messages: [
      {
        role: "user", tag: "kb", id: "m1",
        content: "[Excerpts from the \"docs\" collection relevant to: setup]\n[1] readme.md:1\nInstall via pip.",
      },
      { role: "assistant", content: "Install via pip, per the docs.", id: "m2" },
    ],
  });

  window.exportConversation();
  const text = getBlob();
  assert.doesNotMatch(text, /\*\*You:\*\*\n\n\[Excerpts from/,
    "the retrieved knowledge-base excerpt must NOT be attributed to the user");
  assert.match(text, /\*\*Sources:\*\*\n\n\[Excerpts from the "docs" collection/,
    "the knowledge-base excerpt is labelled Sources");
});

test("export: an attached-document dump is labelled Doc, not You", async () => {
  const { window } = loadAppWithPages();
  const getBlob = captureExport(window);
  setActiveConv(window, {
    id: "c3", title: "Doc chat",
    messages: [
      {
        role: "user", tag: "doc", id: "m1",
        content: "[Attached document: notes.txt]\nMeeting is at 3pm.",
      },
      { role: "user", content: "when's the meeting?", id: "m2" },
    ],
  });

  window.exportConversation();
  const text = getBlob();
  assert.doesNotMatch(text, /\*\*You:\*\*\n\n\[Attached document/,
    "the attached-document text must NOT be attributed to the user");
  assert.match(text, /\*\*Doc:\*\*\n\n\[Attached document: notes\.txt\]/,
    "the attached-document message is labelled Doc");
  assert.match(text, /\*\*You:\*\*\n\nwhen's the meeting\?/,
    "the genuinely user-typed follow-up still reads as You");
});

test("export: dropped/archived branches apply the same label rules", async () => {
  const { window } = loadAppWithPages();
  const getBlob = captureExport(window);
  setActiveConv(window, {
    id: "c4", title: "Branch labels",
    messages: [{ role: "user", content: "hello", id: "m1" }],
    droppedBranches: [
      [
        { role: "user", web: true, id: "a1", content: "[web_search results] archived search hit" },
        { role: "assistant", content: "archived reply", id: "a2" },
      ],
    ],
  });

  window.exportConversation();
  const text = getBlob();
  assert.doesNotMatch(text, /\*\*You:\*\*\n\n\[web_search results\] archived search hit/,
    "an archived web-search message must not be attributed to the user either");
  assert.match(text, /\*\*Web:\*\*\n\n\[web_search results\] archived search hit/);
});
