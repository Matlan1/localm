// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the slash-command palette (app/slash.js): the chat/coder
// dropdown tables and the coder help modal render their catalog-driven
// hint/args text and follow a language change, and the usage-error and
// unknown-command toasts resolve through the catalog too. Before this file,
// slash.js had no exact-text coverage of its own (web.test.mjs drives
// runWebInChat but only asserts on settings-perf.js's system-prompt text), so
// a mistranslation or a typo'd catalog key in CHAT_COMMANDS/CODER_COMMANDS -
// which tests-js/i18n.test.mjs's T_CALL_FILES scan cannot see, since those
// keys are table data rather than a literal argument to a t()/tn() call -
// would otherwise ship with nothing to catch it.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { loadApp, runScript } from "./harness.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const DE = JSON.parse(
  readFileSync(join(ROOT, "localm", "plugins", "gui", "static", "i18n", "de.json"), "utf-8"));

/** Serves the real German catalog; every other URL gets the harness default. */
function fetchWithDe() {
  return async (url) => {
    if (String(url).includes("/i18n/de.json")) {
      return { ok: true, status: 200, json: async () => DE, text: async () => JSON.stringify(DE) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

async function toGerman(win) {
  runScript(win, 'window.__p = applyLanguage("de");');
  await win.__p;
}

function typeSlash(win, textareaId, text) {
  const ta = win.document.getElementById(textareaId);
  ta.value = text;
  ta.dispatchEvent(new win.Event("input", { bubbles: true }));
  return ta;
}

function menuItems(win, textareaId) {
  const ta = win.document.getElementById(textareaId);
  return Array.from(ta.closest(".composer-wrap").querySelectorAll(".slash-menu .slash-item"));
}

const toastText = (win) => win.document.getElementById("toast").textContent;

/* ================================================================ */
/*  Command-table hints (the "/" dropdown)                           */
/* ================================================================ */

test("chat dropdown shows a command's translated hint and args", () => {
  const { window } = loadApp();
  typeSlash(window, "chat-input", "/generate-image");
  const [item] = menuItems(window, "chat-input");
  assert.equal(item.querySelector(".cmd").textContent, "/generate-image <prompt>");
  assert.equal(item.querySelector(".hint").textContent, "generate an image with FLUX");
});

test("coder dropdown shows a no-args command's translated hint", () => {
  const { window } = loadApp();
  typeSlash(window, "coder-input", "/undo");
  const [item] = menuItems(window, "coder-input");
  assert.equal(item.querySelector(".cmd").textContent, "/undo");
  assert.equal(item.querySelector(".hint").textContent, "revert the last file write");
});

test("both dropdowns follow a language change", async () => {
  const { window } = loadApp({ fetchImpl: fetchWithDe() });
  await toGerman(window);

  typeSlash(window, "chat-input", "/generate-image");
  const [image] = menuItems(window, "chat-input");
  assert.equal(image.querySelector(".cmd").textContent, "/generate-image <Prompt>");
  assert.equal(image.querySelector(".hint").textContent, "ein Bild mit FLUX erzeugen");

  typeSlash(window, "chat-input", "/generate-music");
  const [music] = menuItems(window, "chat-input");
  assert.equal(music.querySelector(".cmd").textContent, "/generate-music <Stil-Tags>");

  typeSlash(window, "coder-input", "/undo");
  const [undo] = menuItems(window, "coder-input");
  assert.equal(undo.querySelector(".hint").textContent,
    "die letzte Änderung an einer Datei rückgängig machen");
});

/* ================================================================ */
/*  Coder help modal (/help)                                         */
/* ================================================================ */

test("the coder help modal lists a translated title, hint and footer", () => {
  const { window } = loadApp();
  window.execCoderCommand("help");
  assert.equal(window.document.getElementById("modal-title").textContent, "Coder commands");
  const rows = Array.from(window.document.querySelectorAll("#modal-body .log-entry"));
  const undoRow = rows.find((r) => r.querySelector(".t").textContent === "/undo");
  assert.ok(undoRow, "the /undo row is present");
  assert.equal(undoRow.lastChild.textContent, "revert the last file write");
  assert.equal(window.document.querySelector("#modal-body .sub").textContent,
    "Anything not starting with / is sent to the agent as a task.");
});

test("the coder help modal follows a language change", async () => {
  const { window } = loadApp({ fetchImpl: fetchWithDe() });
  await toGerman(window);
  window.execCoderCommand("help");
  assert.equal(window.document.getElementById("modal-title").textContent, "Coder-Befehle");
  const rows = Array.from(window.document.querySelectorAll("#modal-body .log-entry"));
  const undoRow = rows.find((r) => r.querySelector(".t").textContent === "/undo");
  assert.equal(undoRow.lastChild.textContent,
    "die letzte Änderung an einer Datei rückgängig machen");
});

/* ================================================================ */
/*  Usage-error toasts                                               */
/* ================================================================ */

test("/generate-image with no prompt reports its usage", async () => {
  const { window } = loadApp();
  await window.runImagineInChat("");
  assert.equal(toastText(window), "Usage: /generate-image <prompt>");
});

test("/web with no query reports its usage", async () => {
  const { window } = loadApp();
  await window.runWebInChat("");
  assert.equal(toastText(window), "Usage: /web <query>");
});

test("/web with no model loaded refuses before the search and adds no turn", async () => {
  const calls = [];
  const fetchImpl = async (url) => {
    calls.push(String(url));
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl });
  runScript(window, "modelCache.active = ''; modelCache.resumable = '';");
  await window.runWebInChat("price of bitcoin");
  assert.match(toastText(window), /No model loaded/);
  assert.equal(calls.filter((u) => u.includes("/api/web/")).length, 0,
    "the search must not run when the answer cannot be generated");
  const conv = window.currentConv();
  assert.equal(conv ? conv.messages.length : 0, 0, "no user turn or search event was added");
});

test("/rename with no title reports its usage", () => {
  const { window } = loadApp();
  window.execChatCommand("rename", "");
  assert.equal(toastText(window), "Usage: /rename <title>");
});

/* ================================================================ */
/*  Unknown command                                                  */
/* ================================================================ */

test("an unrecognised command reports 'Unknown command: /<cmd>'", () => {
  const { window } = loadApp();
  window.handleSlashSubmit("/totally-bogus-cmd", window.execChatCommand);
  assert.equal(toastText(window), "Unknown command: /totally-bogus-cmd");
});

/* ================================================================ */
/*  A streaming reply refuses every chat command                     */
/* ================================================================ */

function pressEnter(win, ta) {
  ta.dispatchEvent(new win.KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true }));
}

for (const typed of ["/clear", "/new"]) {
  test(`${typed} picked from the slash menu while a reply streams changes nothing`, () => {
    const { window } = loadApp();
    window.newConversation();
    runScript(window, `currentConv().messages.push({ role: "user", content: "question" });
      chat.busy = true;`);
    const before = window.currentConv();
    const ta = typeSlash(window, "chat-input", typed);
    assert.equal(menuItems(window, "chat-input").length, 1, "the menu offers the command");
    pressEnter(window, ta);
    assert.equal(window.currentConv(), before, "the open chat is still the one that streams");
    assert.equal(before.messages.length, 1, "the question stays in the chat");
    assert.equal(toastText(window), "Wait for the current reply to finish");
  });
}

test("an alias (/imagine) is refused like its command while a reply streams", async () => {
  const calls = [];
  const fetchImpl = async (url) => {
    calls.push(String(url));
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl });
  window.newConversation();
  runScript(window, "chat.busy = true;");
  assert.equal(window.execChatCommand("imagine", "a cat"), true);
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(window.currentConv().messages.length, 0, "no turn was added");
  assert.equal(calls.filter((u) => u.includes("/api/imagine")).length, 0);
  assert.equal(toastText(window), "Wait for the current reply to finish");
});

test("an alias of an inactive plugin's command gets the plugin hint in the chat", () => {
  const calls = [];
  const fetchImpl = async (url) => {
    calls.push(String(url));
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl });
  runScript(window, `pluginCommands.map = { "generate-image": { plugin: "image", active: false } };
    pluginCommands.suggest = true;`);
  window.handleSlashSubmit("/imagine a cat", window.execChatCommand);
  assert.equal(toastText(window),
    "/generate-image needs the image plugin - install or enable it on the Plugins page.");
  assert.equal(calls.filter((u) => u.includes("/api/imagine")).length, 0, "nothing was posted");
});

test("the coder composer gets no chat plugin hint", () => {
  const { window } = loadApp();
  runScript(window, `pluginCommands.map = { "remember": { plugin: "memory", active: false } };
    pluginCommands.suggest = true;`);
  window.handleSlashSubmit("/remember x", (c, a) => window.execCoderCommand(c, a));
  assert.equal(toastText(window), "Unknown command: /remember");
});

test("a name that is not a chat command is still reported unknown while a reply streams", () => {
  const { window } = loadApp();
  runScript(window, "chat.busy = true;");
  assert.equal(window.execChatCommand("totally-bogus-cmd", ""), false);
});

test("/rename <title> with no chat open starts a chat with that title", () => {
  const { window } = loadApp();
  assert.ok(!window.currentConv(), "no chat is open");
  window.execChatCommand("rename", "Trip notes");
  const conv = window.currentConv();
  assert.ok(conv, "a chat was opened");
  assert.equal(conv.title, "Trip notes");
  assert.notEqual(toastText(window), "Usage: /rename <title>");
});

for (const [run, route] of [["runImagineInChat", "/api/imagine"],
                            ["runMusicInChat", "/api/music"],
                            ["runVideoInChat", "/api/video"]]) {
  test(`a failed ${run} shows its error once, in the chat, and no toast`, async () => {
    const fetchImpl = async (url) => {
      if (String(url) === route) {
        return { ok: false, status: 500, statusText: "Internal", json: async () => ({ detail: "boom" }) };
      }
      return { ok: false, status: 404, json: async () => ({}), text: async () => "" };
    };
    const { window } = loadApp({ fetchImpl });
    await window[run]("x");
    const text = window.document.getElementById("chat-messages").textContent;
    assert.match(text, /failed: boom/, "the chat shows the error");
    assert.equal(toastText(window), "", "no second copy of the error as a toast");
  });
}
