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
