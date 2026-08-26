// SPDX-License-Identifier: AGPL-3.0-or-later
// A chat turn surfaces a "used N memories" chip built from the server's
// X-Localm-Memory response header. The chip shows the count and degrade reason,
// and opens the memory modal on click.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function headerResp(payload) {
  const raw = payload === undefined ? null : JSON.stringify(payload);
  return { headers: { get: (k) => (k === "X-Localm-Memory" ? raw : null) } };
}

test("F11: parseMemoryHeader parses the X-Localm-Memory header", () => {
  const { window } = loadApp({});
  const mem = window.parseMemoryHeader(headerResp({
    n: 2, degrade: "no_embedder",
    items: [{ id: "a", text: "User likes tea" }],
  }));
  assert.equal(mem.n, 2);
  assert.equal(mem.degrade, "no_embedder");
  assert.equal(mem.items[0].text, "User likes tea");
});

test("F11: parseMemoryHeader returns null when absent or unparseable", () => {
  const { window } = loadApp({});
  assert.equal(window.parseMemoryHeader(headerResp(undefined)), null);
  assert.equal(window.parseMemoryHeader({ headers: { get: () => "not json" } }), null);
  assert.equal(window.parseMemoryHeader(null), null);
  // n missing -> not a valid summary.
  assert.equal(window.parseMemoryHeader(headerResp({ degrade: "x" })), null);
});

test("F11: buildMemoryChip shows count + facts + degrade reason and opens the modal", () => {
  const { window } = loadApp({});
  const chip = window.buildMemoryChip({
    n: 2, degrade: "no_embedder",
    items: [{ id: "a", text: "User prefers metric units" }],
  });
  assert.equal(chip.textContent, "2");
  assert.ok(chip.querySelector('[data-icon-name="memory"]'), "shows the memory icon");
  assert.match(chip.title, /2 remembered facts/);
  assert.match(chip.title, /User prefers metric units/);
  assert.match(chip.title, /keyword match only/);   // degrade label surfaced

  // Clicking opens the memory modal.
  const modal = window.document.getElementById("modal");
  assert.equal(modal.style.display, "none");
  chip.onclick();
  assert.equal(modal.style.display, "flex");
  assert.match(window.document.getElementById("modal-title").textContent, /Memory/);
});

test("F11: addMessageRow renders the chip only when memory.n > 0", () => {
  const { window } = loadApp({});
  const doc = window.document;

  const boxN = doc.createElement("div");
  window.addMessageRow(boxN, "assistant", "hi", { memory: { n: 3, items: [], degrade: null } });
  assert.equal(boxN.querySelectorAll("button.mem-chip").length, 1);
  assert.equal(boxN.querySelector("button.mem-chip").textContent, "3");

  const boxZero = doc.createElement("div");
  window.addMessageRow(boxZero, "assistant", "hi", { memory: { n: 0, items: [] } });
  assert.equal(boxZero.querySelectorAll("button.mem-chip").length, 0);

  const boxNone = doc.createElement("div");
  window.addMessageRow(boxNone, "assistant", "hi", {});
  assert.equal(boxNone.querySelectorAll("button.mem-chip").length, 0);
});

test("F11: renderChat surfaces the chip from a persisted assistant message", () => {
  const { window } = loadApp({});
  runScript(window, `
    chat.conversations = [{ id: "c1", branches: [], messages: [
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi", model: "m",
        memory: { n: 1, degrade: null, items: [{ id: "a", text: "User likes tea" }] } },
    ] }];
    chat.activeId = "c1";
    renderChat();
  `);
  const chips = window.document.querySelectorAll("#chat-messages button.mem-chip");
  assert.equal(chips.length, 1);
  assert.equal(chips[0].textContent, "1");
});
