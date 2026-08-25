// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the command palette (Ctrl/Cmd+K) in app.js: the palette
// shows and hides, typing filters the list, and running a "Go to <view>"
// command activates that view.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

function goodFetch() {
  return async (url) => {
    const u = String(url);
    if (u.includes("/api/models"))
      return { ok: true, status: 200, json: async () => ({ models: [], active: "" }) };
    if (u.includes("/api/plugins"))
      return { ok: true, status: 200, json: async () => ({ plugins: [] }) };
    return { ok: true, status: 200, json: async () => ({}) };
  };
}

function key(win, k, opts = {}) {
  const ev = new win.KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true, ...opts });
  win.document.dispatchEvent(ev);
  return ev;
}

const isOpen = (win) => win.document.getElementById("cmdk").style.display !== "none";

test("Ctrl+K opens the palette and Escape closes it", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  assert.equal(isOpen(window), false, "palette starts closed");
  key(window, "k", { ctrlKey: true });
  assert.equal(isOpen(window), true, "Ctrl+K opens the palette");
  const items = window.document.querySelectorAll("#cmdk-list .cmdk-item");
  assert.ok(items.length >= 4, "palette lists commands (views + actions)");
  key(window, "Escape");
  assert.equal(isOpen(window), false, "Escape closes the palette");
});

test("Cmd+K (meta) also toggles the palette", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  key(window, "k", { metaKey: true });
  assert.equal(isOpen(window), true, "Cmd+K opens");
  key(window, "k", { metaKey: true });
  assert.equal(isOpen(window), false, "Cmd+K again closes");
});

test("typing filters the command list", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  key(window, "k", { ctrlKey: true });
  const input = window.document.getElementById("cmdk-input");
  input.value = "settings";
  input.dispatchEvent(new window.Event("input", { bubbles: true }));
  const labels = [...window.document.querySelectorAll("#cmdk-list .cmdk-item")]
    .map((n) => n.textContent.toLowerCase());
  assert.ok(labels.length >= 1, "at least one match for 'settings'");
  assert.ok(labels.every((l) => l.includes("settings")), "only matching commands remain");
});

test("running 'Go to Settings' activates the settings view", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  key(window, "k", { ctrlKey: true });
  const input = window.document.getElementById("cmdk-input");
  input.value = "settings";
  input.dispatchEvent(new window.Event("input", { bubbles: true }));
  key(window, "Enter");
  assert.equal(isOpen(window), false, "running a command closes the palette");
  assert.ok(window.document.getElementById("view-settings").classList.contains("active"),
    "the settings view is now active");
});

test("filter that matches nothing leaves an empty list (negative)", () => {
  const { window } = loadApp({ fetchImpl: goodFetch() });
  key(window, "k", { ctrlKey: true });
  const input = window.document.getElementById("cmdk-input");
  input.value = "zzzznotacommand";
  input.dispatchEvent(new window.Event("input", { bubbles: true }));
  assert.equal(window.document.querySelectorAll("#cmdk-list .cmdk-item").length, 0,
    "no items match a nonsense query");
});
