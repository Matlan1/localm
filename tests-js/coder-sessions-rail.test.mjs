// SPDX-License-Identifier: AGPL-3.0-or-later
// DOM behaviour of the coder's right-side open-sessions rail.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function seedTwo(window, { busy2 = true } = {}) {
  runScript(window, `
    coder.sessions.set("aaa111", { info: { id: "aaa111", cwd: "/p/alpha" }, busy: false, feedEl: document.createElement("div") });
    coder.sessions.set("bbb222", { info: { id: "bbb222", cwd: "/p/beta" }, busy: ${busy2}, feedEl: document.createElement("div") });
    coder.activeId = "aaa111";
    renderCoderSessionList();
  `);
}

test("R17: the coder open-sessions rail lists sessions and marks the active one", () => {
  const { window } = loadApp();
  seedTwo(window);
  const items = [...window.document.querySelectorAll("#coder-session-list .coder-session-item")];
  assert.equal(items.length, 2, "both sessions are listed in the rail");
  assert.ok(items[0].classList.contains("active"), "the active session is highlighted");
  assert.ok(!items[1].classList.contains("active"), "the inactive session is not");
  assert.match(items[0].textContent, /alpha/, "labelled by its project directory");
  assert.ok(items[1].querySelector('.badge [data-icon-name="clock"]'),
    "a busy session shows the working badge");
});

test("R17: clicking a rail item activates that session", () => {
  const { window } = loadApp();
  runScript(window, "window.coderState = coder;");
  seedTwo(window, { busy2: false });
  assert.equal(window.coderState.activeId, "aaa111", "starts on the first session");
  const items = [...window.document.querySelectorAll("#coder-session-list .coder-session-item")];
  items[1].click();
  assert.equal(window.coderState.activeId, "bbb222", "clicking the rail item activates that session");
  // and the rail re-renders with the new active highlight
  const reItems = [...window.document.querySelectorAll("#coder-session-list .coder-session-item")];
  assert.ok(reItems.find((i) => /beta/.test(i.textContent)).classList.contains("active"),
    "the newly-activated session is highlighted");
});

test("R17: the rail shows an empty state when there are no sessions at all", () => {
  const { window } = loadApp();
  runScript(window, "coder.sessions.clear(); renderCoderSessionList();");
  const list = window.document.getElementById("coder-session-list");
  // The rail lists both open and past sessions, so the empty state covers both.
  assert.match(list.textContent, /No sessions yet/);
});
