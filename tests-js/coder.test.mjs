// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the coder GUI region of app.js:
//  - CODER-3: renderSessionSelect lists live sessions (the host must see the
//    selector populated, like mobile does).
//  - CODER-2: the dynamically-created "Continue last session" button appears when
//    the chosen directory has a resumable checkpoint, and a resumed session's
//    "history" recap events render as message rows.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const settle = () => new Promise((r) => setTimeout(r, 0));

function okFetch(state = {}) {
  return async (url) => {
    const u = String(url);
    if (u.includes("/api/coder/resumable")) {
      return { ok: true, status: 200,
               json: async () => state.resumable
                 ? { resumable: true, turns: 3, messages: 5,
                     interrupted_at: "2026-06-22T10:00:00" }
                 : { resumable: false } };
    }
    if (u.includes("/api/models"))
      return { ok: true, status: 200, json: async () => ({ models: [], active: "" }) };
    return { ok: true, status: 200, json: async () => ({}) };
  };
}

test("CODER-3: renderSessionSelect lists each live session as an option", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  assert.equal(typeof window.renderSessionSelect, "function");
  // Seed the top-level `coder` state (a const, not on window) from the realm.
  runScript(window, `
    coder.activeId = null;
    coder.sessions.set("sid1", { info: { id: "sid1", cwd: "C:/proj", model: "m",
      mode: "privacy", busy: false, turns: 0, total_tokens: 0, created_at: 0 },
      busy: false });
    renderSessionSelect();
  `);
  const sel = window.document.getElementById("session-select");
  const opt = sel.querySelector('option[value="sid1"]');
  assert.ok(opt, "the live session appears in the selector");
});

test("CODER-2: 'Continue last session' shows when the cwd has a checkpoint", async () => {
  const state = { resumable: true };
  const { window } = loadApp({ fetchImpl: okFetch(state) });
  assert.equal(typeof window.refreshResumable, "function");
  window.document.getElementById("setup-cwd").value = "C:/proj";

  await window.refreshResumable();
  await settle();
  const btn = window.document.querySelector(".coder-continue");
  assert.ok(btn, "the continue button was created");
  assert.equal(btn.style.display, "", "visible when resumable");
  assert.match(btn.textContent, /Continue last session/);
  assert.match(btn.textContent, /3 turns/);

  // No checkpoint -> hidden.
  state.resumable = false;
  await window.refreshResumable();
  await settle();
  assert.equal(btn.style.display, "none", "hidden when nothing to resume");

  // Empty cwd -> hidden, no request needed.
  window.document.getElementById("setup-cwd").value = "";
  await window.refreshResumable();
  assert.equal(btn.style.display, "none");
});

test("CODER-2: a resumed session's history events render as message rows", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  const feedEl = window.document.createElement("div");
  const s = { info: { id: "x" }, feedEl, liveBody: null, liveText: "" };
  window.handleCoderEvent(s, { type: "history", role: "user", text: "build a calc" });
  window.handleCoderEvent(s, { type: "history", role: "assistant", text: "here is the plan" });
  assert.match(feedEl.textContent, /build a calc/);
  assert.match(feedEl.textContent, /here is the plan/);
});

test("CODER-EMPTY-MODEL: a tool-only assistant turn leaves no empty Model row", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  const feedEl = window.document.createElement("div");
  const s = { info: { id: "x" }, feedEl, liveBody: null, liveText: "", pendingCards: [] };
  // The model streams only whitespace (its visible text scrubs to nothing), then a tool call.
  window.handleCoderEvent(s, { type: "token", text: "   " });
  assert.ok(feedEl.querySelector(".msg-row.assistant"), "the token started a Model row");
  window.handleCoderEvent(s, { type: "tool_call", tool: "grep", args: { pattern: "x" } });
  const emptyRows = [...feedEl.querySelectorAll(".msg-row.assistant")]
    .filter((r) => !(r.querySelector(".msg-body")?.textContent || "").trim());
  assert.equal(emptyRows.length, 0, "the empty Model row is dropped (no blank bubble above the tool card)");
  assert.ok(feedEl.querySelector(".tool-card"), "the tool call still renders its card");
});

test("CODER-EMPTY-MODEL: an assistant turn WITH text keeps its Model row", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  const feedEl = window.document.createElement("div");
  const s = { info: { id: "y" }, feedEl, liveBody: null, liveText: "", pendingCards: [] };
  window.handleCoderEvent(s, { type: "token", text: "Here is my analysis." });
  window.handleCoderEvent(s, { type: "tool_call", tool: "grep", args: { pattern: "x" } });
  assert.match(feedEl.textContent, /Here is my analysis/, "a real text turn is preserved");
});
