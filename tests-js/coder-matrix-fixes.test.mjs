// SPDX-License-Identifier: AGPL-3.0-or-later
// Coder page: tool results never land on another call's card, write cards show
// the tool's own result text, a run that ended says why, the turn count has a
// singular, validation errors name their field, attached files stay with their
// session, a session ended elsewhere leaves the page, and the action buttons
// say what happened.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const settle = () => new Promise((r) => setTimeout(r, 0));

function session(window, id = "s1") {
  const feedEl = window.document.createElement("div");
  return { info: { id, cwd: "Z:/proj", mode: "log" }, feedEl, pendingCards: [],
           eventLog: [], confirmCards: new Map(), inlineBrowserAttempted: true };
}

function feedText(s) {
  return [...s.feedEl.querySelectorAll(".feed-info, .feed-final")].map((n) => n.textContent);
}

function fetchWith(routes) {
  return async (url) => {
    const u = String(url);
    for (const [part, reply] of Object.entries(routes)) {
      if (u.includes(part)) return reply();
    }
    return { ok: true, status: 200, json: async () => ({ models: [], active: "" }) };
  };
}

test("a result whose call id has no card gets its own line, not another card", () => {
  const { window } = loadApp();
  const s = session(window);
  window.handleCoderEvent(s, { type: "tool_call", tool: "run_shell", args: {}, call_id: "c1" });
  window.handleCoderEvent(s, { type: "tool_result", tool: "read_file", ok: false,
    summary: "not run: the run was stopped", call_id: "c9" });
  const state = s.feedEl.querySelector(".tool-card .state").textContent;
  assert.equal(state, "…", "the running card was not touched");
  assert.ok(feedText(s).some((t) => t === "read_file: not run: the run was stopped"),
    JSON.stringify(feedText(s)));
});

test("a write card keeps its diff and shows the tool's result text below it", () => {
  const { window } = loadApp();
  const s = session(window);
  window.handleCoderEvent(s, { type: "tool_call", tool: "write_file", call_id: "c1",
    args: { path: "a.py", content: "x" }, diff: "--- a/a.py\n+++ b/a.py\n+x\n" });
  window.handleCoderEvent(s, { type: "tool_result", tool: "write_file", ok: true,
    summary: "wrote a.py", output: "Wrote a.py\n\n[syntax check] Python syntax error: x",
    call_id: "c1" });
  const body = s.feedEl.querySelector(".tool-card .body");
  assert.ok(body.querySelector(".diff"), "the diff stays");
  assert.match(body.querySelector(".tool-output").textContent, /\[syntax check\]/);
});

test("a run stopped by max turns says so, and one turn reads as one turn", () => {
  const { window } = loadApp();
  const s = session(window);
  window.handleCoderEvent(s, { type: "token", text: "working on it" });
  window.handleCoderEvent(s, { type: "final", ok: false, text: "[max_turns=1 reached]",
    turns: 1, total_tokens: 10, changed_files: [] });
  const lines = feedText(s);
  assert.ok(lines.includes("[max_turns=1 reached]"), JSON.stringify(lines));
  const final = lines[lines.length - 1];
  assert.match(final, / - 1 turn, 10 tokens/);
  assert.doesNotMatch(final, /1 turns/);
});

test("a finished run whose text was streamed shows it only once", () => {
  const { window } = loadApp();
  const s = session(window);
  window.handleCoderEvent(s, { type: "token", text: "All done." });
  window.handleCoderEvent(s, { type: "final", ok: false, text: "All done.",
    turns: 2, total_tokens: 10, changed_files: [] });
  assert.ok(!feedText(s).includes("All done."));
});

test("a validation error list names each field", () => {
  const { window } = loadApp();
  const text = window._detailText({ detail: [
    { loc: ["body", "seed"], msg: "Input should be a valid integer" },
    { loc: ["body", "verify_max_retries"], msg: "Input should be less than or equal to 50" },
  ] });
  assert.equal(text, "seed: Input should be a valid integer; "
    + "verify_max_retries: Input should be less than or equal to 50");
  assert.equal(window._detailText({ detail: "plain" }), "plain");
});

test("attached files stay with the session they were attached in", () => {
  const { window } = loadApp();
  runScript(window, `
    const a = { info: { id: "a", cwd: "Z:/a" }, feedEl: document.createElement("div"),
                pendingCards: [], busy: false };
    const b = { info: { id: "b", cwd: "Z:/b" }, feedEl: document.createElement("div"),
                pendingCards: [], busy: false };
    coder.sessions.set("a", a); coder.sessions.set("b", b);
    activateSession("a");
    coder.docs.push({ name: "notes.txt", text: "secret", chars: 6, truncated: false });
    activateSession("b");
    window.__docsInB = coder.docs.length;
    activateSession("a");
    window.__docsInA = coder.docs.map((d) => d.name);
  `);
  assert.equal(window.__docsInB, 0);
  assert.deepEqual(window.__docsInA, ["notes.txt"]);
});

test("a session ended elsewhere is said once and removed from the page", () => {
  const { window } = loadApp();
  const toasts = [];
  window.toast = (msg) => toasts.push(String(msg));
  runScript(window, `
    const a = { info: { id: "a", cwd: "Z:/a" }, feedEl: document.createElement("div"),
                pendingCards: [], busy: false };
    coder.sessions.set("a", a);
    activateSession("a");
    handleCoderEvent(a, { type: "closed" });
    window.__left = coder.sessions.has("a");
  `);
  assert.equal(window.__left, false);
  assert.equal(toasts.filter((t) => t.includes("ended elsewhere")).length, 1, JSON.stringify(toasts));
});

test("undo with nothing to undo and an idle agent does not say it is busy", async () => {
  const { window } = loadApp({ fetchImpl: fetchWith({
    "/undo": () => ({ ok: false, status: 409, statusText: "Conflict",
      json: async () => ({ detail: "Nothing to undo (or agent is busy)" }) }),
  }) });
  const toasts = [];
  window.toast = (msg) => toasts.push(String(msg));
  runScript(window, `
    const a = { info: { id: "a", cwd: "Z:/a" }, feedEl: document.createElement("div"),
                pendingCards: [], busy: false };
    coder.sessions.set("a", a);
    activateSession("a");
  `);
  window.document.getElementById("coder-undo").onclick();
  await settle(); await settle();
  assert.deepEqual(toasts.filter((t) => /undo/i.test(t)), ["Nothing to undo"]);
});

test("stop with the server gone says so instead of failing silently", async () => {
  const { window } = loadApp({ fetchImpl: fetchWith({
    "/stop": () => { throw new TypeError("Failed to fetch"); },
  }) });
  const toasts = [];
  window.toast = (msg) => toasts.push(String(msg));
  runScript(window, `
    const a = { info: { id: "a", cwd: "Z:/a" }, feedEl: document.createElement("div"),
                pendingCards: [], busy: true };
    coder.sessions.set("a", a);
    activateSession("a");
  `);
  await window.document.getElementById("coder-stop").onclick();
  assert.ok(toasts.some((t) => t.startsWith("Could not reach the server: ")), JSON.stringify(toasts));
});

test("the setup form names a missing model instead of a question mark", () => {
  const { window } = loadApp();
  runScript(window, `modelCache.active = ""; populateSetupModels();`);
  const first = window.document.getElementById("setup-model").options[0].textContent;
  assert.equal(first, "active model (none loaded)");
});
