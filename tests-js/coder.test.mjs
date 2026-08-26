// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the coder GUI region of app.js.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const settle = () => new Promise((r) => setTimeout(r, 0));

function okFetch(state = {}) {
  return async (url) => {
    const u = String(url);
    if (u.includes("/api/coder/resumable")) {
      return { ok: true, status: 200,
               json: async () => state.unreadable
                 ? { resumable: false, unreadable: true }
                 : state.resumable
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
  // seed the top-level `coder` state (a const, not on window) from the realm
  runScript(window, `
    coder.activeId = null;
    coder.sessions.set("sid1", { info: { id: "sid1", cwd: "Z:/proj", model: "m",
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
  window.document.getElementById("setup-cwd").value = "Z:/proj";

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

test("CODER-2: an unreadable checkpoint toasts instead of reading as 'nothing to resume'",
  async () => {
    const state = { unreadable: true };
    const { window } = loadApp({ fetchImpl: okFetch(state) });
    const toasts = [];
    window.toast = (msg) => toasts.push(String(msg));
    window.document.getElementById("setup-cwd").value = "Z:/proj";

    await window.refreshResumable();
    await settle();
    const btn = window.document.querySelector(".coder-continue");
    assert.equal(btn.style.display, "none", "nothing to actually resume");
    assert.ok(toasts.some((t) => t.includes("could not be read")),
      `must toast that a checkpoint was found but unreadable, got: ${JSON.stringify(toasts)}`);
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

test("AUD-HIGH-17-3: reasoning events render a collapsible think-block separate from the answer", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  const feedEl = window.document.createElement("div");
  const s = { info: { id: "x" }, feedEl, liveBody: null, liveText: "",
             liveReasoning: "", pendingCards: [] };

  window.handleCoderEvent(s, { type: "reasoning", text: "because " });
  window.handleCoderEvent(s, { type: "reasoning", text: "reasons" });
  window.handleCoderEvent(s, { type: "token", text: "The " });
  window.handleCoderEvent(s, { type: "token", text: "answer." });

  const det = feedEl.querySelector("details.think-block");
  assert.ok(det, "reasoning rendered a collapsible think-block");
  assert.match(det.querySelector("div").textContent, /because reasons/);
  const main = feedEl.querySelector(".md-main");
  assert.match(main.textContent, /The answer\./);
  // the visible answer body never contains the reasoning text
  assert.doesNotMatch(main.textContent, /because reasons/);
});

test("AUD-HIGH-17-3: reasoning events are excluded from the light event log (no spam)", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  const feedEl = window.document.createElement("div");
  const s = { info: { id: "x" }, feedEl, liveBody: null, liveText: "",
             liveReasoning: "", pendingCards: [] };

  window.handleCoderEvent(s, { type: "reasoning", text: "thinking..." });
  window.handleCoderEvent(s, { type: "token", text: "hi" });

  assert.deepEqual(s.eventLog || [], [], "neither token nor reasoning enters eventLog");
});

test("CODER-EMPTY-MODEL: a tool-only assistant turn leaves no empty Model row", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  const feedEl = window.document.createElement("div");
  const s = { info: { id: "x" }, feedEl, liveBody: null, liveText: "", pendingCards: [] };
  // the model streams only whitespace, then a tool call
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

test("CODER-EPISODES: recalled lessons render with the id needed to forget them", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  const feedEl = window.document.createElement("div");
  const s = { info: { id: "z" }, feedEl, liveBody: null, liveText: "", pendingCards: [] };
  window.handleCoderEvent(s, {
    type: "episodes_recalled",
    episodes: [{ id: "ab12cd34ef56", outcome: "ok",
                 lesson: "cap the upload timeout at 30s" }],
  });
  const row = feedEl.querySelector(".feed-info");
  assert.ok(row, "a feed row was rendered");
  assert.match(row.textContent, /Recalled 1 past lesson/);
  assert.match(row.textContent, /cap the upload timeout at 30s/);
  assert.match(row.textContent, /ab12cd34ef56/, "the id is shown so it can be forgotten");
});

test("CODER-EPISODES: an empty recall renders nothing (silence when irrelevant)", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  const feedEl = window.document.createElement("div");
  const s = { info: { id: "z" }, feedEl, liveBody: null, liveText: "", pendingCards: [] };
  window.handleCoderEvent(s, { type: "episodes_recalled", episodes: [] });
  assert.equal(feedEl.querySelector(".feed-info"), null, "no row for an empty recall");
});

function _rejectionSequence(window, s) {
  window.handleCoderEvent(s, { type: "tool_call", tool: "run_shell", args: { command: "rm -rf /" } });
  window.handleCoderEvent(s, { type: "confirm_request", confirm_id: "c1", tool: "run_shell", args: {} });
  window.handleCoderEvent(s, { type: "confirm_resolved", confirm_id: "c1", approved: false, timed_out: false });
  window.handleCoderEvent(s, { type: "tool_result", tool: "run_shell", ok: false, summary: "rejected by user" });
}

test("rejected-2-shell: a rejected call shows ONE card, not two", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  const feedEl = window.document.createElement("div");
  const s = { info: { id: "z" }, feedEl, liveBody: null, liveText: "",
             pendingCards: [], confirmCards: new Map() };
  _rejectionSequence(window, s);

  assert.equal(feedEl.querySelectorAll(".tool-card").length, 0,
    "the tool_call card has nothing left to show once the confirm card narrates the rejection");
  const confirmCards = feedEl.querySelectorAll(".confirm-card");
  assert.equal(confirmCards.length, 1, "the confirm card stays - it is the one useful record");
  assert.match(confirmCards[0].textContent, /Rejected run_shell/);
});

test("rejected-2-shell CONTROL: an APPROVED call keeps both cards (output still matters)", () => {
  const { window } = loadApp({ fetchImpl: okFetch() });
  const feedEl = window.document.createElement("div");
  const s = { info: { id: "z" }, feedEl, liveBody: null, liveText: "",
             pendingCards: [], confirmCards: new Map() };
  window.handleCoderEvent(s, { type: "tool_call", tool: "run_shell", args: { command: "ls" } });
  window.handleCoderEvent(s, { type: "confirm_request", confirm_id: "c2", tool: "run_shell", args: {} });
  window.handleCoderEvent(s, { type: "confirm_resolved", confirm_id: "c2", approved: true, timed_out: false });
  window.handleCoderEvent(s, { type: "tool_result", tool: "run_shell", ok: true, summary: "ok",
                              output: "file1.txt\nfile2.txt" });

  assert.equal(feedEl.querySelectorAll(".tool-card").length, 1,
    "an approved call's real output is not redundant with the confirm checkmark");
  assert.match(feedEl.querySelector(".tool-card").textContent, /file1\.txt/);
  assert.equal(feedEl.querySelectorAll(".confirm-card").length, 1);
  assert.match(feedEl.querySelector(".confirm-card").textContent, /Approved run_shell/);
});
