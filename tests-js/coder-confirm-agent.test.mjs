// SPDX-License-Identifier: AGPL-3.0-or-later
// Sub-agent attribution on a coder approval card: a confirm_request carrying an
// `agent` label renders an attribution line, one without it renders none.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const session = () => ({ info: { id: "sid1" }, confirmCards: new Map() });

const request = (extra = {}) => ({
  type: "confirm_request", confirm_id: "c1", tool: "run_shell",
  args: { cmd: "rm -rf build" }, diff: null, ...extra,
});

test("a child's approval card names the sub-agent that is asking", () => {
  const { window } = loadApp();
  const card = window.buildConfirmCard(session(), request({ agent: "child1" }));
  const asker = card.querySelector(".asker");
  assert.ok(asker, "no attribution line on a sub-agent's approval card");
  assert.equal(asker.textContent, "sub-agent child1 is asking");
  assert.equal(asker.querySelector(".name").textContent, "child1");
  // The card still names the tool being approved.
  assert.equal(card.querySelector(".title .name").textContent, "run_shell");
});

test("FIRES-CONTROL: the session's own prompt carries no attribution line", () => {
  const { window } = loadApp();
  // Same event minus `agent`.
  const card = window.buildConfirmCard(session(), request());
  assert.equal(card.querySelector(".asker"), null,
               "the user's own action was attributed to a sub-agent");
  assert.equal(card.querySelector(".title .name").textContent, "run_shell");
});

test("two children's cards are told apart by their labels", () => {
  const { window } = loadApp();
  const s = session();
  const first = window.buildConfirmCard(s, request({ agent: "child1" }));
  const second = window.buildConfirmCard(
    s, request({ confirm_id: "c2", agent: "child2" }));
  assert.notEqual(first.querySelector(".asker").textContent,
                  second.querySelector(".asker").textContent);
  assert.equal(second.querySelector(".asker .name").textContent, "child2");
});

test("a label is rendered as text, never as markup", () => {
  const { window } = loadApp();
  const card = window.buildConfirmCard(
    session(), request({ agent: "<img src=x onerror=alert(1)>" }));
  const asker = card.querySelector(".asker");
  assert.equal(asker.querySelector("img"), null, "label was injected as HTML");
  assert.equal(asker.querySelector(".name").textContent,
               "<img src=x onerror=alert(1)>");
});

test("the confirm_request event still builds a card through the feed handler", () => {
  const { window } = loadApp();
  const s = session();
  s.feedEl = window.document.createElement("div");
  s.pendingCards = [];
  s.eventLog = [];
  window.handleCoderEvent(s, request({ agent: "child1" }));
  assert.equal(s.feedEl.querySelectorAll(".confirm-card .asker").length, 1);
});
