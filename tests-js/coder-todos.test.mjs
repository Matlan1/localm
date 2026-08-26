// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the task-list surfacing in coder.js: a set_todos tool card
// shows the model's plan progress on its collapsed head line (todoHint), and the
// tool_result fills the card body with the rendered checklist.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const PLAN = ["[x] read the failing test", "[>] fix the parser", "[ ] run the suite"];

test("todoHint: progress count plus the in-progress task", () => {
  const { window } = loadApp();
  assert.equal(window.todoHint(PLAN), "1/3 done · fix the parser");
});

test("todoHint: no in-progress item leaves just the count", () => {
  const { window } = loadApp();
  assert.equal(window.todoHint(["[x] a", "[x] b"]), "2/2 done");
  assert.equal(window.todoHint(["[ ] a", "[ ] b", "[ ] c"]), "0/3 done");
});

test("todoHint: tolerates the marker variants the parser accepts", () => {
  const { window } = loadApp();
  // [X]/[v] count as done and [*]/[~] as in progress, matching tasks.py.
  assert.equal(window.todoHint(["[X] a", "[*] b"]), "1/2 done · b");
  assert.equal(window.todoHint([{ text: "[~] dict form" }]), "0/1 done · dict form");
});

test("todoHint: a task whose own text is bracketed keeps it", () => {
  const { window } = loadApp();
  // Only the leading status marker is stripped, matching tasks.py.
  assert.equal(window.todoHint(["[>] [api] fix the handler", "[ ] test it"]),
               "0/2 done · [api] fix the handler");
  assert.equal(window.todoHint(["[api] fix the handler"]), "0/1 done");
});

test("todoHint: returns nothing for a non-todo tool's args", () => {
  const { window } = loadApp();
  for (const v of [undefined, null, [], "not an array", 42, {}]) {
    assert.equal(window.todoHint(v), "", `expected "" for ${JSON.stringify(v)}`);
  }
});

test("a set_todos tool card shows the plan on its head line", () => {
  const { window } = loadApp();
  const card = window.buildToolCard({ tool: "set_todos", args: { items: PLAN } });
  assert.equal(card.querySelector(".name").textContent, "set_todos");
  assert.equal(card.querySelector(".hint").textContent, "1/3 done · fix the parser");
  // The full plan is still in the body.
  assert.ok(card.querySelector(".body").textContent.includes("[>] fix the parser"));
});

test("the hint never displaces a file/command hint on other tools", () => {
  const { window } = loadApp();
  const write = window.buildToolCard({
    tool: "write_file", args: { path: "src/main.py", content: "x" } });
  assert.equal(write.querySelector(".hint").textContent, "src/main.py");
  const shell = window.buildToolCard({
    tool: "run_shell", args: { command: "pytest -q" } });
  assert.equal(shell.querySelector(".hint").textContent, "pytest -q");
});
