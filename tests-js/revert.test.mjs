// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

// Revert (issue #36): a DESTRUCTIVE undo of the dialogue up to the clicked
// message, dropping that message's text back into the composer, staying in the
// SAME branch (unlike edit, which forks). Reverting past a fork point destroys
// the sibling branches in the removed region, so it must confirm first.

const mk = (role, content, id) => ({ role, content, id });

function freshConv(window) {
  // chat.abort must be falsy for revert to proceed (it bails mid-stream).
  runScript(window, "chat.abort = null;");
}

function setComposer(window, v) { window.document.getElementById("chat-input").value = v; }
function composer(window) { return window.document.getElementById("chat-input").value; }

test("revert with no branches truncates, fills the composer, same branch", () => {
  const { window } = loadApp();
  freshConv(window);
  const conv = {
    id: "c1", title: "t", branches: [],
    messages: [mk("user", "q1", "m0"), mk("assistant", "a1", "m1"),
               mk("user", "q2", "m2"), mk("assistant", "a2", "m3")],
  };
  setComposer(window, "");
  window.revertTo(conv, 2);                       // revert to the 2nd user turn
  assert.deepEqual(conv.messages.map((m) => m.id), ["m0", "m1"], "dropped m2 + everything after");
  assert.equal(composer(window), "q2", "clicked message goes into the composer");
  assert.equal(conv.branches.length, 0, "no branches created (stays in the same branch)");
});

test("branchesLostByRevert counts sibling tails at/after the revert point", () => {
  const { window } = loadApp();
  const conv = {
    messages: [mk("user", "q1", "m0"), mk("assistant", "a1", "m1"),
               mk("user", "q2", "m2"), mk("assistant", "a2", "m3")],
    // a fork at index 2 (parent m1) with one parked alternative tail
    branches: [{ parent: "m1", tails: [[mk("user", "alt", "mA")], null], current: 1 }],
  };
  assert.equal(window.branchesLostByRevert(conv, 2), 1, "1 alternative lost at the fork");
  assert.equal(window.branchesLostByRevert(conv, 4), 0, "nothing after the end is lost");
});

test("reverting past a fork point asks for confirmation and only then destroys it", () => {
  const { window } = loadApp();
  freshConv(window);
  const conv = {
    id: "c1", title: "t",
    messages: [mk("user", "q1", "m0"), mk("assistant", "a1", "m1"),
               mk("user", "q2", "m2"), mk("assistant", "a2", "m3")],
    branches: [{ parent: "m1", tails: [[mk("user", "alt", "mA")], null], current: 1 }],
  };
  window.revertTo(conv, 2);
  // Not applied yet: the confirm modal is open and nothing is destroyed.
  assert.equal(conv.messages.length, 4, "conversation untouched until confirmed");
  assert.equal(window.document.getElementById("modal").style.display, "flex", "confirm modal shown");

  // Confirm -> destroy.
  const ok = window.document.querySelector("#modal-body .btn-danger");
  assert.ok(ok, "danger confirm button present");
  ok.click();
  assert.deepEqual(conv.messages.map((m) => m.id), ["m0", "m1"], "truncated after confirm");
  assert.equal(conv.branches.length, 0, "the fork at the revert point is destroyed");
  assert.equal(composer(window), "q2");
});

test("reverting AFTER an upstream fork keeps that fork (no confirm, branch preserved)", () => {
  const { window } = loadApp();
  freshConv(window);
  const conv = {
    id: "c1", title: "t",
    messages: [mk("user", "q1", "m0"), mk("assistant", "a1", "m1"),
               mk("user", "q2", "m2"), mk("assistant", "a2", "m3")],
    // fork diverges at index 1 (parent m0) - upstream of a revert to index 3
    branches: [{ parent: "m0", tails: [[mk("assistant", "alt", "mA")], null], current: 1 }],
  };
  assert.equal(window.branchesLostByRevert(conv, 3), 0, "upstream fork is not lost");
  window.revertTo(conv, 3);
  assert.deepEqual(conv.messages.map((m) => m.id), ["m0", "m1", "m2"]);
  assert.equal(conv.branches.length, 1, "upstream fork survives the revert");
});

test("revert is blocked while a reply is streaming", () => {
  const { window } = loadApp();
  runScript(window, "chat.abort = () => {};");     // streaming in progress
  const conv = { id: "c1", title: "t", branches: [],
                 messages: [mk("user", "q1", "m0"), mk("assistant", "a1", "m1")] };
  window.revertTo(conv, 0);
  assert.equal(conv.messages.length, 2, "no truncation mid-stream");
});
