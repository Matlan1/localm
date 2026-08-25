// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function summFetch(summary) {
  const calls = [];
  const impl = async (url, opts = {}) => {
    let body = null;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch (e) { body = null; }
    calls.push({ url: String(url), body });
    if (String(url) === "/v1/chat/completions") {
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ choices: [{ message: { content: summary } }] }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  return { impl, calls };
}

function makeConv(n) {
  const msgs = [];
  for (let i = 0; i < n; i++) {
    msgs.push({ role: i % 2 === 0 ? "user" : "assistant",
                content: ("msg-" + i).padEnd(16, ".") });
  }
  return { id: "c1", title: "t", messages: msgs };
}

test("R44: estimateTokens adds a per-message overhead over a bare chars/4", () => {
  const { window: w } = loadApp();
  assert.equal(w.estimateTokens(""), 4);
  assert.equal(w.estimateTokens("a".repeat(40)), 14);   // 40/4 + 4
});

test("R44: truncateAtWord cuts on a word boundary with a marker (no half-words)", () => {
  const { window: w } = loadApp();
  assert.equal(w.truncateAtWord("hello world", 100), "hello world");
  const out = w.truncateAtWord("alpha beta gamma delta", 12);
  assert.match(out, /\.\.\.\[truncated\]$/);
  assert.doesNotMatch(out, /gam ?\.\.\.\[truncated\]$/, "never a mid-word cut");
});

test("R44: compaction keeps more than 4 recent turns and the tail is intact", async () => {
  const { impl } = summFetch("A clean summary of the earlier turns.");
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, "chat.ctxMax = 160;");   // budget ~80 tokens; each msg ~8 tokens
  const conv = makeConv(20);
  const tail = conv.messages.slice();
  const ok = await window.compactConversation(conv);
  assert.equal(ok, true);
  assert.equal(conv.messages[0].role, "user");
  assert.match(conv.messages[0].content, /\[Conversation summary\]/);
  const kept = conv.messages.length - 2;   // minus the 2-message bridge
  assert.ok(kept > 4, `keeps more than the floor of 4 (kept ${kept})`);
  assert.ok(kept < 20, "but still summarised some older turns");
  // conv.messages is rebuilt in the jsdom realm, so compare by value (JSON).
  assert.equal(JSON.stringify(conv.messages.slice(2).map((m) => m.content)),
    JSON.stringify(tail.slice(-kept).map((m) => m.content)), "the recent tail is verbatim");
});

test("R44: reasoning blocks are stripped from the summariser input", async () => {
  const { impl, calls } = summFetch("ok");
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, "chat.ctxMax = 4000;");
  const conv = makeConv(4);
  conv.messages.unshift({ role: "assistant",
    content: "<think>SECRET REASONING HERE</think>The visible answer." });
  await window.compactConversation(conv);
  const summReq = calls.find((c) => c.url === "/v1/chat/completions");
  const sent = summReq.body.messages[0].content;
  assert.doesNotMatch(sent, /SECRET REASONING/, "the <think> content is not summarised");
  assert.match(sent, /The visible answer/, "the visible content is summarised");
});

test("R44: a failed summary keeps the recent turns rather than nuking history", async () => {
  const { impl } = summFetch("");   // empty content = summarisation failed
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, "chat.ctxMax = 160;");
  const conv = makeConv(20);
  const tail = conv.messages.slice();
  await window.compactConversation(conv);
  const kept = conv.messages.length - 2;
  assert.ok(kept > 4, "recent turns are preserved, not collapsed to a one-line note");
  assert.equal(JSON.stringify(conv.messages.slice(2).map((m) => m.content)),
    JSON.stringify(tail.slice(-kept).map((m) => m.content)));
});


test("F5: compaction archives (not silently deletes) branches anchored in the summarised-away region", async () => {
  const { impl } = summFetch("A clean summary of the earlier turns.");
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, "chat.ctxMax = 160;");
  const conv = makeConv(20);
  // give every message a stable id, then park an alternative timeline anchored
  // at an old message that compaction summarises away
  conv.messages.forEach((m) => window.msgId(m));
  conv.branches = [{
    parent: conv.messages[1].id,
    tails: [
      [{ role: "user", content: "the kept live tail", id: "live-x" }],
      [{ role: "user", content: "an alternative timeline I explored", id: "alt-1" }],
    ],
    current: 0,
  }];
  const ok = await window.compactConversation(conv);
  assert.equal(ok, true);
  // the unreachable fork record is gone from navigation
  assert.equal((conv.branches || []).length, 0, "dangling fork record dropped");
  // its alternative content is archived
  const archived = JSON.stringify(conv.droppedBranches || []);
  assert.match(archived, /an alternative timeline I explored/,
    "the summarised-away branch content was archived for recovery");
});

test("F5: pruneBranches returns 0 and archives nothing when every fork still reachable", () => {
  const { window: w } = loadApp();
  const conv = { id: "c", messages: [{ role: "user", content: "a", id: "m1" }] };
  conv.branches = [{ parent: "root", tails: [[{ role: "user", content: "x", id: "m1" }]], current: 0 }];
  const lost = w.pruneBranches(conv);
  assert.equal(lost, 0);
  assert.equal(conv.droppedBranches, undefined);
});
