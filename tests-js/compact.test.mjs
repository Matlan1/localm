// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

function summFetch(summary, finishReason = "stop") {
  const calls = [];
  const impl = async (url, opts = {}) => {
    let body;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = null; }
    calls.push({ url: String(url), body });
    if (String(url) === "/v1/chat/completions") {
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ choices: [{ message: { content: summary },
                                          finish_reason: finishReason }] }) };
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

test("the summarise request's one message is marked origin \"client\", not the user's words", async () => {
  // The server leaves a row the client wrote itself out of the memory recall
  // query and the audit log's user line; unmarked, the summarise prompt plus
  // the excerpt of older turns would be recalled for and logged as if typed.
  const { impl, calls } = summFetch("ok");
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, "chat.ctxMax = 160;");
  await window.compactConversation(makeConv(20));
  const summReq = calls.find((c) => c.url === "/v1/chat/completions");
  assert.ok(summReq, "compaction asked the server for a summary");
  assert.equal(summReq.body.messages.length, 1);
  const [msg] = summReq.body.messages;
  assert.equal(msg.role, "user");
  assert.match(msg.content, /^Summarise the following conversation/);
  assert.equal(msg.origin, "client");
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


test("F-02: an HTTP-200 inference error is not accepted as a summary", async () => {
  // The server reports a failed summarisation as 200 + error content +
  // finish_reason "error"; that text must never replace the older turns.
  const { impl } = summFetch("[inference error: decode failed]", "error");
  const { window } = loadApp({ fetchImpl: impl });
  const toasts = [];
  window.toast = (msg) => toasts.push(String(msg));
  runScript(window, "chat.ctxMax = 160;");
  const conv = makeConv(20);
  const original = conv.messages.map((m) => m.content);
  const ok = await window.compactConversation(conv);
  assert.equal(ok, true, "compaction still ran (hard-trim fallback)");
  const all = JSON.stringify(conv.messages.map((m) => m.content));
  assert.doesNotMatch(all, /inference error/, "the error text is not in the transcript");
  assert.doesNotMatch(all, /\[Conversation summary\]/, "no summary bridge was claimed");
  assert.match(conv.messages[0].content, /trimmed to fit the context window/,
    "the documented hard-trim bridge was used instead");
  assert.ok(toasts.some((t) => /trimmed/.test(t)), "the user is told it was trimmed");
  assert.ok(!toasts.some((t) => /summarised/.test(t)), "no summarised-success message");
  const archived = window.compactedTurns(conv.messages).map((m) => m.content);
  const kept = conv.messages.length - 2;
  assert.equal(JSON.stringify(archived), JSON.stringify(original.slice(0, 20 - kept)),
    "every removed original turn is archived, in order");
});

test("F-02: a finish_reason other than stop (length) is not accepted as a summary", async () => {
  const { impl } = summFetch("A summary that was cut off mid", "length");
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, "chat.ctxMax = 160;");
  const conv = makeConv(20);
  await window.compactConversation(conv);
  assert.doesNotMatch(conv.messages[0].content, /\[Conversation summary\]/);
  assert.match(conv.messages[0].content, /trimmed to fit/);
});

test("F-02: a successful summary still takes the normal path and archives the originals", async () => {
  const { impl } = summFetch("A clean summary of the earlier turns.");
  const { window } = loadApp({ fetchImpl: impl });
  const toasts = [];
  window.toast = (msg) => toasts.push(String(msg));
  runScript(window, "chat.ctxMax = 160;");
  const conv = makeConv(20);
  const original = conv.messages.map((m) => m.content);
  await window.compactConversation(conv);
  assert.match(conv.messages[0].content, /\[Conversation summary\]\nA clean summary/);
  assert.ok(toasts.some((t) => /summarised/.test(t)), "success wording on a real summary");
  const kept = conv.messages.length - 2;
  const archived = window.compactedTurns(conv.messages).map((m) => m.content);
  assert.equal(JSON.stringify(archived), JSON.stringify(original.slice(0, 20 - kept)));
});

test("F-02: a second compaction nests the first bridge; compactedTurns still lists every turn once, oldest first", async () => {
  const { impl } = summFetch("Summary.");
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, "chat.ctxMax = 160;");
  const conv = makeConv(20);
  const original = conv.messages.map((m) => m.content);
  await window.compactConversation(conv);
  const firstKept = conv.messages.length - 2;
  // grow the tail again so a second compaction has something to remove
  for (let i = 20; i < 40; i++) {
    conv.messages.push({ role: i % 2 === 0 ? "user" : "assistant",
                         content: ("msg-" + i).padEnd(16, ".") });
    original.push(("msg-" + i).padEnd(16, "."));
  }
  await window.compactConversation(conv);
  const archived = window.compactedTurns(conv.messages).map((m) => m.content);
  // the archive holds the originals removed by BOTH passes (the first bridge
  // itself is synthetic and is not listed), each exactly once, oldest first
  const removedByFirst = original.slice(0, 20 - firstKept);
  assert.equal(JSON.stringify(archived.slice(0, removedByFirst.length)),
    JSON.stringify(removedByFirst), "first-pass originals come first");
  assert.equal(new Set(archived).size, archived.length, "no turn is archived twice");
  assert.ok(!archived.some((c) => /\[Conversation summary\]/.test(c)),
    "the synthetic bridge is not listed as an archived turn");
  assert.ok(!archived.some((c) => /^Understood\./.test(c)),
    "the bridge's synthetic assistant half is not listed either");
  assert.equal(archived.length, 40 - (conv.messages.length - 2),
    "the archive holds exactly the real turns both passes removed");
  const survivors = conv.messages.slice(2).map((m) => m.content);
  assert.ok(archived.every((c) => !survivors.includes(c)), "archived turns are the removed ones");
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
