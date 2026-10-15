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
  assert.equal(ok, true, "compaction still ran (digest fallback)");
  const all = JSON.stringify(conv.messages.map((m) => m.content));
  assert.doesNotMatch(all, /inference error/, "the error text is not in the transcript");
  assert.doesNotMatch(all, /\[Conversation summary\]/, "no summary bridge was claimed");
  assert.match(conv.messages[0].content, /condensed to fit the context window/,
    "the digest bridge was used instead");
  assert.ok(toasts.some((t) => /trimmed/.test(t)), "the user is told it was trimmed");
  assert.ok(!toasts.some((t) => /summarised/.test(t)), "no summarised-success message");
  const archived = window.compactedTurns(conv.messages).map((m) => m.content);
  const kept = conv.messages.length - 2;
  assert.equal(JSON.stringify(archived), JSON.stringify(original.slice(0, 20 - kept)),
    "every removed original turn is archived, in order");
});

test("a length-cut reply with visible text is accepted as the summary", async () => {
  const { impl } = summFetch("A summary that was cut off mid", "length");
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, "chat.ctxMax = 160;");
  const conv = makeConv(20);
  await window.compactConversation(conv);
  assert.match(conv.messages[0].content, /\[Conversation summary\]\nA summary that was cut off mid/);
});

test("a reasoning-only reply keeps a digest of the removed turns, not a bare note", async () => {
  const { impl } = summFetch("<think>Thinking Process: drafting the", "length");
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, "chat.ctxMax = 160;");
  const conv = makeConv(20);
  const original = conv.messages.map((m) => m.content);
  await window.compactConversation(conv);
  const bridge = conv.messages[0].content;
  assert.doesNotMatch(bridge, /\[Conversation summary\]/);
  assert.doesNotMatch(bridge, /Thinking Process/);
  assert.match(bridge, /condensed to fit the context window/);
  const kept = conv.messages.length - 2;
  for (const c of original.slice(0, 20 - kept)) {
    assert.ok(bridge.includes(c.trim()), `removed turn ${c} is in the digest`);
  }
});

test("the summarise request asks for no reasoning and a reply budget sized to the window", async () => {
  const { impl, calls } = summFetch("ok");
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, "chat.ctxMax = 160;");
  await window.compactConversation(makeConv(20));
  const summReq = calls.find((c) => c.url === "/v1/chat/completions");
  assert.equal(summReq.body.max_tokens, 64);
  assert.equal(JSON.stringify(summReq.body.chat_template_kwargs),
    JSON.stringify({ enable_thinking: false }));
});

test("the kept tail starts at a user turn and the pending request survives", async () => {
  for (const n of [9, 10, 11, 20, 21]) {
    for (const ctx of [40, 80, 160, 400]) {
      const { impl } = summFetch("Summary.");
      const { window } = loadApp({ fetchImpl: impl });
      runScript(window, `chat.ctxMax = ${ctx};`);
      const conv = makeConv(n);
      const last = conv.messages[n - 1].content;
      await window.compactConversation(conv);
      const roles = conv.messages.map((m) => m.role);
      for (let i = 1; i < roles.length; i++) {
        assert.notEqual(roles[i], roles[i - 1], `n=${n} ctx=${ctx}: ${roles.join(",")}`);
      }
      assert.equal(conv.messages[conv.messages.length - 1].content, last);
    }
  }
});

test("compactionCut skips tool events and assistant turns", () => {
  const { window: w } = loadApp();
  const msgs = [
    { role: "user", content: "a" },
    { role: "assistant", content: "b" },
    { role: "user", content: "c" },
    { kind: "tool", role: "user", content: "web" },
    { role: "assistant", content: "d" },
  ];
  assert.equal(w.isToolEvent(msgs[3]), true);
  assert.equal(w.compactionCut(msgs, 4), 2);
  assert.equal(w.compactionCut(msgs, 1), 2);
  assert.equal(w.compactionCut([{ role: "user", content: "a" }, { role: "assistant", content: "b" }], 1), 0);
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

test("compaction shows a Compacting status while the summary is written, then removes it", async () => {
  let seen = null;
  let win = null;
  const impl = async (url) => {
    if (String(url) === "/v1/chat/completions") {
      const ind = win.document.querySelector("#chat-messages .msg-status-indicator");
      seen = ind ? ind.textContent : "";
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ choices: [{ message: { content: "Summary." },
                                          finish_reason: "stop" }] }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl: impl });
  win = window;
  runScript(window, "chat.ctxMax = 160;");
  await window.compactConversation(makeConv(20));
  assert.match(seen || "", /Compacting conversation/);
  assert.equal(window.document.querySelector("#chat-messages .msg-status-indicator"), null);
});

test("a Stopped compaction removes its Compacting status row", async () => {
  const ac = new AbortController();
  const impl = async (url) => {
    if (String(url) === "/v1/chat/completions") {
      ac.abort();
      throw new Error("aborted");
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, "chat.ctxMax = 160;");
  const ok = await window.compactConversation(makeConv(20), ac.signal);
  assert.equal(ok, false);
  assert.equal(window.document.querySelector("#chat-messages .msg-status-indicator"), null);
  assert.equal(window.document.querySelectorAll("#chat-messages .msg-row").length, 0);
});

const TOMATO = "my tomato variety is called Purple Cherokee and I bought the seeds in Lisbon";

/** A long conversation: *n* alternating turns of about *chars* characters,
 *  the second user turn carrying TOMATO. */
function makeLongConv(n, chars = 600) {
  const msgs = [];
  for (let i = 0; i < n; i++) {
    const head = i === 2 ? `Turn ${i}: ${TOMATO}. ` : `Turn ${i}: `;
    let text = head;
    for (let w = 0; text.length < chars; w++) text += `filler${i}x${w} `;
    msgs.push({ role: i % 2 === 0 ? "user" : "assistant", content: text.trim() });
  }
  return { id: "long", title: "t", messages: msgs };
}

/** A summariser that refuses a request whose prompt plus max_tokens is over
 *  *ctxMax* (prompt counted at three characters per token), and otherwise
 *  replies with the facts it can see: TOMATO when the prompt carries it.
 *  *failOn* (1-based) makes that request answer 500. */
function windowedSummariser(ctxMax, { failOn = 0, onPass = null } = {}) {
  const calls = [];
  const impl = async (url, opts = {}) => {
    if (String(url) !== "/v1/chat/completions") {
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    const body = JSON.parse(opts.body);
    calls.push(body);
    if (onPass) onPass(calls.length);
    const content = body.messages[0].content;
    const prompt = Math.ceil(content.length / 3);
    if (failOn === calls.length) {
      return { ok: false, status: 500, json: async () => ({}), text: async () => "boom" };
    }
    if (prompt + body.max_tokens > ctxMax) {
      return { ok: false, status: 413, text: async () => "",
        json: async () => ({ detail: `Prompt (${prompt} tokens) exceeds the model's maximum context capacity (${ctxMax} tokens)` }) };
    }
    const facts = content.includes(TOMATO) ? `Facts: ${TOMATO}.` : "Facts: none.";
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ choices: [{ message: { content: `Pass ${calls.length}. ${facts}` },
                                        finish_reason: "stop" }] }) };
  };
  return { impl, calls };
}

test("a small window summarises the older turns in rolling passes that each fit the window", async () => {
  const ctxMax = 2048;
  const { impl, calls } = windowedSummariser(ctxMax);
  const { window } = loadApp({ fetchImpl: impl });
  const toasts = [];
  window.toast = (msg) => toasts.push(String(msg));
  runScript(window, `chat.ctxMax = ${ctxMax};`);
  const conv = makeLongConv(24);
  const ok = await window.compactConversation(conv);
  assert.equal(ok, true);
  assert.ok(calls.length > 1, `more than one pass (${calls.length})`);
  assert.ok(calls.length <= 8, `at most MAX_SUMMARY_PASSES passes (${calls.length})`);
  for (const body of calls) {
    const prompt = Math.ceil(body.messages[0].content.length / 3);
    assert.ok(prompt + body.max_tokens <= ctxMax - 64,
      `prompt ${prompt} + max_tokens ${body.max_tokens} fits ${ctxMax} with a template margin`);
    assert.ok(body.max_tokens <= 512, `a sized reply budget (${body.max_tokens})`);
    assert.match(body.messages[0].content, /^Summarise the following conversation/);
    assert.equal(body.messages[0].origin, "client");
  }
  const bridge = conv.messages[0].content;
  assert.match(bridge, /^\[Conversation summary\]\n/, "the bridge is a summary, not the digest");
  assert.doesNotMatch(bridge, /condensed to fit the context window/);
  assert.ok(bridge.includes(TOMATO), "the early-turn fact reaches the final summary");
  assert.ok(bridge.includes(`Pass ${calls.length}. `), "the bridge holds the last pass");
  assert.ok(toasts.some((t) => /summarised/.test(t)));
  // Every later pass carries the summary so far.
  for (const body of calls.slice(1)) {
    assert.match(body.messages[0].content, /Summary of the earlier part:\nPass \d+\./);
  }
});

test("a large window summarises the same older turns in one pass", async () => {
  const { impl, calls } = windowedSummariser(32768);
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, "chat.ctxMax = 32768;");
  const conv = makeLongConv(24);
  await window.compactConversation(conv);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].max_tokens, 512);
  assert.match(conv.messages[0].content, /^\[Conversation summary\]\nPass 1\./);
});

test("a very long history at a small window stays within the pass bound and every request fits", async () => {
  const ctxMax = 2048;
  const { impl, calls } = windowedSummariser(ctxMax);
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, `chat.ctxMax = ${ctxMax};`);
  const conv = makeLongConv(240, 1200);
  await window.compactConversation(conv);
  assert.ok(calls.length > 1 && calls.length <= 8, `bounded passes (${calls.length})`);
  for (const body of calls) {
    assert.ok(Math.ceil(body.messages[0].content.length / 3) + body.max_tokens <= ctxMax - 64);
  }
  assert.match(conv.messages[0].content, /^\[Conversation summary\]\n/);
  assert.ok(calls[0].messages[0].content.includes("USER: Turn 0:"), "the oldest turn is in the first pass");
  assert.ok(calls[0].messages[0].content.includes("USER: Turn 2: my tomato variety"),
    "early turns keep the start of their text");
});

test("a failing rolling pass falls back to the digest and makes no further passes", async () => {
  const ctxMax = 2048;
  const { impl, calls } = windowedSummariser(ctxMax, { failOn: 2 });
  const { window } = loadApp({ fetchImpl: impl });
  const toasts = [];
  window.toast = (msg) => toasts.push(String(msg));
  runScript(window, `chat.ctxMax = ${ctxMax};`);
  const conv = makeLongConv(24);
  const ok = await window.compactConversation(conv);
  assert.equal(ok, true);
  assert.equal(calls.length, 2, "the passes stop at the failed one");
  assert.match(conv.messages[0].content, /condensed to fit the context window/);
  assert.doesNotMatch(conv.messages[0].content, /\[Conversation summary\]|Pass 1\./);
  assert.ok(toasts.some((t) => /trimmed/.test(t)));
});

test("aborting between rolling passes leaves the conversation unchanged", async () => {
  const ctxMax = 2048;
  const ac = new AbortController();
  const { impl, calls } = windowedSummariser(ctxMax, {
    onPass: (n) => { if (n === 1) ac.abort(); } });
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, `chat.ctxMax = ${ctxMax};`);
  const conv = makeLongConv(24);
  const before = JSON.stringify(conv.messages);
  const ok = await window.compactConversation(conv, ac.signal);
  assert.equal(ok, false);
  assert.equal(calls.length, 1, "no pass is made after the abort");
  assert.equal(JSON.stringify(conv.messages), before);
  assert.equal(window.document.querySelectorAll("#chat-messages .msg-row").length, 0);
});

test("the summary reply budget is a quarter of a small window, capped at 512", () => {
  const { window: w } = loadApp();
  assert.equal(w.summaryReplyTokens(0), 512);
  assert.equal(w.summaryReplyTokens(2048), 512);
  assert.equal(w.summaryReplyTokens(1024), 256);
  assert.equal(w.summaryReplyTokens(160), 64);
  assert.equal(w.summaryReplyTokens(131072), 512);
});

test("planSummaryPasses keeps the oldest and newest groups and counts what it leaves out past the pass bound", () => {
  const { window: w } = loadApp();
  const entries = [];
  for (let i = 0; i < 400; i++) entries.push(`USER: entry-${i} ` + "word ".repeat(240));
  const groups = w.planSummaryPasses(entries, 2048, 512, 84);
  assert.equal(groups.length, 8);
  assert.ok(groups[0][0].startsWith("USER: entry-0 "), "the oldest entry opens the first pass");
  const last = groups[groups.length - 1];
  assert.ok(last[last.length - 1].startsWith("USER: entry-399 "), "the newest entry closes the last pass");
  const m = /^\((\d+) message\(s\) omitted\)$/.exec(groups[1][0]);
  assert.ok(m, `the gap is counted: ${groups[1][0]}`);
  const kept = groups.reduce((n, g) => n + g.length, 0) - 1;
  assert.equal(kept + Number(m[1]), 400, "kept plus omitted is every entry");
  for (const g of groups) {
    assert.ok(w.budgetTokens(g.join("\n\n")) <= 2048 - 512 - w.templateMargin(2048) - 84 - 512,
      "each group leaves room for the earlier summary and the reply");
  }
});

test("the summariser budget counts each non-ASCII character as a token", () => {
  const { window: w } = loadApp();
  assert.equal(w.budgetTokens("é".repeat(30)), 30);
  assert.equal(w.budgetTokens("番茄".repeat(50) + "x".repeat(9)), 103);
  const cut = w.fitTokens("番茄 ".repeat(500), 40);
  assert.ok(w.budgetTokens(cut) <= 40, `cut text fits (${w.budgetTokens(cut)})`);
  assert.match(cut, /\.\.\.\[truncated\]$/);
});

test("planSummaryPasses makes one group when the window is unknown or the entries fit", () => {
  const { window: w } = loadApp();
  const entries = ["USER: a", "ASSISTANT: b"];
  assert.equal(w.planSummaryPasses(entries, 0, 512, 84).length, 1);
  assert.equal(w.planSummaryPasses(entries, 2048, 512, 84).length, 1);
});
