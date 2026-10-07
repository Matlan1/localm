// SPDX-License-Identifier: AGPL-3.0-or-later
// retrieveKnowledge() (settings-perf.js) injects the selected knowledge
// collection's hits into the conversation. A chunk is injected whole up to
// KB_EXCERPT_CHARS; anything longer, and a question echo longer than
// KB_QUERY_ECHO_CHARS, is cut at a sentence or word boundary and marked "…".
// The query asks the server for relevant_only hits, and when none come back the
// conversation gets a note saying so instead of nothing.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const WORDS = ("retrieval grounds the reply in the indexed documents so the model can cite " +
  "where each claim came from and the reader can check it against the source ").split(" ");

function prose(nChars, { sentences = true } = {}) {
  let out = "";
  let i = 0;
  while (out.length < nChars) {
    const w = WORDS[i % WORDS.length] || "text";
    out += (out ? " " : "") + w;
    i++;
    if (sentences && i % 13 === 0) out += ".";
  }
  return out;
}

function setup(hits) {
  const requests = [];
  const fetchImpl = async (url, opts) => {
    if (String(url).includes("/api/rag/collections/")) {
      requests.push({ url: String(url), body: JSON.parse(opts.body) });
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ collection: "kb1", hits }) };
    }
    return { ok: true, status: 200, text: async () => "", json: async () => ({}) };
  };
  const { window } = loadApp({ fetchImpl });
  const sel = window.document.getElementById("p-kb");
  const opt = window.document.createElement("option");
  opt.value = "kb1";
  opt.textContent = "kb1";
  sel.appendChild(opt);
  sel.value = "kb1";
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "q", id: "m1" }] };
  runScript(window, `chat.conversations = [${JSON.stringify(conv)}]; chat.activeId = "c1";`);
  return { window, requests, conv: window.currentConv() };
}

function injected(conv) {
  const kb = conv.messages.filter((m) => m.tag === "kb");
  assert.equal(kb.length, 1, "exactly one knowledge row is injected");
  return kb[0].content;
}

test("a full-size chunk is injected whole, ending at its real end", async () => {
  const chunk = prose(1150) + " final words of the chunk.";
  assert.ok(chunk.length > 1100 && chunk.length <= 1200);
  const { window, conv, requests } = setup([{ source: "D:/docs/guide.md", pos: 3, text: chunk, score: 1 }]);
  await window.retrieveKnowledge(conv, "how does retrieval cite sources");
  const content = injected(conv);
  assert.ok(content.includes(`[1] guide.md:3\n${chunk}\n\n`),
    "the whole chunk follows its citation, up to its last character");
  assert.ok(!content.includes("…"), "nothing was cut, so no ellipsis");
  assert.equal(requests.length, 1);
  assert.equal(requests[0].body.relevant_only, true, "the chat asks for relevant hits only");
  assert.equal(requests[0].body.k, 4);
});

test("an over-budget excerpt is cut at a sentence end and marked with an ellipsis", async () => {
  const text = prose(3000);
  const { window, conv } = setup([{ source: "a.md", pos: 1, text, score: 1 }]);
  await window.retrieveKnowledge(conv, "retrieval");
  const content = injected(conv);
  const body = content.split("[1] a.md:1\n")[1].split("\n\nUse these excerpts")[0];
  assert.ok(body.endsWith(".…"), `ends on a whole sentence plus the marker: ${body.slice(-30)}`);
  const kept = body.slice(0, -1);
  assert.ok(text.startsWith(kept), "the kept part is an unaltered prefix of the chunk");
  runScript(window, "window.__budget = KB_EXCERPT_CHARS;");
  const budget = window.__budget;
  assert.equal(typeof budget, "number");
  assert.ok(kept.length <= budget && kept.length >= budget * 0.6);
});

test("an over-budget excerpt with no sentence end is cut at a word boundary", () => {
  const { window } = loadApp();
  const text = prose(400, { sentences: false });
  const out = window.clipAtBoundary(text, 100);
  assert.ok(out.endsWith("…"));
  const kept = out.slice(0, -1);
  assert.ok(text.startsWith(kept));
  assert.match(text[kept.length], /\s/, "the cut falls between two words");
  assert.ok(kept.length <= 100 && kept.length >= 60);
});

test("clipAtBoundary leaves text within the budget untouched", () => {
  const { window } = loadApp();
  assert.equal(window.clipAtBoundary("short text", 100), "short text");
  const exact = "x".repeat(100);
  assert.equal(window.clipAtBoundary(exact, 100), exact);
  assert.equal(window.clipAtBoundary("y".repeat(150), 100), "y".repeat(100) + "…",
    "a single unbreakable token is cut at the budget and still marked");
});

test("a long question is echoed cut at a word with an ellipsis; a short one whole", async () => {
  const longQ = "why did some of the web searches succeed while the github page reads failed " +
    "on several occasions even though the network was up the whole time and nothing changed";
  assert.ok(longQ.length > 120);
  let s = setup([{ source: "a.md", pos: 1, text: "retrieval text.", score: 1 }]);
  await s.window.retrieveKnowledge(s.conv, longQ);
  const header = injected(s.conv).split("\n")[0];
  const echo = header.split("relevant to: ")[1].replace(/\]$/, "");
  assert.ok(echo.endsWith("…"), `echo is marked as shortened: ${echo}`);
  const kept = echo.slice(0, -1);
  assert.ok(longQ.startsWith(kept));
  assert.equal(longQ[kept.length], " ", "the echo ends on a whole word");

  s = setup([{ source: "a.md", pos: 1, text: "retrieval text.", score: 1 }]);
  await s.window.retrieveKnowledge(s.conv, "how does retrieval work?");
  assert.equal(injected(s.conv).split("\n")[0],
    '[Excerpts from the "kb1" collection relevant to: how does retrieval work?]');
});

test("no relevant hits injects a note saying so, with no excerpts", async () => {
  const { window, conv } = setup([]);
  await window.retrieveKnowledge(conv, "why did that search fail?");
  const content = injected(conv);
  assert.ok(content.startsWith(
    '[No excerpts from the "kb1" collection were relevant to: why did that search fail?]'));
  assert.ok(!/\[1\]/.test(content), "no excerpt is cited");
});
