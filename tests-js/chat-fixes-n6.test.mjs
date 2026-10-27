// SPDX-License-Identifier: AGPL-3.0-or-later
// Chat fixes: an audio turn after an image turn, math that survives markdown,
// the Qwen3-ASR transcript prefix, compaction that continues from an earlier
// summary and names attachments, and the web settings following a change.

import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadApp, runScript } from "./harness.mjs";

const VENDOR = new URL("../localm/plugins/gui/static/vendor/", import.meta.url);
const readVendored = (n) => fs.readFileSync(new URL(n, VENDOR), "utf8");

const IMG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg";
const WAV_B64 = "UklGRiQAAABXQVZFZm10IA==";

const evalIn = (window, expr) => {
  window.__err = null;
  runScript(window,
    `try { window.__out = (${expr}); } catch (e) { window.__out = undefined; window.__err = String(e); }`);
  if (window.__err) throw new Error(`${expr}: ${window.__err}`);
  return window.__out;
};

function stubStream(window) {
  window.maybeCompactConversation = async () => {};
  window.readSSE = async (_r, onData) => {
    onData(JSON.stringify({ choices: [{ delta: { content: "a reply" } }] }));
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }));
  };
  for (const id of ["p-speak", "p-memory", "p-web"]) window.document.getElementById(id).checked = false;
  runScript(window, "modelCache.active = 'test-model';");
}

const chatCalls = (calls) => calls.filter((c) => c.url === "/v1/chat/completions");

test("an audio turn after an image turn is answered with the earlier images left out of that request", async () => {
  const calls = [];
  const impl = async (url, opts = {}) => {
    let body = null;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = null; }
    calls.push({ url: String(url), body });
    if (String(url) === "/v1/chat/completions") {
      const hasImage = JSON.stringify(body.messages).includes("image_url");
      if (hasImage) {
        return { ok: false, status: 400,
          headers: { get: (k) => (k === "X-Localm-Refusal" ? "image" : null) },
          json: async () => ({ detail: "This model cannot accept image input." }),
          text: async () => "" };
      }
      return { ok: true, status: 200, headers: { get: () => null },
        json: async () => ({}), text: async () => "" };
    }
    return { ok: true, status: 200, headers: { get: () => null },
      json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl: impl });
  stubStream(window);
  const conv = { id: "c1", title: "t", messages: [
    { role: "user", content: [
      { type: "text", text: "what colour?" },
      { type: "image_url", image_url: { url: IMG } },
    ] },
    { role: "assistant", content: "Red." },
    { role: "user", content: [
      { type: "text", text: "transcribe" },
      { type: "input_audio", input_audio: { data: WAV_B64, format: "wav" }, name: "clip.wav", seconds: 2 },
    ] },
  ] };
  await window.runCompletion(conv);

  const posts = chatCalls(calls);
  assert.equal(posts.length, 2, "the refused request is repeated once");
  assert.equal(JSON.stringify(posts[1].body.messages).includes("image_url"), false,
    "the repeat carries no image part");
  assert.equal(JSON.stringify(posts[1].body.messages).includes("input_audio"), true,
    "the repeat still carries the clip");
  assert.equal(window.msgImages(conv.messages[0]).length, 1, "the conversation keeps the image");
  assert.equal(window.msgAudioClips(conv.messages[2]).length, 1, "the conversation keeps the clip");
  assert.ok(conv.messages.some((m) => m.role === "assistant" && m.content === "a reply"),
    "the clip turn gets its reply");
});

test("an image refusal on an audio turn that also carries an image still drops the images", async () => {
  const calls = [];
  const impl = async (url, opts = {}) => {
    let body = null;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = null; }
    calls.push({ url: String(url), body });
    if (String(url) === "/v1/chat/completions") {
      return { ok: false, status: 400,
        headers: { get: (k) => (k === "X-Localm-Refusal" ? "image" : null) },
        json: async () => ({ detail: "This model cannot accept image input." }),
        text: async () => "" };
    }
    return { ok: true, status: 200, headers: { get: () => null }, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl: impl });
  stubStream(window);
  const conv = { id: "c1", title: "t", messages: [
    { role: "user", content: [
      { type: "text", text: "both" },
      { type: "image_url", image_url: { url: IMG } },
      { type: "input_audio", input_audio: { data: WAV_B64, format: "wav" }, name: "clip.wav", seconds: 2 },
    ] },
  ] };
  await window.runCompletion(conv);
  assert.equal(chatCalls(calls).length, 1, "no repeat");
  assert.equal(window.msgImages(conv.messages[0]).length, 0, "the refused image is dropped");
});

function loadRealPipeline() {
  const { window: win } = loadApp();
  for (const f of ["marked.min.js", "purify.min.js", "highlight.min.js",
                   "katex.min.js", "auto-render.min.js"]) {
    const tag = win.document.createElement("script");
    tag.textContent = readVendored(f);
    win.document.head.appendChild(tag);
  }
  win.marked.setOptions({ breaks: true });
  return win;
}

function render(win, text) {
  const target = win.document.createElement("div");
  win.document.body.appendChild(target);
  win.renderMarkdown(target, text, { final: true });
  return target;
}

const tex = (el) => [...el.querySelectorAll("annotation")].map((a) => a.textContent);

test("display math keeps its markdown-escapable TeX", () => {
  const win = loadRealPipeline();
  const out = render(win, String.raw`See $$\int_0^1 x^2\,dx = \frac{1}{3}$$ done.`);
  assert.deepEqual(tex(out), [String.raw`\int_0^1 x^2\,dx = \frac{1}{3}`]);
});

test("inline math, row breaks and underscores survive markdown", () => {
  const win = loadRealPipeline();
  const out = render(win, String.raw`Take $a\,b$ and $$\begin{matrix} x_1 & y_2 \\ z & w \end{matrix}$$`);
  const found = tex(out);
  assert.equal(found.length, 2);
  assert.equal(found[0], String.raw`a\,b`);
  assert.match(found[1], /x_1 & y_2 \\\\ z & w/);
});

test("math inside inline code and fenced code stays literal", () => {
  const win = loadRealPipeline();
  const out = render(win, "Run `echo $a\\,b$` now.\n\n```\n$$x\\,y$$\n```");
  assert.equal(out.querySelectorAll(".katex").length, 0);
  assert.equal(out.querySelector("p code").textContent, "echo $a\\,b$");
  assert.match(out.querySelector("pre").textContent, /\$\$x\\,y\$\$/);
});

test("markup inside a math span is shown as text, never parsed", () => {
  const win = loadRealPipeline();
  const out = render(win, "$$<img src=x onerror=window.__xss=1>$$");
  assert.equal(out.querySelectorAll("img").length, 0);
  assert.equal(win.__xss, undefined);
});

test("math source is never restored into a link or image attribute", () => {
  const win = loadRealPipeline();
  const out = render(win,
    "[click]($javascript:window.__xss=1$) and ![x]($javascript:window.__xss=1$)");
  for (const node of out.querySelectorAll("*")) {
    for (const attr of node.attributes) {
      assert.doesNotMatch(attr.value, /javascript:/i, `${node.tagName}[${attr.name}]`);
    }
  }
});

test("a Qwen3-ASR reply shows the transcript without the language prefix", () => {
  const win = loadRealPipeline();
  const out = render(win, "language English<asr_text>Please remember the meeting.");
  assert.equal(out.textContent.trim(), "Please remember the meeting.");
  const none = render(win, "language None<asr_text>");
  assert.doesNotMatch(none.textContent, /language|asr_text/);
});

function compactionFixture(summaryReply) {
  const prompts = [];
  const impl = async (url, opts = {}) => {
    if (String(url) === "/v1/chat/completions") {
      const b = JSON.parse(opts.body);
      prompts.push(b.messages[0].content);
      if (summaryReply === null) return { ok: false, status: 500, json: async () => ({}) };
      return { ok: true, status: 200,
        json: async () => ({ choices: [{ finish_reason: "stop", message: { content: summaryReply } }] }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl: impl });
  runScript(window, "chat.ctxMax = 600; modelCache.active = 'test-model';");
  const filler = "word ".repeat(40);
  const messages = [
    { role: "user", content: "[Conversation summary]\nThe favourite apple is the Cox Orange Pippin from Ghent.", bridge: true },
    { role: "assistant", content: "Understood. Continuing from this summary.", bridge: true },
  ];
  for (let i = 0; i < 4; i++) {
    messages.push({ role: "user", content: `question ${i} ${filler}` });
    messages.push({ role: "assistant", content: `answer ${i} ${filler}` });
  }
  return { window, prompts, conv: { id: "c1", title: "t", messages } };
}

test("a later compaction continues from the earlier summary", async () => {
  const { window, prompts, conv } = compactionFixture("Cox Orange Pippin, Ghent; later talk covered storage.");
  assert.equal(await window.compactConversation(conv), true);
  assert.equal(prompts.length, 1);
  assert.match(prompts[0], /Summary of the earlier part:\nThe favourite apple is the Cox Orange Pippin from Ghent\./);
  assert.doesNotMatch(prompts[0], /USER: \[Conversation summary\]/,
    "the earlier summary is not summarised as one more turn");
  assert.match(conv.messages[0].content, /^\[Conversation summary\]\nCox Orange Pippin, Ghent/);
});

test("a failed summary keeps the earlier summary and appends a digest", async () => {
  const { window, conv } = compactionFixture(null);
  assert.equal(await window.compactConversation(conv), true);
  assert.match(conv.messages[0].content, /^\[Conversation summary\]\nThe favourite apple is the Cox Orange Pippin from Ghent\./);
  assert.match(conv.messages[0].content, /question 0/);
});

test("compaction text names an audio clip and marks an image", () => {
  const { window } = loadApp();
  const text = window.compactionText({ role: "user", content: [
    { type: "text", text: "remember this" },
    { type: "image_url", image_url: { url: IMG } },
    { type: "input_audio", input_audio: { data: WAV_B64, format: "wav" }, name: "meeting.mp3", seconds: 4 },
    { type: "input_audio", input_audio: { data: WAV_B64, format: "wav" }, name: "", seconds: 4 },
  ] });
  assert.match(text, /remember this/);
  assert.match(text, /\[image attached\]/);
  assert.match(text, /\[audio clip attached: meeting\.mp3\]/);
  assert.match(text, /\[audio clip attached\]$/);
});

test("the web access toggle is hidden when network access is off and shown again otherwise", () => {
  const { window } = loadApp();
  const row = window.document.getElementById("p-web").closest(".check-row");
  window.applyWebPolicy({ net_mode: "off", chat_web_max_lookups: 3 });
  assert.equal(row.style.display, "none");
  assert.equal(evalIn(window, "chat.netOff"), true);
  assert.equal(evalIn(window, "webMaxLookups()"), 3);
  window.applyWebPolicy({ net_mode: "ask", chat_web_max_lookups: null });
  assert.equal(row.style.display, "");
  assert.equal(evalIn(window, "chat.netOff"), false);
  assert.equal(evalIn(window, "webMaxLookups()"), 20, "a null limit means the default");
  window.applyWebPolicy({});
  assert.equal(evalIn(window, "webMaxLookups()"), 20, "a payload without the fields changes nothing");
});

test("a send with web access on reads the lookup limit from the server, not the page-load copy", async () => {
  let limit = 20;
  const calls = [];
  const impl = async (url, opts = {}) => {
    let body = null;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = null; }
    calls.push({ url: String(url), body });
    if (String(url) === "/v1/config") {
      return { ok: true, status: 200, headers: { get: () => null },
        json: async () => ({ net_mode: "ask", chat_web_max_lookups: limit }) };
    }
    return { ok: true, status: 200, headers: { get: () => null }, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl: impl });
  stubStream(window);
  window.document.getElementById("p-web").checked = true;
  limit = 2;
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  await window.runCompletion(conv);
  assert.equal(evalIn(window, "webMaxLookups()"), 2);
});

test("with network access off a checked web toggle sends no web tools", async () => {
  const calls = [];
  const impl = async (url, opts = {}) => {
    let body = null;
    try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = null; }
    calls.push({ url: String(url), body });
    if (String(url) === "/v1/config") {
      return { ok: true, status: 200, headers: { get: () => null },
        json: async () => ({ net_mode: "off" }) };
    }
    return { ok: true, status: 200, headers: { get: () => null }, json: async () => ({}), text: async () => "" };
  };
  const { window } = loadApp({ fetchImpl: impl });
  stubStream(window);
  window.document.getElementById("p-web").checked = true;
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "hi" }] };
  await window.runCompletion(conv);
  const sent = chatCalls(calls)[0].body;
  assert.equal(sent.messages[0].content.includes(evalIn(window, "WEB_TOOL_PROMPT")), false);
  assert.equal(sent.grammar, undefined);
});
