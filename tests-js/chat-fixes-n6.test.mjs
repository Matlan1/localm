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

const noMath = (out, why) => {
  assert.equal(out.querySelectorAll(".katex, .katex-error").length, 0, why);
};

test("price markers of two or more dollar signs keep the markdown between them", () => {
  const win = loadRealPipeline();
  const numbered = render(win,
    "1. **Cafe Luna** ($$) cosy\n2. **Taco Hut** ($) cheap\n3. **Le Bernardin** ($$$$) fancy");
  assert.equal(numbered.querySelectorAll("ol > li").length, 3);
  assert.equal(numbered.querySelectorAll("strong").length, 3);
  assert.equal(numbered.querySelectorAll(".katex-error").length, 0);
  assert.doesNotMatch(numbered.textContent, /\*\*/);

  const bullets = render(win, "- **A** ($$)\n- **B** ($$$)\n- **C** ($)");
  assert.equal(bullets.querySelectorAll("ul > li").length, 3);
  noMath(bullets, "bullet list");

  const paragraphs = render(win, "Price: $$ for A.\n\nPrice: $$$ for B.");
  assert.equal(paragraphs.querySelectorAll("p").length, 2);
  noMath(paragraphs, "paragraphs");

  const heading = render(win, "Cheap ($$)\n\n## Heading\n\nmore ($$)");
  assert.equal(heading.querySelectorAll("h2").length, 1);
  assert.doesNotMatch(heading.textContent, /##/);
  noMath(heading, "heading");

  for (const line of ["Cafe: $$, Taco: $$ and Bar $$.", "Cafe ($$), Taco ($$), Bar ($$)", "A $$ b, C $$ d"]) {
    assert.equal(win.protectMath(line).text, line, line);
  }

  const alone = render(win, "Cheap\n$$\n- Taco\n$$\nfancy");
  noMath(alone, "a list between two lines holding only $$");
});

test("display math on one line or across lines is still typeset", () => {
  const win = loadRealPipeline();
  const block = render(win, "Before\n\n$$\nx^2 + y^2\n$$\n\nAfter");
  assert.deepEqual(tex(block).map((x) => x.trim()), ["x^2 + y^2"]);
  assert.match(block.textContent, /Before/);
  assert.match(block.textContent, /After/);

  const aligned = render(win,
    String.raw`$$\begin{aligned}` + "\na &= b \\\\\n  + c\n" + String.raw`\end{aligned}$$`);
  const found = tex(aligned);
  assert.equal(found.length, 1);
  assert.match(found[0], /a &= b/);
  assert.match(found[0], /\+ c/);
  assert.equal(aligned.querySelectorAll(".katex-error").length, 0);

  const line = render(win, "Then $$a_1\\,b_2$$.");
  assert.deepEqual(tex(line), ["a_1\\,b_2"]);
});

test("single-dollar prices are not math", () => {
  const win = loadRealPipeline();
  for (const text of ["It costs $5 and $10, or $5-$10, not $20.", "From $3 up, $4 for two."]) {
    const out = render(win, text);
    noMath(out, text);
    assert.equal(out.textContent.trim(), text);
  }
});

test("dollar amounts stay text wherever they sit", () => {
  const win = loadRealPipeline();
  for (const src of [
    "$5 and $10", "between $3.50 and $4", "($$$$)", "a ($$ $$) b",
    "pay $5, get $10 back", "costs $3.50 or $4", "cheap ($) and fancy ($$$)",
  ]) {
    const out = render(win, src);
    noMath(out, src);
    assert.equal(out.textContent.trim(), src, src);
    assert.equal(out.querySelectorAll("p").length, 1, `${src} stays one paragraph`);
  }
  const list = render(win, "- pay $5 and $10\n- from $3 to $4");
  noMath(list, "list");
  assert.deepEqual([...list.querySelectorAll("li")].map((li) => li.textContent),
    ["pay $5 and $10", "from $3 to $4"]);
  const table = render(win, "| item | price |\n|---|---|\n| tea | $5 and $10 |\n| cake | $3.50 |");
  noMath(table, "table");
  assert.deepEqual([...table.querySelectorAll("td")].map((td) => td.textContent),
    ["tea", "$5 and $10", "cake", "$3.50"]);
  const bold = render(win, "**$5** and *$10*");
  noMath(bold, "emphasis");
  assert.equal(bold.querySelector("strong").textContent, "$5");
  assert.equal(bold.querySelector("em").textContent, "$10");
});

test("price markers wrapped in bold, italic, quotes or brackets on one line stay text", () => {
  const win = loadRealPipeline();
  const bold = render(win, "Cafe Luna is **$$** and Taco Hut is also **$$** here.");
  noMath(bold, "bold");
  assert.deepEqual([...bold.querySelectorAll("strong")].map((e) => e.textContent), ["$$", "$$"]);
  assert.equal(bold.textContent.trim(), "Cafe Luna is $$ and Taco Hut is also $$ here.");
  for (const src of ["A is *$$* and B is *$$* ok", 'A is "$$" and B is "$$" ok', "A is [$$] and B is [$$] ok",
                     "A is ~~$$~~ and B is ~~$$~~ ok"]) {
    const out = render(win, src);
    noMath(out, src);
    assert.equal(out.textContent.trim().replace(/\s+/g, " "), src.replace(/[*~]/g, "").replace(/\s+/g, " "), src);
  }
});

test("protectMath stays linear on long runs of spaces and tabs", () => {
  const { window } = loadApp();
  const time = (s) => {
    let best = Infinity;
    for (let i = 0; i < 3; i++) {
      const t = process.hrtime.bigint();
      window.protectMath(s);
      best = Math.min(best, Number(process.hrtime.bigint() - t) / 1e6);
    }
    return best;
  };
  for (const ch of [" ", "\t"]) {
    for (const tail of ["y $$ z", "$$ z", "$ $ $", "\n$$"]) {
      const small = time("x\n" + ch.repeat(20000) + tail);
      const big = time("x\n" + ch.repeat(80000) + tail);
      assert.ok(big < 250 + 8 * small, `${JSON.stringify(ch)} + ${JSON.stringify(tail)}: ${small}ms -> ${big}ms`);
    }
  }
});

test("price markers wrapped in bold, italic, quotes or brackets on one line stay text", () => {
  const win = loadRealPipeline();
  const bold = render(win, "Cafe Luna is **$$** and Taco Hut is also **$$** here.");
  noMath(bold, "bold");
  assert.deepEqual([...bold.querySelectorAll("strong")].map((e) => e.textContent), ["$$", "$$"]);
  assert.equal(bold.textContent.trim(), "Cafe Luna is $$ and Taco Hut is also $$ here.");
  for (const src of ["A is *$$* and B is *$$* ok", 'A is "$$" and B is "$$" ok', "A is [$$] and B is [$$] ok",
                     "A is ~~$$~~ and B is ~~$$~~ ok"]) {
    const out = render(win, src);
    noMath(out, src);
    assert.equal(out.textContent.trim().replace(/\s+/g, " "), src.replace(/[*~]/g, "").replace(/\s+/g, " "), src);
  }
});

test("protectMath stays linear on long runs of spaces and tabs", () => {
  const { window } = loadApp();
  const time = (s) => {
    let best = Infinity;
    for (let i = 0; i < 3; i++) {
      const t = process.hrtime.bigint();
      window.protectMath(s);
      best = Math.min(best, Number(process.hrtime.bigint() - t) / 1e6);
    }
    return best;
  };
  for (const ch of [" ", "\t"]) {
    for (const tail of ["y $$ z", "$$ z", "$ $ $", "\n$$"]) {
      const small = time("x\n" + ch.repeat(20000) + tail);
      const big = time("x\n" + ch.repeat(80000) + tail);
      assert.ok(big < 250 + 8 * small, `${JSON.stringify(ch)} + ${JSON.stringify(tail)}: ${small}ms -> ${big}ms`);
    }
  }
});

test("a backslash-escaped dollar is a literal dollar", () => {
  const win = loadRealPipeline();
  const out = render(win, String.raw`\$5 and \$x\$ and \$10`);
  noMath(out, "escaped");
  assert.equal(out.textContent.trim(), "$5 and $x$ and $10");
  const mixed = render(win, String.raw`Cost \$5 but $x^2$ is math`);
  assert.equal(tex(mixed).length, 1);
  assert.match(mixed.textContent, /Cost \$5 but/);
});

test("dollar signs in code stay literal and math beside prices still typesets", () => {
  const win = loadRealPipeline();
  const code = render(win, "Run `echo $5 and $10` or\n\n```\necho $5 $$\n```");
  noMath(code, "code");
  assert.equal(code.querySelector("p code").textContent, "echo $5 and $10");
  assert.equal(code.querySelector("pre").textContent.trim(), "echo $5 $$");

  const both = render(win,
    String.raw`It costs $5 and $10 while $x^2$ and $\alpha + \beta$ and $$\int_0^1 x\,dx$$ are math.`);
  assert.deepEqual(tex(both), ["x^2", String.raw`\alpha + \beta`, String.raw`\int_0^1 x\,dx`]);
  assert.match(both.textContent, /It costs \$5 and \$10 while/);
});

test("a price in a streamed reply never turns into math and back", () => {
  const win = loadRealPipeline();
  const full = "It costs $5 and $10 each, so $3.50 or $4 is fine ($$$$).";
  const target = win.document.createElement("div");
  win.document.body.appendChild(target);
  for (let i = 1; i <= full.length; i++) {
    win.renderMarkdown(target, full.slice(0, i), {});
    assert.equal(target.querySelectorAll(".katex, .katex-error").length, 0, full.slice(0, i));
  }
  assert.equal(target.textContent.trim(), full);
});

test("an empty or blank display block is literal text", () => {
  const win = loadRealPipeline();
  for (const src of ["$$$$", "x $$   $$ y"]) {
    const out = render(win, src);
    noMath(out, src);
    assert.equal(out.textContent.trim(), src);
  }
});

const okResponse = { ok: true, status: 200, headers: { get: () => null },
  json: async () => ({}), text: async () => "" };

test("a Qwen3-ASR prefix is left out of Copy, Speak and the stored reply", async () => {
  const raw = "language English<asr_text>Please remember the meeting.";
  const { window } = loadApp({ fetchImpl: async () => okResponse });
  stubStream(window);
  window.readSSE = async (_r, onData) => {
    onData(JSON.stringify({ choices: [{ delta: { content: "language Eng" } }] }));
    onData(JSON.stringify({ choices: [{ delta: { content: "lish<asr_text>Please remember the meeting." } }] }));
    onData(JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] }));
  };
  const conv = { id: "c1", title: "t", messages: [{ role: "user", content: "transcribe" }] };
  await window.runCompletion(conv);
  const reply = conv.messages.find((m) => m.role === "assistant");
  assert.equal(reply.content, "Please remember the meeting.", "the stored reply");

  const writes = [];
  Object.defineProperty(window.navigator, "clipboard", {
    configurable: true,
    value: { writeText: (t) => { writes.push(t); return Promise.resolve(); } },
  });
  const box = window.document.createElement("div");
  window.document.body.appendChild(box);
  window.addMessageRow(box, "assistant", raw, {});
  box.querySelector(".copy-btn").click();
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(writes, ["Please remember the meeting."], "Copy of a reply stored with the prefix");

  window.speak(raw);
  assert.equal(window.__spoken.length, 1);
  assert.equal(window.__spoken[0].text, "Please remember the meeting.", "Speak");
});

const refuseImageWithAudio = (calls, onlyWhenBoth) => async (url, opts = {}) => {
  let body = null;
  try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = null; }
  calls.push({ url: String(url), body });
  if (String(url) === "/v1/chat/completions") {
    const text = JSON.stringify(body.messages);
    if (text.includes("image_url") && (!onlyWhenBoth || text.includes("input_audio"))) {
      return { ok: false, status: 400,
        headers: { get: (k) => (k === "X-Localm-Refusal" ? "image" : null) },
        json: async () => ({ detail: "This model cannot accept image input." }),
        text: async () => "" };
    }
  }
  return okResponse;
};

const imageThenClip = () => [
  { role: "user", content: [
    { type: "text", text: "what colour?" },
    { type: "image_url", image_url: { url: IMG } },
  ] },
  { role: "assistant", content: "Red." },
  { role: "user", content: [
    { type: "text", text: "transcribe" },
    { type: "input_audio", input_audio: { data: WAV_B64, format: "wav" }, name: "clip.wav", seconds: 2 },
  ] },
  { role: "assistant", content: "hello there" },
];

const setModelRow = (window, row) => {
  window.__row = row;
  runScript(window, "modelCache.models = window.__row ? [window.__row] : [];");
};

test("a text turn after an audio turn leaves the clip out of the request and keeps both attachments", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: refuseImageWithAudio(calls, true) });
  stubStream(window);
  setModelRow(window, { name: "test-model", vision: true, audio_input: false });
  const conv = { id: "c1", title: "t",
    messages: [...imageThenClip(), { role: "user", content: "thanks, and in one word?" }] };
  await window.runCompletion(conv);

  const posts = chatCalls(calls);
  assert.equal(posts.length, 1, "no refused request, no repeat");
  const sent = JSON.stringify(posts[0].body.messages);
  assert.equal(sent.includes("input_audio"), false, "no clip on the wire");
  assert.equal(sent.includes("image_url"), true, "the image is still sent");
  assert.equal(window.msgImages(conv.messages[0]).length, 1, "the conversation keeps the image");
  assert.equal(window.msgAudioClips(conv.messages[2]).length, 1, "the conversation keeps the clip");
  assert.ok(conv.messages.some((m) => m.role === "assistant" && m.content === "a reply"),
    "the text turn is answered");
});

test("a text turn in a chat with clips and no image still sends the clips", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: refuseImageWithAudio(calls, true) });
  stubStream(window);
  const conv = { id: "c1", title: "t",
    messages: [...imageThenClip().slice(2), { role: "user", content: "again?" }] };
  await window.runCompletion(conv);
  const posts = chatCalls(calls);
  assert.equal(posts.length, 1);
  assert.equal(JSON.stringify(posts[0].body.messages).includes("input_audio"), true);
});

for (const [label, row] of [
  ["a model that reads images and hears audio", { name: "test-model", vision: true, audio_input: true }],
  ["a model whose audio support is not recorded", { name: "test-model", vision: true }],
  ["a model with no registry row", null],
]) {
  test(`a text turn after an audio turn still sends the earlier clip to ${label}`, async () => {
    const calls = [];
    const { window } = loadApp({ fetchImpl: async (url, opts = {}) => {
      let body = null;
      try { body = opts.body ? JSON.parse(opts.body) : null; } catch { body = null; }
      calls.push({ url: String(url), body });
      return okResponse;
    } });
    stubStream(window);
    setModelRow(window, row);
    const conv = { id: "c1", title: "t",
      messages: [...imageThenClip(), { role: "user", content: "what did the clip say?" }] };
    await window.runCompletion(conv);
    const posts = chatCalls(calls);
    assert.equal(posts.length, 1);
    const sent = JSON.stringify(posts[0].body.messages);
    assert.equal(sent.includes("input_audio"), true, "the clip is on the wire");
    assert.equal(sent.includes("image_url"), true, "the image is on the wire");
    assert.equal(sent.includes("not included in this request"), false);
  });
}

test("the elided repeat of an audio turn tells the user the images were left out", async () => {
  const calls = [];
  const { window } = loadApp({ fetchImpl: refuseImageWithAudio(calls, false) });
  stubStream(window);
  const toasts = [];
  window.toast = (msg) => { toasts.push(String(msg)); };
  const conv = { id: "c1", title: "t", messages: imageThenClip().slice(0, 3) };
  await window.runCompletion(conv);
  assert.equal(chatCalls(calls).length, 2);
  assert.ok(toasts.some((m) => /cannot read images/.test(m) && /left out/.test(m)), toasts.join(" | "));
});

test("the wire helpers leave the conversation untouched", () => {
  const { window } = loadApp();
  const conv = { messages: imageThenClip() };
  assert.equal(window.latestTurnIsAudioOnly({ messages: imageThenClip().slice(0, 3) }), true);
  assert.equal(window.latestTurnIsAudioOnly({ messages: imageThenClip().slice(0, 1) }), false, "an image turn");
  assert.equal(window.latestTurnIsAudioOnly({ messages: [...imageThenClip(), { role: "user", content: "hi" }] }),
    false, "a text turn");
  assert.equal(window.latestTurnHasAudio({ messages: imageThenClip().slice(0, 3) }), true);
  assert.equal(window.latestTurnHasAudio({ messages: [...imageThenClip(), { role: "user", content: "hi" }] }), false);

  const wire = JSON.parse(JSON.stringify(conv.messages));
  assert.equal(window.wireHasImageAndAudio(wire), true);
  window.elideImagesForWire(wire);
  assert.equal(JSON.stringify(wire).includes("image_url"), false);
  assert.equal(JSON.stringify(wire).includes("input_audio"), true);
  assert.equal(window.wireHasImageAndAudio(wire), false);
  const wire2 = JSON.parse(JSON.stringify(conv.messages));
  window.elideAudioForWire(wire2);
  assert.equal(JSON.stringify(wire2).includes("input_audio"), false);
  assert.equal(JSON.stringify(wire2).includes("image_url"), true);
  assert.equal(JSON.stringify(conv.messages).includes("input_audio"), true, "the source is unchanged");
  assert.equal(JSON.stringify(conv.messages).includes("image_url"), true, "the source is unchanged");
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
