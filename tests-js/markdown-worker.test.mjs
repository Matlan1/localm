// SPDX-License-Identifier: AGPL-3.0-or-later
// Markdown that could take long to parse is parsed off the page in the Markdown
// worker, under a time budget, one parse per message at a time; the page
// sanitises the worker's HTML exactly as it sanitises its own. jsdom has no
// Worker, so the worker boundary is a stand-in that answers with the worker
// module's own reply function over the real vendored marked.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";
import { loadApp, runScript } from "./harness.mjs";

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const STATIC = path.join(ROOT, "localm", "plugins", "gui", "static");
const VENDOR = path.join(STATIC, "vendor");
const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));

function realPipeline() {
  const { window: win } = loadApp();
  for (const f of ["marked.min.js", "purify.min.js"]) {
    const tag = win.document.createElement("script");
    tag.textContent = fs.readFileSync(path.join(VENDOR, f), "utf8");
    win.document.head.appendChild(tag);
  }
  return win;
}

/** Installs a stand-in Worker in *win*. Every constructed worker is kept in
 *  `workers`; each keeps the messages posted to it and answers one with
 *  `answer(i)` through the worker module's markdownWorkerReply. */
function installWorker(win) {
  const workers = [];
  win.Worker = class {
    constructor(url, options) {
      this.url = url;
      this.options = options;
      this.posted = [];
      this.terminated = false;
      workers.push(this);
    }
    postMessage(data) { this.posted.push(data); }
    terminate() { this.terminated = true; }
    answer(i = this.posted.length - 1) {
      this.onmessage({ data: win.markdownWorkerReply(win.marked, this.posted[i]) });
    }
  };
  return workers;
}

function target(win) {
  const el = win.document.createElement("div");
  win.document.body.appendChild(el);
  return el;
}

const LONG_REPLY = Array.from({ length: 120 }, (_, i) =>
  `## Part ${i}\n\nSome **bold** and *it* text with \`code\`, a [link](https://example.com/${i}) and $x^${i}$.`)
  .join("\n\n");

function syncRender(text, opts = {}) {
  const win = realPipeline();
  const t = target(win);
  win.renderMarkdown(t, text, opts);
  return t.innerHTML;
}

test("the worker's parse runs without a DOM and matches marked with the page's options", async () => {
  const mod = await import(new URL("../localm/plugins/gui/static/app/markdown-parse.js", import.meta.url));
  const ctx = {};
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(path.join(VENDOR, "marked.min.js"), "utf8"), ctx);
  assert.deepEqual({ ...mod.MARKED_OPTIONS }, { breaks: true });
  const parts = ["think\nline", "a\nb **c** <div><div>x</div></div>"];
  const reply = mod.markdownWorkerReply(ctx.marked, { id: 7, parts: [null, parts[1]] });
  assert.equal(reply.id, 7);
  assert.deepEqual(reply.html, [null, ctx.marked.parse(parts[1], { breaks: true })]);
  assert.match(reply.html[1], /<br>/, "breaks:true applied");
  assert.equal(reply.depth, 3, "p > div > div");
  const failed = mod.markdownWorkerReply({ parse() { throw new Error("boom"); } }, { id: 8, parts: ["x"] });
  assert.deepEqual(failed, { id: 8, error: "boom" });
});

test("the worker module loads the vendored marked and answers with markdownWorkerReply", () => {
  const src = fs.readFileSync(path.join(STATIC, "app", "markdown-worker.js"), "utf8");
  assert.match(src, /^import "\.\.\/vendor\/marked\.min\.js";\r?$/m);
  assert.match(src, /^import \{ markdownWorkerReply \} from "\.\/markdown-parse\.js";\r?$/m);
  assert.match(src, /self\.postMessage\(markdownWorkerReply\(self\.marked, e\.data\)\)/);
});

test("a long reply is parsed in the worker and renders exactly as on the page", async () => {
  const win = realPipeline();
  const workers = installWorker(win);
  const t = target(win);
  win.renderMarkdown(t, LONG_REPLY, { final: true });
  await settle();
  assert.equal(workers.length, 1);
  assert.equal(workers[0].url, "/app/markdown-worker.js");
  assert.equal(workers[0].options.type, "module");
  assert.equal(workers[0].posted.length, 1);
  assert.equal(t.querySelector("strong"), null, "not formatted before the worker answers");
  workers[0].answer();
  assert.equal(t.innerHTML, syncRender(LONG_REPLY, { final: true }));
  assert.ok(t.querySelectorAll("h2").length === 120);
});

test("a short reply renders on the spot without the worker", () => {
  const win = realPipeline();
  const workers = installWorker(win);
  const t = target(win);
  win.renderMarkdown(t, "short **reply**");
  assert.equal(t.querySelector("strong").textContent, "reply");
  assert.equal(workers.length, 0);
});

test("a reply waiting for the worker holds its plain text hidden, then shows it", async () => {
  const win = realPipeline();
  installWorker(win);
  runScript(win, "MARKDOWN_PLAIN_REVEAL_MS = 20;");
  const t = target(win);
  win.renderMarkdown(t, LONG_REPLY);
  const main = t.querySelector(".md-main");
  assert.ok(main.textContent.includes("## Part 0"), "the text is there for readers of the DOM");
  assert.equal(main.style.visibility, "hidden");
  await settle(60);
  assert.equal(main.style.visibility, "");
  assert.equal(main.getAttribute("style"), null);
});

test("renders of one message coalesce: one parse at a time, the latest text next", async () => {
  const win = realPipeline();
  const workers = installWorker(win);
  const t = target(win);
  win.renderMarkdown(t, LONG_REPLY + "\n\nA");
  await settle();
  win.renderMarkdown(t, LONG_REPLY + "\n\nB");
  win.renderMarkdown(t, LONG_REPLY + "\n\nC");
  await settle();
  const w = workers[0];
  assert.equal(w.posted.length, 1, "no second parse while one runs");
  w.answer(0);
  await settle();
  assert.equal(w.posted.length, 2);
  assert.ok(w.posted[1].parts[1].endsWith("\n\nC"), "the latest text is parsed next");
  w.answer(1);
  assert.match(t.querySelector(".md-main").lastElementChild.textContent, /^C$/);
});

test("an older worker result never replaces a later render applied on the spot", async () => {
  const win = realPipeline();
  const workers = installWorker(win);
  const t = target(win);
  win.renderMarkdown(t, LONG_REPLY);
  await settle();
  win.renderMarkdown(t, "*[error: refused]*");
  workers[0].answer();
  assert.equal(t.querySelector(".md-main").textContent.trim(), "[error: refused]");
});

test("a parse past the budget shows escaped plain text with line breaks and replaces the worker", async () => {
  const win = realPipeline();
  const workers = installWorker(win);
  runScript(win, "MARKDOWN_PARSE_BUDGET_MS = 30;");
  const t = target(win);
  const text = LONG_REPLY + "\n<b>not bold</b>\nlast line";
  win.renderMarkdown(t, text, { final: true });
  await settle(80);
  assert.equal(workers[0].terminated, true, "the stuck worker is stopped");
  const main = t.querySelector(".md-main");
  assert.equal(main.querySelector("b, strong, h2, a"), null, "nothing is formatted");
  assert.ok(main.textContent.includes("<b>not bold</b>"), "markup is shown as text");
  assert.equal(main.querySelectorAll("br").length, text.split("\n").length - 1);
  assert.equal(main.style.visibility, "");
  const note = main.querySelector(".md-plain-note");
  assert.ok(note && note.textContent.length > 0, "a note says formatting was skipped");
  win.renderMarkdown(t, LONG_REPLY + "\n\nmore");
  await settle();
  assert.equal(workers.length, 2, "the next parse gets a new worker");
  workers[1].answer();
  assert.equal(t.querySelectorAll("h2").length, 120);
});

test("a worker that fails to load is recorded and replies are formatted on the page", async () => {
  const win = realPipeline();
  const workers = installWorker(win);
  const t = target(win);
  win.renderMarkdown(t, LONG_REPLY);
  await settle();
  workers[0].onerror({ message: "blocked", preventDefault() {} });
  assert.equal(t.querySelectorAll("h2").length, 120, "the waiting reply is formatted on the page");
  assert.ok(win.recentGuiErrors().some((e) => e.text.includes("Markdown worker unavailable")));
  const t2 = target(win);
  win.renderMarkdown(t2, LONG_REPLY + "\n\nx");
  assert.equal(t2.querySelectorAll("h2").length, 120, "later replies render on the spot");
  assert.equal(workers.length, 1, "no new worker is started");
});

test("a parse that throws or nests too deep shows plain text with a note", () => {
  const win = realPipeline();
  const deep = target(win);
  win.renderMarkdown(deep, "- ".repeat(150) + "x");
  assert.equal(deep.querySelector("ul"), null);
  assert.ok(deep.querySelector(".md-plain-note"));
  const ok = target(win);
  win.renderMarkdown(ok, "- a\n  - b\n    - c");
  assert.equal(ok.querySelectorAll("ul").length, 3, "ordinary nesting still renders");
  const real = win.marked;
  win.marked = { parse() { throw new Error("Maximum call stack size exceeded"); } };
  const thrown = target(win);
  win.renderMarkdown(thrown, "> quote");
  win.marked = real;
  assert.equal(thrown.querySelector(".md-main").firstChild.nodeValue, "> quote");
  assert.ok(thrown.querySelector(".md-plain-note"));
});

test("a reply the worker formatted before renders on the spot in a new message", async () => {
  const win = realPipeline();
  const workers = installWorker(win);
  const first = target(win);
  win.renderMarkdown(first, LONG_REPLY);
  await settle();
  workers[0].answer();
  const again = target(win);
  win.renderMarkdown(again, LONG_REPLY);
  assert.equal(again.querySelectorAll("h2").length, 120);
  await settle();
  assert.equal(workers[0].posted.length, 1, "not parsed twice");
});

test("repeated **$$ does not grow the text protectMath hands to marked", () => {
  const win = realPipeline();
  for (const unit of ["**$$", "~~$$", "$", "\\$", "($$)"]) {
    const text = unit.repeat(Math.ceil(40000 / unit.length));
    const out = win.protectMath(text).text;
    assert.ok(out.length <= text.length, `${unit}: ${text.length} -> ${out.length}`);
  }
  const t = target(win);
  const text = "**$$".repeat(50) + " and \\$5 and $x^2$";
  win.renderMarkdown(t, text);
  assert.equal((t.textContent.match(/\$/g) || []).length, 100 + 1 + 2, "every dollar sign comes back");
  assert.ok(t.textContent.includes("$5") && t.textContent.includes("$x^2$"));
});

test("only the shared parse step calls marked", () => {
  const offenders = [];
  const call = /\bmarked(?:\.(?!min\.js)[A-Za-z_$]|\()/;
  const walk = (dir) => {
    for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, ent.name);
      if (ent.isDirectory()) {
        if (ent.name !== "vendor" && ent.name !== "node_modules") walk(p);
      } else if (/\.(m?js|html)$/.test(ent.name)) {
        const src = fs.readFileSync(p, "utf8")
          .replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:"'])\/\/.*$/gm, "$1");
        if (call.test(src)) offenders.push(path.relative(ROOT, p));
      }
    }
  };
  walk(path.join(ROOT, "localm"));
  assert.deepEqual(offenders, [], "call marked only through parseMarkdownParts (app/markdown-parse.js)");
  const helpers = fs.readFileSync(path.join(STATIC, "app", "helpers.js"), "utf8");
  assert.equal((helpers.match(/parseMarkdownParts\(marked, /g) || []).length, 1);
});
