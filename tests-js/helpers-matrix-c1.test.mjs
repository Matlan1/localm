// SPDX-License-Identifier: AGPL-3.0-or-later
// Shared chat helpers: error toasts leave after a readable time and can be taken
// back, a code block's copy reports a refused clipboard, a formula that starts
// with a digit is typeset, an opened Thoughts box stays open when the reply is
// redrawn, a transparent wallpaper keeps its transparency, the download box words
// a stopped download and names a workflow-file source, a missing native model
// names no node it does not have, and closing the image question is not a
// "Do not load".
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadApp, runScript } from "./harness.mjs";

const ROOT = new URL("../", import.meta.url);
const STATIC = new URL("localm/plugins/gui/static/", ROOT);
const VENDOR = new URL("vendor/", STATIC);
const readVendored = (n) => fs.readFileSync(new URL(n, VENDOR), "utf8");

const evalIn = (win, expr) => {
  win.__err = null;
  runScript(win,
    `try { window.__out = (${expr}); } catch (e) { window.__out = undefined; window.__err = String(e); }`);
  if (win.__err) throw new Error(`${expr}: ${win.__err}`);
  return win.__out;
};

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(cond, what) {
  for (let i = 0; i < 300; i++) {
    if (cond()) return;
    await wait(10);
  }
  assert.fail(`timed out waiting for ${what}`);
}

/* ------------------------------ error toasts ------------------------------ */

/** A fake clock for the window: while a function passed to run() executes,
 *  timers of at least 5 s are held here and run only on advance(ms); Date.now()
 *  returns the fake time from now on. Every other timer stays real. */
function fakeClock(win) {
  const realSet = win.setTimeout.bind(win);
  const realClear = win.clearTimeout.bind(win);
  let now = 1_000_000;
  let next = 1;
  let capturing = false;
  const timers = new Map();
  win.Date.now = () => now;
  win.setTimeout = (fn, ms, ...args) => {
    if (capturing && typeof ms === "number" && ms >= 5000) {
      const id = `fake-${next++}`;
      timers.set(id, { fn, at: now + ms, ms });
      return id;
    }
    return realSet(fn, ms, ...args);
  };
  win.clearTimeout = (id) => {
    if (timers.has(id)) timers.delete(id);
    else realClear(id);
  };
  const run = (fn) => {
    capturing = true;
    try { return fn(); } finally { capturing = false; }
  };
  return {
    timers,
    run,
    advance(ms) {
      now += ms;
      for (;;) {
        const due = [...timers.entries()].filter(([, v]) => v.at <= now)
          .sort((a, b) => a[1].at - b[1].at);
        if (!due.length) return;
        const [id, v] = due[0];
        timers.delete(id);
        run(v.fn);
      }
    },
  };
}

const stack = (win) => [...win.document.querySelectorAll("#toast-errors .toast-error-text")]
  .map((n) => n.textContent);

/** loadApp() once its boot has settled, with no error toast left from it. */
async function boot(opts) {
  const { window: win } = loadApp(opts);
  await wait(60);
  win.dismissAllErrorToasts();
  return win;
}

test("an error toast leaves after a readable time that grows with its text, and stays in Recent errors", async () => {
  const win = await boot();
  const clock = fakeClock(win);
  clock.run(() => win.toast("Enter a prompt", true));
  const life = evalIn(win, 'errorToastLifetimeMs("Enter a prompt")');
  assert.ok(life > evalIn(win, "TOAST_INFO_MS"), "longer than an info toast");
  assert.ok(evalIn(win, 'errorToastLifetimeMs("x".repeat(300))') > life, "a longer text stays longer");
  assert.ok(evalIn(win, 'errorToastLifetimeMs("x".repeat(100000))') <= evalIn(win, "TOAST_ERROR_MAX_MS"));
  assert.deepEqual([...clock.timers.values()].map((v) => v.ms), [life]);

  clock.advance(life - 1);
  assert.deepEqual(stack(win), ["Enter a prompt"], "still shown just before its time");
  clock.advance(1);
  assert.deepEqual(stack(win), [], "gone once its time is up");
  assert.equal(win.document.getElementById("toast-errors").hidden, true);
  assert.ok(win.recentGuiErrors().some((e) => e.text === "Enter a prompt"),
    "the error is still listed in Recent errors");
});

test("an error toast does not leave while it is pointed at or focused", async () => {
  const win = await boot();
  const clock = fakeClock(win);
  const box = win.document.getElementById("toast-errors");
  clock.run(() => win.toast("still loading", true));
  const life = evalIn(win, 'errorToastLifetimeMs("still loading")');

  clock.advance(3000);
  clock.run(() => box.dispatchEvent(new win.MouseEvent("mouseenter")));
  assert.equal(clock.timers.size, 0, "hovering stops the timer");
  clock.advance(60_000);
  assert.deepEqual(stack(win), ["still loading"]);
  clock.run(() => box.dispatchEvent(new win.MouseEvent("mouseleave")));
  assert.deepEqual([...clock.timers.values()].map((v) => v.ms), [life - 3000],
    "the time left goes on from where it stopped");

  clock.run(() => win.document.querySelector("#toast-errors .toast-error-close").focus());
  assert.equal(clock.timers.size, 0, "focus inside the stack stops the timer");
  clock.advance(60_000);
  assert.deepEqual(stack(win), ["still loading"]);
  clock.run(() => win.document.querySelector("#toast-errors .toast-error-close").blur());
  assert.equal(clock.timers.size, 1, "leaving the stack starts the timer again");
  clock.advance(life);
  assert.deepEqual(stack(win), [], "leaves once neither pointer nor focus holds it");
});

test("a repeated error restarts its time, and the returned function takes a toast back", async () => {
  const win = await boot();
  const clock = fakeClock(win);
  const life = evalIn(win, 'errorToastLifetimeMs("same")');
  clock.run(() => win.toast("same", true));
  clock.advance(life - 100);
  clock.run(() => win.toast("same", true));
  clock.advance(life - 100);
  assert.deepEqual(stack(win), ["same"], "the repeat started the time again");
  assert.equal(win.errorToasts()[0].count, 2);

  const retract = clock.run(() => win.toast("Enter a prompt", true));
  assert.equal(typeof retract, "function");
  retract();
  assert.deepEqual(stack(win), ["same"], "only the retracted toast went");
  assert.ok(win.recentGuiErrors().some((e) => e.text === "Enter a prompt"), "and it stays recorded");

  const info = win.toast("Saved");
  assert.equal(win.document.getElementById("toast").className, "show");
  info();
  assert.equal(win.document.getElementById("toast").className, "", "an info toast can be taken back too");
  const old = win.toast("first info");
  win.toast("second info");
  old();
  assert.equal(win.document.getElementById("toast").className, "show",
    "taking back an older info toast leaves the current one");
  assert.equal(win.document.getElementById("toast").textContent, "second info");
});

/* ------------------------------ code copy ------------------------------ */

function codeBlock(win) {
  const pre = win.document.createElement("pre");
  const code = win.document.createElement("code");
  code.textContent = "print(1)";
  pre.appendChild(code);
  win.document.body.appendChild(pre);
  win.enhanceCodeBlock(pre);
  return pre.querySelector(".copy-btn");
}

test("a code block's copy says so when the browser refuses the clipboard", async () => {
  const win = await boot();
  Object.defineProperty(win.navigator, "clipboard", {
    configurable: true,
    value: { writeText: () => Promise.reject(new Error("NotAllowedError")) },
  });
  const btn = codeBlock(win);
  btn.click();
  await until(() => stack(win).length > 0, "the error toast");
  assert.equal(btn.textContent, "copy", "never claims it copied");
  assert.deepEqual(stack(win), ["Could not copy - your browser blocked clipboard access"]);

  Object.defineProperty(win.navigator, "clipboard", { configurable: true, value: undefined });
  win.dismissAllErrorToasts();
  btn.click();
  await until(() => stack(win).length > 0, "the error toast without a clipboard");
  assert.equal(btn.textContent, "copy");
});

test("a code block's copy shows copied when the write succeeds", async () => {
  const win = await boot();
  const writes = [];
  Object.defineProperty(win.navigator, "clipboard", {
    configurable: true,
    value: { writeText: (t) => { writes.push(t); return Promise.resolve(); } },
  });
  const btn = codeBlock(win);
  btn.click();
  await until(() => btn.textContent === "copied", "the copied label");
  assert.equal(writes.length, 1);
  assert.deepEqual(stack(win), []);
});

test("no new clipboard write bypasses the shared copy helper", () => {
  // The most direct navigator.clipboard.writeText calls each file may hold;
  // every other write goes through copyTextWithFeedback.
  const allowed = { "app/chat.js": 2, "app/media-gallery.js": 1, "pages/models.js": 1, "pages/settings.js": 2 };
  const found = {};
  for (const dir of ["app", "pages"]) {
    for (const name of fs.readdirSync(new URL(dir + "/", STATIC))) {
      if (!name.endsWith(".js")) continue;
      const src = fs.readFileSync(new URL(`${dir}/${name}`, STATIC), "utf8");
      const n = (src.match(/clipboard\.writeText\(/g) || []).length;
      if (n) found[`${dir}/${name}`] = n;
    }
  }
  const helpers = fs.readFileSync(new URL("app/helpers.js", STATIC), "utf8");
  assert.equal(found["app/helpers.js"], undefined, "helpers.js writes only inside copyTextWithFeedback");
  assert.equal((helpers.match(/\.writeText\(/g) || []).length, 1);
  assert.match(helpers, /async function copyTextWithFeedback[\s\S]*?clip\.writeText\(text\)/);
  assert.match(helpers, /export function enhanceCodeBlock[\s\S]*?copyTextWithFeedback\(btn,/);
  for (const [file, n] of Object.entries(found)) {
    assert.ok(n <= (allowed[file] || 0), `${file} writes to the clipboard directly; use copyTextWithFeedback`);
  }
});

/* ------------------------------ math ------------------------------ */

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

function renderInto(win, text) {
  const target = win.document.createElement("div");
  win.document.body.appendChild(target);
  win.renderMarkdown(target, text, { final: true });
  return target;
}
const tex = (el) => [...el.querySelectorAll("annotation")].map((a) => a.textContent);

test("an inline formula that starts with a digit is typeset", () => {
  const win = loadRealPipeline();
  for (const [src, want] of [
    [String.raw`So $14 \times 2 = 28$ holds.`, [String.raw`14 \times 2 = 28`]],
    ["Then $2^{10}$ is 1024.", ["2^{10}"]],
    ["And $3 = 3$ too.", ["3 = 3"]],
    ["Here $10_2$ is two.", ["10_2"]],
  ]) {
    assert.deepEqual(tex(renderInto(win, src)), want, src);
  }
});

test("prices that start with a digit stay text", () => {
  const win = loadRealPipeline();
  for (const src of [
    "$5 and $10", "It costs $5 and $10, or $5-$10, not $20.", "$5 each; the $ sign is the currency",
    "Pay $5+$3 shipping", "$5 or $6 = more", "US$5 and US$10 = cheap", String.raw`$5 \$`,
    "Costs $5 + $3 = $8 total.",
  ]) {
    const out = renderInto(win, src);
    assert.equal(out.querySelectorAll(".katex, .katex-error").length, 0, src);
    assert.equal(out.textContent.trim(), src.replace(/\\\$/g, "$"), src);
  }
});

/* ------------------------------ Thoughts box ------------------------------ */

function openByPerson(win, target) {
  const det = target.querySelector("details.think-block");
  det.querySelector("summary").dispatchEvent(new win.Event("click"));
  det.open = !det.open;
  det.dispatchEvent(new win.Event("toggle"));
  return det;
}

test("a Thoughts box the person opened stays open when the turn's final redraw rebuilds it", () => {
  const { window: win } = loadApp();
  const live = win.document.createElement("div");
  win.document.body.appendChild(live);
  win.renderMarkdown(live, "<think>\nstep one");
  win.renderMarkdown(live, "<think>\nstep one, step two\n</think>\nAnswer.");
  assert.equal(live.querySelector("details.think-block").open, false, "collapsed by default when done");
  openByPerson(win, live);
  assert.equal(live.querySelector("details.think-block").open, true);

  const rebuilt = renderInto(win, "<think>\nstep one, step two\n</think>\nAnswer.");
  assert.equal(rebuilt.querySelector("details.think-block").open, true,
    "the rebuilt box keeps the person's choice");

  const other = renderInto(win, "<think>\nother reasoning\n</think>\nAnswer.");
  assert.equal(other.querySelector("details.think-block").open, false, "an untouched box stays shut");
});

test("a Thoughts box opened while the reasoning streams stays open after the turn ends", () => {
  const { window: win } = loadApp();
  const live = win.document.createElement("div");
  win.document.body.appendChild(live);
  win.renderMarkdown(live, "<think>\nfirst");
  const det = live.querySelector("details.think-block");
  det.querySelector("summary").dispatchEvent(new win.Event("click"));
  win.renderMarkdown(live, "<think>\nfirst, second");
  win.renderMarkdown(live, "<think>\nfirst, second, third\n</think>\nDone.");
  assert.equal(det.open, true, "the person's open box was not shut when the reasoning ended");

  const rebuilt = renderInto(win, "<think>\nfirst, second, third\n</think>\nDone.");
  assert.equal(rebuilt.querySelector("details.think-block").open, true);
});

/* ------------------------------ chat background ------------------------------ */

/** Stubs the canvas and image decoding: every picture is *w* x *h*; *alpha*
 *  is the alpha of every pixel; toDataURL returns a URI of *size(type, w, h)*
 *  characters and is recorded in *calls*. */
function stubCanvas(win, { w = 400, h = 200, alpha = 255, size }) {
  const calls = [];
  const reads = [];
  win.Image = class {
    constructor() { this.width = w; this.height = h; }
    set src(v) { this._src = v; setTimeout(() => this.onload && this.onload(), 0); }
    get src() { return this._src; }
  };
  const proto = win.HTMLCanvasElement.prototype;
  proto.getContext = function getContext() {
    return {
      drawImage() {}, clearRect() {},
      getImageData: (_x, _y, cw, ch) => {
        reads.push([cw, ch]);
        const data = new Uint8ClampedArray(cw * ch * 4).fill(255);
        for (let i = 3; i < data.length; i += 4) data[i] = alpha;
        return { data };
      },
    };
  };
  proto.toDataURL = function toDataURL(type = "image/png") {
    calls.push({ type, w: this.width, h: this.height });
    const head = `data:${type};base64,`;
    return head + "A".repeat(Math.max(0, size(type, this.width, this.height) - head.length));
  };
  return { calls, reads };
}

const pngFile = (win, type = "image/png") => new win.File([new Uint8Array([1, 2, 3])], "bg.png", { type });

test("a chat background with transparency is saved as PNG, an opaque one as JPEG", async () => {
  {
    const { window: win } = loadApp();
    const { calls } = stubCanvas(win, { alpha: 0, size: () => 1000 });
    const uri = await win.fileToBackgroundDataUri(pngFile(win));
    assert.match(uri, /^data:image\/png;base64,/, "transparent parts stay transparent");
    assert.deepEqual(calls.map((c) => c.type), ["image/png"]);
  }
  {
    const { window: win } = loadApp();
    const { calls } = stubCanvas(win, { alpha: 255, size: () => 1000 });
    const uri = await win.fileToBackgroundDataUri(pngFile(win));
    assert.match(uri, /^data:image\/jpeg;base64,/, "an opaque PNG is still made small as JPEG");
    assert.deepEqual(calls.map((c) => c.type), ["image/jpeg"]);
  }
  {
    const { window: win } = loadApp();
    const { reads } = stubCanvas(win, { alpha: 0, size: () => 1000 });
    const uri = await win.fileToBackgroundDataUri(pngFile(win, "image/jpeg"));
    assert.match(uri, /^data:image\/jpeg;base64,/);
    assert.equal(reads.length, 0, "a JPEG has no transparency to look for");
  }
});

test("a transparent background too long as PNG is scaled down until the server takes it", async () => {
  const { window: win } = loadApp();
  const limit = evalIn(win, "CHAT_BACKGROUND_MAX_DATA_URI_LEN");
  const { calls } = stubCanvas(win, { w: 1920, h: 1080, alpha: 10, size: (_t, w, h) => w * h * 3 });
  const uri = await win.fileToBackgroundDataUri(pngFile(win));
  assert.match(uri, /^data:image\/png;base64,/);
  assert.ok(uri.length <= limit, `${uri.length} > ${limit}`);
  assert.ok(calls.length > 1, "it was scaled down at least once");
  const last = calls[calls.length - 1];
  assert.ok(last.w / last.h > 1.7 && last.w / last.h < 1.8, "the shape is kept");
});

test("the background size limit matches the server's", () => {
  const { window: win } = loadApp();
  const py = fs.readFileSync(new URL("localm/settings_schema.py", ROOT), "utf8");
  const m = /^_BACKGROUND_MAX_DATA_URI_LEN = ([\d_]+)/m.exec(py);
  assert.ok(m, "settings_schema.py defines _BACKGROUND_MAX_DATA_URI_LEN");
  assert.equal(evalIn(win, "CHAT_BACKGROUND_MAX_DATA_URI_LEN"), Number(m[1].replace(/_/g, "")));
});

/* ------------------------------ download box ------------------------------ */

function sse(events) {
  const bytes = new TextEncoder().encode(
    events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join(""));
  let sent = false;
  return {
    ok: true, status: 200,
    body: { getReader: () => ({ async read() {
      if (sent) return { done: true, value: undefined };
      sent = true;
      return { done: false, value: bytes };
    } }) },
  };
}

function nativePart(source = {}) {
  return {
    filename: "flux-vae.safetensors", native: true, searchable: false,
    source: { repo: "owner/repo", file: "vae/flux-vae.safetensors", spec: "owner/repo:vae/flux-vae.safetensors",
              sha256: "ab".repeat(32), name: null, size_bytes: 1000, model_type: "vae", origin: "curated",
              ...source },
  };
}

function downloadFetch(endStatus) {
  return async (url, opts = {}) => {
    const u = String(url);
    if ((u === "/api/models/pull" || u === "/api/models/pull-comfy-source") && opts.method === "POST") {
      return { ok: true, status: 200, json: async () => ({ job_id: "job-1" }) };
    }
    if (u === "/api/jobs/job-1/events") {
      if (endStatus === "gone") return { ok: false, status: 404, statusText: "Not Found" };
      return sse([{ type: "line", text: "pulling" }, { type: "end", status: endStatus }]);
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

const modalOpen = (win) => win.document.getElementById("modal").style.display === "flex";
const modalText = (win) => win.document.getElementById("modal-body").textContent;
const clickButton = (win, label) => {
  const b = [...win.document.querySelectorAll("#modal-body button")].find((x) => x.textContent === label);
  assert.ok(b, `no "${label}" button: ${modalText(win)}`);
  b.click();
};

for (const [name, open] of [
  ["the native box", (win, part) => win.offerNativeDownloads([part], null)],
  ["the native box for several files", (win, part) => win.offerNativeDownloads([part, nativePart({
    repo: "other/repo", file: "te.gguf", spec: "other/repo:te.gguf" })], null)],
  ["the workflow-model box", (win, part) => win._offerModelDownload({ ...part, native: false }, null, "image")],
]) {
  test(`a stopped download reads as paused with its data kept, in ${name}`, async () => {
    const win = await boot({ fetchImpl: downloadFetch("cancelled") });
    const done = open(win, nativePart());
    await until(() => modalOpen(win), "the download box");
    clickButton(win, "Download");
    await done;
    const toastEl = win.document.getElementById("toast");
    assert.equal(toastEl.textContent,
      "Download paused: flux-vae.safetensors. The data saved so far is kept; downloading it again continues from there.");
    assert.equal(toastEl.className, "show", "a stop the person made is not an error");
    assert.doesNotMatch(toastEl.textContent, /cancelled|disconnected/);
  });
}

test("a lost connection and a failed download are worded, never the raw status", async () => {
  for (const [status, want] of [
    ["gone", "Download interrupted: flux-vae.safetensors. The connection to localm was lost; the download may still be running."],
    ["failed", "Download failed: flux-vae.safetensors"],
    ["error", "Download error: flux-vae.safetensors"],
  ]) {
    const win = await boot({ fetchImpl: downloadFetch(status) });
    const done = win.offerNativeDownloads([nativePart()], null);
    await until(() => modalOpen(win), "the download box");
    clickButton(win, "Download");
    assert.equal(await done, "failed");
    assert.deepEqual(stack(win), [want], status);
  }
});

const WORKFLOW_NOTE = "The workflow file names this download; localm does not list it, so check the repository before you download it. The file's sha256 is checked after the download.";
const WORKFLOW_NO_SHA_NOTE = "The workflow file names this download; localm does not list it and the workflow gives no sha256 to check it against, so check the repository before you download it.";

test("the native download box warns about a source named by a workflow file", async () => {
  for (const [source, want] of [
    [{ origin: "workflow" }, WORKFLOW_NOTE],
    [{ origin: "workflow", sha256: null }, WORKFLOW_NO_SHA_NOTE],
  ]) {
    const { window: win } = loadApp();
    win.offerNativeDownloads([nativePart(source)], null);
    await until(() => modalOpen(win), "the single box");
    assert.ok(modalText(win).includes(want), modalText(win));

    const { window: many } = loadApp();
    many.offerNativeDownloads([nativePart(source), nativePart({ file: "b.gguf", spec: "owner/repo:b.gguf" })], null);
    await until(() => modalOpen(many), "the box for several files");
    const items = [...many.document.querySelectorAll("#modal-body li")].map((li) => li.textContent);
    assert.ok(items[0].includes(want), "the workflow file's entry carries the warning");
    assert.ok(!items[1].includes("workflow file"), "a catalog entry does not");
  }
});

test("both download boxes word a source's origin the same way", async () => {
  for (const source of [{ origin: "curated" }, { origin: "huggingface" }, { origin: "workflow" },
                        { origin: "workflow", sha256: null }]) {
    const notes = [];
    for (const open of [(win, p) => win.offerNativeDownloads([p], null),
                        (win, p) => win._offerModelDownload({ ...p, native: false }, null, "image")]) {
      const { window: win } = loadApp();
      open(win, nativePart(source));
      await until(() => modalOpen(win), "the box");
      notes.push([...win.document.querySelectorAll("#modal-body p.sub")].map((p) => p.textContent));
    }
    assert.deepEqual(notes[0], notes[1], JSON.stringify(source));
  }
});

/* ------------------------------ missing model without a node ------------------------------ */

test("a missing native model with no node names no node", async () => {
  const win = await boot();
  win._reportUncuratedMiss({ filename: "lora-x.safetensors", native: true, slot: "lora" }, null);
  const text = win.document.getElementById("toast").textContent;
  assert.doesNotMatch(text, /undefined|needed by|ComfyUI/);
  assert.equal(text, "'lora-x.safetensors' is missing and the workflow gives no download source for it. Add the file with the Add a model card on the Models page, then try again.");

  win._reportUncuratedMiss({ filename: "c.safetensors", class_type: "CheckpointLoaderSimple",
    input_name: "ckpt_name", reason: "folder" }, null);
  assert.match(win.document.getElementById("toast").textContent, /needed by CheckpointLoaderSimple\.ckpt_name/);

  win._reportUncuratedMiss({ filename: "d.safetensors" }, null, "offline");
  assert.match(win.document.getElementById("toast").textContent, /network access is off/,
    "a message that names no node is kept as it is");
});

/* ------------------------------ image consent ------------------------------ */

function loadAsking() {
  const calls = [];
  const { window: win } = loadApp({
    fetchImpl: async (url, opts) => {
      const u = String(url);
      calls.push({ url: u });
      if (u.includes("/api/image-proxy") && !/[?&]consent=1(&|$)/.test(u)) {
        return { ok: false, status: 428, blob: async () => null,
                 json: async () => ({ detail: "ask first" }) };
      }
      return { ok: true, status: 200, blob: async () => new win.Blob([1, 2, 3]) };
    },
  });
  for (const f of ["marked.min.js", "purify.min.js"]) {
    const tag = win.document.createElement("script");
    tag.textContent = readVendored(f);
    win.document.head.appendChild(tag);
  }
  win.marked.setOptions({ breaks: true });
  if (!win.URL.createObjectURL) {
    win.URL.createObjectURL = () => "blob:mock/1";
    win.URL.revokeObjectURL = () => {};
  }
  return { win, calls };
}

function renderScoped(win, md) {
  const t = win.document.createElement("div");
  win.document.body.appendChild(t);
  win.renderMarkdown(t, md, { final: true, imageScope: "chat:1" });
  return t;
}
const consented = (calls) => calls.filter((c) => /\/api\/image-proxy.*[?&]consent=1(&|$)/.test(c.url));
const closeDialog = async (win) => {
  win.document.getElementById("modal-close").click();
  await wait(350);
};

test("closing the image question is a cancel, not a Do not load", async () => {
  const { win, calls } = loadAsking();
  const first = renderScoped(win, "![a](https://cdn.example.com/1.png)");
  await until(() => modalOpen(win), "the question");
  await closeDialog(win);
  assert.equal(consented(calls).length, 0, "nothing was loaded");
  const note = first.querySelector(".img-blocked");
  assert.ok(note, "the reader sees why the image is missing");
  assert.match(note.textContent, /closed without an answer/);
  assert.doesNotMatch(note.textContent, /You chose not to load/);

  renderScoped(win, "![b](https://cdn.example.com/2.png)");
  await until(() => modalOpen(win), "the question for the next image from that site");
  clickButton(win, "Show images from this site");
  await until(() => consented(calls).length === 1, "the allowed image");
});

test("the Ask again button on a closed question asks once more and shows the image", async () => {
  const { win, calls } = loadAsking();
  const t = renderScoped(win, "![a](https://cdn.example.com/1.png)");
  await until(() => modalOpen(win), "the question");
  await closeDialog(win);
  win.renderMarkdown(t, "![a](https://cdn.example.com/1.png) more", { final: true, imageScope: "chat:1" });
  await wait(30);
  assert.equal(modalOpen(win), false, "a re-render of the same reply does not ask again by itself");

  const again = t.querySelector(".img-blocked-retry");
  assert.ok(again, "the note offers to ask again");
  assert.equal(again.textContent, "Ask again");
  again.click();
  await until(() => modalOpen(win), "the question asked again");
  clickButton(win, "Show images from this site");
  await until(() => /^blob:/.test(t.querySelector("img")?.getAttribute("src") || ""), "the image");
  assert.equal(consented(calls).length, 1);
  assert.equal(t.querySelector(".img-blocked"), null);
});

test("one closed question covers every image of that site in the reply", async () => {
  const { win } = loadAsking();
  const t = renderScoped(win, ["1", "2", "3"].map((n) => `![${n}](https://cdn.example.com/${n}.png)`).join("\n\n"));
  await until(() => modalOpen(win), "the question");
  await closeDialog(win);
  assert.equal(modalOpen(win), false, "closing once does not open the question again for the other images");
  assert.equal(t.querySelectorAll(".img-blocked").length, 3);
});

test("Do not load is still remembered for the site", async () => {
  const { win, calls } = loadAsking();
  renderScoped(win, "![a](https://cdn.example.com/1.png)");
  await until(() => modalOpen(win), "the question");
  clickButton(win, "Do not load");
  await wait(30);
  const t = renderScoped(win, "![b](https://cdn.example.com/2.png)");
  await wait(30);
  assert.equal(modalOpen(win), false);
  assert.match(t.querySelector(".img-blocked").textContent, /You chose not to load images/);
  assert.equal(t.querySelector(".img-blocked-retry"), null, "an answer offers no Ask again");
  assert.equal(consented(calls).length, 0);
});
