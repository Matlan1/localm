// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for the browser plugin client_entry (localm/plugins/builtin/
// browser/static/browser.js).
//
// Same shape as tests-js/jobs.test.mjs: build a jsdom document with a
// <main id="main">, install it as the module's ambient globals, import the
// module and call register(ctx).
//
// The frame path is what these mostly pin. A frame arrives as a string from the
// server and is used as an image source, so it must be able to become a data:
// JPEG and NOTHING else.

import { test } from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";
import { readFile } from "node:fs/promises";

const HERE = dirname(fileURLToPath(import.meta.url));
const BROWSER_JS = join(HERE, "..", "localm", "plugins", "builtin", "browser",
                        "static", "browser.js");
const BROWSER_CSS = join(HERE, "..", "localm", "plugins", "builtin", "browser",
                         "static", "browser.css");

const NODE_SET_TIMEOUT = globalThis.setTimeout;

function makeEnv() {
  const dom = new JSDOM(
    `<!DOCTYPE html><html><body><main id="main"></main></body></html>`,
    { url: "http://localhost:8642/" });
  const win = dom.window;
  const calls = [];
  win.fetch = async (url, opts = {}) => {
    calls.push({ url: String(url), method: (opts.method || "GET").toUpperCase() });
    return { ok: true, status: 200, body: null, json: async () => ({}) };
  };
  global.window = win;
  global.document = win.document;
  global.fetch = win.fetch;
  // Node's timer, NOT jsdom's. Assigning jsdom's window.setTimeout onto the
  // global makes it recurse into itself without bound: its own
  // timerInitializationSteps calls the global setTimeout, which is by then
  // itself, and the first real timer the module schedules blows the stack.
  global.setTimeout = NODE_SET_TIMEOUT;
  return { win, calls };
}

async function load() {
  const mod = await import(pathToFileURL(BROWSER_JS).href + "?t=" + Math.random());
  return mod;
}

test("register builds the browser view once, into #main", async () => {
  const { win } = makeEnv();
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({}) });
  const view = win.document.getElementById("view-browser");
  assert.ok(view, "the view section is built");
  assert.equal(view.parentElement.id, "main");
  mod.register({ toast() {}, authHeaders: () => ({}) });
  assert.equal(win.document.querySelectorAll("#view-browser").length, 1,
    "a second register must not build a second view");
});

test("the view offers a url field and open/stop controls", async () => {
  const { win } = makeEnv();
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({}) });
  const view = win.document.getElementById("view-browser");
  assert.ok(view.querySelector("input.browser-url"));
  const buttons = [...view.querySelectorAll("button")].map((b) => b.textContent);
  assert.deepEqual(buttons, ["Open", "Stop", "Watch the agent"]);
  const watch = [...view.querySelectorAll("button")]
    .find((b) => b.textContent === "Watch the agent");
  assert.equal(watch.hidden, true,
    "the agent offer stays hidden until the server says one is running");
  assert.ok(view.querySelector("img.browser-frame"), "the live view surface");
});

test("register does not reach the network on its own", async () => {
  const { calls } = makeEnv();
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({}) });
  assert.deepEqual(calls, [],
    "opening the tab must not start a browser by itself");
});

// --------------------------------------------------------------------------- //
//  The frame payload is server-originating and becomes an image source.        //
// --------------------------------------------------------------------------- //

test("a base64 frame becomes a data: JPEG source", async () => {
  makeEnv();
  const mod = await load();
  const good = "/9j/4AAQSkZJRg==";
  assert.equal(mod.frameSrc(good), "data:image/jpeg;base64," + good);
});

test("a payload that is not base64 is refused, so it cannot become a source", async () => {
  makeEnv();
  const mod = await load();
  const hostile = [
    "javascript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "\" onerror=alert(1) x=\"",
    "http://evil.example/x.png",
    "abc<def",
    "a b",
    "",
    null,
    undefined,
    123,
  ];
  for (const value of hostile) {
    assert.equal(mod.frameSrc(value), null, String(value));
  }
});

test("a refused frame leaves the previous picture in place", async () => {
  const { win } = makeEnv();
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({}) });
  const img = win.document.querySelector("img.browser-frame");
  const good = mod.frameSrc("/9j/4AAQSkZJRg==");
  img.src = good;
  assert.equal(mod.frameSrc("javascript:alert(1)"), null);
  assert.equal(img.getAttribute("src"), good,
    "a refused frame must not blank or replace the live view");
});

test("the image starts with no source at all", async () => {
  const { win } = makeEnv();
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({}) });
  const img = win.document.querySelector("img.browser-frame");
  assert.equal(img.getAttribute("src"), null,
    "an unpainted live view must not request anything");
  assert.ok(img.alt, "the surface carries alt text");
});


// --------------------------------------------------------------------------- //
//  watchFrames: the reusable SSE-frame reader coder.js's inline mirror also   //
//  calls. None of the tab-level tests above exercise the reader loop at all - //
//  they hand back body: null, which short-circuits before it - so this is the //
//  only coverage of the actual event dispatch.                                //
// --------------------------------------------------------------------------- //

function fakeSSEBody(events) {
  const enc = new TextEncoder();
  const chunks = events.map((ev) => enc.encode("data: " + JSON.stringify(ev) + "\n\n"));
  let i = 0;
  return {
    getReader() {
      return {
        async read() {
          if (i < chunks.length) return { done: false, value: chunks[i++] };
          return { done: true, value: undefined };
        },
      };
    },
  };
}

test("watchFrames dispatches frame, line and end events in order", async () => {
  makeEnv();
  const mod = await load();
  const events = [
    { type: "frame", data: "AAA" },
    { type: "line", line: "hello" },
    { type: "frame", data: "BBB" },
    { type: "end", status: "done" },
  ];
  global.fetch = async () => ({ ok: true, status: 200, body: fakeSSEBody(events) });
  const frames = [];
  const lines = [];
  let endEv = null;
  await mod.watchFrames("j1", {
    authHeaders: () => ({ Authorization: "Bearer x" }),
    onFrame: (d) => frames.push(d),
    onLine: (t) => lines.push(t),
    onEnd: (ev) => { endEv = ev; },
  });
  assert.deepEqual(frames, ["AAA", "BBB"]);
  assert.deepEqual(lines, ["hello"]);
  assert.deepEqual(endEv, { type: "end", status: "done" });
});

test("watchFrames tells a fetch failure and a refused response apart", async () => {
  makeEnv();
  const mod = await load();

  global.fetch = async () => { throw new Error("network down"); };
  let failed = false;
  let unavailable = null;
  await mod.watchFrames("j1", {
    onFetchFailed: () => { failed = true; },
    onUnavailable: (code) => { unavailable = code; },
  });
  assert.equal(failed, true, "a thrown fetch must call onFetchFailed");
  assert.equal(unavailable, null, "onUnavailable fired for a fetch that never returned");

  global.fetch = async () => ({ ok: false, status: 404, body: null });
  failed = false;
  await mod.watchFrames("j1", {
    onFetchFailed: () => { failed = true; },
    onUnavailable: (code) => { unavailable = code; },
  });
  assert.equal(failed, false, "onFetchFailed fired for a response that DID arrive");
  assert.equal(unavailable, 404, "the refusing status code was not passed through");
});

test("watchFrames requests the job's own events route with the caller's auth", async () => {
  makeEnv();
  const mod = await load();
  const calls = [];
  global.fetch = async (url, opts) => {
    calls.push({ url: String(url), opts });
    return { ok: true, status: 200, body: fakeSSEBody([]) };
  };
  const signal = new AbortController().signal;
  await mod.watchFrames("job-42", {
    authHeaders: () => ({ Authorization: "Bearer tok" }),
    signal,
  });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "/api/jobs/job-42/events");
  assert.deepEqual(calls[0].opts.headers, { Authorization: "Bearer tok" });
  assert.equal(calls[0].opts.signal, signal);
});

// --------------------------------------------------------------------------- //
//  Watching the browser the coding agent drives.                              //
// --------------------------------------------------------------------------- //

test("the agent offer appears only when the server says one is running", async () => {
  const { win } = makeEnv();
  let available = false;
  win.fetch = async (url, opts = {}) => {
    const u = String(url);
    if (u.endsWith("/agent") && (opts.method || "GET").toUpperCase() === "GET") {
      return { ok: true, status: 200, json: async () => ({
        available, session_id: available ? "s1" : null }) };
    }
    return { ok: true, status: 200, body: null, json: async () => ({}) };
  };
  global.fetch = win.fetch;
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({}) });
  const view = win.document.getElementById("view-browser");
  const watch = [...view.querySelectorAll("button")]
    .find((b) => b.textContent === "Watch the agent");

  // No agent browser: the tab must not offer something that is not there.
  await win.onViewShown("browser");
  for (let i = 0; i < 8; i++) await Promise.resolve();
  assert.equal(watch.hidden, true,
    "offered to watch an agent browser that is not running");

  // One appears: the offer shows up on the next time the tab is shown.
  available = true;
  await win.onViewShown("browser");
  for (let i = 0; i < 8; i++) await Promise.resolve();
  assert.equal(watch.hidden, false,
    "an agent browser is running and the tab did not offer to show it");
});

test("watching the agent posts to the agent route, not the tab own session", async () => {
  const { win, calls } = makeEnv();
  win.fetch = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, method: (opts.method || "GET").toUpperCase() });
    if (u.endsWith("/agent") && (opts.method || "GET").toUpperCase() === "GET") {
      return { ok: true, status: 200, json: async () => ({ available: true, session_id: "s1" }) };
    }
    if (u.endsWith("/agent")) {
      return { ok: true, status: 200, json: async () => ({ job_id: "j1", session_id: "s1" }) };
    }
    return { ok: true, status: 200, body: null, json: async () => ({}) };
  };
  global.fetch = win.fetch;
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({}) });
  const view = win.document.getElementById("view-browser");
  const watch = [...view.querySelectorAll("button")]
    .find((b) => b.textContent === "Watch the agent");

  watch.onclick();
  for (let i = 0; i < 8; i++) await Promise.resolve();

  const posts = calls.filter((c) => c.method === "POST");
  assert.ok(posts.length > 0, "watching the agent sent no request at all");
  assert.ok(posts.every((c) => !c.url.endsWith("/session")),
    "watching the agent opened a NEW browser instead of attaching to the agent one");
  assert.ok(posts.some((c) => c.url.endsWith("/agent")),
    "the agent route was never called");
});

// --------------------------------------------------------------------------- //
//  Driving the browser to more than one address.                              //
//                                                                             //
//  The tab could reach exactly ONE url per session: nothing ever called        //
//  POST /api/browser/navigate, and the address field was disabled the moment   //
//  a browser opened, so a second address could not even be typed. A second     //
//  Open re-POSTed /session and took that route's "already open" refusal.       //
// --------------------------------------------------------------------------- //

/** A fetch double that records every call and answers the browser routes. */
function wireRoutes(win, calls, { navigateOk = true } = {}) {
  win.fetch = async (url, opts = {}) => {
    const u = String(url);
    const m = (opts.method || "GET").toUpperCase();
    calls.push({ url: u, method: m, body: opts.body });
    if (u.endsWith("/session") && m === "POST") {
      return { ok: true, status: 200, json: async () => ({ job_id: "j1" }) };
    }
    if (u.endsWith("/navigate")) {
      return { ok: true, status: 200,
               json: async () => (navigateOk
                 ? { ok: true, url: "https://second.example/" }
                 : { ok: false, refused: "blocked by policy" }) };
    }
    if (u.endsWith("/agent") && m === "GET") {
      return { ok: true, status: 200,
               json: async () => ({ available: true, session_id: "s1" }) };
    }
    if (u.endsWith("/agent") && m === "POST") {
      return { ok: true, status: 200,
               json: async () => ({ job_id: "aj1", session_id: "s1" }) };
    }
    return { ok: true, status: 200, body: null, json: async () => ({}) };
  };
  global.fetch = win.fetch;
}

async function settle(n = 25) {
  for (let i = 0; i < n; i++) await Promise.resolve();
}

function controls(win) {
  const view = win.document.getElementById("view-browser");
  const buttons = [...view.querySelectorAll("button")];
  return {
    view,
    url: view.querySelector("input.browser-url"),
    go: buttons[0],
    stop: buttons[1],
    watch: buttons[2],
  };
}

test("the address field stays usable once a browser is open", async () => {
  const { win, calls } = makeEnv();
  wireRoutes(win, calls);
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({}) });
  const c = controls(win);

  c.url.value = "https://first.example/";
  c.go.onclick();
  await settle();

  assert.equal(c.url.disabled, false,
    "the address field was disabled while a browser was open, so no second "
    + "address could ever be typed into it");
});

test("a second address navigates the open browser instead of re-opening it", async () => {
  const { win, calls } = makeEnv();
  wireRoutes(win, calls);
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({}) });
  const c = controls(win);

  c.url.value = "https://first.example/";
  c.go.onclick();
  await settle();

  c.url.value = "https://second.example/";
  c.go.onclick();
  await settle();

  const posts = calls.filter((x) => x.method === "POST");
  const opened = posts.filter((x) => x.url.endsWith("/session"));
  const navigated = posts.filter((x) => x.url.endsWith("/navigate"));

  assert.equal(navigated.length, 1,
    "the second address never reached POST /api/browser/navigate, so the tab "
    + "can still only ever show one page per session");
  assert.equal(JSON.parse(navigated[0].body).url, "https://second.example/",
    "the navigate call did not carry the address that was typed");
  assert.equal(opened.length, 1,
    "the second address opened a SECOND browser instead of driving the open one");
});

test("Enter in the address field navigates too, not just the button", async () => {
  const { win, calls } = makeEnv();
  wireRoutes(win, calls);
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({}) });
  const c = controls(win);

  c.url.value = "https://first.example/";
  c.go.onclick();
  await settle();

  c.url.value = "https://second.example/";
  c.url.onkeydown({ key: "Enter" });
  await settle();

  assert.equal(calls.filter((x) => x.url.endsWith("/navigate")).length, 1,
    "Enter did not navigate the open browser");
});

test("a refused destination is reported rather than shown as a blank page", async () => {
  const { win, calls } = makeEnv();
  wireRoutes(win, calls, { navigateOk: false });
  const toasts = [];
  const mod = await load();
  mod.register({ toast: (t) => toasts.push(t), authHeaders: () => ({}) });
  const c = controls(win);

  c.url.value = "https://first.example/";
  c.go.onclick();
  await settle();
  c.url.value = "https://blocked.example/";
  c.go.onclick();
  await settle();

  const status = c.view.querySelector(".browser-status").textContent;
  assert.match(status, /blocked by policy/,
    "a refusal left the status saying nothing about why nothing loaded");
  assert.ok(toasts.length > 0, "a refusal was not surfaced to the user");
});

// --------------------------------------------------------------------------- //
//  The agent's browser is WATCHED, never driven.                              //
//                                                                             //
//  /api/browser/navigate resolves the caller's own gui- session, so a Go while //
//  watching would drive a different browser than the one on screen.           //
// --------------------------------------------------------------------------- //

test("watching the agent leaves the tab read-only", async () => {
  const { win, calls } = makeEnv();
  wireRoutes(win, calls);
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({}) });
  const c = controls(win);

  c.watch.onclick();
  await settle();

  assert.equal(c.url.disabled, true,
    "the address bar is live while watching the agent, so a typed url would "
    + "drive a browser other than the one being shown");
  assert.equal(c.go.disabled, true, "Go is offered while watching the agent");

  c.url.value = "https://elsewhere.example/";
  c.go.onclick();
  await settle();
  assert.deepEqual(calls.filter((x) => x.url.endsWith("/navigate")), [],
    "watching the agent still sent a navigate, driving a browser the viewer "
    + "cannot see");
});

test("stopping cancels the job, so the worker does not outlive the viewer", async () => {
  const { win, calls } = makeEnv();
  wireRoutes(win, calls);
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({}) });
  const c = controls(win);

  c.url.value = "https://first.example/";
  c.go.onclick();
  await settle();
  c.stop.onclick();
  await settle();

  assert.ok(calls.some((x) => x.method === "POST" && x.url.includes("/api/jobs/j1/cancel")),
    "Stop left the browser job running: its worker loops until cancelled, so "
    + "the session and its screencast outlive the viewer");
});

test("stopping the agent view does not close the tab own session", async () => {
  const { win, calls } = makeEnv();
  wireRoutes(win, calls);
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({}) });
  const c = controls(win);

  c.watch.onclick();
  await settle();
  c.stop.onclick();
  await settle();

  assert.ok(calls.some((x) => x.url.includes("/api/jobs/aj1/cancel")),
    "leaving the agent view never cancelled the view job, so the agent "
    + "screencast stays on with nobody watching");
  assert.deepEqual(calls.filter((x) => x.url.endsWith("/browser/stop")), [],
    "leaving the agent view posted /browser/stop, which closes the CALLER own "
    + "browser rather than detaching from the agent one");
});

// --------------------------------------------------------------------------- //
//  The tab ships its own styling.                                             //
// --------------------------------------------------------------------------- //

test("the tab loads its own stylesheet, exactly once", async () => {
  const { win } = makeEnv();
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({}) });

  const sheets = () => [...win.document.querySelectorAll("link[rel=stylesheet]")]
    .filter((l) => String(l.href).endsWith("browser.css"));
  assert.equal(sheets().length, 1,
    "the tab loaded no stylesheet, so every control it renders is unstyled");

  mod.register({ toast() {}, authHeaders: () => ({}) });
  assert.equal(sheets().length, 1, "a second register added a second stylesheet");
});

test("the stylesheet bounds the frame so it cannot overflow its container", async () => {
  const css = await readFile(BROWSER_CSS, "utf8");
  const rule = css.match(/\.browser-frame\s*\{[^}]*\}/);
  assert.ok(rule, "no .browser-frame rule at all");
  assert.match(rule[0], /max-width:\s*100%/,
    "the screencast is up to 1280px wide, so without a max width it overflows "
    + "the tab instead of scaling into it");
});

test("the stylesheet hides [hidden] browser frame to prevent broken image placeholder", async () => {
  const css = await readFile(BROWSER_CSS, "utf8");
  const rule = css.match(/\.browser-frame\[hidden\]\s*\{[^}]*\}/);
  assert.ok(rule, "no .browser-frame[hidden] rule");
  assert.match(rule[0], /display:\s*none\s*!important/,
    "unpainted or idle frame must be display: none");
});

test("frameCoords maps coordinates and letterboxing correctly", async () => {
  const mod = await load();
  const fakeImg = {
    naturalWidth: 1280,
    naturalHeight: 800,
    getBoundingClientRect() {
      // 2:1 aspect ratio element with 16:10 image -> letterboxed left and right
      // element: 800 x 400. Rendered image: 640 x 400. ox = 80, oy = 0.
      return { left: 100, top: 50, width: 800, height: 400 };
    },
  };
  // Click on left letterbox area (clientX = 140, left + 40 < left + 80)
  assert.equal(mod.frameCoords(fakeImg, 140, 200), null);
  // Click on top-left of rendered image (clientX = 180, clientY = 50)
  assert.deepEqual(mod.frameCoords(fakeImg, 180, 50), { x: 0, y: 0 });
  // Click on center (clientX = 500, clientY = 250)
  assert.deepEqual(mod.frameCoords(fakeImg, 500, 250), { x: 640, y: 400 });
});

test("clicking the live frame in own mode posts coordinates to /click", async () => {
  const { win, calls } = makeEnv();
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({ Authorization: "Bearer token" }) });

  const shot = win.document.querySelector("img.browser-frame");
  Object.defineProperty(shot, "naturalWidth", { value: 1280, configurable: true });
  Object.defineProperty(shot, "naturalHeight", { value: 800, configurable: true });
  shot.getBoundingClientRect = () => ({ left: 0, top: 0, width: 1280, height: 800 });

  // In idle mode, clicks are ignored
  shot.dispatchEvent(new win.MouseEvent("click", { clientX: 640, clientY: 400 }));
  assert.equal(calls.filter((c) => c.url.endsWith("/click")).length, 0);

  // Switch to own mode by opening a session
  const go = win.document.querySelector("button.btn-primary");
  await go.onclick();

  shot.dispatchEvent(new win.MouseEvent("click", { clientX: 640, clientY: 400 }));
  const clickCalls = calls.filter((c) => c.url.endsWith("/click"));
  assert.equal(clickCalls.length, 1);
});

test("wheel events on the live frame in own mode post to /scroll", async () => {
  const { win, calls } = makeEnv();
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({ Authorization: "Bearer token" }) });

  const go = win.document.querySelector("button.btn-primary");
  await go.onclick();

  const shot = win.document.querySelector("img.browser-frame");
  const wheelEv = new win.Event("wheel");
  wheelEv.deltaX = 0;
  wheelEv.deltaY = 120;
  wheelEv.preventDefault = () => {};
  shot.dispatchEvent(wheelEv);

  // Wait for 40ms flush
  await new Promise((r) => setTimeout(r, 60));
  const scrollCalls = calls.filter((c) => c.url.endsWith("/scroll"));
  assert.equal(scrollCalls.length, 1);
});

test("keydown on the live frame in own mode posts to /key and /type", async () => {
  const { win, calls } = makeEnv();
  const mod = await load();
  mod.register({ toast() {}, authHeaders: () => ({ Authorization: "Bearer token" }) });

  const go = win.document.querySelector("button.btn-primary");
  await go.onclick();

  const shot = win.document.querySelector("img.browser-frame");
  const enterEv = new win.KeyboardEvent("keydown", { key: "Enter" });
  enterEv.preventDefault = () => {};
  shot.dispatchEvent(enterEv);

  const charEv = new win.KeyboardEvent("keydown", { key: "a" });
  charEv.preventDefault = () => {};
  shot.dispatchEvent(charEv);
  await settle();

  assert.equal(calls.filter((c) => c.url.endsWith("/key")).length, 1);
  assert.equal(calls.filter((c) => c.url.endsWith("/type")).length, 1);
});

// --------------------------------------------------------------------------- //
//  Live-view input reaches the page one request at a time, in the order it    //
//  was made, and a keyboard user can always leave the frame.                  //
// --------------------------------------------------------------------------- //

/** A fetch double for an open browser whose input routes (click, key, type,
 *  scroll) answer only when the test releases them. `fail(entry)` picks the
 *  requests that fail as a network error would. */
function gatedInput(win, { fail = () => false } = {}) {
  const sent = [];
  const pending = [];
  const stats = { inFlight: 0, maxInFlight: 0 };
  win.fetch = (url, opts = {}) => {
    const u = String(url);
    const m = (opts.method || "GET").toUpperCase();
    if (u.endsWith("/session") && m === "POST") {
      return Promise.resolve({ ok: true, status: 200, json: async () => ({ job_id: "j1" }) });
    }
    const input = u.match(/\/api\/browser\/(click|key|type|scroll)$/);
    if (!input) {
      return Promise.resolve({ ok: true, status: 200, body: null, json: async () => ({}) });
    }
    const entry = { route: input[1], body: JSON.parse(opts.body) };
    sent.push(entry);
    stats.inFlight++;
    stats.maxInFlight = Math.max(stats.maxInFlight, stats.inFlight);
    return new Promise((resolve, reject) => {
      pending.push(() => {
        stats.inFlight--;
        if (fail(entry)) reject(new TypeError("Failed to fetch"));
        else resolve({ ok: true, status: 200, json: async () => ({ ok: true }) });
      });
    });
  };
  global.fetch = win.fetch;
  return { sent, pending, stats };
}

/** Release every pending input request, newest first, until none is left. */
async function drain(pending) {
  for (let round = 0; round < 50; round++) {
    await settle();
    if (!pending.length) return;
    for (const release of pending.splice(0).reverse()) release();
  }
  throw new Error("input requests never stopped arriving");
}

/** Register the tab, open a browser in it, and give the frame a 1280x800
 *  picture filling a 1280x800 box. */
async function openOwn(win, mod) {
  const toasts = [];
  mod.register({ toast: (t) => toasts.push(t), authHeaders: () => ({}) });
  const c = controls(win);
  c.url.value = "https://first.example/";
  c.go.onclick();
  await settle();
  const shot = win.document.querySelector("img.browser-frame");
  Object.defineProperty(shot, "naturalWidth", { value: 1280, configurable: true });
  Object.defineProperty(shot, "naturalHeight", { value: 800, configurable: true });
  shot.getBoundingClientRect = () => ({ left: 0, top: 0, width: 1280, height: 800 });
  shot.hidden = false;
  return { c, shot, toasts };
}

function keydown(win, key, extra = {}) {
  return new win.KeyboardEvent("keydown",
    { key, bubbles: true, cancelable: true, ...extra });
}

function frameClick(win) {
  return new win.MouseEvent("click", { clientX: 640, clientY: 400, bubbles: true });
}

test("live-view input goes out one request at a time, in the order it was made", async () => {
  const { win } = makeEnv();
  const { sent, pending, stats } = gatedInput(win);
  const mod = await load();
  const { shot } = await openOwn(win, mod);

  shot.dispatchEvent(frameClick(win));
  shot.dispatchEvent(keydown(win, "a"));
  shot.dispatchEvent(keydown(win, "b"));
  shot.dispatchEvent(keydown(win, "Enter"));
  await drain(pending);

  assert.equal(stats.maxInFlight, 1,
    `${stats.maxInFlight} input requests were in flight at once, so the server `
    + "may apply them in any order");
  // Consecutive typed text is merged here, so this holds with or without
  // coalescing.
  const order = [];
  for (const s of sent) {
    const last = order[order.length - 1];
    if (s.route === "type" && last && last.route === "type") last.text += s.body.text;
    else order.push({ route: s.route, ...s.body });
  }
  assert.deepEqual(order, [
    { route: "click", x: 640, y: 400, button: "left" },
    { route: "type", text: "ab" },
    { route: "key", key: "Enter" },
  ], "input reached the page in a different order from the one it was made in");
});

test("characters typed while an input request is in flight go out as one /type", async () => {
  const { win } = makeEnv();
  const { sent, pending } = gatedInput(win);
  const mod = await load();
  const { shot } = await openOwn(win, mod);

  shot.dispatchEvent(keydown(win, "Enter"));
  for (const ch of "hello") shot.dispatchEvent(keydown(win, ch));
  await drain(pending);

  assert.deepEqual(sent, [
    { route: "key", body: { key: "Enter" } },
    { route: "type", body: { text: "hello" } },
  ]);
});

test("wheel deltas gathered before a click are sent before it", async () => {
  const { win } = makeEnv();
  const { sent, pending } = gatedInput(win);
  const mod = await load();
  const { shot } = await openOwn(win, mod);

  const wheel = new win.Event("wheel", { cancelable: true });
  wheel.deltaX = 0;
  wheel.deltaY = 120;
  shot.dispatchEvent(wheel);
  shot.dispatchEvent(frameClick(win));
  await drain(pending);

  assert.deepEqual(sent.map((s) => s.route), ["scroll", "click"],
    "a click overtook the scroll made before it");
  assert.deepEqual(sent[0].body, { delta_x: 0, delta_y: 120 });
});

test("a failed input request is shown, and the input queued behind it is still sent", async () => {
  const { win } = makeEnv();
  const { sent, pending } = gatedInput(win, { fail: (e) => e.route === "click" });
  const mod = await load();
  const { c, shot, toasts } = await openOwn(win, mod);
  const status = c.view.querySelector(".browser-status");

  shot.dispatchEvent(frameClick(win));
  shot.dispatchEvent(keydown(win, "x"));
  await settle();
  pending.shift()();                    // the click fails
  await settle();

  assert.match(status.textContent, /did not reach the browser/,
    "a failed click left the status line saying nothing about it");
  assert.equal(toasts.length, 1, "a failed click was not surfaced to the user");
  assert.deepEqual(sent.map((s) => s.route), ["click", "type"],
    "the input queued behind a failed request was not sent");

  await drain(pending);
  assert.match(status.textContent, /reaching the browser again/,
    "the failure stayed on screen after input got through again");
  assert.equal(toasts.length, 1, "a request that succeeded raised a toast");
});

test("stopping the browser drops input that has not been sent yet", async () => {
  const { win } = makeEnv();
  const { sent, pending } = gatedInput(win);
  const mod = await load();
  const { c, shot } = await openOwn(win, mod);

  shot.dispatchEvent(keydown(win, "Enter"));
  shot.dispatchEvent(keydown(win, "q"));
  c.stop.onclick();
  await drain(pending);

  assert.deepEqual(sent.map((s) => s.route), ["key"],
    "input made before Stop was sent after it");
});

test("switching to the agent view drops a scroll still being gathered", async () => {
  const { win } = makeEnv();
  const { sent, pending } = gatedInput(win);
  const mod = await load();
  const { c, shot } = await openOwn(win, mod);

  const wheel = new win.Event("wheel", { cancelable: true });
  wheel.deltaX = 0;
  wheel.deltaY = 120;
  shot.dispatchEvent(wheel);
  c.watch.onclick();
  await new Promise((r) => setTimeout(r, 60));
  await drain(pending);

  assert.deepEqual(sent, [],
    "a scroll made in the tab's own browser was sent after the view moved to the agent's");
});

test("Esc releases the frame without reaching the page, and Shift+Tab is left to the browser", async () => {
  const { win } = makeEnv();
  const { sent, pending } = gatedInput(win);
  const mod = await load();
  const { shot } = await openOwn(win, mod);

  shot.focus();
  assert.equal(win.document.activeElement, shot, "precondition: the frame holds focus");
  shot.dispatchEvent(keydown(win, "Escape"));
  await drain(pending);
  assert.notEqual(win.document.activeElement, shot,
    "Esc left keyboard focus in the frame, so a keyboard user cannot leave it");
  assert.deepEqual(sent.filter((s) => s.route === "key" && s.body.key === "Escape"), [],
    "Esc was sent to the page instead of releasing the frame");

  shot.focus();
  const back = keydown(win, "Tab", { shiftKey: true });
  shot.dispatchEvent(back);
  await drain(pending);
  assert.equal(back.defaultPrevented, false,
    "Shift+Tab was cancelled, so focus cannot move back out of the frame");
  assert.deepEqual(sent, [], "Shift+Tab was sent to the page");

  const forward = keydown(win, "Tab");
  shot.dispatchEvent(forward);
  await drain(pending);
  assert.deepEqual(sent, [{ route: "key", body: { key: "Tab" } }],
    "Tab alone still moves through the page's own fields");
});

test("the focused frame shows a hint that names the key which releases it", async () => {
  const { win } = makeEnv();
  gatedInput(win);
  const mod = await load();
  const { shot } = await openOwn(win, mod);
  const hint = win.document.querySelector(".browser-keys-hint");

  assert.ok(hint, "the view has no keyboard hint");
  assert.equal(hint.hidden, true, "the hint shows while the frame is not focused");
  assert.equal(shot.getAttribute("aria-describedby"), hint.id,
    "the frame does not point assistive technology at the hint");
  shot.focus();
  assert.equal(hint.hidden, false, "focusing the frame did not show the hint");
  assert.match(hint.textContent, /\bEsc\b/, "the hint does not name the release key");
  shot.dispatchEvent(keydown(win, "Escape"));
  assert.equal(hint.hidden, true, "the hint stayed after the frame let go of the keyboard");
});
