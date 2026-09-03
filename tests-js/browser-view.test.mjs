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
