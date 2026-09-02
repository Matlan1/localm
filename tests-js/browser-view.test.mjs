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

const HERE = dirname(fileURLToPath(import.meta.url));
const BROWSER_JS = join(HERE, "..", "localm", "plugins", "builtin", "browser",
                        "static", "browser.js");

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
  global.setTimeout = win.setTimeout ? win.setTimeout.bind(win) : setTimeout;
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
  assert.deepEqual(buttons, ["Open", "Stop"]);
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
