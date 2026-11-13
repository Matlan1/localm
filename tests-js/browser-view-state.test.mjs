// SPDX-License-Identifier: AGPL-3.0-or-later
// The Browser tab keeps track of the server: a browser still open after a
// reload is taken back, the agent offer follows the agent while the tab is in
// front, a failed start shows its reason without the job runner's prefix and
// stops polling, and a page that failed to load is not called a refusal.

import { test, mock } from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const BROWSER_JS = join(HERE, "..", "localm", "plugins", "builtin", "browser",
                        "static", "browser.js");
const NODE_SET_TIMEOUT = globalThis.setTimeout;

function makeEnv(route) {
  const dom = new JSDOM(
    `<!DOCTYPE html><html><body><main id="main"></main></body></html>`,
    { url: "http://localhost:8642/" });
  const win = dom.window;
  const calls = [];
  win.fetch = async (url, opts = {}) => {
    const u = String(url);
    const m = (opts.method || "GET").toUpperCase();
    calls.push({ url: u, method: m });
    const r = route(u, m);
    if (r) return r;
    return { ok: true, status: 200, body: null, json: async () => ({}) };
  };
  global.window = win;
  global.document = win.document;
  global.fetch = win.fetch;
  global.setTimeout = NODE_SET_TIMEOUT;
  return { win, calls };
}

function json(body) {
  return { ok: true, status: 200, body: null, json: async () => body };
}

function sse(events) {
  const enc = new TextEncoder();
  const chunks = events.map((ev) => enc.encode("data: " + JSON.stringify(ev) + "\n\n"));
  let i = 0;
  return {
    ok: true, status: 200,
    body: { getReader() { return { async read() {
      return i < chunks.length ? { done: false, value: chunks[i++] }
                               : { done: true, value: undefined };
    } }; } },
  };
}

async function settle(n = 40) {
  for (let i = 0; i < n; i++) await Promise.resolve();
}

async function mount(win, toasts = []) {
  const mod = await import(pathToFileURL(BROWSER_JS).href + "?t=" + Math.random());
  mod.register({ toast: (t) => toasts.push(t), authHeaders: () => ({}) });
  const view = win.document.getElementById("view-browser");
  const buttons = [...view.querySelectorAll("button")];
  return { view, url: view.querySelector("input.browser-url"),
           go: buttons[0], stop: buttons[1], watch: buttons[2],
           status: () => view.querySelector(".browser-status").textContent };
}

test("a browser still open for this key is taken back when the tab is shown", async () => {
  const { win } = makeEnv((u, m) =>
    (u.endsWith("/state") && m === "GET" ? json({ open: true }) : null));
  const c = await mount(win);
  assert.equal(c.stop.disabled, true);
  await win.onViewShown("browser");
  await settle();
  assert.equal(c.stop.disabled, false, "Stop stayed disabled for an open browser");
  assert.equal(c.go.textContent, "Go");
  assert.match(c.status(), /already open for this key/);
});

test("the agent offer follows the agent while the tab is in front", async () => {
  mock.timers.enable({ apis: ["setInterval"] });
  try {
    let available = false;
    const { win } = makeEnv((u, m) =>
      (u.endsWith("/agent") && m === "GET" ? json({ available }) : null));
    const c = await mount(win);
    await win.onViewShown("browser");
    await settle();
    assert.equal(c.watch.hidden, true);
    available = true;
    mock.timers.tick(5000);
    await settle();
    assert.equal(c.watch.hidden, false,
      "an agent browser opened while the tab was in front was never offered");
    available = false;
    mock.timers.tick(5000);
    await settle();
    assert.equal(c.watch.hidden, true, "the offer stayed after the agent closed it");
    await win.onViewShown("chat");
    available = true;
    mock.timers.tick(5000);
    await settle();
    assert.equal(c.watch.hidden, true, "the offer kept refreshing on another page");
  } finally {
    mock.timers.reset();
  }
});

test("a failed start shows its reason without the job runner's prefix", async () => {
  const { win } = makeEnv((u, m) => {
    if (u.endsWith("/session") && m === "POST") return json({ job_id: "j1" });
    if (u.includes("/api/jobs/j1/events")) {
      return sse([{ type: "line", line: "job error: Chromium is not installed." },
                  { type: "end", status: "error" }]);
    }
    return null;
  });
  const toasts = [];
  const c = await mount(win, toasts);
  c.url.value = "https://example.com/";
  c.go.onclick();
  await settle();
  assert.equal(c.status(), "Chromium is not installed.");
  assert.deepEqual(toasts, ["Chromium is not installed."]);
});

test("polling stops once the job has ended by itself", async () => {
  const { win, calls } = makeEnv((u, m) => {
    if (u.endsWith("/session") && m === "POST") return json({ job_id: "j1" });
    if (u.includes("/api/jobs/j1/events")) return sse([{ type: "end", status: "error" }]);
    return null;
  });
  // After makeEnv, which puts Node's own setTimeout back on the global.
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const c = await mount(win);
    c.url.value = "https://example.com/";
    c.go.onclick();
    await settle();
    const before = calls.filter((x) => x.url.endsWith("/state")).length;
    mock.timers.tick(2000);
    await settle();
    mock.timers.tick(2000);
    await settle();
    const after = calls.filter((x) => x.url.endsWith("/state")).length;
    assert.equal(after, before, "a job that ended kept polling the browser state");
  } finally {
    mock.timers.reset();
  }
});

test("a page that did not load is not reported as a refusal", async () => {
  const { win } = makeEnv((u, m) => {
    if (u.endsWith("/session") && m === "POST") return json({ job_id: "j1" });
    if (u.endsWith("/navigate")) {
      return json({ ok: false, url: "https://down.example/", refused: null,
                    error: "net::ERR_CONNECTION_REFUSED" });
    }
    return null;
  });
  const c = await mount(win);
  c.url.value = "https://first.example/";
  c.go.onclick();
  await settle();
  c.url.value = "https://down.example/";
  c.go.onclick();
  await settle();
  assert.equal(c.status(),
    "Could not load https://down.example/: net::ERR_CONNECTION_REFUSED");
});
