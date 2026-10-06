// SPDX-License-Identifier: AGPL-3.0-or-later
// jsdom tests for what the Browser tab shows when the configured browser
// cannot start yet (localm/plugins/builtin/browser/static/browser.js): the
// one-time download of the bundled browser, the list of browsers looked for
// when the system engine finds none, and the switch to the bundled browser.
//
// The server is a fake that answers GET /api/browser/engine, POST
// /api/browser/download, PATCH /v1/config and a job's event stream, so each test
// states what the server says and reads what the tab does with it.

import { test } from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const BROWSER_JS = join(HERE, "..", "localm", "plugins", "builtin", "browser",
                        "static", "browser.js");

const NODE_SET_TIMEOUT = globalThis.setTimeout;

const LOOKED_FOR = ["Google Chrome", "Chromium", "Microsoft Edge", "Brave"];

const MISSING = {
  engine: "bundled", ready: false, problem: "bundled_missing",
  bundled_installed: false, system_browsers: [], looked_for: LOOKED_FOR,
  can_download: true, download_blocked: null, downloading: false,
};
const READY = { ...MISSING, ready: true, problem: null, bundled_installed: true,
                can_download: false };

function sse(events) {
  return events.map((e) => "data: " + JSON.stringify(e) + "\n\n").join("");
}

/** A jsdom window whose fetch is the fake server. `server.engine` is what GET
 *  /api/browser/engine answers next (a function may compute it); `server.jobs`
 *  maps a job id to the events its stream carries. */
function makeEnv(server) {
  const dom = new JSDOM(
    `<!DOCTYPE html><html><body><main id="main"></main></body></html>`,
    { url: "http://localhost:8642/" });
  const win = dom.window;
  const calls = [];
  win.fetch = async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    calls.push({ url: u, method, body: opts.body });
    const json = (status, body) => ({
      ok: status < 400, status, body: null, json: async () => body });
    if (u === "/api/browser/engine" && method === "GET") {
      const eng = typeof server.engine === "function" ? server.engine() : server.engine;
      return json(200, eng);
    }
    if (u === "/api/browser/download" && method === "POST") {
      return server.download ? server.download() : json(200, { job_id: "dl1", status: "started" });
    }
    if (u === "/v1/config" && method === "PATCH") {
      return server.patch ? server.patch() : json(200, { ok: true });
    }
    const job = /^\/api\/jobs\/([^/]+)\/events$/.exec(u);
    if (job && server.jobs && server.jobs[job[1]]) {
      if (server.gate) await server.gate;
      const text = sse(server.jobs[job[1]]);
      return { ok: true, status: 200, body: new Response(text).body };
    }
    return json(200, {});
  };
  global.window = win;
  global.document = win.document;
  global.fetch = win.fetch;
  global.setTimeout = NODE_SET_TIMEOUT;
  return { win, calls };
}

async function loadTab(server) {
  const env = makeEnv(server);
  const mod = await import(pathToFileURL(BROWSER_JS).href + "?t=" + Math.random());
  const toasts = [];
  mod.register({ toast: (msg, isError) => toasts.push({ msg, isError: !!isError }),
                 authHeaders: () => ({}) });
  const view = env.win.document.getElementById("view-browser");
  return {
    ...env, view, toasts,
    setup: view.querySelector(".browser-setup"),
    status: view.querySelector(".browser-status"),
    buttons: () => [...view.querySelectorAll(".browser-setup button")],
    show: () => env.win.onViewShown("browser"),
  };
}

async function until(predicate, what) {
  for (let i = 0; i < 200; i++) {
    if (predicate()) return;
    await new Promise((r) => NODE_SET_TIMEOUT(r, 10));
  }
  assert.fail("timed out waiting for " + what);
}

const textOf = (tab) => tab.setup.textContent;

test("a browser that can start shows no setup area", async () => {
  const tab = await loadTab({ engine: READY });
  await tab.show();
  await until(() => tab.calls.some((c) => c.url === "/api/browser/engine"), "the engine check");
  await new Promise((r) => NODE_SET_TIMEOUT(r, 20));

  assert.equal(tab.setup.hidden, true);
  assert.equal(tab.buttons().length, 0);
});

test("a server answer the tab does not recognise shows no empty box", async () => {
  const tab = await loadTab({ engine: {} });
  await tab.show();
  await until(() => tab.calls.some((c) => c.url === "/api/browser/engine"), "the engine check");
  await new Promise((r) => NODE_SET_TIMEOUT(r, 20));

  assert.equal(tab.setup.hidden, true);
});

test("a missing bundled browser offers the download, in words and a button", async () => {
  const tab = await loadTab({ engine: MISSING });
  await tab.show();
  await until(() => tab.buttons().length === 1, "the download button");

  assert.equal(tab.setup.hidden, false);
  assert.match(textOf(tab), /bundled browser has not been downloaded yet/);
  assert.equal(tab.buttons()[0].textContent, "Download browser");
  assert.ok(!tab.calls.some((c) => c.method === "POST"),
            "showing the offer must not start the download");
});

test("the download posts once, shows its progress, and reports the browser ready", async () => {
  let downloaded = false;
  const tab = await loadTab({
    engine: () => (downloaded ? READY : MISSING),
    jobs: { dl1: [
      { type: "line", text: "Downloading the browser (one-time, a few hundred MB)..." },
      { type: "line", text: "10% of 186.8 MiB" },
      { type: "line", text: "Ready: Chromium installed." },
      { type: "end", status: "done" }] },
    download: () => { downloaded = true;
      return { ok: true, status: 200, json: async () => ({ job_id: "dl1", status: "started" }) }; },
  });
  await tab.show();
  await until(() => tab.buttons().length === 1, "the download button");

  tab.buttons()[0].onclick();
  await until(() => tab.setup.hidden === true, "the setup area to clear");

  assert.equal(tab.calls.filter((c) => c.url === "/api/browser/download").length, 1);
  assert.ok(tab.calls.some((c) => c.url === "/api/jobs/dl1/events"),
            "the job's progress was never read");
  assert.match(tab.status.textContent, /downloaded and ready/);
  assert.deepEqual(tab.toasts, [{ msg: tab.status.textContent, isError: false }]);
});

test("while it downloads the button is disabled and cannot start a second download", async () => {
  let release;
  const gate = new Promise((r) => { release = r; });
  const tab = await loadTab({
    engine: MISSING, gate,
    jobs: { dl1: [{ type: "line", text: "error: stopped" },
                  { type: "end", status: "failed" }] },
  });
  await tab.show();
  await until(() => tab.buttons().length === 1, "the download button");
  const button = tab.buttons()[0];

  button.onclick();
  await until(() => button.disabled, "the button to disable");
  button.click();

  assert.equal(button.textContent, "Downloading the browser...");
  release();
  await until(() => /did not finish/.test(tab.status.textContent), "the job to end");
  assert.equal(tab.calls.filter((c) => c.url === "/api/browser/download").length, 1,
               "a second click started a second download");
});

test("a download that fails says why, leaves the browser missing and offers it again", async () => {
  const tab = await loadTab({
    engine: MISSING,
    jobs: { dl1: [
      { type: "line", text: "error: Could not install Chromium: the installer exited with code 1." },
      { type: "end", status: "failed" }] },
  });
  await tab.show();
  await until(() => tab.buttons().length === 1, "the download button");

  tab.buttons()[0].onclick();
  await until(() => /did not finish/.test(tab.status.textContent), "the failure message");

  assert.equal(tab.status.textContent,
    "The download did not finish: Could not install Chromium: the installer exited with code 1.");
  assert.deepEqual(tab.toasts.map((t) => t.isError), [true]);
  await until(() => tab.buttons().length === 1 && !tab.buttons()[0].disabled,
              "the button to come back");
});

test("a refused download request shows the server's reason and offers it again", async () => {
  const tab = await loadTab({
    engine: MISSING,
    download: () => ({ ok: false, status: 409,
      json: async () => ({ detail: "Network access is disabled (net_mode=off)." }) }),
  });
  await tab.show();
  await until(() => tab.buttons().length === 1, "the download button");

  tab.buttons()[0].onclick();
  await until(() => /did not finish/.test(tab.status.textContent), "the failure message");

  assert.match(tab.status.textContent, /Network access is disabled \(net_mode=off\)\./);
  assert.ok(!tab.calls.some((c) => c.url.startsWith("/api/jobs/")),
            "there was no job to read");
});

test("network access off explains itself and offers no download", async () => {
  const tab = await loadTab({ engine: { ...MISSING, can_download: false,
                                        download_blocked: "network" } });
  await tab.show();
  await until(() => !tab.setup.hidden, "the notice");

  assert.match(textOf(tab), /Network access is off, which blocks the download/);
  assert.match(textOf(tab), /Settings > Server & network/);
  assert.equal(tab.buttons().length, 0);
});

test("missing permission explains itself and offers no download", async () => {
  const tab = await loadTab({ engine: { ...MISSING, can_download: false,
                                        download_blocked: "permission" } });
  await tab.show();
  await until(() => !tab.setup.hidden, "the notice");

  assert.match(textOf(tab), /needs permission to change settings/);
  assert.equal(tab.buttons().length, 0);
});

test("a download already running elsewhere is named, with no second button", async () => {
  const tab = await loadTab({ engine: { ...MISSING, downloading: true,
                                        can_download: true } });
  await tab.show();
  await until(() => !tab.setup.hidden, "the notice");

  assert.match(textOf(tab), /A download is already running/);
  assert.equal(tab.buttons().length, 0);
});

test("a system engine with no browser lists what was looked for and offers the bundled one",
     async () => {
  const tab = await loadTab({ engine: { ...MISSING, engine: "system",
                                        problem: "system_missing" } });
  await tab.show();
  await until(() => tab.buttons().length === 1, "the switch button");

  assert.match(textOf(tab),
    /No supported browser was found on this computer\. localm looked for Google Chrome, Chromium, Microsoft Edge, Brave\./);
  assert.equal(tab.buttons()[0].textContent, "Use the bundled browser instead");
});

test("using the bundled browser saves the setting and re-checks the engine", async () => {
  let engine = "system";
  const tab = await loadTab({
    engine: () => (engine === "system"
      ? { ...MISSING, engine: "system", problem: "system_missing" } : MISSING),
    patch: () => { engine = "bundled";
      return { ok: true, status: 200, json: async () => ({}) }; },
  });
  await tab.show();
  await until(() => tab.buttons().length === 1, "the switch button");

  tab.buttons()[0].onclick();
  await until(() => tab.buttons().length === 1
                    && tab.buttons()[0].textContent === "Download browser",
              "the download offer for the bundled browser");

  const patch = tab.calls.find((c) => c.method === "PATCH");
  assert.equal(patch.url, "/v1/config");
  assert.deepEqual(JSON.parse(patch.body), { browser_engine: "bundled" });
  assert.match(tab.status.textContent, /Switched to the bundled browser/);
});

test("a refused switch shows the reason and leaves the notice in place", async () => {
  const tab = await loadTab({
    engine: { ...MISSING, engine: "system", problem: "system_missing" },
    patch: () => ({ ok: false, status: 403,
                    json: async () => ({ detail: "config:write required" }) }),
  });
  await tab.show();
  await until(() => tab.buttons().length === 1, "the switch button");

  tab.buttons()[0].onclick();
  await until(() => tab.toasts.length === 1, "the refusal");

  assert.equal(tab.toasts[0].msg, "config:write required");
  assert.equal(tab.toasts[0].isError, true);
  assert.equal(tab.buttons()[0].textContent, "Use the bundled browser instead");
});

test("without the browser extra the tab says how to install it", async () => {
  const tab = await loadTab({ engine: { ...MISSING, problem: "playwright_missing",
                                        can_download: false } });
  await tab.show();
  await until(() => !tab.setup.hidden, "the notice");

  assert.match(textOf(tab), /pip install "localm\[browser\]"/);
  assert.equal(tab.buttons().length, 0);
});

test("an open that fails brings the download offer up", async () => {
  const tab = await loadTab({
    engine: MISSING,
    jobs: { open1: [
      { type: "line", text: "job error: The bundled browser has not been downloaded yet." },
      { type: "end", status: "failed" }] },
  });
  const realFetch = tab.win.fetch;
  global.fetch = tab.win.fetch = async (url, opts = {}) => {
    if (String(url) === "/api/browser/session") {
      tab.calls.push({ url: "/api/browser/session", method: "POST" });
      return { ok: true, status: 200, json: async () => ({ job_id: "open1" }) };
    }
    return realFetch(url, opts);
  };
  const open = [...tab.view.querySelectorAll(".browser-bar button")]
    .find((b) => b.textContent === "Open");

  open.onclick();
  await until(() => tab.buttons().length === 1, "the download offer after the failure");

  assert.match(tab.status.textContent, /has not been downloaded yet/);
  assert.equal(tab.buttons()[0].textContent, "Download browser");
});

test("opening the tab shows the offer only when it is shown, not when it is built", async () => {
  const tab = await loadTab({ engine: MISSING });
  await new Promise((r) => NODE_SET_TIMEOUT(r, 30));

  assert.deepEqual(tab.calls, []);
  assert.equal(tab.setup.hidden, true);
});
