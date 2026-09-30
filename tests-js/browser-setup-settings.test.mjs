// SPDX-License-Identifier: AGPL-3.0-or-later
// The "Browser setup" box under Settings > Server & network
// (buildBrowserSetupBox in pages/settings.js): whether the configured browser
// engine can start a browser, the one-time download of the bundled browser, and
// the switch to it when the system engine finds no browser to drive.
//
// The box exists only while the browser plugin contributes its settings, and
// reads GET /api/browser/engine; the fake server below answers that, the
// download route, PATCH /v1/config and a job's event stream.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const BROWSER_ENGINE = {
  key: "browser_engine", widget: "select", label: "Browser to drive", help: "",
  group: "Network", owner: "browser", options: ["bundled", "system"],
  default: "bundled", admin_only: true,
};
const NET_MODE = {
  key: "net_mode", widget: "select", label: "Network access", help: "",
  group: "Network", owner: "core", options: ["off", "ask", "allow"], default: "ask",
};

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

/** Every route the settings render and the box reach. `server.fields` is the
 *  schema; `server.engine` answers the engine check (a function is called each
 *  time); `server.jobs` maps a job id to the events its stream carries. */
function makeFetch(server, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    calls.push({ url: u, method, body: opts.body });
    const json = (body, status = 200) => ({
      ok: status < 400, status, statusText: "", json: async () => body,
      text: async () => "" });
    if (u === "/v1/config/schema") return json({ fields: server.fields });
    if (u === "/v1/media/config") return json({ plugins: [] });
    if (u === "/v1/comfy/status") return json({ alive: false, launched_by_localm: false });
    if (u === "/v1/tts/config") return json({ plugin: "tts", active: false, fields: [] });
    if (u === "/v1/plugins/settings") return json({ plugins: [] });
    if (u === "/api/browser/engine") {
      const eng = typeof server.engine === "function" ? server.engine() : server.engine;
      return eng === null ? json({ detail: "no" }, 500) : json(eng);
    }
    if (u === "/api/browser/download" && method === "POST") {
      return server.download ? server.download() : json({ job_id: "dl1", status: "started" });
    }
    if (u === "/v1/config" && method === "PATCH") {
      return server.patch ? server.patch() : json({});
    }
    const job = /^\/api\/jobs\/([^/]+)\/events$/.exec(u);
    if (job && server.jobs && server.jobs[job[1]]) {
      return { ok: true, status: 200, body: new Response(sse(server.jobs[job[1]])).body };
    }
    return json({ models: [], active: "", conversations: [], plugins: [] });
  };
}

async function render(server) {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(server, calls) });
  runScript(win, "refreshSettingsPage();");
  for (let i = 0; i < 30; i++) await new Promise((r) => setTimeout(r, 0));
  return { win, calls };
}

const box = (win) => win.document.querySelector(".browser-setup-box");
const status = (win) => box(win).querySelector(".browser-setup-status").textContent;
const buttons = (win) => [...box(win).querySelectorAll("button")];

async function until(predicate, what) {
  for (let i = 0; i < 300; i++) {
    if (predicate()) return;
    await new Promise((r) => setTimeout(r, 10));
  }
  assert.fail("timed out waiting for " + what);
}

test("the box sits inside the Network section when the browser plugin has settings", async () => {
  const { win } = await render({ fields: [NET_MODE, BROWSER_ENGINE], engine: MISSING });
  await until(() => box(win) && buttons(win).length === 1, "the box");

  assert.equal(box(win).closest("#settings-sec-core-Network") !== null, true);
  assert.match(status(win), /bundled browser has not been downloaded yet/);
  assert.equal(buttons(win)[0].textContent, "Download browser");
});

test("without the browser plugin's settings there is no box and no engine request", async () => {
  const { win, calls } = await render({ fields: [NET_MODE], engine: MISSING });

  assert.equal(box(win), null);
  assert.ok(!calls.some((c) => c.url === "/api/browser/engine"));
});

test("the box appears once per render, not once per setting", async () => {
  const { win } = await render({ fields: [NET_MODE, BROWSER_ENGINE], engine: READY });
  await until(() => box(win), "the box");

  assert.equal(win.document.querySelectorAll(".browser-setup-box").length, 1);
});

test("a ready bundled browser says so and offers nothing", async () => {
  const { win } = await render({ fields: [BROWSER_ENGINE], engine: READY });
  await until(() => box(win) && /ready/.test(status(win)), "the status");

  assert.equal(status(win), "The bundled browser is downloaded and ready.");
  assert.equal(buttons(win).length, 0);
});

test("a ready system engine names the browser it will use", async () => {
  const { win } = await render({ fields: [BROWSER_ENGINE], engine: {
    ...READY, engine: "system", system_browsers: ["Brave", "Google Chrome"] } });
  await until(() => box(win) && /will use/.test(status(win)), "the status");

  assert.equal(status(win), "localm will use Brave, found on this computer.");
  assert.equal(buttons(win).length, 0);
});

test("network access off explains itself and offers no download", async () => {
  const { win } = await render({ fields: [BROWSER_ENGINE], engine: {
    ...MISSING, can_download: false, download_blocked: "network" } });
  await until(() => box(win) && /Network access is off/.test(status(win)), "the status");

  assert.match(status(win), /Settings > Server & network/);
  assert.equal(buttons(win).length, 0);
});

test("missing permission explains itself and offers no download", async () => {
  const { win } = await render({ fields: [BROWSER_ENGINE], engine: {
    ...MISSING, can_download: false, download_blocked: "permission" } });
  await until(() => box(win) && /needs permission/.test(status(win)), "the status");

  assert.equal(buttons(win).length, 0);
});

test("a system engine with no browser lists what was looked for and offers the bundled one",
     async () => {
  const { win } = await render({ fields: [BROWSER_ENGINE], engine: {
    ...MISSING, engine: "system", problem: "system_missing" } });
  await until(() => box(win) && buttons(win).length === 1, "the switch button");

  assert.match(status(win), /localm looked for Google Chrome, Chromium, Microsoft Edge, Brave\./);
  assert.equal(buttons(win)[0].textContent, "Use the bundled browser instead");
});

test("without the browser extra the box says how to install it", async () => {
  const { win } = await render({ fields: [BROWSER_ENGINE], engine: {
    ...MISSING, problem: "playwright_missing", can_download: false } });
  await until(() => box(win) && /pip install/.test(status(win)), "the status");

  assert.equal(buttons(win).length, 0);
});

test("a failed engine check is reported, not hidden", async () => {
  const { win } = await render({ fields: [BROWSER_ENGINE], engine: null });
  await until(() => box(win) && /Could not check/.test(status(win)), "the status");

  assert.equal(status(win), "Could not check the browser.");
});

test("the download posts once, reads the job, and leaves the box ready", async () => {
  let downloaded = false;
  const { win, calls } = await render({
    fields: [BROWSER_ENGINE],
    engine: () => (downloaded ? READY : MISSING),
    download: () => { downloaded = true;
      return { ok: true, status: 200, json: async () => ({ job_id: "dl1", status: "started" }) }; },
    jobs: { dl1: [
      { type: "line", text: "Downloading the browser (one-time, a few hundred MB)..." },
      { type: "line", text: "Ready: Chromium installed." },
      { type: "end", status: "done" }] },
  });
  await until(() => box(win) && buttons(win).length === 1, "the download button");

  buttons(win)[0].click();
  await until(() => /ready/.test(status(win)) && buttons(win).length === 0, "the box to be ready");

  assert.equal(calls.filter((c) => c.url === "/api/browser/download").length, 1);
  assert.ok(calls.some((c) => c.url === "/api/jobs/dl1/events"));
  const toast = win.document.getElementById("toast");
  assert.equal(toast.textContent, "The browser is downloaded and ready.");
  assert.ok(!toast.className.includes("error"));
});

test("a download that fails toasts the reason and offers the button again", async () => {
  const { win } = await render({
    fields: [BROWSER_ENGINE], engine: MISSING,
    jobs: { dl1: [
      { type: "line", text: "error: Could not install Chromium: the installer exited with code 1." },
      { type: "end", status: "failed" }] },
  });
  await until(() => box(win) && buttons(win).length === 1, "the download button");

  buttons(win)[0].click();
  const toast = win.document.getElementById("toast");
  await until(() => /did not finish/.test(toast.textContent), "the failure toast");

  assert.equal(toast.textContent,
    "The download did not finish: Could not install Chromium: the installer exited with code 1.");
  assert.ok(toast.className.includes("error"));
  await until(() => buttons(win).length === 1 && !buttons(win)[0].disabled,
              "the button to come back");
});

test("a refused download request toasts the server's reason", async () => {
  const { win, calls } = await render({
    fields: [BROWSER_ENGINE], engine: MISSING,
    download: () => ({ ok: false, status: 409, statusText: "Conflict",
      json: async () => ({ detail: "Network access is disabled (net_mode=off)." }) }),
  });
  await until(() => box(win) && buttons(win).length === 1, "the download button");

  buttons(win)[0].click();
  const toast = win.document.getElementById("toast");
  await until(() => /did not finish/.test(toast.textContent), "the failure toast");

  assert.match(toast.textContent, /Network access is disabled \(net_mode=off\)\./);
  assert.ok(!calls.some((c) => c.url.startsWith("/api/jobs/")));
});

test("using the bundled browser saves the engine setting and re-renders", async () => {
  let engine = "system";
  const { win, calls } = await render({
    fields: [BROWSER_ENGINE],
    engine: () => (engine === "system"
      ? { ...MISSING, engine: "system", problem: "system_missing" } : MISSING),
    patch: () => { engine = "bundled";
      return { ok: true, status: 200, json: async () => ({}) }; },
  });
  await until(() => box(win) && buttons(win).length === 1
                    && buttons(win)[0].textContent === "Use the bundled browser instead",
              "the switch button");

  buttons(win)[0].click();
  await until(() => box(win) && buttons(win).length === 1
                    && buttons(win)[0].textContent === "Download browser",
              "the download offer for the bundled browser");

  const patch = calls.find((c) => c.method === "PATCH");
  assert.equal(patch.url, "/v1/config");
  assert.deepEqual(JSON.parse(patch.body), { browser_engine: "bundled" });
});

test("a refused switch toasts the reason and leaves the button", async () => {
  const { win } = await render({
    fields: [BROWSER_ENGINE],
    engine: { ...MISSING, engine: "system", problem: "system_missing" },
    patch: () => ({ ok: false, status: 403, statusText: "Forbidden",
                    json: async () => ({ detail: "config:write required" }) }),
  });
  await until(() => box(win) && buttons(win).length === 1, "the switch button");

  buttons(win)[0].click();
  const toast = win.document.getElementById("toast");
  await until(() => toast.textContent === "config:write required", "the refusal toast");

  assert.ok(toast.className.includes("error"));
  assert.equal(buttons(win)[0].disabled, false);
});
