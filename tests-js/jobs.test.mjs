// jsdom tests for the jobs plugin client_entry (localm/plugins/builtin/jobs/
// static/jobs.js).
//
// Unlike the GUI's app.js (a top-level script), jobs.js is an ES MODULE that
// exports register(ctx). loadClientPlugins() in app.js does
//   const mod = await import(`${base}/${p.client_entry}`); await mod.register(ctx)
// with ctx = { registerTTS, toast, authHeaders, voicesChanged }. We mirror that
// here: build a jsdom document that has a <main id="main"> (the GUI views
// container), install it as the global document/window, stub fetch with canned
// /api/jobs data, import the module, call register(ctx), trigger the jobs view,
// and assert the list renders + Run-now POSTs.

import { test } from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const JOBS_JS = join(HERE, "..", "localm", "plugins", "builtin", "jobs", "static", "jobs.js");

// A canned job (matches the backend Job dict: asdict(Job)).
const JOB = {
  id: "abc123",
  name: "Nightly digest",
  schedule_kind: "interval",
  schedule: 3600,
  task_kind: "chat",
  prompt: "summarise my notes",
  model: null,
  cwd: null,
  scope: null,
  enabled: true,
  created: 1700000000,
  last_run: 1700003600,
  last_status: "ok",
  last_result_id: "2026-06-17T00-00-00",
};

// Build a fresh jsdom env per test, install it as the module's globals, and
// return { window, calls } where calls records every fetch (method + url).
function makeEnv({ jobs = [JOB] } = {}) {
  const dom = new JSDOM(
    `<!DOCTYPE html><html><body><main id="main"></main></body></html>`,
    { url: "http://localhost:8642/" });
  const win = dom.window;
  const calls = [];

  win.confirm = () => true;           // Delete uses confirm(); auto-accept.
  win.fetch = async (url, opts = {}) => {
    const method = (opts.method || "GET").toUpperCase();
    calls.push({ url: String(url), method, body: opts.body });
    if (method === "GET" && /\/api\/jobs$/.test(url)) {
      return { ok: true, status: 200, json: async () => ({ jobs }) };
    }
    if (method === "POST" && /\/api\/jobs\/[^/]+\/run$/.test(url)) {
      return { ok: true, status: 200,
               json: async () => ({ result_id: "r1", status: "ok", output: "done" }) };
    }
    if (method === "GET" && /\/api\/jobs\/[^/]+\/results$/.test(url)) {
      return { ok: true, status: 200,
               json: async () => ({ id: JOB.id, results: [
                 { result_id: "r1", status: "ok", output: "the result text",
                   error: null, finished: 1700003600 }] }) };
    }
    if (method === "PUT" || method === "DELETE" || method === "POST") {
      return { ok: true, status: 200, json: async () => ({}) };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };

  // Expose the jsdom realm as the module's ambient globals. jobs.js only touches
  // document / window / fetch / setTimeout, all sourced from here.
  global.window = win;
  global.document = win.document;
  global.fetch = win.fetch;
  global.confirm = win.confirm;
  return { win, calls };
}

// Import the module fresh (cache-busted) so each test starts with no shared
// module-level state and the just-installed globals.
async function importJobs() {
  const url = pathToFileURL(JOBS_JS).href + `?t=${Date.now()}_${Math.random()}`;
  return import(url);
}

// A small async settle so awaited fetch microtasks in refresh() resolve before
// we assert against the DOM.
const settle = () => new Promise((r) => setTimeout(r, 0));

test("register builds a #view-jobs container in #main", async () => {
  const { win } = makeEnv();
  const mod = await importJobs();
  assert.equal(typeof mod.register, "function", "jobs.js must export register()");
  await mod.register({ toast: () => {}, authHeaders: () => ({}) });
  const view = win.document.getElementById("view-jobs");
  assert.ok(view, "register() must create #view-jobs");
  assert.ok(win.document.getElementById("main").contains(view),
    "#view-jobs lives inside #main");
});

test("the jobs view renders canned job names via textContent", async () => {
  const { win } = makeEnv();
  const mod = await importJobs();
  await mod.register({ toast: () => {}, authHeaders: () => ({}) });

  // Trigger the view exactly as app.js does: the chained window.onViewShown.
  assert.equal(typeof win.onViewShown, "function",
    "register() must install a chained window.onViewShown");
  win.onViewShown("jobs");
  await settle();

  const view = win.document.getElementById("view-jobs");
  assert.match(view.textContent, /Nightly digest/, "job name is rendered");
});

test("Run now issues a POST to /api/jobs/<id>/run", async () => {
  const { win, calls } = makeEnv();
  const mod = await importJobs();
  await mod.register({ toast: () => {}, authHeaders: () => ({}) });
  win.onViewShown("jobs");
  await settle();

  // Find the Run-now button for the rendered job (data-action marks intent).
  const btn = win.document.querySelector(
    `#view-jobs [data-action="run"][data-id="${JOB.id}"]`);
  assert.ok(btn, "a Run-now button is rendered for the job");
  btn.click();
  await settle();

  const ran = calls.find(
    (c) => c.method === "POST" && c.url.endsWith(`/api/jobs/${JOB.id}/run`));
  assert.ok(ran, "clicking Run now POSTs to /api/jobs/<id>/run");
});

test("authHeaders() is applied to the list fetch", async () => {
  const { win, calls } = makeEnv();
  const mod = await importJobs();
  let used = false;
  await mod.register({ toast: () => {}, authHeaders: () => { used = true; return {}; } });
  win.onViewShown("jobs");
  await settle();
  assert.ok(used, "register/refresh calls ctx.authHeaders()");
  assert.ok(calls.some((c) => c.method === "GET" && /\/api\/jobs$/.test(c.url)),
    "the list is fetched from GET /api/jobs");
});
