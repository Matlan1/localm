// SPDX-License-Identifier: AGPL-3.0-or-later
// The jobs plugin renders server-supplied strings (a job's name, its prompt,
// a run's status). jobs.js asserts at its top that every such string reaches
// the DOM "via textContent (or value), never innerHTML".
//
// That claim had never been driven through a payload battery, only read. This
// drives it: hostile markup is fed through the real render path in jsdom and
// the DOM is asked whether any element was PARSED out of it.
//
// The battery carries its own positive control (the last test): the same
// detector, pointed at a document built by parsing that markup, must report
// INJECTED. Without it, a clean sweep here could mean the payloads are safe OR
// that the detector cannot see anything at all.

import { test } from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const JOBS_JS = join(HERE, "..", "localm", "plugins", "builtin", "jobs",
                     "static", "jobs.js");

const PAYLOADS = [
  '<img src=x onerror="window.__INJECTED=1">',
  '<svg onload="window.__INJECTED=1"></svg>',
  '"><b>bold</b>',
  '<iframe src="javascript:window.__INJECTED=1"></iframe>',
];

function jobWith(payload) {
  return {
    id: "abc123",
    name: payload,
    schedule_kind: "interval",
    schedule: 3600,
    task_kind: "chat",
    prompt: payload,
    model: null, cwd: null, scope: null,
    enabled: true,
    created: 1700000000,
    last_run: 1700003600,
    last_status: "ok",
    last_result_id: "2026-06-17T00-00-00",
  };
}

function makeEnv(job) {
  const dom = new JSDOM(
    `<!DOCTYPE html><html><body><main id="main"></main></body></html>`,
    { url: "http://localhost:8642/" });
  const win = dom.window;
  win.confirm = () => true;
  win.fetch = async (url, opts = {}) => {
    const method = (opts.method || "GET").toUpperCase();
    if (method === "GET" && /\/api\/jobs$/.test(url)) {
      return { ok: true, status: 200, json: async () => ({ jobs: [job] }) };
    }
    if (method === "GET" && /\/api\/models$/.test(url)) {
      return { ok: true, status: 200,
               json: async () => ({ models: [{ name: "m", active: true }],
                                    active: "m" }) };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };
  global.window = win;
  global.document = win.document;
  global.fetch = win.fetch;
  global.confirm = win.confirm;
  return win;
}

async function importJobs() {
  const url = pathToFileURL(JOBS_JS).href + `?t=${Date.now()}_${Math.random()}`;
  return import(url);
}

const settle = () => new Promise((r) => setTimeout(r, 0));

// The detector: did the payload become ELEMENTS, or stay text?
function injectedElements(root) {
  return root.querySelectorAll("img, script, svg, b, iframe").length;
}

test("the jobs view never parses server-supplied strings into elements", async () => {
  for (const payload of PAYLOADS) {
    const win = makeEnv(jobWith(payload));
    const mod = await importJobs();
    await mod.register({ toast: () => {}, authHeaders: () => ({}) });
    win.onViewShown("jobs");
    await settle();

    const view = win.document.getElementById("view-jobs");
    assert.ok(view, "the jobs view rendered");
    assert.equal(injectedElements(view), 0,
      `payload was parsed into DOM elements: ${payload}`);
    assert.equal(win.__INJECTED, undefined, `payload executed: ${payload}`);
    // It must still be VISIBLE as text. A render that silently DROPS hostile
    // input is not the same as one that escapes it, and only one of those is
    // honest about what the job is actually named.
    assert.ok(view.textContent.includes(payload),
      `payload neither rendered as text nor escaped, it vanished: ${payload}`);
  }
});

test("POSITIVE CONTROL: the same detector reports elements when they exist", async () => {
  // Proves the detector can report INJECTED. Without this, "0 elements" could
  // mean the payloads are safe OR that the query is simply blind. Built with
  // DOMParser rather than an unsafe assignment, so the proof costs the repo
  // no dangerous sink of its own.
  const win = makeEnv(jobWith("x"));
  for (const payload of PAYLOADS) {
    const parsed = new win.DOMParser().parseFromString(
      `<body>${payload}</body>`, "text/html");
    assert.ok(injectedElements(parsed.body) > 0,
      `the detector must see an element parsed out of: ${payload}`);
  }
});
