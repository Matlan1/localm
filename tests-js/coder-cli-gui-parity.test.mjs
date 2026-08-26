// SPDX-License-Identifier: AGPL-3.0-or-later
// GUI controls for the six coder-only CLI options:
//
//   --estimate       "estimate" button beside the composer
//   --patch-mode     "Patch mode" checkbox + a "patch" download button
//   --native-tools   "Native tools API" checkbox, with the server's verdict relayed
//   --output-format  export offers markdown OR the last task's result JSON
//   --episodes       "lessons" button in the setup panel
//   --until          verification command + fix-attempt cap in the setup form

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function makeFetch(calls, { sessionInfo = {}, routes = {} } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    calls.push({ url: u, method, body: opts.body ? JSON.parse(opts.body) : null });
    if (u === "/api/coder/sessions" && method === "POST") {
      return {
        ok: true, status: 200,
        json: async () => ({ id: "s1", cwd: "/tmp/project", notes: [], ...sessionInfo }),
        text: async () => "",
      };
    }
    for (const [prefix, resp] of Object.entries(routes)) {
      if (u.startsWith(prefix)) return resp;
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

function createCalls(calls) {
  return calls.filter((c) => c.url === "/api/coder/sessions" && c.method === "POST");
}

async function startSession(win, calls) {
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.startCoderSession();
  await new Promise((r) => setTimeout(r, 0));
  return createCalls(calls)[0].body;
}

/* ------------------------------------------------------------------ */
/*  --until                                                            */
/* ------------------------------------------------------------------ */

test("verification command: the setup form has one, and blank omits it", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  assert.ok(win.document.getElementById("setup-verify"),
    "there must be a control for the exit-code oracle");
  const body = await startSession(win, calls);
  assert.ok(!("verify" in body),
    "blank must omit verify so the project's detected check stays the default");
  assert.ok(!("auto_verify" in body), "and auto-detection stays on");
});

test("verification command typed: it reaches the session POST", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  win.document.getElementById("setup-verify").value = "pytest -x";
  const body = await startSession(win, calls);
  assert.equal(body.verify, "pytest -x");
});

test("skip verification: sends auto_verify=false and does not also send a command", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  win.document.getElementById("setup-verify").value = "pytest -x";
  win.document.getElementById("setup-no-verify").checked = true;
  const body = await startSession(win, calls);
  assert.equal(body.auto_verify, false);
  assert.ok(!("verify" in body),
    "--no-verify wins over a typed command, exactly as the CLI flag does");
});

test("fix-attempt cap: blank omits it, a number is sent (the CLI's --goal-max-iters)", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  const field = win.document.getElementById("setup-verify-retries");
  assert.equal(field.value, "", "no client-side duplicate of the server default");
  let body = await startSession(win, calls);
  assert.ok(!("verify_max_retries" in body));

  const calls2 = [];
  const { window: win2 } = loadAppWithPages({ fetchImpl: makeFetch(calls2) });
  win2.document.getElementById("setup-verify-retries").value = "7";
  body = await startSession(win2, calls2);
  assert.equal(body.verify_max_retries, 7);
});

/* ------------------------------------------------------------------ */
/*  --patch-mode                                                       */
/* ------------------------------------------------------------------ */

test("patch mode: the checkbox reaches the session POST, default off", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  let body = await startSession(win, calls);
  assert.equal(body.patch_mode, false);

  const calls2 = [];
  const { window: win2 } = loadAppWithPages({ fetchImpl: makeFetch(calls2) });
  win2.document.getElementById("setup-patch").checked = true;
  body = await startSession(win2, calls2);
  assert.equal(body.patch_mode, true);
});

test("the patch button is shown only for a patch-mode session", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, { sessionInfo: { patch_mode: true } }),
  });
  const btn = win.document.getElementById("coder-patch");
  assert.equal(btn.style.display, "none", "hidden with no session");
  await startSession(win, calls);
  assert.notEqual(btn.style.display, "none",
    "a patch-mode session must offer its patch");

  const calls2 = [];
  const { window: win2 } = loadAppWithPages({
    fetchImpl: makeFetch(calls2, { sessionInfo: { patch_mode: false } }),
  });
  await startSession(win2, calls2);
  assert.equal(win2.document.getElementById("coder-patch").style.display, "none");
});

/* ------------------------------------------------------------------ */
/*  --native-tools                                                     */
/* ------------------------------------------------------------------ */

test("native tools: the checkbox is sent, and the server's 'not applied' note is shown", async () => {
  const calls = [];
  const toasts = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, {
      sessionInfo: {
        native_tools: false,
        native_tools_requested: true,
        notes: ["native_tools was not applied: this server does not implement the OpenAI tools API."],
      },
    }),
  });
  const realToast = win.toast;
  win.toast = (msg, isErr) => { toasts.push(String(msg)); return realToast?.(msg, isErr); };
  win.document.getElementById("setup-native-tools").checked = true;
  const body = await startSession(win, calls);
  assert.equal(body.native_tools, true, "the request is sent for real");
  assert.ok(toasts.some((t) => t.includes("native_tools was not applied")),
    "an option the server could not honour must be SAID, not swallowed - a "
    + "ticked box and an ignored one would otherwise look identical");
});

test("no notes, no noise: a plain session start toasts nothing about options", async () => {
  const calls = [];
  const toasts = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  win.toast = (msg) => toasts.push(String(msg));
  const body = await startSession(win, calls);
  assert.equal(body.native_tools, false);
  assert.equal(toasts.length, 0);
});

/* ------------------------------------------------------------------ */
/*  --estimate                                                         */
/* ------------------------------------------------------------------ */

test("estimate: posts the composer text to the estimate route and leaves it there", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, {
      routes: {
        "/api/coder/sessions/s1/estimate": {
          ok: true, status: 200,
          json: async () => ({ estimate: "PLAN: do the thing", total_tokens: 12 }),
          text: async () => "",
        },
      },
    }),
  });
  await startSession(win, calls);
  win.document.getElementById("coder-input").value = "add a --foo flag";
  await win.estimateCoderTask();
  const est = calls.find((c) => c.url.endsWith("/estimate"));
  assert.ok(est, "the estimate route was called");
  assert.equal(est.method, "POST");
  assert.equal(est.body.text, "add a --foo flag");
  assert.equal(win.document.getElementById("coder-input").value, "add a --foo flag",
    "the task stays in the composer - an estimate is a pre-flight, and the "
    + "usual next action is sending the same text for real");
});

test("estimate with an empty composer never reaches the server", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  await startSession(win, calls);
  win.document.getElementById("coder-input").value = "   ";
  await win.estimateCoderTask();
  assert.equal(calls.filter((c) => c.url.endsWith("/estimate")).length, 0);
});

test("an estimate event renders as a labelled, non-executed plan", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  await startSession(win, calls);
  // `coder` is a module-scope binding, not a window property.
  runScript(win, "window.coderState = coder;");
  const s = win.coderState.sessions.get("s1");
  win.handleCoderEvent(s, {
    type: "estimate", task: "add a --foo flag",
    text: "PLAN: touch cli.py", prompt_tokens: 5, total_tokens: 9,
  });
  const rendered = s.feedEl.textContent;
  assert.match(rendered, /Estimate for: add a --foo flag/,
    "labelled, so a replayed plan cannot be mistaken for a turn that ran");
  assert.match(rendered, /PLAN: touch cli\.py/);
  assert.match(rendered, /Nothing was run or written/);
});

/* ------------------------------------------------------------------ */
/*  --output-format json                                               */
/* ------------------------------------------------------------------ */

test("result JSON export reads the server's result route, not the client event log", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, {
      routes: {
        "/api/coder/sessions/s1/result": {
          ok: true, status: 200,
          json: async () => ({ ok: true, response: "done", turns: 3, total_tokens: 90 }),
          text: async () => "",
        },
      },
    }),
  });
  await startSession(win, calls);
  await win.exportCoderResultJson();
  const res = calls.find((c) => c.url.endsWith("/result"));
  assert.ok(res, "a tab that joined late never saw the final event, so the "
    + "result has to come from the server rather than a local log");
});

/* ------------------------------------------------------------------ */
/*  --episodes                                                         */
/* ------------------------------------------------------------------ */

test("lessons: the setup panel queries the episodes route for the typed cwd", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, {
      routes: {
        "/api/coder/episodes": {
          ok: true, status: 200,
          json: async () => ({
            cwd: "/tmp/project",
            episodes: [{ id: "ep123", outcome: "ok", lesson: "run the tests first",
                         summary: "", task: "fix parser", turns: 3, merged: 0 }],
          }),
          text: async () => "",
        },
      },
    }),
  });
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.openEpisodesModal();
  const ep = calls.find((c) => c.url.startsWith("/api/coder/episodes"));
  assert.ok(ep, "the episodes route was called");
  assert.match(ep.url, /cwd=%2Ftmp%2Fproject/);
  const body = win.document.getElementById("modal-body").textContent;
  assert.match(body, /run the tests first/);
  assert.match(body, /ep123/,
    "the id has to be shown: it is what --forget-episode / --restore-episode take");
});

test("lessons with no cwd never reaches the server", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  win.document.getElementById("setup-cwd").value = "";
  await win.openEpisodesModal();
  assert.equal(calls.filter((c) => c.url.startsWith("/api/coder/episodes")).length, 0);
});
