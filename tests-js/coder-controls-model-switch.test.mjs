// SPDX-License-Identifier: AGPL-3.0-or-later
// The coder session-controls model switcher: the active-model tag must go
// through i18n rather than a hardcoded English literal, and picking a model
// already offered must re-check the LIVE session state rather than a
// snapshot taken when the modal opened - otherwise switching back to the
// model the session started on, in the same still-open modal, is a no-op.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function reply(status, data) {
  return { ok: status < 400, status, json: async () => data, text: async () => "" };
}

function makeFetch(calls, { model = [], de = null } = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    if (u.includes("/i18n/de.json")) return reply(200, de || {});
    if (/^\/api\/coder\/sessions\/[^/]+\/model$/.test(u) && method === "POST") {
      calls.push({ url: u, body: JSON.parse(opts.body) });
      return reply(...model.shift());
    }
    return reply(200, {});
  };
}

function settle(ms = 0) { return new Promise((r) => setTimeout(r, ms)); }
async function waitFor(fn, timeout = 1000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await settle(10); }
  return false;
}

// A running session s1 for /p/alpha, made the active one.
function seedSession(win, model = "model-a") {
  runScript(win, `
    coder.sessions.set("s1", { info: { id: "s1", cwd: "/p/alpha", model: ${JSON.stringify(model)},
                                       total_tokens: 0, turns: 0, patch_mode: false,
                                       backend_info: { backend: "local" } },
                               busy: false, feedEl: document.createElement("div") });
    activateSession("s1");
  `);
}

function seedModels(win, models) {
  runScript(win, `modelCache.models = ${JSON.stringify(models)};`);
}

test("session controls: the active-model tag is translated, not hardcoded English", async () => {
  const calls = [];
  const de = { "models.tag.active": "AKTIV-MARKER" };
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch(calls, { de }),
    seedLocalStorage: { "localm.language": "de" },
  });
  await waitFor(() => win.document.documentElement.lang === "de");
  seedSession(win);
  seedModels(win, [{ name: "model-a", active: true }, { name: "model-b", active: false }]);

  win.openSessionControls();

  const select = win.document.getElementById("ctl-model");
  assert.ok(select, "the model select renders");
  const activeOpt = Array.from(select.options).find((o) => o.value === "model-a");
  assert.ok(activeOpt, "the active model is offered");
  assert.match(activeOpt.textContent, /AKTIV-MARKER/,
    `expected the translated active tag, got: ${activeOpt.textContent}`);
  assert.doesNotMatch(activeOpt.textContent, /\(active\)/,
    "the English literal must not survive under a German catalog");
});

test("session controls: switching back to the original model in the same open modal is not a no-op", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, {
    model: [[200, { id: "s1", model: "model-b" }], [200, { id: "s1", model: "model-a" }]],
  }) });
  seedSession(win, "model-a");
  seedModels(win, [{ name: "model-a", active: false }, { name: "model-b", active: true }]);

  win.openSessionControls();
  const select = win.document.getElementById("ctl-model");
  const btn = select.nextElementSibling;
  assert.equal(btn.tagName, "BUTTON", "the switch button sits right after the select");

  select.value = "model-b";
  await btn.onclick();
  assert.equal(calls.length, 1, "the first switch posts");
  assert.deepEqual(calls[0].body, { model: "model-b" });

  select.value = "model-a";
  await btn.onclick();
  assert.equal(calls.length, 2, "switching back must also post, not silently return");
  assert.deepEqual(calls[1].body, { model: "model-a" });
});

test("slash /model with no arg reports the live model through i18n", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  seedSession(win, "model-a");

  win.execCoderCommand("model", "");

  const toastText = win.document.getElementById("toast").textContent;
  assert.equal(toastText, win.t("slash.coderCmd.currentModel", { model: "model-a" }));
});

test("slash /model with no arg and no model loaded names \"none\" through i18n, not a literal", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  seedSession(win, "");

  win.execCoderCommand("model", "");

  const toastText = win.document.getElementById("toast").textContent;
  assert.equal(toastText,
    win.t("slash.coderCmd.currentModel", { model: win.t("chat.none") }));
  assert.doesNotMatch(toastText, /: none$/, "the raw English literal must not appear");
});
