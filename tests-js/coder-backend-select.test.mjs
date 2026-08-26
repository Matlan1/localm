// SPDX-License-Identifier: AGPL-3.0-or-later
// ADR-0013 stage 1: the coder setup form chooses WHICH model server answers the
// session. Choosing anything other than this localm is a trust-boundary change,
// so the properties pinned here are the consent surface and the default, not the
// happy path:
//
//   - the DEFAULT form is byte-for-byte the request it always sent (no backend
//     field at all), because a feature that moves the default is a different
//     feature;
//   - the consequence of an off-machine choice is VISIBLE, on select, and stays
//     visible - not a tooltip, because hover does not exist on touch and phones
//     are a supported target;
//   - the privacy warning re-renders when the PERSISTENCE choice changes, since
//     that is what decides whether the backend choice is allowed at all;
//   - the API key does not linger in the DOM after it has been sent.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function makeFetch(calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u === "/api/coder/sessions" && (opts.method || "GET") === "POST") {
      calls.push(JSON.parse(opts.body));
      return {
        ok: true, status: 200, text: async () => "",
        json: async () => ({ id: "s1", cwd: "/tmp/project", notes: [],
                             backend_info: { backend: "openai", leaves_machine: true,
                                             target: "https://api.openai.com/v1",
                                             model: "gpt-4o" } }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

const pick = (win, value) => {
  const sel = win.document.getElementById("setup-backend");
  sel.value = value;
  sel.dispatchEvent(new win.Event("change"));
  return win.document.getElementById("setup-backend-hint");
};

const shown = (win, id) =>
  win.document.getElementById(id).style.display !== "none";

test("the default is this localm and the extra fields stay out of the way", () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  assert.equal(win.document.getElementById("setup-backend").value, "local");
  for (const id of ["setup-backend-url-wrap", "setup-backend-key-wrap",
                    "setup-backend-model-wrap", "setup-backend-hint"]) {
    assert.equal(shown(win, id), false, id + " is hidden for the local default");
  }
});

test("an unchanged form sends the request it always sent", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.startCoderSession();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(calls.length, 1);
  for (const k of ["backend", "backend_url", "backend_model", "backend_api_key"]) {
    assert.ok(!(k in calls[0]), k + " must be omitted when the default is unchanged");
  }
});

test("picking a provider reveals only the fields it needs", () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  pick(win, "openai");
  assert.equal(shown(win, "setup-backend-key-wrap"), true, "a provider needs a key");
  assert.equal(shown(win, "setup-backend-model-wrap"), true, "and a model name");
  assert.equal(shown(win, "setup-backend-url-wrap"), false,
    "a fixed provider has no URL to type - offering one invites a wrong answer");

  pick(win, "url");
  assert.equal(shown(win, "setup-backend-url-wrap"), true);
});

test("the consequence of leaving the machine is stated, not implied", () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  const hint = pick(win, "anthropic");
  assert.equal(shown(win, "setup-backend-hint"), true, "shown on select, and it stays");
  assert.match(hint.textContent, /Anthropic/, "names WHO receives the data");
  assert.match(hint.textContent, /leave this machine/,
    "the consent surface has to say the thing it is asking consent for");
  // Grammar-constrained tool calls are a localm-server capability. Losing them
  // silently is the rule-5 failure this line exists to prevent.
  assert.match(hint.textContent, /Grammar-constrained tool calls are off/);
});

test("in privacy mode the hint says it will be refused, and names the setting", () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  assert.equal(win.document.getElementById("setup-mode").value, "privacy",
    "privacy is the default, which is exactly why this warning has to exist");
  const hint = pick(win, "openai");
  assert.match(hint.textContent, /refuse/, "says what will happen");
  assert.match(hint.textContent, /log or\s+full/,
    "and what to change - a dead-end warning is worse than none");
});

test("changing the persistence choice re-renders the warning", () => {
  // The two controls are independent, so the hint has to listen to BOTH. Wiring
  // it only to the backend select leaves a stale "this will be refused" sitting
  // under a form that would now succeed.
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  const hint = pick(win, "openai");
  assert.match(hint.textContent, /refuse/);
  const mode = win.document.getElementById("setup-mode");
  mode.value = "log";
  mode.dispatchEvent(new win.Event("change"));
  assert.doesNotMatch(hint.textContent, /refuse/,
    "log mode allows an off-machine model, so the refusal warning must clear");
});

test("the chosen backend is sent, and the key does not linger in the DOM", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  win.document.getElementById("setup-mode").value = "log";
  pick(win, "openai");
  win.document.getElementById("setup-backend-model").value = "gpt-4o";
  win.document.getElementById("setup-backend-key").value = "sk-secret-value";
  await win.startCoderSession();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(calls[0].backend, "openai");
  assert.equal(calls[0].backend_model, "gpt-4o");
  assert.equal(calls[0].backend_api_key, "sk-secret-value");
  // It had one job. Left in the field it survives every later screenshot,
  // screen share and stray autofill for the life of the page.
  assert.equal(win.document.getElementById("setup-backend-key").value, "",
    "the key is cleared once the session has been created");
});


// --------------------------------------------------------------------------
//  The running-session badge
// --------------------------------------------------------------------------
// The setup hint is consent at the moment of choosing. This is the part that has
// to hold afterwards: while you are typing into a session, it says where the
// words are going. A badge that is always on says nothing, so both states are
// asserted here, not just the interesting one.

function seed(window, backendInfo) {
  runScript(window, `
    coder.sessions.set("s1", { info: { id: "s1", cwd: "/p/alpha", total_tokens: 0,
                                       turns: 0, patch_mode: false,
                                       backend_info: ${JSON.stringify(backendInfo)} },
                               busy: false, feedEl: document.createElement("div") });
    activateSession("s1");
  `);
  return window.document.getElementById("coder-remote");
}

test("a session whose model is off this machine is marked while it runs", () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  const badge = seed(win, { backend: "anthropic", leaves_machine: true,
                            target: "https://api.anthropic.com/v1",
                            model: "claude-opus-4-5" });
  assert.notEqual(badge.style.display, "none", "the marker is visible");
  // The HOST, not the whole URL: the full target crowded the session bar hard
  // enough to push the End button off the right edge.
  assert.equal(badge.textContent, "remote: api.anthropic.com");
  assert.match(badge.title, /api\.anthropic\.com\/v1/,
    "the full target survives in the tooltip - abbreviating must not lose it");
});

test("a session on this machine carries no marker at all", () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  // A LOOPBACK URL is still a custom backend, and still must not be marked
  // remote: this is the case a naive "any custom URL is cloud" rule gets wrong.
  const badge = seed(win, { backend: "url", leaves_machine: false,
                            target: "http://127.0.0.1:11434/v1", model: "qwen" });
  assert.equal(badge.style.display, "none",
    "marking a local session remote would train the user to ignore the marker");
});
