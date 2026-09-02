// SPDX-License-Identifier: AGPL-3.0-or-later
// setup-max-turns renders as an empty field with a placeholder, and is omitted
// from the session POST when blank.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    if (u === "/api/coder/sessions" && (opts.method || "GET") === "POST") {
      calls.push(JSON.parse(opts.body));
      return { ok: true, status: 200, json: async () => ({ session_id: "s1" }), text: async () => "" };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

test("setup-max-turns: index.html carries no hardcoded value, matching its temperature sibling", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  const el = win.document.getElementById("setup-max-turns");
  assert.equal(el.value, "", "no solid pre-filled value");
  assert.match(el.placeholder, /default \(40\)/, "the server's real default is named in the placeholder");
});

test("setup-max-turns left blank: the session POST omits max_turns (server applies its own default)", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.startCoderSession();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(calls.length, 1);
  assert.ok(!("max_turns" in calls[0]),
    "max_turns must be OMITTED, not sent as a client-guessed 40");
});

test("setup-max-turns typed: the session POST sends exactly what the user entered", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  win.document.getElementById("setup-max-turns").value = "12";
  await win.startCoderSession();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(calls[0].max_turns, 12);
});
