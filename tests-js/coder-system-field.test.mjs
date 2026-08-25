// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// The coder session-setup form's "Custom instructions" textarea is sent as
// custom_instructions in POST /api/coder/sessions.

function setup() {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    if (u === "/api/coder/sessions" && opts.method === "POST")
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ id: "s1", cwd: "/proj", resumed: false, mode: "privacy" }) };
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", sessions: [] }) };
  };
  const { window } = loadAppWithPages({ fetchImpl });
  return { window, calls };
}

const tick = () => new Promise((r) => setTimeout(r, 0));
function createBody(calls) {
  const c = calls.find(
    (x) => x.url === "/api/coder/sessions" && x.opts.method === "POST");
  return c ? JSON.parse(c.opts.body) : null;
}

test("the setup form has a custom-instructions textarea", () => {
  const { window } = setup();
  const ta = window.document.getElementById("setup-system");
  assert.ok(ta, "setup-system exists");
  assert.equal(ta.tagName, "TEXTAREA");
});

test("startCoderSession sends custom_instructions when filled (trimmed)", async () => {
  const { window, calls } = setup();
  window.document.getElementById("setup-cwd").value = "/proj";
  window.document.getElementById("setup-system").value = "  Be terse.  ";
  runScript(window, "startCoderSession();");
  await tick();
  await tick();
  const body = createBody(calls);
  assert.ok(body, "a create POST was sent");
  assert.equal(body.custom_instructions, "Be terse.");
});

test("startCoderSession omits custom_instructions when the field is blank", async () => {
  const { window, calls } = setup();
  window.document.getElementById("setup-cwd").value = "/proj";
  window.document.getElementById("setup-system").value = "   ";
  runScript(window, "startCoderSession();");
  await tick();
  await tick();
  const body = createBody(calls);
  assert.ok(body, "a create POST was sent");
  assert.equal("custom_instructions" in body, false,
    "no custom_instructions key when blank (server falls back to the file)");
});
