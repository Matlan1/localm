// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

// NET-1: over a network bind the SPA boots without a key (the loopback key is
// never auto-seeded to remote clients). The boot must show an IN-PAGE key gate
// so a phone/LAN client can enter a key - NOT window.prompt(), which mobile/PWA
// browsers suppress (leaving a blank page). These tests pin that behaviour.

const tick = () => new Promise((r) => setTimeout(r, 100));

const keyless401 = () => Promise.resolve({
  ok: false, status: 401,
  json: async () => ({ detail: "Missing or invalid API key" }),
  text: async () => "Missing or invalid API key",
});
const allOk = () => Promise.resolve({
  ok: true, status: 200,
  json: async () => ({ models: [], active: "" }),
  text: async () => "",
});

test("NET-1: a keyless (401) boot shows the in-page key gate, not window.prompt", async () => {
  const { window } = loadApp({ fetchImpl: keyless401 });
  let promptCalls = 0;
  window.prompt = () => { promptCalls++; return null; };
  await tick();
  const gate = window.document.getElementById("key-gate");
  assert.ok(gate, "#key-gate element exists in the shell");
  assert.notEqual(gate.style.display, "none", "key gate is visible after a 401 boot");
  assert.equal(promptCalls, 0, "window.prompt() is NOT used (mobile/PWA suppress it)");
});

test("NET-1: a working backend leaves the gate hidden (loopback case unaffected)", async () => {
  const { window } = loadApp({ fetchImpl: allOk });
  await tick();
  const gate = window.document.getElementById("key-gate");
  assert.ok(gate, "#key-gate element exists");
  assert.equal(gate.style.display, "none", "key gate stays hidden when fetches succeed");
});

test("NET-1: submitting the gate stores the key (trimmed) for the next boot", async () => {
  const { window } = loadApp({ fetchImpl: keyless401 });
  try { Object.defineProperty(window.location, "reload", { configurable: true, value: () => {} }); } catch { /* jsdom no-op nav */ }
  await tick();
  window.document.getElementById("key-gate-input").value = "  my-secret-key  ";
  window.document.getElementById("key-gate-submit").click();
  assert.equal(window.localStorage.getItem("localm.apiKey"), "my-secret-key",
    "submitting stores the trimmed key so the reloaded boot is authenticated");
});
