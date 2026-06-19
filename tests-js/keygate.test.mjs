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

test("NET-1: over HTTPS the gate offers a one-tap Install-certificate link", async () => {
  const { window } = loadApp({ fetchImpl: keyless401, url: "https://192.168.0.5:8651/" });
  await tick();
  const cert = window.document.getElementById("key-gate-cert");
  const link = window.document.getElementById("key-gate-cert-link");
  assert.ok(cert, "#key-gate-cert exists in the shell");
  assert.notEqual(cert.style.display, "none", "cert link is shown over HTTPS (built-in TLS)");
  assert.equal(link.getAttribute("href"), "/localm-ca.crt", "links to the CA download");
});

test("NET-1: over plain HTTP the cert link stays hidden (loopback dev gate)", async () => {
  const { window } = loadApp({ fetchImpl: keyless401, url: "http://localhost:8642/" });
  await tick();
  const cert = window.document.getElementById("key-gate-cert");
  assert.ok(cert, "#key-gate-cert exists");
  assert.equal(cert.style.display, "none", "no CA to install over plain HTTP, so hidden");
});

test("NET-1 hard gate: a keyless (401) boot HIDES the app shell - nothing of localm loads behind the gate", async () => {
  const { window } = loadApp({ fetchImpl: keyless401 });
  await tick();
  const app = window.document.getElementById("app");
  assert.equal(app.style.display, "none", "#app is hidden while locked (no shell behind the onboarding)");
  assert.equal(window.__localmLocked, true, "the lock flag is set so late boot steps bail too");
  const gate = window.document.getElementById("key-gate");
  assert.notEqual(gate.style.display, "none", "the onboarding/key gate is shown");
});

test("NET-1 onboarding: the guided cert-install step + per-platform help render over HTTPS", async () => {
  const { window } = loadApp({ fetchImpl: keyless401, url: "https://192.168.0.5:8651/" });
  await tick();
  const help = window.document.querySelector("#key-gate-cert .onboard-help");
  assert.ok(help, "per-platform cert-install help is present");
  assert.match(help.textContent, /iOS|iPadOS/, "includes the iOS trust steps");
  const titles = [...window.document.querySelectorAll(".onboard-step-title")]
    .map((e) => e.textContent);
  assert.ok(titles.some((t) => /Trust this device/.test(t)), "cert step titled");
  assert.ok(titles.some((t) => /Enter your API key/.test(t)), "key-entry step titled");
});

test("NET-1 hard gate: a working (200) boot reveals the app and hides the gate", async () => {
  const { window } = loadApp({ fetchImpl: allOk });
  await tick();
  const app = window.document.getElementById("app");
  assert.notEqual(app.style.display, "none", "#app is visible when authed / open mode");
  assert.equal(window.__localmLocked, false, "unlocked");
  const gate = window.document.getElementById("key-gate");
  assert.equal(gate.style.display, "none", "gate hidden when authed / open mode");
});
