// SPDX-License-Identifier: AGPL-3.0-or-later
// theme.js and logo.js read/write localStorage at ES-module evaluation time
// with no guard, and init.js reads it at boot before the app shell is
// revealed. A browser that throws a SecurityError from localStorage (a
// blocked/disabled store: private browsing, a strict storage policy, quota)
// must degrade those preferences to in-memory defaults, never abort boot.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const OK = { ok: true, status: 200, json: async () => ({}), text: async () => "" };
const settle = (ms = 50) => new Promise((r) => setTimeout(r, ms));

test("theme still applies its default and wires the toggle when storage is blocked", () => {
  const { window } = loadApp({ breakStorage: true, fetchImpl: async () => OK });
  assert.equal(window.document.documentElement.dataset.theme, "dark",
    "applyTheme's in-memory default must still apply");
  assert.equal(typeof window.document.getElementById("theme-toggle").onclick, "function",
    "the toggle handler must still be wired - proves theme.js ran past its storage read");
});

test("the logo wordmark and style picker still render when storage is blocked", () => {
  const { window } = loadApp({ breakStorage: true, fetchImpl: async () => OK });
  const logo = window.document.getElementById("logo");
  assert.equal(logo.textContent, "LocaLM", "the default wordmark still renders");
  assert.equal(logo.querySelector("span").textContent, "M");
  const tiles = window.document.querySelectorAll("#logo-style-picker .logo-tile");
  assert.equal(tiles.length, 3,
    "renderLogoPicker() must still run - proves logo.js ran to its final line");
});

test("applyLogoStyle and renderLogoPicker still work directly when storage is blocked", () => {
  const { window } = loadApp({ breakStorage: true, fetchImpl: async () => OK });
  assert.doesNotThrow(() => window.applyLogoStyle("localm"));
  assert.equal(window.document.getElementById("logo").textContent, "localm");
  assert.doesNotThrow(() => window.renderLogoPicker());
});

test("the main interface still boots to the app shell when storage is blocked", async () => {
  const { window } = loadApp({ breakStorage: true, fetchImpl: async () => OK });
  await settle();
  assert.equal(typeof window.bootAuthProbe, "function",
    "init.js's own top-level code must still have run (a throw at module-eval " +
    "time in a real ES module would also abort every module that imports it)");
  assert.equal(window.document.getElementById("app").style.display, "",
    "bootAuthProbe's success path must still reveal the app shell");
  assert.equal(window.document.getElementById("setup-cwd").value, "",
    "a blocked coderCwd read degrades to the empty default instead of throwing");
  assert.equal((window.__localmClientLog || []).some((l) => /securityerror|storage is blocked/i.test(l)),
    false, "no storage error may escape uncaught anywhere in the boot chain " +
    "(a throw past this point would silently stop init.js's script, including " +
    "the renderConvList()/renderChat() calls at its very end)");
});

test("a boot-time key gate (401) still shows instead of an unrecovered throw when storage is blocked", async () => {
  const { window } = loadApp({
    breakStorage: true,
    fetchImpl: async (url) => (String(url).includes("/api/models")
      ? { ok: false, status: 401, json: async () => ({}), text: async () => "" } : OK),
  });
  await settle();
  assert.equal(window.document.getElementById("key-gate").style.display, "flex",
    "the auth gate must still render - proves the boot chain reached it");
});

test("a read that throws only for an unrelated key does not stop the rest of init.js", async () => {
  // instanceCacheTrusted() succeeds (a real cached instance id), so
  // _instanceTrusted is true and the guarded coderCwd/activeView reads are
  // actually reached - the harder case than "everything throws", where
  // instanceCacheTrusted() itself already fails closed and those two reads
  // are never attempted at all.
  const { window } = loadApp({
    seedLocalStorage: { "localm.instanceId": "abc123" },
    breakStorage: ["localm.coderCwd", "localm.activeView"],
    fetchImpl: async () => OK,
  });
  await settle();
  assert.equal(typeof window.bootAuthProbe, "function",
    "init.js's own script must run to completion despite the blocked coderCwd read");
  assert.equal(window.document.getElementById("setup-cwd").value, "",
    "the blocked read degrades to the empty default instead of throwing");
  assert.equal((window.__localmClientLog || []).some((l) => l.includes("unhandledrejection")), false,
    "the blocked activeView read (inside the boot IIFE) must not surface as an unhandled rejection");
});
