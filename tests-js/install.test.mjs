// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

// applyInstallUI() picks one branch of the Settings "Install app" affordance:
//   standalone -> "installed" confirmation; canPrompt -> native Install button;
//   ios -> Add-to-Home-Screen steps; otherwise the generic browser-menu hint.

const allOk = () => Promise.resolve({
  ok: true, status: 200, json: async () => ({ models: [], active: "" }), text: async () => "",
});

const els = (window) => ({
  btn: window.document.getElementById("install-app"),
  hint: window.document.getElementById("install-hint"),
  ios: window.document.getElementById("install-ios"),
});
const shown = (e) => e.style.display !== "none";

test("P2c: a fired install prompt reveals the native Install button only", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  window.applyInstallUI({ canPrompt: true });
  const { btn, hint, ios } = els(window);
  assert.ok(shown(btn), "Install button is shown");
  assert.ok(!shown(hint), "generic hint is hidden");
  assert.ok(!shown(ios), "iOS steps are hidden");
});

test("P2c: iOS shows Add-to-Home-Screen steps (no beforeinstallprompt on iOS)", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  window.applyInstallUI({ ios: true });
  const { btn, hint, ios } = els(window);
  assert.ok(shown(ios), "iOS steps are shown");
  assert.ok(!shown(btn), "no native button on iOS");
  assert.ok(!shown(hint), "generic hint is hidden");
  assert.match(ios.textContent, /Add to Home Screen/i);
});

test("P2c: an already-installed (standalone) launch confirms it, hides install paths", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  window.applyInstallUI({ standalone: true });
  const { btn, hint, ios } = els(window);
  assert.ok(!shown(btn), "no install button when already installed");
  assert.ok(!shown(ios), "no iOS steps when already installed");
  assert.ok(shown(hint), "the installed confirmation is shown");
  assert.match(hint.textContent, /installed app/i);
});

test("P2c: default (no prompt yet, not iOS) falls back to the generic hint", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  window.applyInstallUI({});
  const { btn, hint, ios } = els(window);
  assert.ok(shown(hint), "generic hint is the fallback");
  assert.ok(!shown(btn), "no button until beforeinstallprompt fires");
  assert.ok(!shown(ios), "no iOS steps off iOS");
});

test("P2c: standalone wins even if a stale prompt flag is passed", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  window.applyInstallUI({ standalone: true, canPrompt: true, ios: true });
  const { btn, hint, ios } = els(window);
  assert.ok(!shown(btn) && !shown(ios), "installed state suppresses both install paths");
  assert.match(hint.textContent, /installed app/i);
});

test("PWA cert gate: HTTPS without a trusted cert points at the certificate step", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  window.applyInstallUI({ certNeeded: true });
  const { btn, hint, ios } = els(window);
  assert.ok(shown(hint), "the cert hint is shown");
  assert.ok(!shown(btn) && !shown(ios), "no install paths until the cert is trusted");
  assert.match(hint.textContent, /certificate/i, "steers the user to install the certificate");
});

test("PWA cert gate: a usable install prompt overrides certNeeded", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  window.applyInstallUI({ canPrompt: true, certNeeded: true });
  const { btn, hint } = els(window);
  assert.ok(shown(btn), "an install prompt wins over the cert hint");
  assert.ok(!shown(hint));
});

test("PWA cert gate: an installed app whose cert stopped being trusted is told so", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  window.applyInstallUI({ standalone: true, certUntrusted: true });
  const { btn, hint, ios } = els(window);
  assert.ok(shown(hint), "the cert-untrusted hint is shown");
  assert.ok(!shown(btn) && !shown(ios), "still no install paths once already installed");
  assert.match(hint.textContent, /certificate/i, "names the actual problem");
  assert.doesNotMatch(hint.textContent, /^Running as an installed app\.$/,
    "does not silently claim everything is fine");
});

test("PWA cert gate: an installed app with a trusted cert still gets the plain confirmation", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  window.applyInstallUI({ standalone: true, certUntrusted: false });
  const { hint } = els(window);
  assert.match(hint.textContent, /installed app/i);
});
