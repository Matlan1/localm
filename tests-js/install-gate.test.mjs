// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

// After auth, a phone that has not installed localm lands on a one-time install
// gate (Install on Android, Add-to-Home-Screen steps on iOS) and Continue enters
// the app. Desktop, installed and returning visits skip it.

const allOk = () => Promise.resolve({
  ok: true, status: 200, json: async () => ({ models: [], active: "" }), text: async () => "",
});

const gateEls = (w) => ({
  install: w.document.getElementById("install-gate-install"),
  ios: w.document.getElementById("install-gate-ios"),
  hint: w.document.getElementById("install-gate-hint"),
  gate: w.document.getElementById("install-gate"),
});
const shown = (e) => e.style.display !== "none";

test("install gate UI: a captured prompt reveals the Install button only", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  window.applyInstallGateUI({ canPrompt: true });
  const { install, ios, hint } = gateEls(window);
  assert.ok(shown(install), "Install button shown");
  assert.ok(!shown(ios), "iOS steps hidden");
  assert.ok(!shown(hint), "generic hint hidden");
});

test("install gate UI: iOS shows Add-to-Home-Screen steps (no install prompt)", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  window.applyInstallGateUI({ ios: true });
  const { install, ios, hint } = gateEls(window);
  assert.ok(shown(ios), "iOS steps shown");
  assert.ok(!shown(install), "no Install button on iOS");
  assert.match(ios.textContent, /Add to Home Screen/i);
});

test("install gate UI: default falls back to the generic hint", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  window.applyInstallGateUI({});
  const { install, ios, hint } = gateEls(window);
  assert.ok(shown(hint), "generic hint is the fallback");
  assert.ok(!shown(install) && !shown(ios));
});

test("shouldShowInstallGate: false on desktop (no touch, fine pointer)", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  assert.equal(window.shouldShowInstallGate(), false);
});

test("shouldShowInstallGate: true on a coarse-pointer device not yet onboarded", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  window.matchMedia = (q) => ({
    matches: q.includes("coarse"), addEventListener() {}, removeEventListener() {},
  });
  assert.equal(window.shouldShowInstallGate(), true);
});

test("shouldShowInstallGate: false once the onboarded flag is set (no nagging)", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  window.matchMedia = (q) => ({ matches: q.includes("coarse") });
  // The onboarded flag is honoured only once a localm.instanceId is cached.
  window.localStorage.setItem("localm.instanceId", "instance-a");
  window.localStorage.setItem("localm.onboarded", "1");
  assert.equal(window.shouldShowInstallGate(), false);
});

test("shouldShowInstallGate: AUD-INSTANCEID - an onboarded flag with no confirmed " +
     "instance id is NOT trusted (a fresh backend pairing still onboards)", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  window.matchMedia = (q) => ({ matches: q.includes("coarse") });
  // No localm.instanceId cached.
  window.localStorage.setItem("localm.onboarded", "1");
  assert.equal(window.shouldShowInstallGate(), true,
    "an unverified onboarded flag must not silently skip onboarding for a new pairing");
});

test("show/dismiss: gate covers the app, Continue reveals it and remembers", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  const app = window.document.getElementById("app");
  window.showInstallGate();
  assert.equal(app.style.display, "none", "app hidden behind the landing");
  assert.ok(shown(gateEls(window).gate), "gate shown");
  window.dismissInstallGate();
  assert.equal(app.style.display, "", "app revealed on Continue");
  assert.equal(gateEls(window).gate.style.display, "none", "gate hidden");
  assert.equal(window.localStorage.getItem("localm.onboarded"), "1");
});

test("privacy mode: Continue does NOT persist the onboarded flag (no traces)", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  runScript(window, "chat.privacy = true;");
  window.localStorage.removeItem("localm.onboarded");
  window.showInstallGate();
  window.dismissInstallGate();
  assert.equal(window.localStorage.getItem("localm.onboarded"), null,
    "no localStorage trace written in privacy mode");
});
