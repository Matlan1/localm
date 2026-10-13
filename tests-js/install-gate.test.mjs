// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadApp, runScript } from "./harness.mjs";

// After auth, a phone that has not installed localm lands on a one-time install
// gate (Install on Android, Add-to-Home-Screen steps on iOS) and Continue enters
// the app. Desktop, installed and returning visits skip it.

// Lets the boot's /v1/config round trip land, so the session mode is
// confirmed and localStorage writes are no longer held back.
const drain = async (n = 12) => {
  for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0));
};

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
  const { install, ios } = gateEls(window);
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

test("show/dismiss: gate covers the app, Continue reveals it and remembers", async () => {
  const { window } = loadApp({ fetchImpl: allOk });
  await drain();
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

const asDevice = (w, { touchPoints = 0, coarse = false, ua, mobile } = {}) => {
  Object.defineProperty(w.navigator, "maxTouchPoints", { value: touchPoints, configurable: true });
  if (ua !== undefined) Object.defineProperty(w.navigator, "userAgent", { value: ua, configurable: true });
  if (mobile !== undefined) {
    Object.defineProperty(w.navigator, "userAgentData", { value: { mobile }, configurable: true });
  }
  w.matchMedia = (q) => ({
    matches: coarse ? q.includes("coarse") : (q.includes("fine") || q.includes("hover: hover")),
    addEventListener() {}, removeEventListener() {},
  });
};

const WIN_CHROME = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) " +
  "Chrome/155.0.0.0 Safari/537.36";

test("shouldShowInstallGate: false on a Windows PC with touch input and a mouse", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  asDevice(window, { touchPoints: 10, coarse: false, ua: WIN_CHROME, mobile: false });
  assert.equal(window.shouldShowInstallGate(), false);
});

test("shouldShowInstallGate: true when the browser reports a mobile device", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  asDevice(window, { touchPoints: 5, coarse: false, ua: WIN_CHROME, mobile: true });
  assert.equal(window.shouldShowInstallGate(), true);
});

test("shouldShowInstallGate: true for an Android phone user agent", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  asDevice(window, {
    touchPoints: 5, coarse: false,
    ua: "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) " +
      "Chrome/155.0.0.0 Mobile Safari/537.36",
  });
  assert.equal(window.shouldShowInstallGate(), true);
});

test("shouldShowInstallGate: true for iPadOS that reports itself as a Mac", () => {
  const { window } = loadApp({ fetchImpl: allOk });
  asDevice(window, {
    touchPoints: 5, coarse: false,
    ua: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) " +
      "Version/18.0 Safari/605.1.15",
  });
  Object.defineProperty(window.navigator, "platform", { value: "MacIntel", configurable: true });
  assert.equal(window.shouldShowInstallGate(), true);
});

test("the camera button is hidden on a mouse-pointer device", () => {
  const css = fs.readFileSync(new URL("../localm/plugins/gui/static/style.css", import.meta.url), "utf8");
  const rule = /\.composer \.chat-camera\s*\{\s*display:\s*none;\s*\}/;
  assert.match(css, rule, "the hide rule is as specific as .composer .attach");
  const show = css.match(/@media \(pointer: coarse\)\s*\{\s*\.composer \.chat-camera\s*\{\s*display:\s*flex;/);
  assert.ok(show, "a coarse primary pointer shows it");
});
