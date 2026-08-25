// SPDX-License-Identifier: AGPL-3.0-or-later
// Exercises updateKeyGateCertStep() across the three __swFailed states and loopback HTTP.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const gate401 = async () => ({ ok: false, status: 401, json: async () => ({}), text: async () => "" });
const certShown = (window) =>
  window.document.getElementById("key-gate-cert").style.display !== "none";

test("HTTPS + UNKNOWN trust (mobile / clicked-through cert) -> cert download offered", () => {
  const { window } = loadApp({ url: "https://localhost:8642/", fetchImpl: gate401 });
  // jsdom has no service worker, so __swFailed stays undefined.
  window.updateKeyGateCertStep();
  assert.ok(certShown(window), "unknown trust over HTTPS must still offer the certificate");
  assert.match(window.document.getElementById("key-gate-cert-link").href,
    /^https:\/\/[^/]+\/localm-ca\.crt$/, "download pinned to an absolute https URL");
});

test("HTTPS + CONFIRMED trusted (service worker registered) -> no cert nag", () => {
  const { window } = loadApp({ url: "https://localhost:8642/", fetchImpl: gate401 });
  runScript(window, `window.__swFailed = false;`);
  window.updateKeyGateCertStep();
  assert.equal(certShown(window), false);
});

test("HTTPS + service worker failed (cert untrusted) -> cert download offered", () => {
  const { window } = loadApp({ url: "https://localhost:8642/", fetchImpl: gate401 });
  runScript(window, `window.__swFailed = true;`);
  window.updateKeyGateCertStep();
  assert.ok(certShown(window));
});

test("loopback HTTP needs no certificate -> step stays hidden", () => {
  const { window } = loadApp({ url: "http://127.0.0.1:8642/", fetchImpl: gate401 });
  window.updateKeyGateCertStep();
  assert.equal(certShown(window), false);
});
