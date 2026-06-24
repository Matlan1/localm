// SPDX-License-Identifier: AGPL-3.0-or-later
// R25: a real first-load progress indicator. The cold-start wait (the first
// /api/models can block for seconds while the model loads) used to show a blank
// shell with no feedback; the only overlay was the "server is down" reconnect
// one. Now a distinct startup overlay shows immediately at boot and hides once
// the model list resolves.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

async function drain(times = 10) {
  for (let i = 0; i < times; i++) await new Promise((r) => setTimeout(r, 0));
}

test("R25: the startup overlay shows at boot and hides once the model list loads", async () => {
  // loadApp runs app.js; the boot IIFE shows the overlay synchronously before its
  // first await, so it is already visible when loadApp returns.
  const { window } = loadApp();
  const ov = window.document.getElementById("startup-overlay");
  assert.ok(ov, "the startup overlay is created at boot");
  assert.equal(ov.style.display, "flex", "it is visible during the first-load wait");
  await drain();
  assert.equal(ov.style.display, "none", "it hides once the model list resolves");
});

test("R25: show/hide helpers toggle the overlay", () => {
  const { window } = loadApp();
  window.hideStartupOverlay();
  const ov = window.document.getElementById("startup-overlay");
  assert.equal(ov.style.display, "none");
  window.showStartupOverlay();
  assert.equal(ov.style.display, "flex");
});

test("R25: a failed boot probe hides the overlay (no stuck spinner)", async () => {
  // server unreachable -> bootAuthProbe returns false; the overlay must come down
  // so the reconnect overlay / key gate is not stacked under it.
  const { window } = loadApp({ fetchImpl: async () => { throw new Error("offline"); } });
  await drain();
  const ov = window.document.getElementById("startup-overlay");
  assert.ok(ov, "the overlay was created at boot");
  assert.equal(ov.style.display, "none", "it is hidden after an unreachable boot");
});
