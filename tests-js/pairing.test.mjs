// SPDX-License-Identifier: AGPL-3.0-or-later
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

// The "Pair a phone" block in Settings fetches the server-rendered key QR and
// shows it, and hides when no key is configured and the endpoint 404s.

const tick = () => new Promise((r) => setTimeout(r, 50));

function fetchFor(pairingResp) {
  return (url) => {
    if (String(url).includes("/api/pairing/qr")) return Promise.resolve(pairingResp);
    return Promise.resolve({ ok: true, status: 200,
      json: async () => ({ models: [], active: "" }), text: async () => "" });
  };
}

test("P2a: pairing QR is shown when the server returns an SVG", async () => {
  const svg = '<svg id="qr-svg"><rect/></svg>';
  const { window } = loadAppWithPages({ fetchImpl: fetchFor(
    { ok: true, status: 200, text: async () => svg, json: async () => ({}) }) });
  await tick();
  await window.refreshPairingQR();
  assert.equal(window.document.getElementById("pairing").style.display, "block");
  assert.ok(window.document.querySelector("#pairing-qr svg"), "the SVG QR is injected");
});

test("P2a: pairing block stays hidden when there is no key (404)", async () => {
  const { window } = loadAppWithPages({ fetchImpl: fetchFor(
    { ok: false, status: 404, text: async () => "", json: async () => ({}) }) });
  await tick();
  await window.refreshPairingQR();
  assert.equal(window.document.getElementById("pairing").style.display, "none");
});
