// SPDX-License-Identifier: AGPL-3.0-or-later
// The sidebar status pill used to be shown in every state, including "ok" -
// where its text is just the active model's name (or "no model"), which the
// MODEL dropdown directly above already displays, placeholder included. That
// was two controls saying one thing, so "ok" now renders as the select alone.
//
// THE POINT OF THIS FILE IS THE OTHER HALF: "busy" and "err" must SURVIVE.
// Folding them into the same hide would delete the only surface localm has for
// "server unreachable", "load failed", "unload failed", "models unavailable"
// and "page out of date" - a rule 5 violation dressed as a decluttering.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadApp } from "./harness.mjs";

const box = (win) => win.document.getElementById("model-status");
const pill = (win) => win.document.getElementById("status-text");

// loadApp() boots the real app, which kicks off refreshModels(). Letting that
// settle BEFORE asserting is not cosmetic: a test that ends with it in flight
// touches an already-closed window from the root after-hook and surfaces as an
// unhandledRejection rather than as a clean failure.
const drain = async () => { for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 0)); };

async function boot(fetchImpl = async () => ({ ok: true, json: async () => ({}) })) {
  const { window: win } = loadApp({ fetchImpl });
  await drain();
  return win;
}

test("status pill: 'ok' is hidden (the dropdown already says it)", async () => {
  const win = await boot();
  win.setStatus("ok", "some-model");
  assert.equal(box(win).hidden, true, "an 'ok' pill duplicates the dropdown and is hidden");
  // Text and class are still written, so a hidden pill holds the last state.
  assert.equal(pill(win).textContent, "some-model");
  assert.equal(pill(win).className, "job-state st-ok");
});

test("status pill: 'no model' is hidden too (placeholder covers it)", async () => {
  const win = await boot();
  win.setStatus("ok", "no model");
  assert.equal(box(win).hidden, true,
    "the dropdown's disabled 'No model loaded' placeholder already states this");
});

test("status pill: 'busy' STAYS VISIBLE (the dropdown cannot express it)", async () => {
  const win = await boot();
  win.setStatus("busy", "loading some-model…");
  assert.equal(box(win).hidden, false, "a transient load/unload must remain visible");
  assert.equal(pill(win).textContent, "loading some-model…");
});

test("status pill: every 'err' STAYS VISIBLE (rule 5)", async () => {
  const win = await boot();
  for (const msg of ["server unreachable", "load failed", "unload failed",
                     "models unavailable (HTTP 500)", "page out of date - reload to reconnect"]) {
    win.setStatus("err", msg);
    assert.equal(box(win).hidden, false, `error must never be hidden: ${msg}`);
    assert.equal(pill(win).textContent, msg);
  }
});

test("status pill: a real unreachable server surfaces through refreshModels", async () => {
  // Driven through the actual caller rather than setStatus directly, so the
  // wiring is exercised, not just the setter.
  const win = await boot(async () => { throw new Error("boom"); });
  win.setStatus("ok", "some-model");
  assert.equal(box(win).hidden, true, "precondition: hidden while ok");
  await win.refreshModels();
  assert.equal(box(win).hidden, false, "the server going away must un-hide the pill");
  assert.equal(pill(win).textContent, "server unreachable");
});

test("status pill: recovering from an error re-hides the pill", async () => {
  const win = await boot();
  win.setStatus("err", "load failed");
  assert.equal(box(win).hidden, false);
  win.setStatus("ok", "some-model");
  assert.equal(box(win).hidden, true, "a resolved error must not leave a stale visible pill");
});

// Bound to the REAL shipped stylesheet, not a fixture. `#model-status` sets
// `display: flex`, which BEATS the user-agent `[hidden] { display: none }` - so
// without an explicit override the `hidden` toggles above are a silent no-op in
// the browser while every jsdom assertion here still passes. jsdom does not
// apply the cascade, so this is the only place that gap can be caught.
test("status pill: style.css overrides display:flex for [hidden]", () => {
  const css = readFileSync(
    fileURLToPath(new URL("../localm/plugins/gui/static/style.css", import.meta.url)), "utf8");
  assert.match(css, /#model-status\[hidden\]\s*\{[^}]*display:\s*none/,
    "#model-status[hidden] must force display:none or hiding never takes effect");
});
