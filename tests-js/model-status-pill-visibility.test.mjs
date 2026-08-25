// SPDX-License-Identifier: AGPL-3.0-or-later
// The sidebar status pill is hidden in the "ok" state, where its text is the
// active model's name (or "no model") that the MODEL dropdown above already
// displays. "busy" and "err" stay visible.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadApp } from "./harness.mjs";

const box = (win) => win.document.getElementById("model-status");
const pill = (win) => win.document.getElementById("status-text");

// loadApp() boots the real app, which kicks off refreshModels(). Let that
// settle before asserting: a test that ends with it in flight touches an
// already-closed window.
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

// Reads the shipped stylesheet. `#model-status` sets `display: flex`, which
// beats the user-agent `[hidden] { display: none }`, so hiding needs an
// explicit override. jsdom does not apply the cascade.
test("status pill: style.css overrides display:flex for [hidden]", () => {
  const css = readFileSync(
    fileURLToPath(new URL("../localm/plugins/gui/static/style.css", import.meta.url)), "utf8");
  assert.match(css, /#model-status\[hidden\]\s*\{[^}]*display:\s*none/,
    "#model-status[hidden] must force display:none or hiding never takes effect");
});
