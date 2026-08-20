// SPDX-License-Identifier: AGPL-3.0-or-later
// The Settings > Updates "App launcher" sub-block (S8, PARITY-AUDIT-CLI-GUI-
// 2026-08-19 #16): the GUI form of `localm make-launcher --force`. Unlike Roll
// back it has no read-only probe (rebuilding is idempotent and cheap to just
// try) and no danger confirm (it only replaces a cosmetic launcher copy, not
// running code or user data) - a single blocking POST that always reports the
// server's own structured result, success or not.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(rebuildPost, calls) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = (opts.method || "GET").toUpperCase();
    if (u.startsWith("/api/app/rebuild-launcher")) {
      calls.push({ url: u, method });
      const body = typeof rebuildPost === "function" ? rebuildPost() : rebuildPost;
      const status = (body && body.__status) || 200;
      return { ok: status < 400, status, text: async () => "", json: async () => body };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function flush() {
  for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0));
}

test("App launcher: the block is always shown, unlike the probe-gated Roll back block", async () => {
  const { window } = loadAppWithPages({ fetchImpl: makeFetch({ ok: true }, []) });
  const doc = window.document;
  assert.equal(doc.getElementById("app-launcher-block").hidden, false,
    "local disk state, not a proxy capability - same rule as the runtime block");
});

test("Rebuild launcher: a click POSTs with force=true and reports the path", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch(
      { ok: true, path: "C:/fake/LocaLM.exe", desktop_file: null,
        icon_stamped: true, notes: ["built LocaLM.exe"] }, calls) });
  const doc = window.document;

  doc.getElementById("rebuild-launcher").click();
  await flush();

  assert.equal(calls.length, 1, "exactly one POST");
  assert.ok(calls[0].url.includes("force=true"),
    "a clicked Rebuild must mean a rebuild, not the idempotent no-op force=false "
    + "would return on a launcher already on disk: " + calls[0].url);
  const text = doc.getElementById("rebuild-launcher-status").textContent;
  assert.match(text, /Launcher rebuilt/);
  assert.match(text, /LocaLM\.exe/, "says WHERE it was built: " + text);
  assert.equal(doc.getElementById("rebuild-launcher").disabled, false,
    "unlike Roll back this does not restart the server - the button stays usable");
});

test("Rebuild launcher: a structured failure (never raises) is shown, not swallowed", async () => {
  const calls = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch(
      { ok: false, path: null, desktop_file: null, icon_stamped: false,
        notes: ["could not locate the base interpreter to copy"] }, calls) });
  const doc = window.document;

  doc.getElementById("rebuild-launcher").click();
  await flush();

  const text = doc.getElementById("rebuild-launcher-status").textContent;
  assert.match(text, /Could not rebuild/);
  assert.match(text, /could not locate the base interpreter/,
    "make_launcher()'s own notes must reach the user, not a generic failure: " + text);
});

test("Rebuild launcher: a 409 (concurrent rebuild) is reported and leaves the button usable", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch(
      { __status: 409, detail: "A launcher rebuild is already in progress." }, []) });
  const doc = window.document;

  doc.getElementById("rebuild-launcher").click();
  await flush();

  const text = doc.getElementById("rebuild-launcher-status").textContent;
  assert.match(text, /already in progress/);
  assert.equal(doc.getElementById("rebuild-launcher").disabled, false);
});
