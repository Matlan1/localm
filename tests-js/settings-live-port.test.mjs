// SPDX-License-Identifier: AGPL-3.0-or-later
// The Settings "Server port" field used to render ONLY the persisted config
// default (GET /v1/config/schema's field.default), even when the server was
// actually started with an explicit -p override or auto-bumped onto a
// different free port - neither ever gets written back to disk, so the field
// silently showed the wrong port. refreshSettingsPage() now also fetches
// GET /v1/config for the live instance_port and buildSettingControl() shows
// it alongside the persisted value when the two differ.
// Same harness/fetch-mock style as settings-default-placeholder.test.mjs.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

// No shipped_default: this is the "legacy schema rows with no shipped_default
// key" shape from settings-default-placeholder.test.mjs, which always renders
// SOLID (old behavior) - keeps these tests about the live-port note alone,
// not entangled with the separate shipped-default-placeholder feature.
const SCHEMA_WITH_PORT = {
  fields: [
    { key: "port", widget: "number", label: "Server port",
      help: "Port the API/GUI server binds to (default 8642); auto-bumps to " +
            "the next free port if busy.",
      group: "Server", owner: "core", min: 1, max: 65535, step: 1,
      default: 8642 },
  ],
};

// No "port" field at all: every OTHER settings test's fixture shape.
const SCHEMA_NO_PORT = {
  fields: [
    { key: "n_ctx", widget: "number", label: "Context window", help: "",
      group: "Engine", owner: "core", min: 512, step: 512,
      default: 4096 },
  ],
};

function makeFetch({ schema = SCHEMA_WITH_PORT, instancePort, configFails = false,
                      configThrows = false } = {}) {
  return async (url, opts = {}) => {
    if (url === "/v1/config/schema") {
      // A fresh clone every call: the code under test mutates a field object
      // in place (portField.live_port = ...) exactly as a real page render
      // would on its own one-shot schema fetch - sharing the literal object
      // across tests would leak that mutation into whichever test runs next.
      return { ok: true, status: 200, json: async () => JSON.parse(JSON.stringify(schema)),
               text: async () => "" };
    }
    if (url === "/v1/config" && (opts.method || "GET") === "GET") {
      if (configThrows) throw new TypeError("network error");
      if (configFails) {
        return { ok: false, status: 500, json: async () => ({}), text: async () => "err" };
      }
      return { ok: true, status: 200, json: async () => ({ instance_port: instancePort }),
                text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

async function render(win) {
  runScript(win, "refreshSettingsPage();");
  await new Promise((r) => setTimeout(r, 0));
}

test("a live port that DIFFERS from the persisted default shows a note naming both", async () => {
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch({ instancePort: 1111 }),
  });
  await render(win);
  const wrap = win.document.querySelector('input[data-key="port"]').closest("div");
  const subs = [...wrap.querySelectorAll(".sub")].map((n) => n.textContent);
  const note = subs.find((t) => t.includes("Currently running"));
  assert.ok(note, "expected a live-port note when 1111 (live) != 8642 (persisted)");
  assert.match(note, /port 1111/);
  assert.match(note, /8642/, "the persisted value must still be named for comparison");
});

test("a live port that MATCHES the persisted default shows no extra note", async () => {
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch({ instancePort: 8642 }),
  });
  await render(win);
  const wrap = win.document.querySelector('input[data-key="port"]').closest("div");
  const subs = [...wrap.querySelectorAll(".sub")].map((n) => n.textContent);
  assert.ok(!subs.some((t) => t.includes("Currently running")),
    "no reason to call out a live port that already matches what is shown");
});

test("the persisted value still renders normally regardless of the live-port note", async () => {
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch({ instancePort: 1111 }),
  });
  await render(win);
  const el = win.document.querySelector('input[data-key="port"]');
  assert.equal(el.value, "8642", "the field itself still shows/edits the persisted setting");
});

// ---------------------------------------------------------------- //
//  Best-effort: a failed or absent live-port fetch degrades to      //
//  today's behavior (persisted value only), never breaks the page.  //
// ---------------------------------------------------------------- //

test("a non-ok GET /v1/config leaves the port field exactly as before (no note, no crash)", async () => {
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch({ configFails: true }),
  });
  await render(win);
  const el = win.document.querySelector('input[data-key="port"]');
  assert.equal(el.value, "8642");
  const wrap = el.closest("div");
  assert.ok(![...wrap.querySelectorAll(".sub")].some((n) => n.textContent.includes("Currently running")));
});

test("a thrown/network-failed GET /v1/config leaves the port field exactly as before", async () => {
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch({ configThrows: true }),
  });
  await render(win);
  const el = win.document.querySelector('input[data-key="port"]');
  assert.equal(el.value, "8642", "the schema fetch alone must still render the page");
});

test("instance_port missing from the response (older server) is treated as unknown, no note", async () => {
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch({ instancePort: undefined }),
  });
  await render(win);
  const wrap = win.document.querySelector('input[data-key="port"]').closest("div");
  assert.ok(![...wrap.querySelectorAll(".sub")].some((n) => n.textContent.includes("Currently running")));
});

// ---------------------------------------------------------------- //
//  Backward compatibility: every OTHER settings fixture in this repo //
//  has no "port" field - confirm that shape still renders cleanly    //
//  with no live-port note anywhere (nothing for it to attach to).    //
// ---------------------------------------------------------------- //

test("a schema with no port field renders normally, no live-port note anywhere", async () => {
  const { window: win } = loadAppWithPages({
    fetchImpl: makeFetch({ schema: SCHEMA_NO_PORT }),
  });
  await render(win);
  assert.ok(win.document.querySelector('input[data-key="n_ctx"]'), "the real field still renders");
  const allSubs = [...win.document.querySelectorAll(".sub")].map((n) => n.textContent);
  assert.ok(!allSubs.some((t) => t.includes("Currently running")));
});
