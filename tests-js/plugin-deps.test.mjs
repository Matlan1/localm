// SPDX-License-Identifier: AGPL-3.0-or-later
// The Plugins page surfaces a plugin that is missing its pip extras and offers a
// host-side "Install dependencies" button that POSTs the install and streams
// progress over SSE. A remote client (403) is told to install on the host.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function pluginsPayload() {
  return {
    auto_install_plugin_deps: true,
    plugins: [
      { name: "beta", label: "Beta", builtin: true, installed: true,
        enabled: true, active: true, protected: false, description: "needs pkgs",
        requires: [], missing_requires: [],
        requires_extras: ["fakeextra"], missing_deps: ["x>=1"] },
      { name: "gamma", label: "Gamma", builtin: true, installed: true,
        enabled: true, active: true, protected: false, description: "ready",
        requires: [], missing_requires: [], requires_extras: [], missing_deps: [] },
    ],
  };
}

function sseResp(frames) {
  const enc = new TextEncoder();
  let i = 0;
  return {
    ok: true, status: 200,
    body: { getReader: () => ({
      read: async () => (i < frames.length
        ? { done: false, value: enc.encode(frames[i++]) }
        : { done: true, value: undefined }),
    }) },
  };
}

function okEndFrames() {
  return [
    'data: {"type":"log","line":"resolving..."}\n\n',
    'data: {"type":"log","line":"installed x"}\n\n',
    'data: {"type":"end","ok":true,"installed":["x>=1"],"failed":[],"error":""}\n\n',
  ];
}

function makeFetch(calls, { local = true } = {}) {
  return async (url, opts = {}) => {
    if (url === "/api/plugins") {
      return { ok: true, status: 200, json: async () => pluginsPayload(), text: async () => "" };
    }
    // The SSE progress stream (GET).
    if (typeof url === "string" && url.includes("/install-deps/events")) {
      calls.push(url);
      return sseResp(okEndFrames());
    }
    // Any plugin action POST (install/enable/install-deps/...): record it.
    if (typeof url === "string" && url.startsWith("/api/plugins/")) {
      calls.push(url);
      if (url.includes("/install-deps")) {
        if (!local) {
          return { ok: false, status: 403, json: async () => ({ detail: "host only" }), text: async () => "" };
        }
        return { ok: true, status: 200, text: async () => "",
                 json: async () => ({ status: "running", name: "beta", missing: ["x>=1"], lines: [] }) };
      }
      return { ok: true, status: 200, text: async () => "",
               json: async () => ({ status: "ok" }) };
    }
    return { ok: true, status: 200, text: async () => "",
             json: async () => ({ models: [], active: "", conversations: [], plugins: [], commands: [] }) };
  };
}

async function renderCatalog(win) {
  runScript(win, "_catalogStaggerMs = 0; renderCatalogPlugins();");
  for (let i = 0; i < 20; i++) {
    await new Promise((r) => setTimeout(r, 0));
    const s = win.document.querySelector(".catalog-status");
    if (s && !/Loading/.test(s.textContent)) break;
  }
}

async function drain(n = 12) {
  for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0));
}

test("catalog flags missing pip extras and offers an Install dependencies button", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch([]) });
  await renderCatalog(win);
  const box = win.document.getElementById("catalog-table");
  assert.match(box.textContent, /needs Python packages: x>=1/, "names the missing package");
  assert.ok(box.querySelector('[data-depsfor="beta"]'), "beta has an Install dependencies button");
  assert.equal(box.querySelector('[data-depsfor="gamma"]'), null,
    "a plugin with no missing deps gets no button");
});

test("clicking Install dependencies POSTs the install and streams the events", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  await renderCatalog(win);
  const box = win.document.getElementById("catalog-table");
  box.querySelector('[data-depsfor="beta"]').click();
  await drain();
  assert.ok(calls.some((u) => u === "/api/plugins/beta/install-deps"), "started the install");
  assert.ok(calls.some((u) => u.includes("/install-deps/events")), "opened the progress stream");
});

test("a remote client (403) does not open the event stream", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls, { local: false }) });
  await renderCatalog(win);
  const box = win.document.getElementById("catalog-table");
  box.querySelector('[data-depsfor="beta"]').click();
  await drain();
  assert.ok(calls.some((u) => u === "/api/plugins/beta/install-deps"), "attempted the install");
  assert.equal(calls.some((u) => u.includes("/install-deps/events")), false,
    "no stream after a 403");
});

test("enabling a plugin auto-installs its deps when the setting is on", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  await renderCatalog(win);
  // Simulate the enable action completing, which should auto-kick the dep install.
  runScript(win, "pluginCatalogAction('enable', 'beta');");
  await drain(16);
  assert.ok(calls.some((u) => u === "/api/plugins/beta/enable"), "enabled the plugin");
  assert.ok(calls.some((u) => u === "/api/plugins/beta/install-deps"),
    "auto-started the dependency install");
});
