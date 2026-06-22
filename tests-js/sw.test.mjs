// SPDX-License-Identifier: AGPL-3.0-or-later
// Tests for the PWA service worker (localm/plugins/gui/static/sw.js).
//
// sw.js is a classic service-worker script (uses the `self`, `caches`, `fetch`
// globals), not an ES module. We run it in a sandboxed VM context with mocked
// globals, capture the event handlers it registers, then drive its fetch handler
// with fake events to assert the routing decisions - in particular that the CA
// cert is never served from the SW (the J2 "Install certificate downloads .html"
// bug).

import { test } from "node:test";
import assert from "node:assert/strict";
import vm from "node:vm";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const SW_SRC = readFileSync(
  join(HERE, "..", "localm", "plugins", "gui", "static", "sw.js"), "utf8");

// Run sw.js in a sandbox with mocked SW globals and return the handlers it
// registers via self.addEventListener(...). Only the top-level const defs +
// addEventListener calls run here (install/activate/fetch bodies run only when
// dispatched), so the cache/fetch mocks are never actually exercised at load.
function loadSW() {
  const handlers = {};
  const self = {
    location: { origin: "https://localhost:8642" },
    addEventListener: (type, fn) => { handlers[type] = fn; },
    skipWaiting: () => {},
    clients: { claim: () => {} },
  };
  const caches = {
    open: async () => ({ put: () => {}, add: () => {}, addAll: () => {} }),
    match: async () => undefined,
    keys: async () => [],
    delete: async () => {},
  };
  const sandbox = {
    self, caches, URL, Promise, console,
    fetch: async () => ({ ok: true, clone: () => ({}) }),
  };
  vm.runInNewContext(SW_SRC, sandbox, { filename: "sw.js" });
  return handlers;
}

// Drive the fetch handler with a fake GET event for `path` and report whether
// the SW took over the response (called respondWith) or passed it through to the
// network (returned without responding).
function swHandles(handlers, path, mode = "no-cors") {
  let handled = false;
  handlers.fetch({
    request: { method: "GET", url: "https://localhost:8642" + path, mode },
    respondWith: () => { handled = true; },
  });
  return handled;
}

test("the service worker never intercepts /localm-ca.crt (J2)", () => {
  const h = loadSW();
  assert.equal(swHandles(h, "/localm-ca.crt"), false,
    "the CA cert must pass through to the network, not be SW-handled");
  assert.equal(swHandles(h, "/localm-ca.crt", "navigate"), false,
    "even a navigate-mode cert request must not fall back to cached index.html");
});

test("the service worker still bypasses API traffic and handles shell assets", () => {
  const h = loadSW();
  assert.equal(swHandles(h, "/api/models"), false, "/api is always live (bypassed)");
  assert.equal(swHandles(h, "/v1/models"), false, "/v1 is always live (bypassed)");
  assert.equal(swHandles(h, "/plugins/jobs/jobs.js"), false, "/plugins is live (bypassed)");
  assert.equal(swHandles(h, "/app.js"), true, "shell assets are SW-handled (cache-first)");
});
