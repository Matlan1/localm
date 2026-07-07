// SPDX-License-Identifier: AGPL-3.0-or-later
// AUD-INSTANCEID: browser localStorage is scoped by ORIGIN (protocol+host+port)
// only, never by which backend DATA DIRECTORY is actually running behind it.
// localm's server binds to a fixed default port that only changes when it is
// already busy, so a fresh install opened after a prior instance closed
// typically reuses the SAME origin - and therefore the same localStorage
// bucket - as a totally unrelated data directory. Confirmed root cause: a
// fresh install showed a PRIOR instance's private conversation history, and in
// non-privacy mode would even PERMANENTLY UPLOAD it into the new install's own
// data directory (initServerConversations' "!remote" branch).
//
// The fix: /v1/config now reports a stable per-data-directory instance_id; the
// GUI compares it against the id it last confirmed for this origin
// (localStorage["localm.instanceId"]) and discards every instance-scoped
// cached key on any mismatch - or when no id has ever been confirmed yet
// (a brand-new pairing, exactly this scenario) - before rendering, merging, or
// uploading any of it.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, runScript } from "./harness.mjs";

const drain = async (n = 12) => {
  for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0));
};

const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, timeout = 800) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await settle(15); }
  return false;
}

// A conversation that must never leak across instances: real title + message
// content, exactly the payload the bug report described.
const FOREIGN_CONV = {
  id: "foreign-1", title: "Someone else's private chat", updated_at: 999,
  pinned: false, folder: null, branches: [],
  messages: [{ role: "user", content: "my secret plan" },
             { role: "assistant", content: "here is the reply" }],
};

function makeFetch({ instanceId, putCalls, indexConversations = [] } = {}) {
  return async (url, opts) => {
    const u = String(url);
    if (u === "/v1/config") {
      return { ok: true, status: 200, text: async () => "", json: async () => (
        { effective_mode: "log", n_ctx_max: 16384, instance_id: instanceId }) };
    }
    if (u.startsWith("/api/conversations?")) {
      return { ok: true, status: 200, text: async () => "", json: async () => (
        { enabled: true, total: indexConversations.length,
          conversations: indexConversations }) };
    }
    if (opts && opts.method === "PUT" && u.startsWith("/api/conversations/")) {
      if (putCalls) putCalls.push(u);
      return { ok: true, status: 200, text: async () => "", json: async () => ({}) };
    }
    return { ok: true, status: 200, text: async () => "", json: async () => (
      { models: [], active: "", conversations: [], plugins: [] }) };
  };
}

test("AUD-INSTANCEID: a brand-new pairing (no cached instance id) never renders " +
     "or uploads a foreign cached conversation", async () => {
  const putCalls = [];
  const { window } = loadApp({
    fetchImpl: makeFetch({ instanceId: "backend-new", putCalls }),
    // Simulate a PRIOR, unrelated backend having populated this browser origin's
    // cache before this fixed client code ever ran here - no localm.instanceId
    // cached yet, the exact reported scenario.
    seedLocalStorage: { "localm.conversations": JSON.stringify([FOREIGN_CONV]) },
  });
  runScript(window, "window.chatState = chat;");
  await drain();

  assert.equal(window.chatState.conversations.length, 0,
    "the foreign conversation must never enter the in-memory list");
  const listText = window.document.getElementById("conv-list").textContent;
  assert.ok(!listText.includes("Someone else's private chat"),
    "the foreign conversation title must never be painted to the sidebar");
  assert.equal(putCalls.length, 0,
    "the foreign conversation must never be uploaded to the new backend's store");
  // The wipe removes the key outright; a later saveConversations() (from the
  // now-corrected, empty chat.conversations) may re-write it as "[]" - either
  // way the foreign content must be gone from disk.
  const persisted = JSON.parse(window.localStorage.getItem("localm.conversations") || "[]");
  assert.deepEqual(persisted, [],
    "the stale cache is wiped on disk too, not merely ignored in memory");
  assert.equal(window.localStorage.getItem("localm.instanceId"), "backend-new",
    "the confirmed backend id is now cached for the next restart");
});

test("AUD-INSTANCEID: a mismatched instance id (a DIFFERENT backend now behind " +
     "this origin) discards the old cache and never uploads it", async () => {
  const putCalls = [];
  const { window } = loadApp({
    fetchImpl: makeFetch({ instanceId: "backend-b", putCalls }),
    seedLocalStorage: {
      "localm.instanceId": "backend-a",   // a PREVIOUSLY confirmed, different backend
      "localm.conversations": JSON.stringify([FOREIGN_CONV]),
    },
  });
  runScript(window, "window.chatState = chat;");
  await drain();

  assert.equal(window.chatState.conversations.length, 0,
    "the other backend's conversation is discarded once the mismatch is confirmed");
  assert.equal(putCalls.length, 0, "never re-uploaded to the new backend");
  assert.equal(window.localStorage.getItem("localm.instanceId"), "backend-b");
  const persisted = JSON.parse(window.localStorage.getItem("localm.conversations") || "[]");
  assert.deepEqual(persisted, []);
});

test("AUD-INSTANCEID: a CONFIRMED same-instance restart still renders and " +
     "syncs a not-yet-uploaded local-only chat (legitimate offline-sync case)", async () => {
  const putCalls = [];
  const { window } = loadApp({
    fetchImpl: makeFetch({ instanceId: "backend-same", putCalls }),
    seedLocalStorage: {
      "localm.instanceId": "backend-same",   // already confirmed for THIS backend
      "localm.conversations": JSON.stringify([{
        id: "local-only-1", title: "Not yet synced", updated_at: 1, pinned: false,
        folder: null, branches: [], messages: [{ role: "user", content: "hi" }],
      }]),
    },
  });
  runScript(window, "window.chatState = chat;");
  await drain();

  assert.equal(window.chatState.conversations.length, 1,
    "a legitimate offline-composed chat is kept for the SAME confirmed instance");
  const listText = window.document.getElementById("conv-list").textContent;
  assert.ok(listText.includes("Not yet synced"));

  // pushConversation debounces 600ms - ride it out with margin.
  await new Promise((r) => setTimeout(r, 900));
  assert.ok(putCalls.some((u) => u.includes("local-only-1")),
    "a confirmed same-instance restart still syncs the not-yet-uploaded chat " +
    "back to ITS OWN server (the legitimate use case must survive the fix)");
});

test("AUD-INSTANCEID: privacy mode clears the in-memory list and repaints, " +
     "not just the on-disk cache", async () => {
  const { window } = loadApp({
    fetchImpl: makeFetch({ instanceId: "backend-x" }),
    seedLocalStorage: {
      "localm.instanceId": "backend-x",
      "localm.conversations": JSON.stringify([FOREIGN_CONV]),
    },
  });
  // Override /v1/config to report privacy mode for THIS test.
  runScript(window, `
    window.fetch = async (url) => {
      const u = String(url);
      if (u === "/v1/config") {
        return { ok: true, status: 200, text: async () => "",
          json: async () => ({ effective_mode: "privacy", n_ctx_max: 16384,
                                instance_id: "backend-x" }) };
      }
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
    };
    window.chatState = chat;
    refreshCtxLimit();
  `);
  await drain();

  assert.equal(window.chatState.privacy, true);
  assert.equal(window.chatState.conversations.length, 0,
    "the in-memory list must be cleared too, not just the localStorage key");
  const listText = window.document.getElementById("conv-list").textContent;
  assert.ok(!listText.includes("Someone else's private chat"),
    "the sidebar must be repainted, not left showing the stale list");
  assert.equal(window.localStorage.getItem("localm.conversations"), null);
});

test("AUD-INSTANCEID: a mismatch also clears the Coder tab's stale " +
     "\"Project directory\" input, not just localStorage", async () => {
  // init.js seeds $("setup-cwd").value from localm.coderCwd synchronously at
  // boot (before any network round trip can resolve) whenever ANY instance id
  // was previously cached - so this input is a second place a foreign
  // install's data (here, a host filesystem path) can linger even after
  // reconcileInstanceId wipes the underlying localStorage key.
  const { window } = loadApp({
    fetchImpl: makeFetch({ instanceId: "backend-b" }),
    seedLocalStorage: {
      "localm.instanceId": "backend-a",
      "localm.coderCwd": "D:\\example\\other-install\\project",
    },
  });
  await drain();
  assert.equal(window.document.getElementById("setup-cwd").value, "",
    "the foreign coderCwd path must not remain in the Coder tab's input " +
    "after the instance-id mismatch is confirmed");
});

test("reconcileInstanceId: a matching id is a no-op (nothing wiped)", () => {
  const { window } = loadApp();
  window.localStorage.setItem("localm.instanceId", "same-id");
  window.localStorage.setItem("localm.conversations", JSON.stringify([FOREIGN_CONV]));
  const trusted = window.reconcileInstanceId("same-id");
  assert.equal(trusted, true);
  assert.ok(window.localStorage.getItem("localm.conversations"),
    "cache is left untouched on a confirmed match");
});

test("reconcileInstanceId: a missing server id is a no-op (older server, no " +
     "information to act on)", () => {
  const { window } = loadApp();
  window.localStorage.setItem("localm.conversations", JSON.stringify([FOREIGN_CONV]));
  const trusted = window.reconcileInstanceId(undefined);
  assert.equal(trusted, true);
  assert.ok(window.localStorage.getItem("localm.conversations"),
    "nothing is wiped without a server id to compare against");
});

test("reconcileInstanceId: a mismatch wipes every instance-scoped key and " +
     "re-caches the new id", () => {
  const { window } = loadApp();
  window.localStorage.setItem("localm.instanceId", "old");
  runScript(window, "window.__keys = INSTANCE_SCOPED_KEYS;");
  for (const key of window.__keys) window.localStorage.setItem(key, "x");
  const trusted = window.reconcileInstanceId("new");
  assert.equal(trusted, false);
  for (const key of window.__keys) {
    assert.equal(window.localStorage.getItem(key), null, `${key} must be wiped on a mismatch`);
  }
  assert.equal(window.localStorage.getItem("localm.instanceId"), "new");
});

test("instanceCacheTrusted: false with no cached id, true once one is cached", () => {
  const { window } = loadApp();
  assert.equal(window.instanceCacheTrusted(), false);
  window.localStorage.setItem("localm.instanceId", "abc");
  assert.equal(window.instanceCacheTrusted(), true);
});

test("AUD-INSTANCEID: the startup overlay stays up until refreshCtxLimit's " +
     "instance-id check resolves, even when refreshModels resolves FIRST " +
     "(closes the flash-of-stale-content race, not just the no-cache case)", async () => {
  // This origin already confirmed pairing with a DIFFERENT prior backend
  // ("backend-a"), so instanceCacheTrusted() is true at boot and the raw
  // cached (foreign) conversation list is painted into the DOM synchronously,
  // exactly like a real fresh install reusing a previously-paired browser
  // origin/port. The only thing standing between that paint and the user's
  // eyes is the opaque startup overlay - it must not come down before
  // refreshCtxLimit has had a chance to detect the mismatch and correct the DOM.
  let resolveConfig;
  const configGate = new Promise((r) => (resolveConfig = r));
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u === "/v1/config") {
      await configGate;   // deliberately held back to win the race against models
      return { ok: true, status: 200, text: async () => "", json: async () => (
        { effective_mode: "log", n_ctx_max: 16384, instance_id: "backend-new" }) };
    }
    if (u.startsWith("/api/conversations?")) {
      return { ok: true, status: 200, text: async () => "", json: async () => (
        { enabled: true, total: 0, conversations: [] }) };
    }
    // /api/models, /api/session, and everything else resolve immediately -
    // this is what makes refreshModels() the FIRST of the two chains to settle.
    return { ok: true, status: 200, text: async () => "", json: async () => (
      { models: [], active: "", conversations: [], plugins: [] }) };
  };
  const { window } = loadApp({
    fetchImpl,
    seedLocalStorage: {
      "localm.instanceId": "backend-a",
      "localm.conversations": JSON.stringify([FOREIGN_CONV]),
    },
  });

  const ov = window.document.getElementById("startup-overlay");
  // Let bootAuthProbe, refreshCsrf, and refreshModels (all held-open-free) run
  // to completion while /v1/config is still gated.
  await waitFor(() => window.document.getElementById("conv-list").textContent
    .includes("Someone else's private chat"));
  assert.equal(ov.style.display, "flex",
    "the overlay must still be covering the shell: refreshCtxLimit has not " +
    "resolved yet, so the mismatch has not been detected/corrected");

  resolveConfig();   // let /v1/config (and the instance-id reconciliation) proceed
  assert.ok(await waitFor(() => ov.style.display === "none"),
    "the overlay hides once refreshCtxLimit (and initServerConversations) settle too");
  assert.ok(!window.document.getElementById("conv-list").textContent
    .includes("Someone else's private chat"),
    "by the time the overlay comes down, the foreign conversation is gone from the DOM");
});
