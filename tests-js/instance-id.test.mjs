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

// configStatus / configThrows exist because this fixture used to HARDCODE
// /v1/config to {ok:true, status:200}: the value space had no room for a FAILED
// round trip at all, so the "or a failed round trip" case named in chat.js's own
// comments was untestable by construction and the fail-open guard below went
// unmeasured. The two failures are kept separate deliberately - a non-ok answer
// and a rejected fetch reach refreshCtxLimit through different paths (the
// `if (r.ok)` branch versus the `catch`), so one of them passing proves nothing
// about the other.
function makeFetch({ instanceId, putCalls, indexConversations = [],
                    configStatus = 200, configThrows = false } = {}) {
  return async (url, opts) => {
    const u = String(url);
    if (u === "/v1/config") {
      if (configThrows) throw new TypeError("Failed to fetch");
      if (configStatus !== 200) {
        return { ok: false, status: configStatus, text: async () => "",
                 json: async () => ({}) };
      }
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

test("GUI-LIVE-WIPE: a SECOND refreshCtxLimit() call in privacy mode (the 30s " +
     "poll, init.js) must not wipe a conversation the poll itself did not start " +
     "- it should only ever wipe stale content the FIRST time privacy mode is " +
     "confirmed for this page load", async () => {
  const { window } = loadApp({
    fetchImpl: makeFetch({ instanceId: "backend-x" }),
  });
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
    "sanity check: the first call still wipes stale content as before");

  // Simulate the user's OWN live conversation - the first message just landed
  // (or is still streaming) in this same tab, no reload, no other instance
  // involved. Rendered explicitly here since real message-send flows through
  // runCompletion()'s own renderChat()/renderConvList(), not refreshCtxLimit().
  runScript(window, `
    chat.conversations.push({
      id: "live-1", title: "Say the word BANANA", updated_at: 1000,
      pinned: false, folder: null, branches: [],
      messages: [{ role: "user", content: "Say the word BANANA and nothing else." },
                 { role: "assistant", content: "Banana" }],
    });
    chat.activeId = "live-1";
    renderConvList();
  `);
  assert.equal(window.chatState.conversations.length, 1);

  // The 30s poll (init.js: setInterval(refreshCtxLimit, 30000)) fires again -
  // still privacy mode, nothing about the backend changed.
  runScript(window, "refreshCtxLimit();");
  await drain();

  assert.equal(window.chatState.conversations.length, 1,
    "a second privacy-mode confirmation must NOT wipe the tab's own live " +
    "conversation - only the first confirmation may reset stale content");
  assert.equal(window.chatState.activeId, "live-1");
  const listText = window.document.getElementById("conv-list").textContent;
  assert.ok(listText.includes("Say the word BANANA"),
    "the live conversation must still be rendered in the sidebar after the " +
    "second poll tick");
});

test("GUI-LIVE-WIPE: leaving privacy mode re-arms the wipe latch, so a LATER " +
     "return to privacy mode wipes newly-accumulated non-privacy leftovers " +
     "again (AUD-PRIV-2)", async () => {
  const { window } = loadApp({
    fetchImpl: makeFetch({ instanceId: "backend-x" }),
  });

  // 1) First privacy-mode confirmation: wipes (as usual) and arms the latch.
  runScript(window, `
    window.__mode = "privacy";
    window.fetch = async (url) => {
      const u = String(url);
      if (u === "/v1/config") {
        return { ok: true, status: 200, text: async () => "",
          json: async () => ({ effective_mode: window.__mode, n_ctx_max: 16384,
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
  assert.equal(window.chatState.conversations.length, 0);

  // 2) The server restarts OUT of privacy mode (log/full); the tab goes on to
  // accumulate a new conversation exactly as a real non-privacy session would.
  runScript(window, `window.__mode = "log"; refreshCtxLimit();`);
  await drain();
  assert.equal(window.chatState.privacy, false,
    "sanity check: the tab now sees non-privacy mode");

  runScript(window, `
    chat.conversations.push({
      id: "leftover-1", title: "Non-privacy leftover", updated_at: 1000,
      pinned: false, folder: null, branches: [],
      messages: [{ role: "user", content: "hi" }, { role: "assistant", content: "hello" }],
    });
    chat.activeId = "leftover-1";
    renderConvList();
  `);
  assert.equal(window.chatState.conversations.length, 1);

  // 3) The server restarts back INTO privacy mode - the exact AUD-PRIV-2
  // scenario the wipe exists to prevent. Without re-arming privacyWiped on the
  // way out of privacy mode (step 2), this second confirmation is silently
  // suppressed and the non-privacy leftover stays painted in the sidebar for
  // the rest of the tab's life.
  runScript(window, `window.__mode = "privacy"; refreshCtxLimit();`);
  await drain();

  assert.equal(window.chatState.privacy, true);
  assert.equal(window.chatState.conversations.length, 0,
    "the second privacy-mode confirmation must wipe the non-privacy leftover, " +
    "not leave it painted in the sidebar for the rest of the tab's life");
  const listText2 = window.document.getElementById("conv-list").textContent;
  assert.ok(!listText2.includes("Non-privacy leftover"),
    "the sidebar must be repainted clean on the second privacy confirmation too");
});

test("reconcileInstanceId: a matching id returns \"confirmed\" (nothing wiped)", () => {
  const { window } = loadApp();
  window.localStorage.setItem("localm.instanceId", "same-id");
  window.localStorage.setItem("localm.conversations", JSON.stringify([FOREIGN_CONV]));
  const state = window.reconcileInstanceId("same-id");
  assert.equal(state, "confirmed");
  assert.ok(window.localStorage.getItem("localm.conversations"),
    "cache is left untouched on a confirmed match");
});

test("reconcileInstanceId: a missing server id returns \"unknown\" (older " +
     "server, no information to act on)", () => {
  const { window } = loadApp();
  window.localStorage.setItem("localm.conversations", JSON.stringify([FOREIGN_CONV]));
  const state = window.reconcileInstanceId(undefined);
  assert.equal(state, "unknown");
  assert.ok(window.localStorage.getItem("localm.conversations"),
    "nothing is wiped without a server id to compare against");
});

test("reconcileInstanceId: a mismatch returns \"mismatched\", wipes every " +
     "instance-scoped key and re-caches the new id", () => {
  const { window } = loadApp();
  window.localStorage.setItem("localm.instanceId", "old");
  runScript(window, "window.__keys = INSTANCE_SCOPED_KEYS;");
  for (const key of window.__keys) window.localStorage.setItem(key, "x");
  const state = window.reconcileInstanceId("new");
  assert.equal(state, "mismatched");
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

// --------------------------------------------------------------------------- //
//  Residual 1 (COORDINATOR-DISPATCH-2026-08-11): a confirmed mismatch never   //
//  re-asserted the landing PAGE, only the conversation cache + cwd input.     //
// --------------------------------------------------------------------------- //

test("AUD-INSTANCEID residual 1: a confirmed mismatch corrects the landing PAGE " +
     "too, not just the cache - a savedView from a DIFFERENT prior backend was " +
     "already restored optimistically before the round trip could catch it", async () => {
  // Same held-back-/v1/config technique as the startup-overlay race test above:
  // in the REAL browser the synchronous setTimeout(0) savedView restore always
  // wins the race against a genuine network round trip, so the test must force
  // that same ordering rather than let the stub fetch resolve instantly (which
  // would let the mismatch-driven correction win first via pure microtasks and
  // never actually exercise the residual).
  let resolveConfig;
  const configGate = new Promise((r) => (resolveConfig = r));
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u === "/v1/config") {
      await configGate;
      return { ok: true, status: 200, text: async () => "", json: async () => (
        { effective_mode: "log", n_ctx_max: 16384, instance_id: "backend-b" }) };
    }
    if (u.startsWith("/api/conversations?")) {
      return { ok: true, status: 200, text: async () => "", json: async () => (
        { enabled: true, total: 0, conversations: [] }) };
    }
    return { ok: true, status: 200, text: async () => "", json: async () => (
      { models: [], active: "", conversations: [], plugins: [] }) };
  };
  const { window } = loadApp({
    fetchImpl,
    seedLocalStorage: {
      "localm.instanceId": "backend-a",   // a PREVIOUSLY confirmed, DIFFERENT backend
      "localm.activeView": "models",      // that backend's last-open page
    },
  });

  assert.ok(await waitFor(() =>
    window.document.getElementById("view-models").classList.contains("active")),
    "precondition: the foreign savedView is restored optimistically (init.js's " +
    "setTimeout(0) fast path), before the instance-id round trip can catch it");

  resolveConfig();   // let refreshCtxLimit's round trip proceed and detect the mismatch
  assert.ok(await waitFor(() =>
    window.document.getElementById("view-chat").classList.contains("active")),
    "the confirmed mismatch must correct the landing page back to chat once " +
    "detected, not leave the foreign install's page showing indefinitely");
  assert.equal(window.document.getElementById("view-models").classList.contains("active"),
    false, "the foreign view is no longer the active one");
});

// --------------------------------------------------------------------------- //
//  Residual 2 (COORDINATOR-DISPATCH-2026-08-11): "unknown" and "confirmed"    //
//  used to collapse into the same boolean, so a failed/old-server round trip  //
//  could authorise an upload meant only for a real confirmed match.           //
// --------------------------------------------------------------------------- //

test("AUD-INSTANCEID residual 2: an UNKNOWN instance state (old server, no " +
     "instance_id field) still renders an already-cached local-only conversation " +
     "but must NEVER upload it - only a CONFIRMED match may write to the backend", async () => {
  const putCalls = [];
  const { window } = loadApp({
    // instanceId left undefined: cfg.instance_id comes back missing/falsy,
    // exactly what an older server (or a failed round trip) looks like.
    fetchImpl: makeFetch({ instanceId: undefined, putCalls, indexConversations: [] }),
    seedLocalStorage: {
      // A previously-confirmed pairing with SOME backend, so the synchronous
      // boot-time fast path in init.js does not wipe the cache outright - the
      // server THIS load talks to just happens not to report instance_id at
      // all, so reconcileInstanceId can neither confirm nor refute the match.
      "localm.instanceId": "backend-old",
      "localm.conversations": JSON.stringify([{
        id: "local-only-2", title: "Not yet synced (old server)", updated_at: 1,
        pinned: false, folder: null, branches: [], messages: [{ role: "user", content: "hi" }],
      }]),
    },
  });
  runScript(window, "window.chatState = chat;");
  await drain();

  assert.equal(window.chatState.conversations.length, 1,
    "an unconfirmed (unknown) instance state still renders whatever is already " +
    "cached - it is not a confirmed mismatch, so nothing is wiped");
  const listText = window.document.getElementById("conv-list").textContent;
  assert.ok(listText.includes("Not yet synced (old server)"));

  await new Promise((r) => setTimeout(r, 900));   // ride out pushConversation's 600ms debounce
  assert.equal(putCalls.length, 0,
    "an UNCONFIRMED (unknown) instance state must never upload a local-only " +
    "conversation to the backend's own store - only a CONFIRMED match may");
});

// --------------------------------------------------------------------------- //
//  Residual 2, the FAILED-ROUND-TRIP half: the test above proves an old server //
//  (answers, no instance_id) cannot authorise an upload. A /v1/config that     //
//  never answers at all took a different path - it skipped the reconciliation  //
//  entirely and left BOTH flags at their permissive boot defaults, so the      //
//  guard read "confirmed" for a backend it had never heard from.               //
// --------------------------------------------------------------------------- //

// A conversation cached under a PREVIOUSLY confirmed pairing. Whether the
// backend now behind this origin is that same one is exactly what the failed
// round trip leaves unanswered.
const CACHED_LOCAL_ONLY = {
  id: "local-only-3", title: "Cached under an unverified pairing", updated_at: 1,
  pinned: false, folder: null, branches: [], messages: [{ role: "user", content: "hi" }],
};
const CACHED_SEED = {
  "localm.instanceId": "backend-a",
  "localm.conversations": JSON.stringify([CACHED_LOCAL_ONLY]),
};

test("AUD-INSTANCEID residual 2: a NON-OK /v1/config (HTTP 500) leaves the " +
     "instance UNCONFIRMED - the cached conversation still renders, but must " +
     "never be uploaded to a backend that was never identified", async () => {
  const putCalls = [];
  const { window } = loadApp({
    fetchImpl: makeFetch({ instanceId: "backend-a", putCalls, configStatus: 500 }),
    seedLocalStorage: { ...CACHED_SEED },
  });
  runScript(window, "window.chatState = chat;");
  await drain();
  // Ride out pushConversation's 600ms debounce BEFORE reading putCalls: without
  // this the assertion is vacuously true even on the unfixed code, because the
  // upload has not been attempted yet.
  await new Promise((r) => setTimeout(r, 900));

  // The loss first, the mechanism second (a status flag is a proxy; the write
  // into another install's data directory is the property).
  assert.equal(putCalls.length, 0,
    "a failed /v1/config round trip cannot authorise writing a cached " +
    "conversation into the store of a backend whose identity was never confirmed");
  assert.equal(window.chatState.instanceState, "unknown",
    "a round trip that FAILED is not a confirmed match - it is no information");
  assert.equal(window.chatState.instanceMatch, false,
    "instanceMatch is the upload gate and must be false without a real match");

  // ...and the other direction: this must not become a data-losing fail-CLOSED.
  assert.equal(window.chatState.conversations.length, 1,
    "nothing is wiped on a failed round trip - it is not a confirmed mismatch");
  assert.ok(window.document.getElementById("conv-list").textContent
    .includes("Cached under an unverified pairing"),
    "the user's own cached conversation still renders while the server is unreachable");
  assert.ok(window.localStorage.getItem("localm.conversations"),
    "and it is still on disk, so a later confirmed boot can still sync it");
});

test("AUD-INSTANCEID residual 2: a THROWN /v1/config fetch (server down, the " +
     "connection never completes) leaves the instance UNCONFIRMED too - the " +
     "reject path reaches the guard through catch, not through if (r.ok)", async () => {
  const putCalls = [];
  const { window } = loadApp({
    fetchImpl: makeFetch({ instanceId: "backend-a", putCalls, configThrows: true }),
    seedLocalStorage: { ...CACHED_SEED },
  });
  runScript(window, "window.chatState = chat;");
  await drain();
  await new Promise((r) => setTimeout(r, 900));   // pushConversation's debounce

  assert.equal(putCalls.length, 0,
    "a rejected /v1/config cannot authorise an upload either - the catch used " +
    "to swallow it and leave the boot defaults standing as though confirmed");
  assert.equal(window.chatState.instanceState, "unknown");
  assert.equal(window.chatState.instanceMatch, false);

  assert.equal(window.chatState.conversations.length, 1,
    "an unreachable server still shows the user their own cached conversations");
  assert.ok(window.localStorage.getItem("localm.conversations"));
});

test("AUD-INSTANCEID residual 2: the unconfirmed WARNING is one per breakage, " +
     "but the unconfirmed STATE is re-asserted on every failed poll", async () => {
  // refreshCtxLimit is polled every 30s for the tab's whole lifetime (init.js),
  // so the warning has to be deduped or an outage floods the console for hours.
  // The hazard in deduping it is doing so one line too early and skipping the
  // flags as well, which would silently restore the fail-open this unit closes.
  const { window } = loadApp({ fetchImpl: makeFetch({ instanceId: "backend-a" }) });
  runScript(window, `
    window.chatState = chat;
    window.__warns = [];
    console.warn = (m) => window.__warns.push(String(m));
    window.fetch = async (url) => {
      if (String(url) === "/v1/config") return { ok: false, status: 503,
        text: async () => "", json: async () => ({}) };
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
    };
  `);

  runScript(window, "refreshCtxLimit();");
  await drain();
  assert.equal(window.__warns.length, 1, "the first failed poll says so once");
  assert.equal(window.chatState.instanceMatch, false);

  // Simulate the flag having drifted back to permissive between polls: the
  // second failed poll must still put it back, warning or no warning.
  runScript(window, `chat.instanceMatch = true; chat.instanceState = "confirmed";`);
  runScript(window, "refreshCtxLimit();");
  await drain();

  assert.equal(window.chatState.instanceMatch, false,
    "every failed poll re-asserts the gate - deduping the LINE must never " +
    "dedupe the STATE, or the second poll silently re-opens the upload path");
  assert.equal(window.chatState.instanceState, "unknown");
  assert.equal(window.__warns.length, 1,
    "...while the console still carries exactly one line for this outage");
});
