// SPDX-License-Identifier: AGPL-3.0-or-later
// A browser allows 6 connections per host. One open events stream per coder
// session froze the GUI at 5 sessions, so only a bounded number of sessions hold
// a stream: the one on screen and the busy ones.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const settle = async (n = 30) => { for (let i = 0; i < n; i++) await Promise.resolve(); };

function eventsFetch() {
  const open = new Set();
  const opened = [];
  const impl = async (url, opts = {}) => {
    const m = /\/api\/coder\/sessions\/([^/]+)\/events(\?replay=true)?/.exec(String(url));
    if (!m) return { ok: true, status: 200, json: async () => ({ sessions: [] }) };
    const id = m[1];
    open.add(id);
    opened.push({ id, replay: !!m[2] });
    if (opts.signal) opts.signal.addEventListener("abort", () => open.delete(id));
    return {
      ok: true, status: 200,
      body: { getReader() { return { read: () => new Promise(() => {}), cancel() {} }; } },
    };
  };
  return { impl, open, opened };
}

function register(window, id, busy = false) {
  window.__info = { id, cwd: "Z:/" + id, busy };
  window.registerSession(window.__info, { replay: false });
}

test("with more sessions than the stream budget, only the active one holds a stream", async () => {
  const { impl, open } = eventsFetch();
  const { window } = loadApp({ fetchImpl: impl });
  for (const id of ["a", "b", "c", "d", "e"]) register(window, id);
  window.activateSession("e");
  await settle();
  assert.deepEqual([...open], ["e"]);
});

test("a busy session keeps its stream and a parked one is re-read in full when shown", async () => {
  const { impl, open, opened } = eventsFetch();
  const { window } = loadApp({ fetchImpl: impl });
  for (const id of ["a", "b", "c", "d", "e"]) register(window, id, id === "b");
  window.activateSession("e");
  await settle();
  assert.deepEqual([...open].sort(), ["b", "e"]);
  window.activateSession("a");
  await settle();
  assert.deepEqual([...open].sort(), ["a", "b"], "e did not park when a took the screen");
  const last = opened.filter((o) => o.id === "a").pop();
  assert.equal(last.replay, true, "a was parked once the fourth session registered, so it is re-read in full");
  window.activateSession("e");
  await settle();
  assert.equal(opened.filter((o) => o.id === "e").pop().replay, true,
    "a parked session must be re-read in full when it returns");
});

test("up to the budget every session keeps streaming", async () => {
  const { impl, open } = eventsFetch();
  const { window } = loadApp({ fetchImpl: impl });
  for (const id of ["a", "b", "c"]) register(window, id);
  window.activateSession("c");
  await settle();
  assert.deepEqual([...open].sort(), ["a", "b", "c"]);
});
