import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages } from "./harness.mjs";

// loadApp's startup fetches resolve on a later microtask, so every test here is
// async and awaits a settle tick before it returns.
const settle = () => new Promise((r) => setTimeout(r, 0));

const PAYLOAD = {
  privacy_note: "Privacy-mode sessions are never recorded and never appear here.",
  projects: [
    {
      path: "/work/here", name: "here", available: true, current: true,
      sessions: [
        { id: "aaa111", title: "build a calculator", interrupted_at: null,
          turns: 4, total_tokens: 10, messages: 8, changed_files: 1 },
      ],
    },
    {
      path: "/work/elsewhere", name: "elsewhere", available: true, current: false,
      sessions: [
        { id: "bbb222", title: "write a csv parser", interrupted_at: null,
          turns: 2, total_tokens: 5, messages: 4, changed_files: 0 },
      ],
    },
    {
      path: "/work/vanished", name: "vanished", available: false, current: false,
      sessions: [
        { id: "ccc333", title: "an old experiment", interrupted_at: null,
          turns: 1, total_tokens: 1, messages: 2, changed_files: 0 },
      ],
    },
  ],
};

function boot(payload = PAYLOAD) {
  const posts = [];
  const { window } = loadApp({
    fetchImpl: async (url, opts = {}) => {
      const u = String(url);
      if (opts.method && opts.method !== "GET") posts.push({ url: u, opts });
      const body = u.startsWith("/api/coder/dormant") ? payload : {};
      return {
        ok: true, status: 200, json: async () => body, text: async () => "",
        headers: { get: () => null },
      };
    },
  });
  return { window, posts };
}

const rail = (window) => window.document.getElementById("coder-session-list");
const texts = (window) =>
  [...rail(window).querySelectorAll(".coder-rail-head")].map((n) => n.textContent);

test("rail: past sessions from OTHER projects are listed, not only the current one", async () => {
  const { window } = boot();
  await settle();
  await window.refreshDormant();

  assert.deepEqual(texts(window), ["Past sessions here", "Other projects"],
    "with no live session there is no Open group, and both dormant groups render");

  const titles = [...rail(window).querySelectorAll(".coder-session-item .title")]
    .map((n) => n.textContent);
  assert.ok(titles.includes("build a calculator"), "the current project's past session");
  assert.ok(titles.includes("write a csv parser"), "another project's past session");

  const groups = [...rail(window).querySelectorAll(".coder-rail-project summary .title")]
    .map((n) => n.textContent);
  assert.deepEqual(groups, ["elsewhere", "vanished"],
    "other projects are collapsible groups, the current one is not");
});

test("rail: continuing another project's session uses THAT project, not the form's", async () => {
  const { window, posts } = boot();
  await settle();
  // The form points at a different project than the row clicked below.
  window.document.getElementById("setup-cwd").value = "/work/here";
  await window.refreshDormant();

  const row = [...rail(window).querySelectorAll(".coder-session-item.dormant")]
    .find((n) => n.querySelector(".title").textContent === "write a csv parser");
  assert.ok(row, "the other-project row rendered");
  row.onclick();
  await settle();

  const create = posts.find((p) => p.url.startsWith("/api/coder/sessions"));
  assert.ok(create, "clicking a past session starts one");
  const sent = JSON.parse(create.opts.body);
  assert.equal(sent.cwd, "/work/elsewhere",
    "started in the row's own project, not the one in the form");
  assert.equal(sent.resume_checkpoint_id, "bbb222",
    "and continues the session that was clicked");
  assert.equal(sent.resume, true);
});

test("rail: a session whose folder is gone is shown but not offered", async () => {
  const { window, posts } = boot();
  await settle();
  await window.refreshDormant();

  const row = [...rail(window).querySelectorAll(".coder-session-item.dormant")]
    .find((n) => n.querySelector(".title").textContent === "an old experiment");
  assert.ok(row, "the row is present rather than silently dropped");
  assert.ok(row.classList.contains("unavailable"));
  assert.equal(row.onclick, null, "no click handler - it would fail at the server");
  assert.match(row.title, /folder is missing/i, "and it says why");
  assert.equal(posts.length, 0);
});

test("rail: the privacy note is the server's text and shows on a NON-empty list", async () => {
  const { window } = boot();
  await settle();
  await window.refreshDormant();

  const note = window.document.getElementById("coder-rail-note");
  assert.equal(note.textContent, PAYLOAD.privacy_note);
  assert.ok(rail(window).querySelectorAll(".coder-session-item").length > 0,
    "this arm must be the NON-empty one - a note that only appears on an empty "
    + "list reads as an excuse for a short list rather than a property");
});

test("rail: a failed dormant fetch leaves what is already shown alone", async () => {
  // The dormant fetch succeeds once, then fails. Assertions read the DOM.
  let fail = false;
  const { window } = loadApp({
    fetchImpl: async (url) => {
      const u = String(url);
      if (u.startsWith("/api/coder/dormant")) {
        if (fail) {
          return { ok: false, status: 500, json: async () => ({}),
                   text: async () => "", headers: { get: () => null } };
        }
        return { ok: true, status: 200, json: async () => PAYLOAD,
                 text: async () => "", headers: { get: () => null } };
      }
      return { ok: true, status: 200, json: async () => ({}), text: async () => "",
               headers: { get: () => null } };
    },
  });
  await settle();
  await window.refreshDormant();
  const before = rail(window).querySelectorAll(".coder-session-item").length;
  assert.ok(before > 0, "precondition: rows are on screen to lose");

  fail = true;
  await window.refreshDormant();

  assert.equal(rail(window).querySelectorAll(".coder-session-item").length, before,
    "a failed refresh must not clear the rail");
  assert.equal(window.document.getElementById("coder-rail-note").textContent,
    PAYLOAD.privacy_note, "and the note stays with the list it describes");
});

test("rail: the side toggle persists the choice and reverts if the save fails", async () => {
  const { window, posts } = boot();
  await settle();
  const view = window.document.getElementById("view-coder");
  window.document.getElementById("coder-rail-flip").onclick();
  await settle();

  assert.equal(view.dataset.rail, "left", "flipped immediately");
  const patch = posts.find((p) => p.url === "/v1/config");
  assert.ok(patch && patch.opts.method === "PATCH", "and persisted");
  assert.equal(JSON.parse(patch.opts.body).coder_rail_side, "left",
    "as the SAME setting the Settings page edits, not a second one");
});

test("rail: a failed side save puts the rail back rather than lying about it", async () => {
  const { window } = loadApp({
    fetchImpl: async (url, opts = {}) => {
      const u = String(url);
      if (u === "/v1/config" && opts.method === "PATCH") {
        return { ok: false, status: 500, json: async () => ({}),
                 text: async () => "", headers: { get: () => null } };
      }
      return { ok: true, status: 200, json: async () => ({ projects: [] }),
               text: async () => "", headers: { get: () => null } };
    },
  });
  await settle();
  const view = window.document.getElementById("view-coder");
  window.document.getElementById("coder-rail-flip").onclick();
  await settle();

  assert.equal(view.dataset.rail, undefined,
    "the unsaved flip was undone, so the screen matches what was stored");
});

test("rail: clicking the same dormant row twice reuses the session, not a second one", async () => {
  // Simulates the pre-fix backend too: every POST /api/coder/sessions
  // returns a DIFFERENT id, exactly as observed in the real bug report (six
  // clicks on one row, six different session ids). The fix must stop the
  // SECOND click from ever reaching this POST at all.
  let nextId = 1;
  const posts = [];
  const { window } = loadApp({
    fetchImpl: async (url, opts = {}) => {
      const u = String(url);
      if (u === "/api/coder/sessions" && opts.method === "POST") {
        posts.push({ url: u, opts });
        const sent = JSON.parse(opts.body);
        return { ok: true, status: 200,
                 json: async () => ({
                   id: "sess" + nextId++, cwd: sent.cwd, resumed: true, notes: [],
                 }),
                 text: async () => "", headers: { get: () => null } };
      }
      const body = u.startsWith("/api/coder/dormant") ? PAYLOAD : {};
      return { ok: true, status: 200, json: async () => body, text: async () => "",
               headers: { get: () => null } };
    },
  });
  await settle();
  await window.refreshDormant();

  const row = () => [...rail(window).querySelectorAll(".coder-session-item.dormant")]
    .find((n) => n.querySelector(".title").textContent === "build a calculator");
  assert.ok(row(), "precondition: the dormant row is present");

  row().onclick();
  await settle();
  row().onclick();   // the row list is a snapshot - it still shows after being resumed
  await settle();
  row().onclick();
  await settle();

  assert.equal(posts.length, 1,
    `clicking an already-open session must not start another - got ${posts.length} POSTs`);
});

test("rail: a rapid double-click on a dormant row cannot race past the reuse check", async () => {
  // The reuse check above only sees an already-open session once its POST has
  // resolved and registerSession() has run - a click fired before that must
  // still not slip a second POST through.
  let resolvePost;
  const posts = [];
  const { window } = loadApp({
    fetchImpl: async (url, opts = {}) => {
      const u = String(url);
      if (u === "/api/coder/sessions" && opts.method === "POST") {
        posts.push({ url: u, opts });
        const sent = JSON.parse(opts.body);
        await new Promise((res) => { resolvePost = res; });
        return { ok: true, status: 200,
                 json: async () => ({ id: "sessA", cwd: sent.cwd, resumed: true, notes: [] }),
                 text: async () => "", headers: { get: () => null } };
      }
      const body = u.startsWith("/api/coder/dormant") ? PAYLOAD : {};
      return { ok: true, status: 200, json: async () => body, text: async () => "",
               headers: { get: () => null } };
    },
  });
  await settle();
  await window.refreshDormant();

  const row = () => [...rail(window).querySelectorAll(".coder-session-item.dormant")]
    .find((n) => n.querySelector(".title").textContent === "build a calculator");
  row().onclick();
  await settle();
  row().onclick();   // fired while the first POST is still in flight
  await settle();
  resolvePost();
  await settle();

  assert.equal(posts.length, 1,
    `a click while the first POST was still in flight must not queue a second one - got ${posts.length}`);
});

test("rail: arriving at the coder view loads past sessions by itself", async () => {
  // loadAppWithPages, not loadApp: `onViewShown` is installed only by
  // pages/dispatch.js.
  const posts = [];
  const { window } = loadAppWithPages({
    fetchImpl: async (url, opts = {}) => {
      const u = String(url);
      if (opts.method && opts.method !== "GET") posts.push({ url: u, opts });
      const body = u.startsWith("/api/coder/dormant") ? PAYLOAD : {};
      return { ok: true, status: 200, json: async () => body,
               text: async () => "", headers: { get: () => null } };
    },
  });
  await settle();
  // refreshDormant() is not called here: arriving at the view must trigger it.
  window.onViewShown("coder");
  await settle();

  const titles = [...rail(window).querySelectorAll(".coder-session-item .title")]
    .map((n) => n.textContent);
  assert.ok(titles.includes("build a calculator"),
    "opening the coder view must populate the rail without any further input");
});
