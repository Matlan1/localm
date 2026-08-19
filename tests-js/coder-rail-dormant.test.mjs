import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

// ASYNC + a settle tick, for the reason banked while building the rail-side
// unit: loadApp boots the real app, whose startup fetches resolve on a later
// microtask. A test that returns first is torn down underneath them and the
// continuation dies on an undefined `document`, which node:test reports as
// "asynchronous activity after the test ended" - reading exactly like a defect
// in the code under test, and not being one.
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

  // The headings are what stop "Open" and "Past" reading as one list.
  assert.deepEqual(texts(window), ["Past sessions here", "Other projects"],
    "with no live session there is no Open group, and both dormant groups render");

  const titles = [...rail(window).querySelectorAll(".coder-session-item .title")]
    .map((n) => n.textContent);
  assert.ok(titles.includes("build a calculator"), "the current project's past session");
  // THE POINT OF THE FEATURE: reachable without typing the project path first.
  assert.ok(titles.includes("write a csv parser"), "another project's past session");

  const groups = [...rail(window).querySelectorAll(".coder-rail-project summary .title")]
    .map((n) => n.textContent);
  assert.deepEqual(groups, ["elsewhere", "vanished"],
    "other projects are collapsible groups, the current one is not");
});

test("rail: continuing another project's session uses THAT project, not the form's", async () => {
  const { window, posts } = boot();
  await settle();
  // The form points somewhere else entirely - which is the normal case when
  // you click a row under "Other projects".
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
  // ASSERT ON THE DESTINATION FIRST. Taking the form's cwd would start a real
  // session in the WRONG FOLDER while every status code stayed 200 - a silent
  // wrong-directory start, not a visible error.
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
  // Listed, because the conversation outlives the directory and losing the
  // folder is exactly when someone wants it back.
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
  // Not a string in the JS: one wording, which cannot drift from what the
  // endpoint actually guarantees.
  assert.equal(note.textContent, PAYLOAD.privacy_note);
  assert.ok(rail(window).querySelectorAll(".coder-session-item").length > 0,
    "this arm must be the NON-empty one - a note that only appears on an empty "
    + "list reads as an excuse for a short list rather than a property");
});

test("rail: a failed dormant fetch leaves what is already shown alone", async () => {
  // Succeed once, then fail. Asserting on the DOM rather than on internal
  // state, because the DOM is the property that matters and because a
  // top-level `const` in a classic script is never a window property - the
  // first version of this test read `window.dormant` and got undefined, which
  // would have "passed" against almost any behaviour once coerced.
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

  // A listing that blanks on a transient error looks exactly like "you have no
  // past work" - which is a lie about the user's own history.
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

  // Back where it started: a side that silently reverts on the next page load
  // is worse than one that refuses now.
  assert.equal(view.dataset.rail, undefined,
    "the unsaved flip was undone, so the screen matches what was stored");
});
