// SPDX-License-Identifier: AGPL-3.0-or-later
// GUI controls for the coder CLI flags:
//
//   --seed                  a setup-form field
//   --interactive-confirm   a setup-form checkbox
//   --episodes-archive      the lessons modal's "dropped" tab
//   --forget-episode        a per-lesson forget button
//   --restore-episode       a per-lesson restore button on the dropped tab
//   --forget-episodes       an "erase all" button behind confirmDanger
//   --consolidate-episodes  a "consolidate" button that reports what it did

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(calls, routes = {}) {
  return async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    calls.push({ url: u, method, body: opts.body ? JSON.parse(opts.body) : null });
    if (u === "/api/coder/sessions" && method === "POST") {
      return {
        ok: true, status: 200,
        json: async () => ({ id: "s1", cwd: "/tmp/project", notes: [] }),
        text: async () => "",
      };
    }
    for (const [prefix, resp] of Object.entries(routes)) {
      if (u.startsWith(prefix)) return typeof resp === "function" ? resp(method) : resp;
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

const ok = (body) => ({ ok: true, status: 200, json: async () => body, text: async () => "" });

async function startSession(win, calls) {
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.startCoderSession();
  await new Promise((r) => setTimeout(r, 0));
  return calls.filter((c) => c.url === "/api/coder/sessions" && c.method === "POST")[0].body;
}

/* ------------------------------------------------------------------ */
/*  --seed                                                             */
/* ------------------------------------------------------------------ */

test("seed: the field exists and a blank one is OMITTED, not sent as 0", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  const field = win.document.getElementById("setup-seed");
  assert.ok(field, "there must be a control for reproducible runs");
  assert.equal(field.value, "");
  const body = await startSession(win, calls);
  assert.ok(!("seed" in body),
    "blank means NO seed; sending 0 would silently pin every session that left "
    + "the field alone, because 0 is a real and reproducible seed");
});

test("seed typed: it reaches the session POST, and 0 survives as a real value", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  win.document.getElementById("setup-seed").value = "4242";
  assert.equal((await startSession(win, calls)).seed, 4242);

  const c2 = [];
  const { window: w2 } = loadAppWithPages({ fetchImpl: makeFetch(c2) });
  w2.document.getElementById("setup-seed").value = "0";
  assert.equal((await startSession(w2, c2)).seed, 0);
});

/* ------------------------------------------------------------------ */
/*  --interactive-confirm                                              */
/* ------------------------------------------------------------------ */

test("interactive-confirm: the checkbox reaches the session POST, default off", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  assert.equal((await startSession(win, calls)).interactive_confirm, false);

  const c2 = [];
  const { window: w2 } = loadAppWithPages({ fetchImpl: makeFetch(c2) });
  w2.document.getElementById("setup-interactive-confirm").checked = true;
  assert.equal((await startSession(w2, c2)).interactive_confirm, true);
});

/* ------------------------------------------------------------------ */
/*  the lessons modal: archive / forget / restore / erase / consolidate */
/* ------------------------------------------------------------------ */

function lessonsFetch(calls, extra = {}) {
  return makeFetch(calls, {
    "/api/coder/episodes/archive": ok({
      cwd: "/tmp/project",
      archived: [{ id: "arc1", reason: "forget", lesson: "the dropped one" }],
    }),
    "/api/coder/episodes/consolidate": extra.consolidate
      || ok({ groups: 1, merged: 1, replaced: 3, archived: 3, skipped: 0 }),
    "/api/coder/episodes/": extra.perEpisode || ok({ forgotten: "ep1", recoverable: true }),
    "/api/coder/episodes": extra.list
      || ok({ cwd: "/tmp/project", episodes: [{ id: "ep1", outcome: "ok", turns: 2, lesson: "run the tests" }] }),
  });
}

test("the dropped tab reads the archive route and shows what can be brought back", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: lessonsFetch(calls) });
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.openEpisodesModal("archive");
  assert.ok(calls.some((c) => c.url.startsWith("/api/coder/episodes/archive")));
  const body = win.document.getElementById("modal-body").textContent;
  assert.match(body, /the dropped one/);
  assert.match(body, /arc1/, "the id is what --restore-episode takes");
});

test("forget posts the cwd in the BODY, not the URL", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: lessonsFetch(calls) });
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.openEpisodesModal("live");
  const btn = [...win.document.querySelectorAll("#modal-body button")]
    .find((b) => b.textContent === "forget");
  assert.ok(btn, "each stored lesson offers a forget");
  btn.click();
  await new Promise((r) => setTimeout(r, 10));
  const f = calls.find((c) => c.url.endsWith("/forget"));
  assert.ok(f);
  assert.equal(f.method, "POST",
    "state-changing, so it must be an unsafe method - that is what the CSRF "
    + "check applies to, and a destructive URL is one someone can be walked into");
  assert.equal(f.body.cwd, "/tmp/project");
});

test("a forget the server could not make recoverable is SAID, not swallowed", async () => {
  const calls = [];
  const toasts = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: lessonsFetch(calls, {
      perEpisode: ok({
        forgotten: "ep1", recoverable: false,
        warning: "The lesson was dropped from recall, but the archive could not "
          + "be written - so this one cannot be restored.",
      }),
    }),
  });
  win.toast = (m) => toasts.push(String(m));
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.openEpisodesModal("live");
  [...win.document.querySelectorAll("#modal-body button")]
    .find((b) => b.textContent === "forget").click();
  await new Promise((r) => setTimeout(r, 10));
  assert.ok(toasts.some((t) => t.includes("cannot be restored")),
    "a user who believed this was undoable would find out at the worst moment");
});

test("erase all is behind a confirmation and never fires on the click alone", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: lessonsFetch(calls) });
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.openEpisodesModal("live");
  const wipe = [...win.document.querySelectorAll("#modal-body button")]
    .find((b) => b.textContent === "erase all");
  assert.ok(wipe, "the erase-everything path exists");
  assert.ok(wipe.classList.contains("btn-danger"),
    "an irreversible action carries the danger styling");
  wipe.click();
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(calls.filter((c) => c.method === "DELETE").length, 0,
    "clicking must only OPEN the confirmation - this one cannot be undone");

  // Confirming does fire it.
  const confirm = [...win.document.querySelectorAll("#modal-body button")]
    .find((b) => b.textContent === "Erase everything");
  assert.ok(confirm, "the confirmation names what it will do");
  confirm.click();
  await new Promise((r) => setTimeout(r, 10));
  const d = calls.find((c) => c.method === "DELETE");
  assert.ok(d && d.url === "/api/coder/episodes");
  assert.equal(d.body.cwd, "/tmp/project");
});

test("consolidate reports the skipped groups rather than only the good news", async () => {
  const calls = [];
  const toasts = [];
  const { window: win } = loadAppWithPages({
    fetchImpl: lessonsFetch(calls, {
      consolidate: ok({ groups: 2, merged: 2, replaced: 5, archived: 5, skipped: 1,
                        warning: "one group was left alone" }),
    }),
  });
  win.toast = (m) => toasts.push(String(m));
  win.document.getElementById("setup-cwd").value = "/tmp/project";
  await win.openEpisodesModal("live");
  [...win.document.querySelectorAll("#modal-body button")]
    .find((b) => b.textContent === "consolidate").click();
  await new Promise((r) => setTimeout(r, 10));
  assert.ok(calls.some((c) => c.url.endsWith("/consolidate")));
  assert.ok(toasts.some((t) => t.includes("5 merged into 2")),
    "memory that rewrites itself without saying so is how a bad merge hides");
  assert.ok(toasts.some((t) => t.includes("left untouched")),
    "a group the model returned nothing usable for is COUNTED, not dropped");
});

test("lessons with no project directory never reaches the server", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: lessonsFetch(calls) });
  win.document.getElementById("setup-cwd").value = "";
  await win.openEpisodesModal("live");
  assert.equal(calls.filter((c) => c.url.startsWith("/api/coder/episodes")).length, 0);
});
