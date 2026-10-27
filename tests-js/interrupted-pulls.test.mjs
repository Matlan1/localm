// SPDX-License-Identifier: AGPL-3.0-or-later
// Covers the interrupted-downloads list (app/interrupted-pulls.js): the
// Activity pill and modal, the Models page card, and Resume when the source
// changed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));

const DL = {
  id: "a".repeat(32), spec: "owner/repo:tiny.gguf", name: "tiny", source: "hf-file",
  urls: ["https://huggingface.co/owner/repo/resolve/abc/tiny.gguf"],
  bytes_on_disk: 1500, expected_size: 4000, dismissed: false, resumable: true,
};

function makeFetch(list, onPost = () => ({ ok: true, body: {} })) {
  const posts = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u.startsWith("/api/models/pulls/interrupted?")) {
      return { ok: true, status: 200, json: async () => list() };
    }
    if ((opts.method || "GET") === "POST" && u.startsWith("/api/models/pulls/")) {
      const body = opts.body ? JSON.parse(opts.body) : {};
      posts.push({ url: u, body });
      const r = onPost(u, body);
      return { ok: r.ok, status: r.ok ? 200 : 409, json: async () => r.body };
    }
    if (u === "/api/activity") {
      return { ok: true, status: 200, json: async () => ({ operations: [] }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  return { fetchImpl, posts };
}

test("an interrupted download shows in the Activity pill and modal with its actions", async () => {
  const { fetchImpl } = makeFetch(() => ({ downloads: [DL], orphans: [] }));
  const { window: win } = loadApp({ fetchImpl });
  await win.pollActivity();
  await win.loadInterrupted();
  const pill = win.document.getElementById("activity-pill");
  assert.notEqual(pill.style.display, "none");
  assert.match(pill.textContent, /1 interrupted download/);

  win.showActivityDetails();
  const body = win.document.getElementById("modal-body");
  assert.match(body.textContent, /owner\/repo:tiny\.gguf/);
  assert.match(body.textContent, /huggingface\.co\/owner\/repo/);
  const labels = [...body.querySelectorAll(".interrupted-row button")].map((b) => b.textContent);
  assert.deepEqual(labels, ["Resume", "Delete", "Dismiss"]);
});

test("a dismissed download does not show the pill", async () => {
  const { fetchImpl } = makeFetch(() => ({ downloads: [{ ...DL, dismissed: true }], orphans: [] }));
  const { window: win } = loadApp({ fetchImpl });
  await win.pollActivity();
  await win.loadInterrupted();
  assert.equal(win.document.getElementById("activity-pill").style.display, "none");
});

test("the Models card lists it, and shows only a link once it is dismissed", async () => {
  let dismissed = false;
  const { fetchImpl, posts } = makeFetch(
    () => ({ downloads: [{ ...DL, dismissed }], orphans: [] }),
    (u, body) => { dismissed = body.dismissed; return { ok: true, body: { ...DL, dismissed } }; });
  const { window: win } = loadAppWithPages({ fetchImpl });
  await win.loadInterrupted();
  const card = win.document.getElementById("interrupted-card");
  assert.equal(card.hidden, false);
  const dismiss = [...card.querySelectorAll("button")].find((b) => b.textContent === "Dismiss");
  dismiss.click();
  for (let i = 0; i < 5; i++) await tick();
  assert.deepEqual(posts.map((p) => p.url), [`/api/models/pulls/interrupted/${DL.id}/dismiss`]);
  assert.equal(posts[0].body.dismissed, true);
  assert.equal(card.hidden, false);
  assert.equal(card.querySelectorAll(".interrupted-row").length, 0);
  assert.match(card.textContent, /Show 1 dismissed download/);
});

test("Resume of a changed source asks, and Download again sends fresh", async () => {
  const { fetchImpl, posts } = makeFetch(
    () => ({ downloads: [DL], orphans: [] }),
    (u, body) => (body.fresh
      ? { ok: true, body: { status: "started", job_id: "j1", spec: DL.spec, note: "" } }
      : { ok: true, body: { status: "changed", detail: "changed at the source", spec: DL.spec } }));
  const { window: win } = loadApp({ fetchImpl });
  const started = win.startResume(DL);
  for (let i = 0; i < 5; i++) await tick();
  const modal = win.document.getElementById("modal-body");
  assert.match(modal.textContent, /changed at the source/);
  const again = [...modal.querySelectorAll("button")].find((b) => b.textContent === "Download again");
  again.click();
  const data = await started;
  assert.equal(data.job_id, "j1");
  assert.deepEqual(posts.map((p) => p.body.fresh), [false, true]);
});

test("Resume of a changed source that is declined starts nothing", async () => {
  const { fetchImpl, posts } = makeFetch(
    () => ({ downloads: [DL], orphans: [] }),
    () => ({ ok: true, body: { status: "changed", detail: "changed at the source" } }));
  const { window: win } = loadApp({ fetchImpl });
  const started = win.startResume(DL);
  for (let i = 0; i < 5; i++) await tick();
  const modal = win.document.getElementById("modal-body");
  [...modal.querySelectorAll("button")].find((b) => b.textContent === "Cancel").click();
  assert.equal(await started, null);
  assert.equal(posts.length, 1);
});
