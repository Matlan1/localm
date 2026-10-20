// SPDX-License-Identifier: AGPL-3.0-or-later
// The "Add a model" card can pause, resume and cancel a download in flight.
// Pause stops the job and keeps the partial data (Resume sends the same request
// again); Cancel stops it and asks the server to delete the partial data.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

function makeFetch(calls) {
  return async (url, opts = {}) => {
    if (url === "/api/models/pull" || url === "/api/models/pull/discard"
        || String(url).endsWith("/cancel")) {
      calls.push({ url, body: opts.body ? JSON.parse(opts.body) : null });
      return { ok: true, status: 200, json: async () => ({ job_id: "j" + calls.length }),
               text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

const tick = () => new Promise((r) => setTimeout(r, 0));
async function settle() { for (let i = 0; i < 6; i++) await tick(); }

// A streamJob stand-in that reports one progress frame, then waits for the cancel
// request before ending the job as "cancelled" (or finishes at once when the
// window was told to).
const WAITING_STREAM = `
  window.__ended = null;
  streamJob = (jobId, onLine, onProgress) => new Promise((resolve) => {
    onProgress({ phase: "download", downloaded: 167000000, total: 491000000, pct: 34.0,
                 rate: 22000000, eta: 15 });
    window.__finishJob = (status) => resolve({ status });
  });
`;

test("pull panel: Pause cancels the job, keeps the partial data and offers Resume", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  runScript(win, WAITING_STREAM);
  const $ = (id) => win.document.getElementById(id);

  $("pull-spec").value = "owner/repo:m.gguf";
  $("pull-start").click();
  await settle();

  assert.equal($("pull-pause").hidden, false, "Pause shows while a download runs");
  assert.equal($("pull-cancel").hidden, false, "Cancel shows while a download runs");
  assert.equal($("pull-resume").hidden, true, "Resume is hidden while a download runs");
  assert.match($("pull-pct").textContent, /34%.*21\.0 MB\/s.*ETA 15s/s,
    "the server's rate and ETA are shown");

  $("pull-pause").click();
  await settle();
  assert.ok(calls.some((c) => c.url === "/api/jobs/j1/cancel"), "the job was cancelled");
  win.__finishJob("cancelled");
  await settle();

  assert.equal($("pull-resume").hidden, false, "Resume shows after a pause");
  assert.equal($("pull-pause").hidden, true);
  assert.ok(!calls.some((c) => c.url === "/api/models/pull/discard"),
    "pausing keeps the partial data");
  assert.match($("pull-pct").textContent, /paused/);

  $("pull-resume").click();
  await settle();
  const pulls = calls.filter((c) => c.url === "/api/models/pull");
  assert.equal(pulls.length, 2, "Resume sends the pull request again");
  assert.deepEqual(pulls[1].body, pulls[0].body, "with the same request");
});

test("pull panel: Cancel cancels the job and asks the server to delete the partial data", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  runScript(win, WAITING_STREAM);
  const $ = (id) => win.document.getElementById(id);

  $("pull-spec").value = "owner/repo:m.gguf";
  $("pull-name").value = "mine";
  $("pull-start").click();
  await settle();

  $("pull-cancel").click();
  await settle();
  win.__finishJob("cancelled");
  await settle();

  const discard = calls.find((c) => c.url === "/api/models/pull/discard");
  assert.ok(discard, "the partial data was discarded");
  assert.deepEqual(discard.body, { spec: "owner/repo:m.gguf", name: "mine" });
  assert.equal($("pull-resume").hidden, true);
  assert.equal($("pull-cancel").hidden, true, "nothing left to cancel");
  assert.match($("pull-pct").textContent, /cancelled/);
});

test("pull panel: Cancel on a paused download deletes the partial data without a job", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  runScript(win, WAITING_STREAM);
  const $ = (id) => win.document.getElementById(id);

  $("pull-spec").value = "owner/repo:m.gguf";
  $("pull-start").click();
  await settle();
  $("pull-pause").click();
  await settle();
  win.__finishJob("cancelled");
  await settle();

  $("pull-cancel").click();
  await settle();

  assert.ok(calls.some((c) => c.url === "/api/models/pull/discard"));
  assert.equal($("pull-resume").hidden, true);
});

test("pull panel: a failed download shows no Pause, Resume or Cancel", async () => {
  const calls = [];
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch(calls) });
  runScript(win, `streamJob = async () => ({ status: "failed", returncode: 1 });`);
  const $ = (id) => win.document.getElementById(id);

  $("pull-spec").value = "owner/repo:m.gguf";
  $("pull-start").click();
  await settle();

  assert.equal($("pull-pause").hidden, true);
  assert.equal($("pull-resume").hidden, true);
  assert.equal($("pull-cancel").hidden, true);
});
