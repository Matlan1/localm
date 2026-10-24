// SPDX-License-Identifier: AGPL-3.0-or-later
// A folder import reports what it is doing: the "Add a model" card shows the
// import frame's detail text (with byte progress while a file is hashed) instead
// of a bare "starting...", and the Activity dialog shows the same detail.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

function makeFetch() {
  return async (url) => {
    if (url === "/api/models/pull") {
      return { ok: true, status: 200, json: async () => ({ job_id: "j1" }), text: async () => "" };
    }
    return {
      ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }),
    };
  };
}

const tick = () => new Promise((r) => setTimeout(r, 0));
async function settle() { for (let i = 0; i < 6; i++) await tick(); }

test("pull panel: an import frame shows its detail, with bytes while hashing", async () => {
  const { window: win } = loadAppWithPages({ fetchImpl: makeFetch() });
  runScript(win, `
    window.__frame = null;
    streamJob = (jobId, onLine, onProgress) => new Promise((resolve) => {
      window.__send = (ev) => onProgress(ev);
      window.__finishJob = (status) => resolve({ status });
    });
  `);
  const $ = (id) => win.document.getElementById(id);

  $("pull-spec").value = "/some/folder";
  $("pull-start").click();
  await settle();
  assert.match($("pull-pct").textContent, /starting/);

  win.__send({ phase: "import", detail: "Scanning /some/folder for models",
               downloaded: 0, total: 0, pct: null });
  assert.equal($("pull-pct").textContent, "Scanning /some/folder for models");
  assert.ok($("pull-bar").classList.contains("indeterminate"), "no total: a busy bar");

  win.__send({ phase: "import", detail: "Hashing big.gguf for duplicate detection (2 of 5, 4,000 MB)",
               downloaded: 1000000000, total: 4000000000, pct: 25.0, rate: 500000000 });
  const text = $("pull-pct").textContent;
  assert.match(text, /Hashing big\.gguf for duplicate detection \(2 of 5/);
  assert.match(text, /25%/);
  assert.match(text, /\/s/, "the hashing speed is shown");
  assert.equal($("pull-bar").style.width, "25%");
  assert.ok(!$("pull-bar").classList.contains("indeterminate"));

  win.__finishJob("done");
  await settle();
});

test("activity dialog: a running import shows its detail line", async () => {
  const ops = [{ id: "j1", kind: "pull", label: "Model pull /some/folder", status: "running",
                 cancellable: true, created_at: 100, phase: "import",
                 detail: "Hashing big.gguf for duplicate detection (2 of 5, 4,000 MB)" }];
  const fetchImpl = async (url) => {
    if (String(url) === "/api/activity") {
      return { ok: true, status: 200, json: async () => ({ operations: ops, now: 160 }) };
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
  const { window: win } = loadApp({ fetchImpl });
  await win.pollActivity();
  win.showActivityDetails();
  assert.match(win.document.getElementById("modal-body").textContent,
    /Hashing big\.gguf for duplicate detection \(2 of 5/);
});
