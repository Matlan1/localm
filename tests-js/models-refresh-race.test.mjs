// SPDX-License-Identifier: AGPL-3.0-or-later
// refreshModelsPage() clears #models-table with box.replaceChildren() at its
// very start, then only appends its freshly-built <table> ~280 lines later,
// after an awaited /api/models fetch. Two overlapping calls (a rapid
// double-click on a sortable header, a set-type change firing while a prior
// refresh's fetch is still in flight, ...) each clear once at their own start
// and then each append their own table - nothing stops both from surviving,
// rendering the model list duplicated (phantom models). Reproduced live via a
// rapid double-click on the Name header before this fix: two <table
// class="data-table"> elements landed in #models-table for 3 registered
// models. See models-remove-disconnect.test.mjs for the harness pattern this
// borrows (win.refreshModelsPage() driven directly against a controlled
// fetchImpl).
//
// NOTE: init.js fires its own one-off, unrelated fetch("/api/models") calls
// during startup (an instance-pairing check, unrelated to #models-table) that
// use the exact same URL as this page's own default (unfiltered) fetch. The
// harness below only starts gating "/api/models" once told to (via
// startRacing()), after draining enough ticks for that startup noise to
// settle - so the two gated calls it counts are only the two
// refreshModelsPage() invocations the test itself triggers.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function deferred() {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  return { promise, resolve };
}

async function tick(n = 2) { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); }

// Gates ONLY the page-level (exact "/api/models", no query - the default
// "all types" filter) fetch, and only once startRacing() has been called.
// Everything else (the sidebar's "/api/models?type=llm", init.js's own
// unrelated "/api/models" startup calls, any other endpoint) resolves
// immediately, so it never blocks reaching the gated calls under test.
function makeRaceHarness(resultFor) {
  const gates = [deferred(), deferred()];
  let pageCallIndex = 0;
  let racing = false;
  const fetchImpl = async (url) => {
    const u = String(url);
    if (racing && u === "/api/models") {
      const i = pageCallIndex++;
      await gates[i].promise;
      return resultFor(i);
    }
    return { ok: true, status: 200, json: async () => ({ models: [], active: "" }), text: async () => "" };
  };
  return {
    fetchImpl,
    startRacing: () => { racing = true; },
    resolve: (i) => gates[i].resolve(),
  };
}

function tableCount(win) {
  return win.document.querySelectorAll("#models-table table.data-table").length;
}

function nameCells(win) {
  return [...win.document.querySelectorAll("#models-table tbody tr .name")].map((n) => n.textContent);
}

test("refreshModelsPage: two overlapping refreshes leave exactly one table (the newer one), not two", async () => {
  const payloads = [
    { models: [{ name: "model-a", model_type: "llm" }] },
    { models: [{ name: "model-b", model_type: "llm" }] },
  ];
  const harness = makeRaceHarness((i) =>
    ({ ok: true, status: 200, json: async () => payloads[i], text: async () => "" }));
  const { window: win } = loadAppWithPages({ fetchImpl: harness.fetchImpl });
  await tick(5);   // drain init.js's own unrelated startup fetches first
  harness.startRacing();

  const p1 = win.refreshModelsPage();               // call A: starts, blocks on gate 0
  await tick();
  const p2 = win.refreshModelsPage();                // call B: starts while A is still in flight
  await tick();

  // The classic race shape: the OLDER call's request is the slower one - B
  // (started second) resolves first, A (started first) resolves last, well
  // after B has already finished rendering.
  harness.resolve(1);
  await tick();
  harness.resolve(0);
  await Promise.all([p1, p2]);
  await tick();

  assert.equal(tableCount(win), 1,
    "both calls' box.replaceChildren()+appendChild ran unguarded - a stale " +
    "call must not leave its own table sitting alongside (or in place of) " +
    "the newer call's");
  assert.deepEqual(nameCells(win), ["model-b"],
    "the SURVIVING content must be the newer call's (B), not the stale " +
    "call's (A) that merely happened to finish last");
});

test("refreshModelsPage: a stale call's error path does not overwrite a newer call's success", async () => {
  // The error/empty branches also write into `box` - a stale error render
  // painting over a fresh success is the same bug wearing different clothes.
  const harness = makeRaceHarness((i) => i === 0
    ? { ok: false, status: 500, json: async () => ({}), text: async () => "" }
    : { ok: true, status: 200, json: async () => ({ models: [{ name: "model-b", model_type: "llm" }] }), text: async () => "" });
  const { window: win } = loadAppWithPages({ fetchImpl: harness.fetchImpl });
  await tick(5);
  harness.startRacing();

  const p1 = win.refreshModelsPage();   // call A: will 500, resolves LAST
  await tick();
  const p2 = win.refreshModelsPage();   // call B: succeeds, resolves FIRST
  await tick();

  harness.resolve(1);
  await tick();
  harness.resolve(0);
  await Promise.all([p1, p2]);
  await tick();

  assert.equal(tableCount(win), 1, "B's table must still be standing");
  assert.deepEqual(nameCells(win), ["model-b"]);
  const boxText = win.document.getElementById("models-table").textContent;
  assert.doesNotMatch(boxText, /Could not load models/,
    "call A's stale HTTP-error message must not land after B's real success");
});
