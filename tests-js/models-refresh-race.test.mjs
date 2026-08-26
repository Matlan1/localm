// SPDX-License-Identifier: AGPL-3.0-or-later
// init.js fires its own unrelated fetch("/api/models") calls during startup on
// the same URL as this page's default fetch, so the harness below only gates
// "/api/models" once startRacing() is called, after those ticks have drained.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function deferred() {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  return { promise, resolve };
}

async function tick(n = 2) { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); }

// Gates only the page-level fetch (exact "/api/models", no query), and only
// once startRacing() has been called. Everything else resolves immediately.
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

  // B (started second) resolves first; A (started first) resolves last
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
  // the error/empty branches also write into `box`
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
