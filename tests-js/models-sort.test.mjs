// SPDX-License-Identifier: AGPL-3.0-or-later
// sortModels(models, sortKey, sortDir) in pages/models.js: the Models page's
// per-column sort. size_bytes and mtime are nullable, and a null sorts LAST in
// both directions.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

const okFetch = async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => "" });

function sorter() {
  const { window } = loadAppWithPages({ fetchImpl: okFetch });
  return window.sortModels;
}

const M = (name, extra = {}) => ({ name, model_type: "llm", source: "local", ...extra });

test("sortModels: name ascending is case-insensitive alphabetical", () => {
  const sortModels = sorter();
  const out = sortModels([M("Zebra"), M("apple"), M("Mango")], "name", "asc");
  assert.deepEqual(out.map((m) => m.name), ["apple", "Mango", "Zebra"]);
});

test("sortModels: name descending reverses it", () => {
  const sortModels = sorter();
  const out = sortModels([M("Zebra"), M("apple"), M("Mango")], "name", "desc");
  assert.deepEqual(out.map((m) => m.name), ["Zebra", "Mango", "apple"]);
});

test("sortModels: does not mutate the input array", () => {
  const sortModels = sorter();
  const input = [M("b"), M("a")];
  const out = sortModels(input, "name", "asc");
  assert.deepEqual(input.map((m) => m.name), ["b", "a"], "input order must be untouched");
  assert.deepEqual(out.map((m) => m.name), ["a", "b"]);
});

test("sortModels: size_bytes ascending, nulls LAST", () => {
  const sortModels = sorter();
  const models = [
    M("big", { size_bytes: 300 }),
    M("unknown", { size_bytes: null }),
    M("small", { size_bytes: 100 }),
    M("mid", { size_bytes: 200 }),
  ];
  const out = sortModels(models, "size_bytes", "asc");
  assert.deepEqual(out.map((m) => m.name), ["small", "mid", "big", "unknown"],
    "a null size must never be coerced to 0 and sort as the smallest file");
});

test("sortModels: size_bytes descending, nulls STILL LAST (not first)", () => {
  const sortModels = sorter();
  const models = [
    M("big", { size_bytes: 300 }),
    M("unknown", { size_bytes: null }),
    M("small", { size_bytes: 100 }),
  ];
  const out = sortModels(models, "size_bytes", "desc");
  assert.deepEqual(out.map((m) => m.name), ["big", "small", "unknown"],
    "reversing the direction must not move the null row to the front");
});

test("sortModels: mtime ascending and descending, nulls LAST both ways", () => {
  const sortModels = sorter();
  const models = [
    M("newest", { mtime: 300 }),
    M("missing", { mtime: null }),
    M("oldest", { mtime: 100 }),
  ];
  assert.deepEqual(
    sortModels(models, "mtime", "asc").map((m) => m.name),
    ["oldest", "newest", "missing"]);
  assert.deepEqual(
    sortModels(models, "mtime", "desc").map((m) => m.name),
    ["newest", "oldest", "missing"]);
});

test("sortModels: all-null numeric column is a stable no-throw no-op ordering", () => {
  const sortModels = sorter();
  const models = [M("a", { size_bytes: null }), M("b", { size_bytes: null })];
  const out = sortModels(models, "size_bytes", "asc");
  assert.deepEqual(out.map((m) => m.name), ["a", "b"]);
});

test("sortModels: model_type (Role) and source sort case-insensitively", () => {
  const sortModels = sorter();
  const byRole = sortModels(
    [M("a", { model_type: "unknown" }), M("b", { model_type: "Embedding" }), M("c", { model_type: "llm" })],
    "model_type", "asc");
  assert.deepEqual(byRole.map((m) => m.model_type), ["Embedding", "llm", "unknown"]);

  const bySource = sortModels(
    [M("a", { source: "Local" }), M("b", { source: "huggingface" })], "source", "asc");
  assert.deepEqual(bySource.map((m) => m.source), ["huggingface", "Local"]);
});

test("sortModels: an unknown/non-sortable sortKey degrades to an unsorted copy, never throws", () => {
  const sortModels = sorter();
  const models = [M("b"), M("a")];
  const out = sortModels(models, "not-a-real-column", "asc");
  assert.deepEqual(out.map((m) => m.name), ["b", "a"]);
  const out2 = sortModels(models, "actions", "asc");   // a real but non-sortable column
  assert.deepEqual(out2.map((m) => m.name), ["b", "a"]);
});
