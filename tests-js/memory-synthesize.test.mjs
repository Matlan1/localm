// SPDX-License-Identifier: AGPL-3.0-or-later
// The memory modal's "Synthesize now" button POSTs /api/memory/consolidate and
// reports the result in its status element.

import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages } from "./harness.mjs";

function makeFetch(posts, consolidateResp, memText) {
  return async (url, _opts = {}) => {
    const u = String(url);
    if (u === "/api/memory/consolidate") {
      posts.push(u);
      return { ok: true, status: 200, text: async () => "",
        json: async () => consolidateResp };
    }
    if (u === "/api/memory") {
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ text: memText, writable: true, items: [] }) };
    }
    return { ok: true, status: 200, text: async () => "",
      json: async () => ({ models: [], active: "", conversations: [], plugins: [] }) };
  };
}

async function flush() {
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
}

test("F7: Synthesize now POSTs consolidate and reports the added count", async () => {
  const posts = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch(posts, { status: "ok", added: 2, facts: ["a", "b"] },
      "- a\n- b"),
  });
  const status = { textContent: "" };
  const data = await window.synthesizeMemoryNow(status);
  await flush();
  assert.equal(posts.length, 1, "one consolidate POST");
  assert.equal(data.added, 2);
  assert.match(status.textContent, /Added 2 new facts/);
});

test("F7: Synthesize now surfaces the privacy-skip honestly", async () => {
  const posts = [];
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch(posts, { status: "skipped", reason: "privacy", added: 0 }, ""),
  });
  const status = { textContent: "" };
  await window.synthesizeMemoryNow(status);
  assert.match(status.textContent, /privacy mode/);
});

test("F7: Synthesize now reports when nothing new was found", async () => {
  const { window } = loadAppWithPages({
    fetchImpl: makeFetch([], { status: "ok", added: 0, facts: [] }, ""),
  });
  const status = { textContent: "" };
  await window.synthesizeMemoryNow(status);
  assert.match(status.textContent, /No new durable facts/);
});

test("Synthesize now sends the conversation on screen, not web results or reasoning", async () => {
  const bodies = [];
  const { window } = loadAppWithPages({
    fetchImpl: async (url, opts = {}) => {
      if (String(url) === "/api/memory/consolidate") {
        bodies.push(JSON.parse(opts.body));
        return { ok: true, status: 200, text: async () => "",
          json: async () => ({ status: "ok", added: 1, facts: ["x"] }) };
      }
      return { ok: true, status: 200, text: async () => "",
        json: async () => ({ text: "", writable: true, items: [], models: [], active: "",
          conversations: [], plugins: [] }) };
    },
  });
  window.eval(`
    chat.conversations = [{ id: "c1", branches: [], messages: [
      { role: "user", content: "remember that i love to play rpg style games" },
      { role: "assistant", content: "<think>hidden</think>Sounds fun!" },
      { role: "user", content: "web result text", origin: "tool" },
      { role: "user", content: "attached doc", tag: "doc" },
    ] }];
    chat.currentId = "c1";
  `);
  await window.synthesizeMemoryNow({ textContent: "" });
  assert.equal(bodies.length, 1);
  const rows = bodies[0].messages;
  assert.equal(rows[0].role, "user");
  assert.equal(rows[0].content, "remember that i love to play rpg style games");
  assert.equal(rows[1].role, "assistant");
  assert.doesNotMatch(rows[1].content, /hidden/);
  assert.equal(rows.length, 2, "tool and document rows are left out");
});
