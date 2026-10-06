import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const settle = () => new Promise((r) => setTimeout(r, 0));

function boot() {
  const calls = [];
  const { window } = loadApp({
    fetchImpl: async (url, opts = {}) => {
      const u = String(url);
      calls.push({ url: u, opts });
      const body = u === "/api/rag/extract"
        ? { filename: "notes.txt", text: "hello", chars: 5, truncated: false }
        : {};
      return {
        ok: true, status: 200, json: async () => body, text: async () => "",
        headers: { get: () => null },
      };
    },
  });
  return { window, calls };
}

const extractCalls = (calls) => calls.filter((c) => c.url === "/api/rag/extract");

test("coder attach: an image is refused client-side with an accurate message", async () => {
  const { window, calls } = boot();
  await settle();
  for (const [name, type] of [["shot.png", "image/png"], ["shot.JPG", ""], ["x.webp", "image/webp"]]) {
    const file = new window.File([new Uint8Array([1, 2, 3])], name, { type });
    await assert.rejects(window.attachCoderDocument(file), (err) => {
      assert.match(err.message, /cannot take images/i);
      assert.doesNotMatch(err.message, /load a vision/i);
      return true;
    });
  }
  assert.deepEqual(extractCalls(calls), [], "no image reached /api/rag/extract");
});

test("coder attach: a text file still goes through extraction", async () => {
  const { window, calls } = boot();
  await settle();
  const file = new window.File(["hello"], "notes.txt", { type: "text/plain" });
  await window.attachCoderDocument(file);
  assert.equal(extractCalls(calls).length, 1);
});
