// SPDX-License-Identifier: AGPL-3.0-or-later
// The Models page's Browse starts in a folder only for a path, refuses without
// host disk access, offers a LoRA only the bases the attach route accepts, and
// asks before removing an uploaded file.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));
const settle = async () => { for (let i = 0; i < 6; i++) await tick(); };
const ok = (body) => ({ ok: true, status: 200, text: async () => "", json: async () => body });

function setup(routes = {}) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    for (const [part, fn] of Object.entries(routes)) if (u.includes(part)) return fn(u, opts);
    return ok({});
  };
  const { window } = loadAppWithPages({ fetchImpl });
  return { window, calls };
}

test("a path is told apart from a Hugging Face spec and a URL", () => {
  const { window: win } = setup();
  for (const p of ["C:\\models", "D:/m", "/home/me/m", "~/m", "./m", "..\\m"]) {
    assert.equal(win.looksLikeLocalPath(p), true, p);
  }
  for (const s of ["owner/repo:file.gguf", "owner/repo", "https://x/y.gguf", ""]) {
    assert.equal(win.looksLikeLocalPath(s), false, s);
  }
});

test("Browse with a repo spec typed starts the picker at its default, not at the spec", async () => {
  const { window: win } = setup();
  runScript(win, `caps.fsAccess = "host"; globalThis.__start = null;
    pickDirectory = (title, start) => { globalThis.__start = start; return Promise.resolve(null); };`);
  win.document.getElementById("pull-spec").value = "owner/repo:file.gguf";
  win.document.getElementById("pull-browse").click();
  await settle();
  assert.equal(win.__start, "");
});

test("Browse without host disk access says so and opens no picker", async () => {
  const { window: win } = setup();
  runScript(win, `caps.fsAccess = "none"; globalThis.__picked = false;
    pickDirectory = () => { globalThis.__picked = true; return Promise.resolve(null); };`);
  win.document.getElementById("pull-browse").click();
  await settle();
  assert.equal(win.__picked, false);
  assert.match(win.document.getElementById("toast-errors").textContent, /Type the path instead/);
});

test("a LoRA is offered only the chat models of its own architecture", () => {
  const { window: win } = setup();
  runScript(win, `_modelsSnapshot = [
    { name: "llama-base", model_type: "llm", architecture: "llama" },
    { name: "qwen-base", model_type: "llm", architecture: "qwen2" },
    { name: "hf-folder", model_type: "llm", architecture: "LlamaForCausalLM" },
    { name: "lora", model_type: "lora", architecture: "llama", adapter: true },
  ];
  globalThis.__bases = adapterBaseCandidates({ name: "lora", architecture: "llama" }).map((m) => m.name);
  globalThis.__any = adapterBaseCandidates({ name: "lora" }).map((m) => m.name);`);
  assert.deepEqual([...win.__bases], ["llama-base"]);
  assert.deepEqual([...win.__any], ["llama-base", "qwen-base", "hf-folder"]);
});

test("Remove on an uploaded file asks first and deletes nothing until confirmed", async () => {
  const { window: win, calls } = setup({
    "/api/uploads": (u, opts) => (opts.method === "DELETE" ? ok({}) : ok({ items: [{ name: "a.txt", bytes: 3 }] })),
  });
  await win.refreshUploadsList();
  await settle();
  const del = win.document.querySelector(".upload-del");
  assert.ok(del, "the list shows a Remove button");
  del.click();
  await settle();
  assert.equal(win.document.getElementById("modal").style.display, "flex");
  assert.match(win.document.getElementById("modal-title").textContent, /Remove a\.txt\?/);
  assert.equal(calls.filter((c) => c.opts.method === "DELETE").length, 0);
});
