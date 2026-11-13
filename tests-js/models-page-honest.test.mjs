// SPDX-License-Identifier: AGPL-3.0-or-later
// The Models page's Browse starts in a folder only for a path, refuses without
// host disk access, offers a LoRA only the bases the attach route accepts, asks
// before removing an uploaded file, names the HTTP error of a pull whose answer
// is not JSON, and raises the reconnect overlay after an update that restarts.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));
const settle = async () => { for (let i = 0; i < 6; i++) await tick(); };
const ok = (body) => ({ ok: true, status: 200, text: async () => "", json: async () => body });
async function waitFor(fn, timeout = 3000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (fn()) return true; await new Promise((r) => setTimeout(r, 15)); }
  return false;
}

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

const hostCaps = { "/api/capabilities": () => ok({ fs_access: "host" }) };

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
  const { window: win } = setup(hostCaps);
  runScript(win, `globalThis.__start = null;
    pickDirectory = (title, start) => { globalThis.__start = start; return Promise.resolve(null); };`);
  win.document.getElementById("pull-spec").value = "owner/repo:file.gguf";
  win.document.getElementById("pull-browse").click();
  await settle();
  assert.equal(win.__start, "");
});

test("Browse without host disk access says so and opens no picker", async () => {
  const { window: win } = setup({ "/api/capabilities": () => ok({ fs_access: "none" }) });
  runScript(win, `globalThis.__picked = false;
    pickDirectory = () => { globalThis.__picked = true; return Promise.resolve(null); };`);
  win.document.getElementById("pull-browse").click();
  await settle();
  assert.equal(win.__picked, false);
  assert.match(win.document.getElementById("toast-errors").textContent, /Type the path instead/);
});

test("Browse clicked before the capabilities answer waits for it instead of refusing", async () => {
  let answer;
  const late = new Promise((r) => { answer = r; });
  const { window: win } = setup({ "/api/capabilities": () => late });
  runScript(win, `globalThis.__picked = false;
    pickDirectory = () => { globalThis.__picked = true; return Promise.resolve(null); };`);
  win.document.getElementById("pull-browse").click();
  await settle();
  assert.equal(win.__picked, false, "nothing decided before the answer");
  answer(ok({ fs_access: "host" }));
  assert.ok(await waitFor(() => win.__picked), "the picker opens once host access is known");
  assert.doesNotMatch(win.document.getElementById("toast-errors").textContent, /Type the path instead/);
});

test("a pull answered with a non-JSON error page names the HTTP status", async () => {
  const { window: win } = setup({
    "/api/models/pull": () => ({ ok: false, status: 502, statusText: "Bad Gateway",
      text: async () => "<html>", json: async () => { throw new SyntaxError("Unexpected token <"); } }),
  });
  runScript(win, `globalThis.__err = null;
    postPull({ spec: "owner/repo" }).catch((e) => { globalThis.__err = e.message; });`);
  await settle();
  assert.equal(win.__err, "HTTP 502 Bad Gateway");
});

test("a successful pull answer that is not JSON is reported, not parsed into a crash", async () => {
  const { window: win } = setup({
    "/api/models/pull": () => ({ ok: true, status: 200, statusText: "OK",
      text: async () => "<html>", json: async () => { throw new SyntaxError("Unexpected token <"); } }),
  });
  runScript(win, `globalThis.__err = null;
    postPull({ spec: "owner/repo" }).catch((e) => { globalThis.__err = e.message; });`);
  await settle();
  assert.match(win.__err || "", /could not be read/);
});

test("Update now that restarts the server raises the reconnect overlay for the new process", async () => {
  const { window: win } = setup({
    "/whoami": () => ok({ instance_id: "old-proc" }),
    "/api/update/apply": () => ok({ applied: true, restarting: true, version: "9.9.9" }),
  });
  const seen = [];
  win.onServerUnreachable = (opts) => { seen.push(opts); };
  await win.updateApply();
  assert.match(win.document.getElementById("update-status").textContent, /Updated to 9\.9\.9/);
  assert.ok(await waitFor(() => seen.length > 0), "the overlay is raised");
  assert.equal(seen.length, 1);
  assert.equal(seen[0].priorInstanceId, "old-proc", "it waits for a process other than the old one");
});

test("Update now that does not restart raises no overlay", async () => {
  const { window: win } = setup({
    "/whoami": () => ok({ instance_id: "old-proc" }),
    "/api/update/apply": () => ok({ applied: true, restarting: false, klass: "setup", version: "9.9.9" }),
  });
  let raised = 0;
  win.onServerUnreachable = () => { raised += 1; };
  await win.updateApply();
  await new Promise((r) => setTimeout(r, 1000));
  assert.equal(raised, 0);
  assert.match(win.document.getElementById("update-status").textContent, /setup\.sh on Linux and macOS/);
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

test("CivitAI 'credit required' reads unknown when CivitAI sends null", () => {
  const { window: win } = setup();
  runScript(win, `globalThis.__rows = [null, true, false].map((v) =>
    discCivitaiRow({ name: "m", allowNoCredit: v }).textContent);`);
  assert.match(win.__rows[0], /credit required: unknown/);
  assert.match(win.__rows[1], /credit required: no/);
  assert.match(win.__rows[2], /credit required: yes/);
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

test("files the server did not take are reported after an upload", async () => {
  const { window: win } = setup({
    "/api/upload": () => ok({ uploaded: [{ name: "one.txt", bytes: 1 }] }),
  });
  const input = win.document.getElementById("upload-input");
  const files = [new win.File(["a"], "one.txt"), new win.File([""], "empty.txt")];
  Object.defineProperty(input, "files", { value: files, configurable: true });
  win.document.getElementById("upload-send").click();
  await settle();
  assert.match(win.document.getElementById("toast-errors").textContent, /1 file was not uploaded/);
});

test("Cancelling Browse in Import from ComfyUI re-opens the dialog with the typed path", async () => {
  const { window: win } = setup();
  runScript(win, `pickDirectory = () => Promise.resolve(null);`);
  win.openImportComfyModal("");
  const input = win.document.querySelector("#modal-body .import-comfy input[type=text]");
  input.value = "D:/ComfyUI";
  [...win.document.querySelectorAll("#modal-body .import-comfy button")]
    .find((b) => /Browse/.test(b.textContent)).click();
  await settle();
  assert.equal(win.document.getElementById("modal").style.display, "flex");
  assert.equal(win.document.querySelector("#modal-body .import-comfy input[type=text]").value, "D:/ComfyUI");
});

test("the logs export with nothing to export says so once, without an error toast", async () => {
  const { window: win } = setup({
    "/api/logs/export": () => ok({ copied: 0, message: "No log files found (server text)." }),
  });
  runScript(win, `pickDirectory = () => Promise.resolve("D:/out");`);
  win.document.getElementById("logs-export").click();
  await settle();
  assert.equal(win.document.getElementById("logs-result").textContent, "No logs found to export.");
  assert.equal(win.document.getElementById("toast-errors").textContent, "");
});
