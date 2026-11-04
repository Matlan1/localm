// SPDX-License-Identifier: AGPL-3.0-or-later
// A native media model that is not downloaded is offered in the usual download
// box, never as a command to run: several missing files share ONE box and one
// confirmation, declining downloads nothing, and a job that stopped for a
// missing file offers the same box from its "missing_model" event.
import { test as nodeTest } from "node:test";
import assert from "node:assert/strict";
import { loadApp } from "./harness.mjs";

const test = (name, fn) => nodeTest(name, { timeout: 15000 }, fn);

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function until(cond, what) {
  for (let i = 0; i < 200; i++) {
    if (cond()) return;
    await wait(10);
  }
  assert.fail(`timed out waiting for ${what}`);
}

function sse(events) {
  const bytes = new TextEncoder().encode(
    events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join(""));
  let sent = false;
  return {
    ok: true, status: 200,
    body: { getReader: () => ({ async read() {
      if (sent) return { done: true, value: undefined };
      sent = true;
      return { done: false, value: bytes };
    } }) },
  };
}

const PARTS = [
  { filename: "wan2.1_t2v_1.3B_fp16.safetensors", native: true, searchable: false,
    source: { repo: "Comfy-Org/Wan-2.1", file: "split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
              spec: "Comfy-Org/Wan-2.1:split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
              sha256: "be531024cd9018cb5b48c40cfbb6a6191645b1c792eb8bf4f8c1c6e10f924dc5",
              name: null, size_bytes: 2838303560, model_type: "diffusion-unet", origin: "curated" } },
  { filename: "wan_2.1_vae.safetensors", native: true, searchable: false,
    source: { repo: "Comfy-Org/Wan-2.1", file: "split_files/vae/wan_2.1_vae.safetensors",
              spec: "Comfy-Org/Wan-2.1:split_files/vae/wan_2.1_vae.safetensors",
              sha256: "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
              name: null, size_bytes: 253815318, model_type: "vae", origin: "curated" } },
  { filename: "umt5-xxl-encoder-Q4_K_M.gguf", native: true, searchable: false,
    source: { repo: "city96/umt5-xxl-encoder-gguf", file: "umt5-xxl-encoder-Q4_K_M.gguf",
              spec: "city96/umt5-xxl-encoder-gguf:umt5-xxl-encoder-Q4_K_M.gguf",
              sha256: "17cf97a5bbbc60a646d6105b832b6f657ce904a8a1ad970e4b59df0c67584a40",
              name: null, size_bytes: 3655145312, model_type: "text-encoder", origin: "curated" } },
];

function makeFetch({ missing, pulls, pullStatus = "done", pullOk = true }) {
  return async (url, opts = {}) => {
    const method = opts.method || "GET";
    if (String(url) === "/api/media/video/preflight" && method === "POST") {
      return { ok: true, status: 200,
               json: async () => ({ status: "verified", missing, warning: null }) };
    }
    if (String(url) === "/api/models/pull" && method === "POST") {
      pulls.push(JSON.parse(opts.body));
      if (!pullOk) {
        return { ok: false, status: 403, statusText: "Forbidden",
                 json: async () => ({ detail: "network access is off" }) };
      }
      return { ok: true, status: 200, json: async () => ({ job_id: `job-${pulls.length}` }) };
    }
    const m = /^\/api\/jobs\/(job-\d+)\/events$/.exec(String(url));
    if (m) {
      return sse([{ type: "line", text: `pulling ${m[1]}` },
                  { type: "end", status: pullStatus }]);
    }
    return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
  };
}

const modalOpen = (win) => win.document.getElementById("modal").style.display !== "none";
const buttons = (win) => [...win.document.querySelectorAll("#modal-body button")];
const click = (win, label) => buttons(win).find((b) => b.textContent === label).click();

function logEl(win) {
  const log = win.document.createElement("pre");
  win.document.body.appendChild(log);
  return log;
}

test("several missing native files share one box and one confirmation", async () => {
  const pulls = [];
  const { window: win } = loadApp({ fetchImpl: makeFetch({ missing: PARTS, pulls }) });
  const log = logEl(win);
  const result = win.checkModelsBeforeGenerate("video", log, {});
  await until(() => modalOpen(win), "the download box");
  const text = win.document.getElementById("modal-body").textContent;
  for (const p of PARTS) assert.ok(text.includes(p.filename), `lists ${p.filename}`);
  assert.match(win.document.getElementById("modal-title").textContent, /3 files/);
  assert.match(text, /GB in total/);
  assert.doesNotMatch(text, /localm pull/);
  assert.deepEqual(buttons(win).map((b) => b.textContent), ["Not now", "Download"]);

  click(win, "Download");
  assert.equal(await result, true, "generation goes ahead once every file is downloaded");
  assert.equal(pulls.length, 3);
  assert.deepEqual(pulls.map((p) => p.sha256), PARTS.map((p) => p.source.sha256));
  assert.deepEqual(pulls.map((p) => p.spec), PARTS.map((p) => p.source.spec));
  assert.deepEqual(pulls.map((p) => p.model_type), ["diffusion-unet", "vae", "text-encoder"]);
  assert.ok(pulls.every((p) => p.type_is_hint === true));
  assert.match(log.textContent, /Downloading 1 of 3: wan2\.1_t2v/);
  assert.match(log.textContent, /Downloading 3 of 3: umt5-xxl/);
  assert.match(log.textContent, /pulling job-1[\s\S]*pulling job-2[\s\S]*pulling job-3/);
  assert.doesNotMatch(log.textContent, /localm pull/);
});

test("declining downloads nothing and does not start the generation", async () => {
  const pulls = [];
  const { window: win } = loadApp({ fetchImpl: makeFetch({ missing: PARTS, pulls }) });
  const log = logEl(win);
  const result = win.checkModelsBeforeGenerate("video", log, {});
  await until(() => modalOpen(win), "the download box");
  click(win, "Not now");
  assert.equal(await result, false);
  assert.equal(pulls.length, 0);
  assert.match(log.textContent, /Nothing was downloaded/);
  assert.doesNotMatch(log.textContent, /localm pull/);
  assert.equal(modalOpen(win), false);
});

test("closing the box counts as declining", async () => {
  const pulls = [];
  const { window: win } = loadApp({ fetchImpl: makeFetch({ missing: PARTS, pulls }) });
  const result = win.checkModelsBeforeGenerate("video", logEl(win), {});
  await until(() => modalOpen(win), "the download box");
  win.document.getElementById("modal").style.display = "none";
  assert.equal(await result, false);
  assert.equal(pulls.length, 0);
});

test("a download that does not finish stops the rest and does not start the generation", async () => {
  const pulls = [];
  const { window: win } = loadApp({
    fetchImpl: makeFetch({ missing: PARTS, pulls, pullStatus: "failed" }) });
  const result = win.checkModelsBeforeGenerate("video", logEl(win), {});
  await until(() => modalOpen(win), "the download box");
  click(win, "Download");
  assert.equal(await result, false);
  assert.equal(pulls.length, 1, "later files are not started after one fails");
});

test("a refused pull is reported and does not start the generation", async () => {
  const pulls = [];
  const { window: win } = loadApp({
    fetchImpl: makeFetch({ missing: PARTS, pulls, pullOk: false }) });
  const result = win.checkModelsBeforeGenerate("video", logEl(win), {});
  await until(() => modalOpen(win), "the download box");
  click(win, "Download");
  assert.equal(await result, false);
  assert.match(win.document.getElementById("toast").textContent, /network access is off/);
  assert.equal(modalOpen(win), false);
});

test("with network access off it says why and offers no box", async () => {
  const pulls = [];
  const blocked = PARTS.map((p) => ({ ...p, blocked: "net_mode_off" }));
  const { window: win } = loadApp({ fetchImpl: makeFetch({ missing: blocked, pulls }) });
  const log = logEl(win);
  const result = await win.checkModelsBeforeGenerate("video", log, {});
  assert.equal(result, false);
  assert.equal(modalOpen(win), false);
  assert.equal(pulls.length, 0);
  assert.match(log.textContent, /network access is off/);
  assert.match(log.textContent, /wan_2\.1_vae\.safetensors/);
});

test("a single missing native file keeps its own wording", async () => {
  const pulls = [];
  const { window: win } = loadApp({ fetchImpl: makeFetch({ missing: [PARTS[1]], pulls }) });
  const result = win.checkModelsBeforeGenerate("video", logEl(win), {});
  await until(() => modalOpen(win), "the download box");
  assert.match(win.document.getElementById("modal-body").textContent,
    /Generating needs 'wan_2\.1_vae\.safetensors' \(.+\), which isn't downloaded yet\./);
  click(win, "Download");
  assert.equal(await result, true);
  assert.equal(pulls.length, 1);
});

test("a job that stopped for a missing file offers the same box and hides the command line", async () => {
  const pulls = [];
  const { window: win } = loadApp({ fetchImpl: makeFetch({ missing: [], pulls }) });
  const log = logEl(win);
  const command = "No native video model is set up: localm pull Comfy-Org/Wan-2.1:x --sha256 abc";
  const offer = win.createMissingModelOffer(log);
  offer.onEvent({ type: "missing_model", missing: PARTS, text: command });
  assert.equal(offer.show(command), false, "the line that names the command is left out");
  assert.equal(offer.show("Video backend: native"), true);
  const done = offer.offer();
  await until(() => modalOpen(win), "the download box");
  assert.match(win.document.getElementById("modal-title").textContent, /3 files/);
  click(win, "Download");
  assert.equal(await done, "done");
  assert.equal(pulls.length, 3);
  assert.match(win.document.getElementById("toast").textContent, /Press Generate again/);
  assert.equal(await offer.offer(), null, "an offer is made once");
});

test("a job that named nothing missing offers nothing", async () => {
  const { window: win } = loadApp({ fetchImpl: makeFetch({ missing: [], pulls: [] }) });
  const offer = win.createMissingModelOffer(logEl(win));
  offer.onEvent({ type: "line", text: "x" });
  offer.onEvent({ type: "missing_model", missing: [], text: "y" });
  assert.equal(offer.show("y"), true);
  assert.equal(await offer.offer(), null);
  assert.equal(modalOpen(win), false);
});

function slashFetch({ kind, missing, posts, jobEvents }) {
  const base = makeFetch({ missing: [], pulls: [] });
  const api = { video: "/api/video", image: "/api/imagine", music: "/api/music" }[kind];
  return async (url, opts = {}) => {
    const u = String(url);
    if (u === `/api/media/${kind}/preflight`) {
      return { ok: true, status: 200,
               json: async () => ({ status: "verified", missing, warning: null }) };
    }
    if (u === api && (opts.method || "GET") === "POST") {
      posts.push(JSON.parse(opts.body));
      return { ok: true, status: 200, json: async () => ({ job_id: "job-9" }) };
    }
    if (u === "/api/jobs/job-9/events") return sse(jobEvents);
    return base(url, opts);
  };
}

for (const [kind, run] of [["video", "runVideoInChat"], ["image", "runImagineInChat"],
                           ["music", "runMusicInChat"]]) {
  test(`/${kind} in chat offers the download box and stops when it is declined`, async () => {
    const posts = [];
    const { window: win } = loadApp({
      fetchImpl: slashFetch({ kind, missing: PARTS, posts, jobEvents: [] }) });
    const done = win[run]("a fox");
    await until(() => modalOpen(win), "the download box");
    assert.match(win.document.getElementById("modal-title").textContent, /3 files/);
    click(win, "Not now");
    await done;
    assert.equal(posts.length, 0, "no job is started");
    assert.match(win.document.getElementById("chat-messages").textContent,
      /Not generated: the model files it needs are not downloaded/);
  });

  test(`/${kind} in chat keeps a command line out of the reply and offers the box`, async () => {
    const posts = [];
    const command = "No native model is set up: localm pull owner/repo:x --sha256 abc";
    const { window: win } = loadApp({
      fetchImpl: slashFetch({ kind, missing: [], posts, jobEvents: [
        { type: "missing_model", missing: PARTS, text: command },
        { type: "line", text: command },
        { type: "end", status: "failed" }] }) });
    const box = win.document.getElementById("chat-messages");
    const seen = [];
    new win.MutationObserver(() => seen.push(box.textContent))
      .observe(box, { subtree: true, childList: true, characterData: true });
    const done = win[run]("a fox");
    await until(() => modalOpen(win), "the download box");
    assert.equal(posts.length, 1);
    click(win, "Not now");
    await done;
    await wait(20);
    assert.ok(seen.length > 0, "the observer saw the reply change");
    assert.ok(seen.every((text) => !/localm pull/.test(text)), "no reply text ever named the command");
  });
}
