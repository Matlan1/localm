// SPDX-License-Identifier: AGPL-3.0-or-later
// Native media on the Images and Video pages: the live preview of a generating
// job (streamJob frames, createJobPreview), steps and clip skip in the Images
// Advanced fold, and the component picker's "(other family)" marking.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadApp, loadAppWithPages, runScript } from "./harness.mjs";

const tick = () => new Promise((r) => setTimeout(r, 0));
const settle = async (n = 6) => { for (let i = 0; i < n; i++) await tick(); };

function sseFetch(attempts, extra = () => null) {
  let call = 0;
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ url: u, opts });
    const other = extra(u, opts);
    if (other) return other;
    if (!/\/api\/jobs\/[^/]+\/events$/.test(u)) {
      return { ok: true, status: 200, json: async () => ({}), text: async () => "" };
    }
    const scenario = attempts[Math.min(call, attempts.length - 1)];
    call++;
    const frames = scenario.events.map((ev) => `data: ${JSON.stringify(ev)}\n\n`);
    let idx = 0;
    const enc = new TextEncoder();
    return {
      ok: true, status: 200,
      body: { getReader() {
        return {
          async read() {
            if (idx < frames.length) return { done: false, value: enc.encode(frames[idx++]) };
            if (scenario.thenThrow) { const e = new Error("drop"); e.name = "TypeError"; throw e; }
            return { done: true, value: undefined };
          },
          async cancel() {},
        };
      } },
    };
  };
  return { fetchImpl, calls };
}

const FRAME = (step, data = "AAAA") => ({ type: "frame", data, mime: "image/png", step, steps: 20 });

test("streamJob hands preview frames to onFrame and keeps them out of the line stream", async () => {
  const { fetchImpl } = sseFetch([{ events: [
    { type: "line", text: "a" }, FRAME(3), { type: "line", text: "b" }, FRAME(6),
    { type: "end", status: "done", result: "x.png" }] }]);
  const { window: win } = loadApp({ fetchImpl, shellToken: "tok" });
  const lines = [];
  const frames = [];
  const end = await win.streamJob("j", (l) => lines.push(l), undefined, (f) => frames.push(f.step));
  assert.deepEqual(lines, ["a", "b"]);
  assert.deepEqual(frames, [3, 6]);
  assert.equal(end.status, "done");
});

test("a reconnect does not skip history lines because frames were delivered", async () => {
  // Attempt 1 delivers line a, two frames and line b, then drops. The server
  // replays history (a, b, c) plus only the latest frame on reconnect: c must
  // arrive exactly once and a, b must not repeat.
  const { fetchImpl } = sseFetch([
    { events: [{ type: "line", text: "a" }, FRAME(2), FRAME(4), { type: "line", text: "b" }],
      thenThrow: true },
    { events: [{ type: "line", text: "a" }, { type: "line", text: "b" },
               { type: "line", text: "c" }, FRAME(6),
               { type: "end", status: "done" }] },
  ]);
  const { window: win } = loadApp({ fetchImpl, shellToken: "tok" });
  runScript(win, "JOB_RECONNECT_DELAY_MS = 2;");
  const lines = [];
  await win.streamJob("j", (l) => lines.push(l), undefined, () => {});
  assert.deepEqual(lines, ["a", "b", "c"]);
});

test("a stream with no frame callback still works", async () => {
  const { fetchImpl } = sseFetch([{ events: [FRAME(1), { type: "line", text: "x" },
                                              { type: "end", status: "done" }] }]);
  const { window: win } = loadApp({ fetchImpl, shellToken: "tok" });
  const lines = [];
  const end = await win.streamJob("j", (l) => lines.push(l));
  assert.deepEqual(lines, ["x"]);
  assert.equal(end.status, "done");
});

test("createJobPreview shows the latest frame with its step and clears", () => {
  const { window: win } = loadApp({ fetchImpl: async () => ({ ok: true, json: async () => ({}) }) });
  const mount = win.document.getElementById("img-preview");
  assert.ok(mount && mount.hidden, "the Images page has a hidden preview area");
  const preview = win.createJobPreview("img-preview");
  preview.show(FRAME(6, "QUJD"));
  const img = mount.querySelector("img");
  assert.equal(img.src, "data:image/png;base64,QUJD");
  assert.match(mount.textContent, /step 6 of 20/);
  assert.equal(mount.hidden, false);
  preview.show(FRAME(9, "REVG"));
  assert.equal(mount.querySelectorAll("img").length, 1, "one picture is replaced, not stacked");
  assert.equal(mount.querySelector("img").src, "data:image/png;base64,REVG");
  preview.clear();
  assert.equal(mount.hidden, true);
  assert.equal(mount.children.length, 0);
});

test("createJobPreview ignores a frame without picture data and never trusts the mime", () => {
  const { window: win } = loadApp({ fetchImpl: async () => ({ ok: true, json: async () => ({}) }) });
  const mount = win.document.getElementById("img-preview");
  const preview = win.createJobPreview("img-preview");
  preview.show({ type: "frame", step: 1, steps: 2 });
  preview.show({ type: "frame", data: 5 });
  assert.equal(mount.hidden, true);
  preview.show({ type: "frame", data: "QUJD", mime: "text/html;base64,evil" });
  assert.match(mount.querySelector("img").src, /^data:image\/png;base64,QUJD$/);
});

test("createJobPreview on a page without the area is a no-op", () => {
  const { window: win } = loadApp({ fetchImpl: async () => ({ ok: true, json: async () => ({}) }) });
  const preview = win.createJobPreview("no-such-area");
  preview.show(FRAME(1));
  preview.clear();
});

// ---------------------------------------------------------------- images page

function imagesPage({ backend = "native", defaults = { steps: 20 }, jobEvents } = {}) {
  const extra = (u) => {
    if (u === "/api/imagine") return { ok: true, status: 200, json: async () => ({ job_id: "j1" }) };
    if (u.startsWith("/api/imagine/backend")) {
      return { ok: true, status: 200, json: async () => ({
        choice: backend, active: backend, native: { model: "m", recommended: {} } }) };
    }
    if (u.startsWith("/api/imagine/native-models")) {
      return { ok: true, status: 200, json: async () => ({
        models: [], slots: {}, loras: [], current: {},
        preview: defaults ? { family: "flux", family_label: "FLUX", needs: [], components: {},
                              notes: [], defaults } : null }) };
    }
    return null;
  };
  const { fetchImpl, calls } = sseFetch([{ events: jobEvents || [{ type: "end", status: "failed" }] }], extra);
  const { window } = loadAppWithPages({ fetchImpl, shellToken: "tok" });
  return { window, calls };
}

const nativeFields = (win) => [...win.document.querySelectorAll("#view-images .img-native-only")];

test("the Images fold shows steps and clip skip only for the native backend", async () => {
  const native = imagesPage({ backend: "native" });
  await native.window.refreshImageBackend();
  assert.equal(nativeFields(native.window).length, 2);
  assert.ok(nativeFields(native.window).every((n) => !n.hidden));
  const comfy = imagesPage({ backend: "comfy" });
  await comfy.window.refreshImageBackend();
  assert.ok(nativeFields(comfy.window).every((n) => n.hidden));
});

test("steps placeholder shows the picked model family's default", async () => {
  const { window: win } = imagesPage({ defaults: { steps: 28 } });
  await win.refreshImageBackend();
  await settle();
  assert.equal(win.document.getElementById("img-steps").placeholder, "default (28)");
});

test("steps placeholder falls back to the plain wording without a family default", async () => {
  const { window: win } = imagesPage({ defaults: null });
  await win.refreshImageBackend();
  await settle();
  assert.equal(win.document.getElementById("img-steps").placeholder, "model default");
});

test("steps and clip skip are sent to POST /api/imagine for the native backend", async () => {
  const { window: win, calls } = imagesPage();
  await win.refreshImageBackend();
  win.document.getElementById("img-prompt").value = "a fox";
  win.document.getElementById("img-steps").value = "12";
  win.document.getElementById("img-clip-skip").value = "2";
  win.document.getElementById("img-generate").onclick();
  await settle();
  const body = JSON.parse(calls.find((c) => c.url === "/api/imagine").opts.body);
  assert.equal(body.steps, 12);
  assert.equal(body.clip_skip, 2);
});

test("blank steps and clip skip are sent as null", async () => {
  const { window: win, calls } = imagesPage();
  await win.refreshImageBackend();
  win.document.getElementById("img-prompt").value = "a fox";
  win.document.getElementById("img-generate").onclick();
  await settle();
  const body = JSON.parse(calls.find((c) => c.url === "/api/imagine").opts.body);
  assert.equal(body.steps, null);
  assert.equal(body.clip_skip, null);
});

test("ComfyUI requests do not carry steps or clip skip", async () => {
  const { window: win, calls } = imagesPage({ backend: "comfy" });
  await win.refreshImageBackend();
  win.document.getElementById("img-prompt").value = "a fox";
  win.document.getElementById("img-steps").value = "12";
  win.document.getElementById("img-generate").onclick();
  await settle();
  const body = JSON.parse(calls.find((c) => c.url === "/api/imagine").opts.body);
  assert.ok(!("steps" in body) && !("clip_skip" in body));
});

test("the preview shows while the job runs and is gone when it ends", async () => {
  const { window: win } = imagesPage({ jobEvents: [
    { type: "line", text: "Step 3/20" }, FRAME(3), { type: "end", status: "failed" }] });
  await win.refreshImageBackend();
  win.document.getElementById("img-prompt").value = "a fox";
  const seen = [];
  const mount = win.document.getElementById("img-preview");
  new win.MutationObserver(() => { if (!mount.hidden && mount.querySelector("img")) seen.push(mount.textContent); })
    .observe(mount, { childList: true, attributes: true, subtree: true });
  win.document.getElementById("img-generate").onclick();
  await settle(12);
  assert.ok(seen.some((t) => /step 3 of 20/.test(t)), "the frame was shown during the job");
  assert.equal(mount.hidden, true, "and removed once the job ended");
});

// ------------------------------------------------------------- picker marking

test("components of another family are listed last and marked", () => {
  const { window: win } = loadApp({ fetchImpl: async () => ({ ok: true, json: async () => ({}) }) });
  const sel = win.nativeModelSelect([
    { name: "ae", file: "ae.safetensors", family_label: "FLUX", fits: true },
    { name: "wan_vae", file: "wan_vae.safetensors", family_label: "Wan", fits: false },
    { name: "sdxl_vae", file: "sdxl_vae.safetensors", family_label: "SDXL", fits: false },
  ], "", "Automatic");
  const texts = [...sel.options].map((o) => o.textContent);
  assert.deepEqual(texts, [
    "Automatic",
    "ae (ae.safetensors)",
    "wan_vae (wan_vae.safetensors) (other family)",
    "sdxl_vae (sdxl_vae.safetensors) (other family)",
  ]);
  assert.deepEqual([...sel.options].map((o) => o.value), ["", "ae", "wan_vae", "sdxl_vae"]);
});

test("a list without fit information is grouped by family as before", () => {
  const { window: win } = loadApp({ fetchImpl: async () => ({ ok: true, json: async () => ({}) }) });
  const sel = win.nativeModelSelect([
    { name: "a", file: "a.gguf", family_label: "FLUX" },
    { name: "b", file: "b.gguf", family_label: "Wan" },
  ], "b", "Automatic");
  assert.equal(sel.querySelectorAll("optgroup").length, 2);
  assert.equal(sel.value, "b");
  assert.ok(![...sel.options].some((o) => /other family/.test(o.textContent)));
});

test("a saved choice that is an other-family component stays selected", () => {
  const { window: win } = loadApp({ fetchImpl: async () => ({ ok: true, json: async () => ({}) }) });
  const sel = win.nativeModelSelect([
    { name: "ae", file: "ae.safetensors", family_label: "FLUX", fits: true },
    { name: "wan_vae", file: "wan_vae.safetensors", family_label: "Wan", fits: false },
  ], "wan_vae", "Automatic");
  assert.equal(sel.value, "wan_vae");
  assert.equal([...sel.options].filter((o) => o.value === "wan_vae").length, 1);
});
