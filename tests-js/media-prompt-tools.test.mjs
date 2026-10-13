// SPDX-License-Identifier: AGPL-3.0-or-later
// "Enhance prompt" on the Images, Video and Music pages (app/native-models.js),
// "Import parameters from image" on the Images page (pages/images.js), and the
// Music picker's list of checkpoints the native backend cannot run.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadAppWithPages, runScript } from "./harness.mjs";

const IMAGE_OPTIONS = {
  models: [
    { name: "fluxunchained", file: "fluxunchained-dev-q8-0-v2.gguf", family: "flux", family_label: "FLUX.1" },
    { name: "dreamshaper_8", file: "dreamshaper_8.safetensors", family: "sd1", family_label: "Stable Diffusion 1.x" },
  ],
  slots: { clip_l: [], clip_g: [], llm: [], t5xxl: [], vae: [] },
  loras: [{ name: "detail", file: "detail.safetensors" }],
  current: { model: "", clip_l: "", clip_g: "", t5xxl: "", llm: "", vae: "" },
};

function previewFor(model) {
  if (model === "dreamshaper_8") return { model, kind: "checkpoint", family: "sd1", family_label: "Stable Diffusion 1.x", needs: [], components: {}, notes: [] };
  if (model) return { model, kind: "diffusion", family: "flux", family_label: "FLUX.1", needs: [], components: {}, notes: [] };
  return null;
}

function makeApp({ active = "qwen3", reply, chatStatus = 200, importResult, importStatus = 200 } = {}) {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    calls.push({ u, opts });
    const ok = (body, status = 200) => ({ ok: status < 400, status, statusText: "ERR", json: async () => body, text: async () => "" });
    if (u.startsWith("/api/models")) return ok({ models: active ? [{ name: active }] : [], active });
    if (u.includes("/v1/chat/completions")) {
      if (chatStatus !== 200) return ok({ detail: "model is busy" }, chatStatus);
      return ok({ choices: [{ message: { content: reply ?? "an enhanced prompt" } }] });
    }
    if (u.includes("/api/imagine/import-params")) return ok(importResult || {}, importStatus);
    if (u.includes("/api/music/native-models")) {
      return ok({ components: { dit: [], text_encoder: [], vae: [], lm: [] },
                  unsupported: [{ name: "ace_step_v1_3.5b.safetensors", file: "ace_step_v1_3.5b.safetensors",
                                  reason: "ACE-Step v1: ComfyUI only" }],
                  current: {} });
    }
    if (u.includes("/native-models")) {
      const model = new URL(u, "http://x").searchParams.get("model");
      return ok({ ...IMAGE_OPTIONS, preview: previewFor(model) });
    }
    if (u.includes("/api/imagine/backend")) return ok({ choice: "auto", active: "native", native: { runtime: "vulkan" } });
    return ok({ models: [], active: "", conversations: [], plugins: [] });
  };
  const { window: win } = loadAppWithPages({ fetchImpl });
  return { win, calls, $: (id) => win.document.getElementById(id) };
}

const settle = (ms = 20) => new Promise((r) => setTimeout(r, ms));
const chatCalls = (calls) => calls.filter((c) => c.u.includes("/v1/chat/completions"));
const enhanceBtn = ($, field) => $(field).nextElementSibling.querySelector("button");

async function click(win, btn) {
  btn.click();
  for (let i = 0; i < 20; i++) { await settle(10); if (!btn.disabled) break; }
}

test("Enhance rewrites the prompt through the loaded chat model and Undo restores it", async () => {
  const { win, calls, $ } = makeApp({ reply: "\"a watercolor fox, soft light\"" });
  $("img-prompt").value = "a fox";
  await click(win, enhanceBtn($, "img-prompt"));
  assert.equal($("img-prompt").value, "a watercolor fox, soft light");
  const sent = JSON.parse(chatCalls(calls)[0].opts.body);
  assert.equal(sent.model, "qwen3");
  assert.equal(sent.stream, false);
  assert.equal(sent.pin_model, true);
  assert.equal(sent.messages[0].role, "system");
  assert.equal(sent.messages[1].content, "a fox");
  assert.equal(sent.messages[1].origin, "client");
  const undo = $("img-prompt").nextElementSibling.querySelectorAll("button")[1];
  assert.equal(undo.hidden, false);
  undo.click();
  assert.equal($("img-prompt").value, "a fox");
  assert.equal(undo.hidden, true);
});

test("Enhance with no chat model loaded fails loudly and leaves the prompt alone", async () => {
  const { win, calls, $ } = makeApp({ active: "" });
  $("img-prompt").value = "a fox";
  await click(win, enhanceBtn($, "img-prompt"));
  assert.equal($("img-prompt").value, "a fox");
  assert.equal(chatCalls(calls).length, 0);
  assert.match($("toast").textContent, /No chat model is loaded/);
  assert.ok($("toast").className.includes("error"));
});

test("a chat error and an empty reply are reported and leave the prompt alone", async () => {
  const bad = makeApp({ chatStatus: 503 });
  bad.$("img-prompt").value = "a fox";
  await click(bad.win, enhanceBtn(bad.$, "img-prompt"));
  assert.equal(bad.$("img-prompt").value, "a fox");
  assert.match(bad.$("toast").textContent, /model is busy/);
  const empty = makeApp({ reply: "<think>hmm</think>   " });
  empty.$("img-prompt").value = "a fox";
  await click(empty.win, enhanceBtn(empty.$, "img-prompt"));
  assert.equal(empty.$("img-prompt").value, "a fox");
  assert.match(empty.$("toast").textContent, /empty reply/);
});

test("an empty prompt is refused before any model call", async () => {
  const { win, calls, $ } = makeApp();
  $("img-prompt").value = "   ";
  await click(win, enhanceBtn($, "img-prompt"));
  assert.equal(chatCalls(calls).length, 0);
  assert.match($("toast").textContent, /short prompt first/);
});

test("the system prompt follows the model family: tags for SD, sentences for FLUX", async () => {
  const { win } = makeApp();
  runScript(win, `globalThis.__s = {
    sd: enhanceSystemPrompt("image", "sd1", "Stable Diffusion 1.x"),
    flux: enhanceSystemPrompt("image", "flux", "FLUX.1"),
    unknown: enhanceSystemPrompt("image", "", ""),
    video: enhanceSystemPrompt("video", "wan2.2", "Wan 2.2"),
    music: enhanceSystemPrompt("music", "", ""),
  };`);
  const s = win.__s;
  assert.match(s.sd, /comma-separated/);
  assert.match(s.sd, /Stable Diffusion 1\.x/);
  assert.doesNotMatch(s.flux, /comma-separated phrases/);
  assert.match(s.flux, /natural language/);
  assert.match(s.flux, /FLUX\.1/);
  assert.match(s.unknown, /natural language/);
  assert.match(s.video, /motion/);
  assert.match(s.video, /Wan 2\.2/);
  assert.match(s.music, /ACE-Step/);
  assert.match(s.music, /bpm/);
});

test("the image page sends the picked model's family to the rewrite", async () => {
  const { win, calls, $ } = makeApp();
  runScript(win, "globalThis.__p = (async () => { await refreshImageBackend(); await imageModelPicker.setModels({ model: 'dreamshaper_8' }); })();");
  await win.__p;
  $("img-prompt").value = "a fox";
  await click(win, enhanceBtn($, "img-prompt"));
  const sent = JSON.parse(chatCalls(calls)[0].opts.body);
  assert.match(sent.messages[0].content, /Stable Diffusion 1\.x/);
  assert.match(sent.messages[0].content, /comma-separated/);
});

test("Video and Music pages have the button, and Music enhances the style tags only", async () => {
  const { win, calls, $ } = makeApp();
  assert.ok(enhanceBtn($, "video-prompt"));
  $("music-lyrics").value = "[verse] la la";
  $("music-tags").value = "lofi";
  await click(win, enhanceBtn($, "music-tags"));
  assert.equal($("music-tags").value, "an enhanced prompt");
  assert.equal($("music-lyrics").value, "[verse] la la");
  assert.match(JSON.parse(chatCalls(calls)[0].opts.body).messages[0].content, /ACE-Step/);
});

test("cleanEnhancedPrompt strips reasoning, fences and quotes only", async () => {
  const { win } = makeApp();
  runScript(win, `globalThis.__c = [
    cleanEnhancedPrompt("<think>x</think>\\nfoo, bar"),
    cleanEnhancedPrompt("\`\`\`\\nfoo bar\\n\`\`\`"),
    cleanEnhancedPrompt("\\u201cfoo bar\\u201d"),
    cleanEnhancedPrompt("it's a \\"quoted\\" word"),
    cleanEnhancedPrompt(null),
  ];`);
  assert.deepEqual([...win.__c], ["foo, bar", "foo bar", "foo bar", "it's a \"quoted\" word", ""]);
});

async function importWith(result, { picker = true } = {}) {
  const app = makeApp({ importResult: result });
  runScript(app.win, `globalThis.__p = (async () => { await refreshImageBackend(); ${picker ? "await imageModelPicker.refresh();" : ""} })();`);
  await app.win.__p;
  runScript(app.win, `globalThis.__r = (async () => {
    const f = new File([new Uint8Array([1, 2, 3])], "x.png", { type: "image/png" });
    return await importParamsFromFile(f);
  })().catch((e) => ({ error: e.message }));`);
  await app.win.__r;
  return app;
}

test("importing a localm image fills the fields, the model and the LoRAs", async () => {
  const { win, calls, $ } = await importWith({
    source: "localm",
    params: { prompt: "a harbour", negative_prompt: "blur", seed: 77, cfg: 4, guidance: 3.5,
              width: 640, height: 480, model: "fluxunchained", loras: [{ name: "detail", strength: 0.6 }] },
  });
  const post = calls.find((c) => c.u.includes("import-params"));
  assert.equal(post.opts.method, "POST");
  assert.equal(post.opts.headers["Content-Type"], "application/octet-stream");
  assert.equal($("img-prompt").value, "a harbour");
  assert.equal($("img-negative").value, "blur");
  assert.equal($("img-seed").value, "77");
  assert.equal($("img-cfg").value, "4");
  assert.equal($("img-size").value, "640x480");
  runScript(win, "globalThis.__c = JSON.stringify(imageModelPicker.choice());");
  assert.deepEqual(JSON.parse(win.__c), {
    native_models: { model: "fluxunchained" }, loras: [{ name: "detail", strength: 0.6 }] });
  assert.match($("toast").textContent, /Imported parameters from a localm image\./);
});

test("importing a foreign image matches the model by file stem and reports what it could not apply", async () => {
  const { win, $ } = await importWith({
    source: "a1111",
    params: { prompt: "1girl", seed: 5, steps: 30, sampler: "Euler a", model: "dreamshaper_8",
              loras: [{ name: "not-installed", strength: 1 }] },
  });
  runScript(win, "globalThis.__c = JSON.stringify(imageModelPicker.choice());");
  assert.deepEqual(JSON.parse(win.__c), { native_models: { model: "dreamshaper_8" } });
  const toast = $("toast").textContent;
  assert.match(toast, /A1111\/Forge/);
  assert.match(toast, /Not applied: .*steps/);
  assert.match(toast, /sampler/);
  assert.match(toast, /LoRA not-installed/);
});

test("an unmatched model is reported, not silently dropped", async () => {
  const { win, $ } = await importWith({
    source: "comfyui", params: { prompt: "x", model: "sd_xl_base_1.0.safetensors" } });
  runScript(win, "globalThis.__c = JSON.stringify(imageModelPicker.choice());");
  assert.deepEqual(JSON.parse(win.__c), {});
  assert.match($("toast").textContent, /model sd_xl_base_1\.0\.safetensors/);
});

test("a refused import shows the server's reason and changes nothing", async () => {
  const app = makeApp({ importResult: { detail: "Cannot import parameters: not a PNG image" }, importStatus: 422 });
  app.$("img-prompt").value = "keep me";
  runScript(app.win, `globalThis.__r = (async () => {
    try { await importParamsFromFile(new File([new Uint8Array([1])], "x.gif")); return "ok"; }
    catch (e) { return e.message; }
  })();`);
  assert.match(await app.win.__r, /not a PNG image/);
  assert.equal(app.$("img-prompt").value, "keep me");
});

test("the Music picker lists checkpoints it cannot run, with the reason", async () => {
  const { win, $ } = makeApp();
  runScript(win, "globalThis.__p = musicModelPicker.refresh();");
  await win.__p;
  const text = $("music-native-models").querySelector(".native-models-unsupported").textContent;
  assert.match(text, /Not runnable by the native backend/);
  assert.match(text, /ace_step_v1_3\.5b\.safetensors: ACE-Step v1: ComfyUI only/);
  assert.equal($("music-native-models").querySelectorAll("select").length, 4);
});

test("a foreign image without prompt text keeps the user's prompt and says it was not applied", async () => {
  const app = makeApp({ importResult: { source: "comfyui", params: { seed: 5, width: 64, height: 64 }, notes: [] } });
  app.$("img-prompt").value = "keep me";
  app.$("img-negative").value = "keep neg";
  runScript(app.win, `globalThis.__p = (async () => { await refreshImageBackend(); await imageModelPicker.refresh(); })();`);
  await app.win.__p;
  runScript(app.win, `globalThis.__r = importParamsFromFile(new File([new Uint8Array([1])], "x.png"));`);
  await app.win.__r;
  assert.equal(app.$("img-prompt").value, "keep me");
  assert.equal(app.$("img-negative").value, "keep neg");
  assert.equal(app.$("img-seed").value, "5");
  assert.match(app.$("toast").textContent, /Not applied: prompt/);
});

test("a localm record clears the fields it does not hold, and finds its model by file name", async () => {
  const app = makeApp({ importResult: { source: "localm",
    params: { prompt: "p", seed: 3, model: "fluxunchained-dev-q8-0-v2.gguf" }, notes: [] } });
  app.$("img-negative").value = "old negative";
  app.$("img-cfg").value = "9";
  runScript(app.win, `globalThis.__p = (async () => { await refreshImageBackend(); await imageModelPicker.refresh(); })();`);
  await app.win.__p;
  runScript(app.win, `globalThis.__r = importParamsFromFile(new File([new Uint8Array([1])], "x.png"));`);
  await app.win.__r;
  assert.equal(app.$("img-negative").value, "");
  assert.equal(app.$("img-cfg").value, "");
  runScript(app.win, "globalThis.__c = JSON.stringify(imageModelPicker.choice());");
  assert.deepEqual(JSON.parse(app.win.__c), { native_models: { model: "fluxunchained" } });
  assert.doesNotMatch(app.$("toast").textContent, /Not applied/);
});

test("an unregistered model of a localm record is reported and does not wipe the picker", async () => {
  const app = makeApp({ importResult: { source: "localm", params: { prompt: "p", model: "gone-model.gguf" }, notes: [] } });
  runScript(app.win, `globalThis.__p = (async () => { await refreshImageBackend(); await imageModelPicker.setModels({ model: 'dreamshaper_8' }); })();`);
  await app.win.__p;
  runScript(app.win, `globalThis.__r = importParamsFromFile(new File([new Uint8Array([1])], "x.png"));`);
  await app.win.__r;
  runScript(app.win, "globalThis.__c = JSON.stringify(imageModelPicker.choice());");
  assert.deepEqual(JSON.parse(app.win.__c), { native_models: { model: "dreamshaper_8" } });
  assert.match(app.$("toast").textContent, /model gone-model\.gguf/);
});
