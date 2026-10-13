// SPDX-License-Identifier: AGPL-3.0-or-later
// The default Kokoro model and its voices are fetched by the BROWSER from
// Hugging Face at a pinned commit, not at the repository's moving default
// branch (localm/plugins/builtin/tts/static/{tts-util.js,tts.js} and the
// vendored kokoro.min.js).

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const STATIC = join(HERE, "..", "localm", "plugins", "builtin", "tts", "static");
const UTIL = join(STATIC, "tts-util.js");
const TTS_JS = join(STATIC, "tts.js");
const BUNDLE = join(STATIC, "vendor", "kokoro.min.js");
const { KOKORO_DEFAULT_MODEL, KOKORO_PINNED_REVISION, pinnedRemotePathTemplate, rekeyCachedModel } =
  await import(pathToFileURL(UTIL).href);

const FAKE_LIBRARY = "data:text/javascript," + encodeURIComponent(`
  export const env = { backends: { onnx: { wasm: {} } },
                       remotePathTemplate: "{model}/resolve/{revision}/" };
  globalThis.__kokoroEnv = env;
  export class TextSplitterStream {}
  export class KokoroTTS {
    static async from_pretrained(model) {
      globalThis.__kokoroLoad = { model, template: env.remotePathTemplate };
      return {};
    }
  }
`);

test("the pinned revision is a full commit id and the template names it", () => {
  assert.match(KOKORO_PINNED_REVISION, /^[0-9a-f]{40}$/);
  assert.equal(KOKORO_DEFAULT_MODEL, "onnx-community/Kokoro-82M-v1.0-ONNX");
  assert.equal(pinnedRemotePathTemplate(KOKORO_DEFAULT_MODEL),
    `{model}/resolve/${KOKORO_PINNED_REVISION}/`);
});

test("a model other than the default is not pinned", () => {
  assert.equal(pinnedRemotePathTemplate("someone/else-ONNX"), null);
  assert.equal(pinnedRemotePathTemplate(undefined), null);
});

class FakeCache {
  constructor(entries) { this.map = new Map(Object.entries(entries)); }
  async keys() { return [...this.map.keys()].map((url) => ({ url })); }
  async match(req) { const v = this.map.get(typeof req === "string" ? req : req.url); return v === undefined ? undefined : { clone: () => v }; }
  async put(url, res) { this.map.set(url, res); }
  async delete(req) { return this.map.delete(typeof req === "string" ? req : req.url); }
}

test("a model cached under the default branch is moved to the pinned commit, once", async () => {
  const base = `https://huggingface.co/${KOKORO_DEFAULT_MODEL}/resolve/`;
  const cache = new FakeCache({
    [`${base}main/onnx/model.onnx`]: "weights", [`${base}main/config.json`]: "cfg",
    [`${base}other/config.json`]: "unrelated", "https://example.com/x": "other host" });
  assert.equal(await rekeyCachedModel(cache, KOKORO_DEFAULT_MODEL, KOKORO_PINNED_REVISION), 2);
  const pinned = `${base}${KOKORO_PINNED_REVISION}/`;
  assert.deepEqual([...cache.map.keys()].sort(), [
    `${base}other/config.json`, `${pinned}config.json`, `${pinned}onnx/model.onnx`,
    "https://example.com/x"].sort());
  assert.equal(cache.map.get(`${pinned}onnx/model.onnx`), "weights");
  assert.equal(await rekeyCachedModel(cache, KOKORO_DEFAULT_MODEL, KOKORO_PINNED_REVISION), 0);
});

test("a file already cached at the pinned commit is not overwritten by a main entry", async () => {
  const base = `https://huggingface.co/${KOKORO_DEFAULT_MODEL}/resolve/`;
  const cache = new FakeCache({
    [`${base}main/config.json`]: "old", [`${base}${KOKORO_PINNED_REVISION}/config.json`]: "pinned" });
  assert.equal(await rekeyCachedModel(cache, KOKORO_DEFAULT_MODEL, KOKORO_PINNED_REVISION), 0);
  assert.equal(cache.map.get(`${base}${KOKORO_PINNED_REVISION}/config.json`), "pinned");
  assert.equal(cache.map.has(`${base}main/config.json`), false);
});

test("the vendored bundle fetches the voices at the pinned commit and caches them under the unchanged key", () => {
  const src = readFileSync(BUNDLE, "utf8");
  assert.ok(src.includes(`.replace("/resolve/main/","/resolve/${KOKORO_PINNED_REVISION}/")`),
    "kokoro.min.js must fetch the voices at the pinned commit (a re-vendored bundle needs the patch re-applied; see vendor/NOTICE.md)");
  assert.ok(src.includes("/resolve/main/voices/${o}.bin"),
    "the voices cache key must stay the same so voices already cached are still found");
});

test("the real bundle fetches the model files at the template's commit", async () => {
  const seen = [];
  globalThis.fetch = async (url) => { seen.push(String(url)); throw new Error("stop"); };
  const kokoro = await import(pathToFileURL(BUNDLE).href);
  Object.assign(kokoro.env, { useFSCache: false, allowLocalModels: false, useBrowserCache: false });
  kokoro.env.remotePathTemplate = pinnedRemotePathTemplate(KOKORO_DEFAULT_MODEL);
  await kokoro.KokoroTTS.from_pretrained(KOKORO_DEFAULT_MODEL, { dtype: "q8", device: "wasm" })
    .catch(() => {});
  const base = `https://huggingface.co/${KOKORO_DEFAULT_MODEL}/resolve/${KOKORO_PINNED_REVISION}/`;
  assert.ok(seen.length > 0, "the bundle requested nothing");
  for (const url of seen) assert.ok(url.startsWith(base), `${url} is not at the pinned commit`);
});

function installShell() {
  const dom = new JSDOM(`<!DOCTYPE html><html><body></body></html>`, { url: "http://localhost:8642/" });
  global.window = dom.window;
  global.document = dom.window.document;
  dom.window.Audio = class { play() { return Promise.resolve(); } pause() {} removeAttribute() {} };
  global.Audio = dom.window.Audio;
  return dom.window;
}

async function loadWith(config) {
  const win = installShell();
  win.fetch = async (url) => {
    const u = String(url);
    if (u.includes("/api/tts/config")) {
      return { ok: true, status: 200, json: async () => (
        { net_mode: "allow", library: FAKE_LIBRARY, ...config }) };
    }
    return { ok: false, status: 404, json: async () => ({}) };
  };
  global.fetch = win.fetch;
  delete globalThis.__kokoroLoad;
  if (globalThis.__kokoroEnv) globalThis.__kokoroEnv.remotePathTemplate = "{model}/resolve/{revision}/";
  const calls = { registerTTS: [] };
  const ctx = { authHeaders: () => ({}), toast: () => {}, registerTTS: (p) => calls.registerTTS.push(p) };
  const { register } = await import(pathToFileURL(TTS_JS).href + `?t=${Date.now()}_${Math.random()}`);
  await register(ctx);
  await calls.registerTTS[0].ready();
  return globalThis.__kokoroLoad;
}

test("load() points the default model at the pinned commit before it is built", async () => {
  const seen = await loadWith({});
  assert.equal(seen.model, KOKORO_DEFAULT_MODEL);
  assert.equal(seen.template, `{model}/resolve/${KOKORO_PINNED_REVISION}/`);
});

test("load() leaves a model the admin chose on the bundle's own template", async () => {
  const seen = await loadWith({ model: "someone/else-ONNX" });
  assert.equal(seen.model, "someone/else-ONNX");
  assert.equal(seen.template, "{model}/resolve/{revision}/");
});

test("load() moves a model cached under the default branch to the pinned commit before building it", async () => {
  const base = `https://huggingface.co/${KOKORO_DEFAULT_MODEL}/resolve/`;
  const cache = new FakeCache({ [`${base}main/onnx/model.onnx`]: "weights" });
  globalThis.caches = { open: async () => cache };
  try {
    await loadWith({});
  } finally {
    delete globalThis.caches;
  }
  assert.deepEqual([...cache.map.keys()], [`${base}${KOKORO_PINNED_REVISION}/onnx/model.onnx`]);
});
