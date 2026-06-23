// SPDX-License-Identifier: AGPL-3.0-or-later
// Unit tests for the pure TTS decision logic shared by tts.js and the Kokoro
// load path (localm/plugins/builtin/tts/static/tts-util.js):
//   R06 dtype default (fp32, not the cracking q8), R07 blocked-huggingface.co
//   detection, R08 cache-aware / honest load toast.

import { test } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const UTIL = join(HERE, "..", "localm", "plugins", "builtin", "tts", "static", "tts-util.js");
const { pickDevice, pickDtype, classifyLoadError, loadToast } = await import(pathToFileURL(UTIL).href);

// ---- integration: the whole tts.js module graph loads ------------------ //
test("tts.js loads as a module and exports register (the tts-util import resolves)", async () => {
  const tts = join(HERE, "..", "localm", "plugins", "builtin", "tts", "static", "tts.js");
  const mod = await import(pathToFileURL(tts).href);
  assert.equal(typeof mod.register, "function");
});

// ---- R06: dtype -------------------------------------------------------- //
test("R06: WASM/CPU path defaults to fp32, not the cracking q8", () => {
  assert.equal(pickDtype({ dtype: "auto" }, "wasm"), "fp32");
  assert.equal(pickDtype({}, "wasm"), "fp32");
  assert.equal(pickDtype({ dtype: "auto" }, "webgpu"), "fp32");
});

test("R06: an explicit dtype override is respected", () => {
  assert.equal(pickDtype({ dtype: "q8" }, "wasm"), "q8");
  assert.equal(pickDtype({ dtype: "fp16" }, "wasm"), "fp16");
});

test("pickDevice honours auto and explicit overrides", () => {
  assert.equal(pickDevice({ device: "auto" }, true), "webgpu");
  assert.equal(pickDevice({ device: "auto" }, false), "wasm");
  assert.equal(pickDevice({}, false), "wasm");
  assert.equal(pickDevice({ device: "wasm" }, true), "wasm");
});

// ---- R07: blocked huggingface.co detection ----------------------------- //
test("R07: a network/fetch failure becomes an actionable allow-huggingface.co message", () => {
  const r = classifyLoadError(new TypeError("Failed to fetch"), { cached: false, online: true });
  assert.equal(r.blocked, true);
  assert.match(r.message, /huggingface\.co/);
});

test("R07: a 403 from a filtering proxy is treated as blocked", () => {
  const r = classifyLoadError(new Error("Request failed with status 403"), { cached: false });
  assert.equal(r.blocked, true);
  assert.match(r.message, /allow huggingface\.co/i);
});

test("R07: being offline (and uncached) is reported as a blocked download", () => {
  const r = classifyLoadError(new Error("whatever"), { cached: false, online: false });
  assert.equal(r.blocked, true);
});

test("R07: a non-network fault when the model is cached keeps its real message", () => {
  const r = classifyLoadError(new Error("ONNX runtime: bad graph"), { cached: true, online: true });
  assert.equal(r.blocked, false);
  assert.match(r.message, /bad graph/);
});

test("R07: a non-network fault while online+uncached is NOT mislabelled as blocked", () => {
  const r = classifyLoadError(new Error("Unsupported model opset"), { cached: false, online: true });
  assert.equal(r.blocked, false);
  assert.match(r.message, /opset/i);
});

// ---- R08: honest, cache-aware load toast ------------------------------- //
test("R08: an uncached secure context promises a one-time download", () => {
  const t = loadToast({ cached: false, secureContext: true });
  assert.match(t, /first run/i);
});

test("R08: a cached model does NOT claim a fresh first-time download", () => {
  const t = loadToast({ cached: true, secureContext: true });
  assert.doesNotMatch(t, /first run/i);
  assert.match(t, /cached/i);
});

test("R08: an insecure (plain-HTTP) context explains why the model will not persist", () => {
  const t = loadToast({ cached: false, secureContext: false });
  assert.match(t, /HTTPS or localhost/i);
});
