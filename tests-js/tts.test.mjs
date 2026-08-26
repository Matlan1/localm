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
const { pickDevice, pickDtype, classifyLoadError, loadToast, repairAudioTransient } = await import(pathToFileURL(UTIL).href);

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

// AUTO picks wasm even WITH a GPU present: Kokoro on the WebGPU EP produces
// corrupted audio (measured 2026-08-13 on AMD RDNA2 - a 34-word sentence peaked
// at 74,355,872 with 25 samples outside [-1,1], vs 0.572 and 0 on wasm), a known
// open upstream defect (hexgrad/kokoro#98, #193, microsoft/onnxruntime#29807).
test("pickDevice: auto does NOT select webgpu even when a GPU is present", () => {
  assert.equal(pickDevice({ device: "auto" }, true), "wasm");
  assert.equal(pickDevice({ device: "auto" }, false), "wasm");
  assert.equal(pickDevice({}, true), "wasm");
  assert.equal(pickDevice({}, false), "wasm");
});

test("pickDevice: an EXPLICIT device still wins, including webgpu", () => {
  // Never silently override a user's explicit choice - they get what they asked
  // for, and tts.js warns if the corruption shows up.
  assert.equal(pickDevice({ device: "webgpu" }, true), "webgpu");
  assert.equal(pickDevice({ device: "webgpu" }, false), "webgpu");
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

test("R07: a runtime error merely mentioning 'prefetch'/'fetching' is NOT blocked", () => {
  for (const msg of ["error prefetching shard 2", "failed while fetching tensor data"]) {
    const r = classifyLoadError(new Error(msg), { cached: false, online: true });
    assert.equal(r.blocked, false, `should not be blocked: ${msg}`);
  }
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

// --- repairAudioTransient: the WebGPU leading-transient click ------------- //
// The fixture is the REAL measured signal shape, not an invented one: on AMD
// RDNA2 / Chrome 151 the webgpu backend put -24.98 at index 9 (3 fresh loads out
// of 3), preceded by small non-zero garbage, while wasm produced exact zeros
// there.
const SR = 24000;

function gpuLikeChunk() {                        // measured head, then quiet speech
  const a = new Float32Array(SR);                // 1 s
  const head = [0.001, -0.002, 0.002, 0.002, -0.014, 0.036, -0.062, 0.013, 0.943, -24.975];
  head.forEach((v, i) => { a[i] = v; });
  for (let i = head.length; i < a.length; i++) a[i] = 0.3 * Math.sin(i / 8);
  return a;
}

test("repairs the measured WebGPU transient: nothing survives outside [-1,1]", () => {
  const a = gpuLikeChunk();
  const rep = repairAudioTransient(a, SR);
  assert.equal(rep.count, 1, "exactly the one out-of-range sample was seen");
  assert.equal(rep.firstIndex, 9);
  assert.ok(rep.peak > 24 && rep.peak < 26, `peak reported as ${rep.peak}`);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Number.isFinite(a[i]) && Math.abs(a[i]) <= 1,
      `sample ${i} is ${a[i]}, outside [-1,1]`);
  }
});

test("kills the CLICK, not just the range: no full-scale step remains", () => {
  const a = gpuLikeChunk();
  repairAudioTransient(a, SR);
  let maxStep = 0;
  for (let i = 1; i < a.length; i++) maxStep = Math.max(maxStep, Math.abs(a[i] - a[i - 1]));
  // Clamping ALONE would leave a 1.0-to-neighbour step here - i.e. still a click.
  // This is the assertion only the fade can satisfy, so it pins the actual fix
  // rather than the range contract that a bare clamp would also satisfy.
  assert.ok(maxStep < 0.2, `max adjacent step is ${maxStep}, still an audible step`);
});

test("a CLEAN (wasm-like) chunk is left completely untouched", () => {
  const a = new Float32Array(SR);
  for (let i = 0; i < a.length; i++) a[i] = 0.5 * Math.sin(i / 10);
  const before = Float32Array.from(a);
  const rep = repairAudioTransient(a, SR);
  assert.equal(rep.count, 0);
  assert.equal(rep.zeroedThrough, -1);
  assert.deepEqual(Array.from(a), Array.from(before),
    "a clean chunk must not be modified - over-repair would mute real speech onsets");
});

test("a LATE out-of-range sample is clamped, never zeroed away", () => {
  const a = new Float32Array(SR);
  for (let i = 0; i < a.length; i++) a[i] = 0.4;
  const late = 12000;                            // 0.5 s in, mid-utterance
  a[late] = -9.5;
  const rep = repairAudioTransient(a, SR);
  assert.equal(rep.count, 1);
  assert.equal(rep.firstIndex, late);
  assert.equal(rep.zeroedThrough, -1, "a mid-utterance fault must not zero the head");
  assert.equal(a[late], -1, "clamped to the range floor");
  // Float32Array rounds 0.4 to 0.4000000059604645, so compare with a tolerance
  // rather than exactly - an exact compare here fails on storage precision, not
  // on behaviour.
  assert.ok(Math.abs(a[0] - 0.4) < 1e-6, "untouched audio before it stays untouched");
});

test("non-finite samples are reported and neutralised", () => {
  const a = new Float32Array([NaN, Infinity, 0.1, 0.2]);
  const rep = repairAudioTransient(a, SR);
  assert.equal(rep.count, 2);
  for (const v of a) assert.ok(Number.isFinite(v), "a NaN/Inf must not reach the encoder");
});

test("an empty buffer is safe and reports nothing", () => {
  const rep = repairAudioTransient(new Float32Array(0), SR);
  assert.equal(rep.count, 0);
  assert.equal(rep.firstIndex, -1);
});
