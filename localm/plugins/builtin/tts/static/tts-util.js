// SPDX-License-Identifier: AGPL-3.0-or-later
/* Pure, dependency-free helpers shared by tts.js (main thread) and the Kokoro
 * loading path. No DOM/Worker/Cache globals are touched at module scope, so this
 * module imports and runs under plain node for unit tests. */

// R06: the WASM/CPU path defaulted to the q8 quantized model, which produced
// audible cracks/pings instead of a clean voice for some users. fp32 is the
// clean default (it is what the WebGPU path already uses); q8/fp16 remain
// available via the `dtype` config for users who want a smaller/faster download.
// AUTO NOW PICKS WASM EVEN WHEN WebGPU IS AVAILABLE. Kokoro on the WebGPU
// execution provider produces corrupted audio, and it is worst on exactly the
// input chat generates: long sentences.
//
// MEASURED 2026-08-13 on AMD RDNA2 / Chrome 151, identical text, voice, model
// and dtype (fp32), changing only `device`:
//
//                              webgpu        wasm
//     34-word sentence
//       samples outside [-1,1]      25           0
//       peak                 74,355,872       0.572
//       high-frequency share     2.1198      0.1729
//     short sentence
//       samples outside [-1,1]       1           0
//       peak                     24.975       0.506
//
// A peak of 74 million is not a transient, it is numerical blowup, and 12x the
// high-frequency energy is the "congested"/noise signature. Corruption scales
// with sentence LENGTH, so a chat reply is the worst case, which is why this
// reads to a user as "no speech at all, just clicks".
//
// This is a KNOWN, OPEN, UNFIXED upstream defect, not a localm bug and not
// specific to this box: hexgrad/kokoro#98 ("distorted and unusable", open since
// 2025-02, WASM fine), hexgrad/kokoro#193 ("when using device 'webgpu' no dtype
// works"), and microsoft/onnxruntime#29807 (deterministic Kokoro corruption on
// the WebGPU EP, with the same length correlation). Reports span Intel Iris Xe
// and several AMD parts. Forcing WASM is the workaround every one of them lands
// on, and it is the only one with evidence behind it.
//
// COST, measured the same session: WASM generated 6.3 s of audio in 4762 ms vs
// 456 ms on WebGPU, so 10.4x slower - but that is 0.76x realtime, i.e. still
// faster than playback, so streaming keeps up after the first sentence. Slower
// and correct beats fast and unintelligible.
//
// `hasGPU` is retained deliberately: it is the switch to flip back the moment
// upstream fixes this, and callers already pass it. An EXPLICIT device setting
// still wins - a user who sets "webgpu" gets webgpu (never silently override a
// user's choice), and tts.js warns when it detects the corruption.
export function pickDevice(cfg, hasGPU) {
  if (cfg.device && cfg.device !== "auto") return cfg.device;
  return "wasm";
}
export function pickDtype(cfg, device) {
  if (cfg.dtype && cfg.dtype !== "auto") return cfg.dtype;
  // R06: fp32 on both paths. The WASM path used q8 (audible cracks/pings); fp32
  // matches the clean WebGPU path. `device` is accepted for symmetry with
  // pickDevice and a possible future per-device default.
  return "fp32";
}

// R07: turn a model-load failure into an actionable message. A blocked or
// failed huggingface.co download (script/ad blocker, firewall, offline, or a
// 403 from a filtering proxy) surfaces as a network-class error; tell the user
// to allow huggingface.co rather than dumping a raw error. A genuine runtime
// fault keeps its real message so we do not hide the actual problem.
//
// `cached` gates only the `!online` fallback. A `networkish` match is
// unconditional and is never suppressed by `cached`.
export function classifyLoadError(err, { cached = false, online = true } = {}) {
  const raw = (err && (err.message || err.name)) ? String(err.message || err.name) : String(err || "");
  const m = raw.toLowerCase();
  const networkish =
    err instanceof TypeError ||                       // fetch() rejects with TypeError
    // Specific network signatures only. NOT a bare "fetch" - that would mislabel
    // a genuine runtime error mentioning "prefetch"/"fetching shard" as blocked.
    /failed to fetch|networkerror|network error|load failed|err_|net::|blocked|forbidden|\b403\b|cors/.test(m);
  if (networkish || (!cached && !online)) {
    return {
      blocked: true,
      message:
        "Voice model download blocked - allow huggingface.co. A script/ad " +
        "blocker, firewall or offline network is stopping the one-time Kokoro " +
        "download. Allow huggingface.co (and hf.co) and try again.",
    };
  }
  return {
    blocked: false,
    message: "Kokoro voice failed to load (" + (raw || "unknown error") + ")",
  };
}

// R-NET: the browser fetches the Kokoro model directly from Hugging Face, so
// localm's server-side net_mode enforcement (netpolicy.py) never sees this
// request. planModelFetch is the client-side gate: "allow" (already cached,
// net_mode=allow, or net_mode=off with allowDownloadsWhenOff) proceeds with
// no prompt; "refuse" (net_mode=off, not exempted) throws without ever
// fetching; "confirm" (net_mode=ask, or any other value) needs an explicit
// one-time user action before the fetch is allowed to proceed.
export function planModelFetch(mode, cached, allowDownloadsWhenOff) {
  if (cached) return "allow";
  if (mode === "allow") return "allow";
  if (mode === "off") return allowDownloadsWhenOff ? "allow" : "refuse";
  return "confirm";
}

// Thrown by tts.js's load() when planModelFetch returns "refuse" or the user
// declines a "confirm" prompt. A distinct class so the catch handler in
// ensureLoaded() can show this message as-is instead of running it through
// classifyLoadError, whose networkish regex (e.g. /blocked/) would otherwise
// relabel it as a generic connectivity failure.
export class NetGateError extends Error {
  constructor(message) {
    super(message);
    this.name = "NetGateError";
  }
}

// R08: only promise a download when the model is NOT already cached, so a hard
// reload that loads from the browser cache does not falsely claim a fresh ~90 MB
// "first time setup". `secureContext` false (plain-HTTP LAN origin) means the
// Cache API is unavailable and the model genuinely cannot persist - say so
// instead of silently re-downloading every load (RULE 5: surface the reason).
export function loadToast({ cached = false, secureContext = true } = {}) {
  if (!secureContext) {
    return "Loading Kokoro voice. Note: caching needs HTTPS or localhost, so " +
      "over plain HTTP the model re-downloads each visit.";
  }
  return cached
    ? "Loading the cached Kokoro voice..."
    : "Loading Kokoro voice (first run downloads the model, then cached).";
}

// Repair the leading transient the WebGPU backend can emit, IN PLACE, and report
// what was wrong so the caller can surface it (RULE 5: never repair silently).
//
// MEASURED 2026-08-13 on AMD RDNA2 / Chrome 151, same text, voice, model and
// dtype, changing only `device`:
//     webgpu  1 sample outside [-1,1], at index 9, value -24.98, max step 25.9
//     wasm    0 samples outside [-1,1], and the head is exact zeros
// A fresh model's FIRST utterance reproduced it 3 times out of 3 at exactly
// -24.98; a warm model still hit 3 of 6, at 1.62 / 4.16 / 1.99. Both backends
// synthesise real speech (autocorrelation peak 0.75-0.79 at a 209-229 Hz lag),
// so the model is fine and only this one sample is not.
//
// It is audible out of all proportion to its length: that single sample carried
// 97% of the file's energy (24.98^2 = 624 of 645), so the user hears a
// full-scale click over comparatively quiet speech. tts.js streams one WAV PER
// SENTENCE, so it repeats per sentence. `toBlob()` writes a FLOAT32 wav, so the
// value is not wrapped by integer encoding - it survives verbatim and clips at
// the output stage.
//
// Audio samples outside [-1,1] are invalid by definition, whatever produced
// them, so this is a contract repair rather than a cosmetic mask. The leading
// window is ZEROED (with a short fade back in) because the fault sits in the
// first few samples, ahead of any speech; a late sample is only CLAMPED, never
// zeroed, so a mid-utterance fault cannot be silently deleted.
//
// THIS IS A BACKSTOP, NOT THE FIX, and it must not be read as one. On a long
// sentence the WebGPU output is corrupted throughout (25 out-of-range samples,
// peak 74 million, 12x the high-frequency energy) - clamping that yields QUIETER
// GARBAGE, not speech. The actual fix is pickDevice() no longer selecting
// WebGPU automatically; see its comment. This function exists to (a) protect the
// path where a user has EXPLICITLY chosen webgpu, and (b) make the fault
// audible-to-the-log rather than silent, per rule 5. If it ever fires on the
// default path, something has regressed.
export function repairAudioTransient(samples, sampleRate) {
  const report = { count: 0, peak: 0, firstIndex: -1, zeroedThrough: -1 };
  const n = samples ? samples.length : 0;
  if (!n) return report;
  const bad = (v) => !Number.isFinite(v) || Math.abs(v) > 1;
  for (let i = 0; i < n; i++) {
    const v = samples[i];
    if (!bad(v)) continue;
    report.count++;
    if (report.firstIndex < 0) report.firstIndex = i;
    const a = Number.isFinite(v) ? Math.abs(v) : Infinity;
    if (a > report.peak) report.peak = a;
  }
  if (!report.count) return report;
  // 25 ms: comfortably past the observed index-9 fault, still ahead of speech.
  const head = Math.min(n, Math.max(1, Math.round(sampleRate * 0.025)));
  let last = -1;
  for (let i = 0; i < head; i++) if (bad(samples[i])) last = i;
  if (last >= 0) {
    for (let i = 0; i <= last; i++) samples[i] = 0;
    report.zeroedThrough = last;
    // Ramp back in over 2 ms: cutting straight from silence to full amplitude
    // would substitute one step discontinuity for another, i.e. another click.
    const fade = Math.max(1, Math.round(sampleRate * 0.002));
    for (let i = 0; i < fade; i++) {
      const j = last + 1 + i;
      if (j >= n) break;
      samples[j] = Number.isFinite(samples[j]) ? samples[j] * (i / fade) : 0;
    }
  }
  for (let i = 0; i < n; i++) {           // anything left out of range: clamp only
    const v = samples[i];
    if (!Number.isFinite(v)) samples[i] = 0;
    else if (v > 1) samples[i] = 1;
    else if (v < -1) samples[i] = -1;
  }
  return report;
}
