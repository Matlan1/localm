// SPDX-License-Identifier: AGPL-3.0-or-later
/* Pure, dependency-free helpers shared by tts.js (main thread) and the Kokoro
 * loading path. No DOM/Worker/Cache globals are touched at module scope, so this
 * module imports and runs under plain node for unit tests. */

// R06: the WASM/CPU path defaulted to the q8 quantized model, which produced
// audible cracks/pings instead of a clean voice for some users. fp32 is the
// clean default (it is what the WebGPU path already uses); q8/fp16 remain
// available via the `dtype` config for users who want a smaller/faster download.
export function pickDevice(cfg, hasGPU) {
  if (cfg.device && cfg.device !== "auto") return cfg.device;
  return hasGPU ? "webgpu" : "wasm";
}
export function pickDtype(cfg, device) {
  if (cfg.dtype && cfg.dtype !== "auto") return cfg.dtype;
  return device === "webgpu" ? "fp32" : "fp32";   // R06: fp32 on WASM too (was q8)
}

// R07: turn a model-load failure into an actionable message. A blocked or
// failed huggingface.co download (script/ad blocker, firewall, offline, or a
// 403 from a filtering proxy) surfaces as a network-class error; tell the user
// to allow huggingface.co rather than dumping a raw error. A genuine runtime
// fault (already cached, or a non-network error) keeps its real message so we
// do not hide the actual problem.
export function classifyLoadError(err, { cached = false, online = true } = {}) {
  const raw = (err && (err.message || err.name)) ? String(err.message || err.name) : String(err || "");
  const m = raw.toLowerCase();
  const networkish =
    err instanceof TypeError ||                       // fetch() rejects with TypeError
    /failed to fetch|networkerror|network error|load failed|err_|net::|fetch|blocked|forbidden|\b403\b|cors/.test(m);
  if (!cached && (networkish || !online)) {
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
