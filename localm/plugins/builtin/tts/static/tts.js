// SPDX-License-Identifier: AGPL-3.0-or-later
/* Kokoro text-to-speech, rendered entirely in the browser.
 *
 * The SPA's plugin loader import()s this module for the active `tts` plugin and
 * calls register(ctx). We fetch the resolved config (/api/tts/config), build a
 * TTS provider backed by the vendored kokoro-js, and hand it to the app via
 * ctx.registerTTS(); from then on the chat "speak" button and auto-speak use
 * Kokoro instead of the browser's robotic offline voices.
 *
 * The model (~86 MB) is fetched from Hugging Face on first use and cached by the
 * browser, so synthesis is fully local thereafter. No text ever leaves the
 * machine and nothing is written to the server, so privacy mode stays intact.
 * The fetch itself is gated on the resolved net_mode (net_mode=off refuses it;
 * net_mode=ask requires a one-time confirmation) before it is ever made.
 */

import { NetGateError, classifyLoadError, loadToast, planModelFetch, pickDevice, pickDtype, repairAudioTransient, shouldAbortForCorruption, shouldWarmPassively } from "./tts-util.js";
import { isNonEnglishVoice, streamNonEnglish } from "./g2p.js";

const VENDOR_VOICES = new URL("vendor/voices.json", import.meta.url);

// R-NET: net_mode=ask gate. Reaches the GUI shell's shared in-page modal via
// window (app/main.js exposes every app/helpers.js export as window.<name>,
// the same bridge the jobs plugin's own confirmDangerous() uses, including
// its native confirm() fallback for a document with no shell - this module's
// own isolated unit test). No closure dependency on register()'s cfg/ctx, so
// it is a plain module-level export rather than nested like the rest of this
// file's helpers.
export function requestDownloadConsent() {
  if (typeof window === "undefined" || typeof window.openModal !== "function" ||
      typeof window.$ !== "function" || typeof window.el !== "function") {
    return Promise.resolve(typeof confirm === "function" && confirm(
      "Download the Kokoro voice model (~86 MB) from huggingface.co now?"));
  }
  return new Promise((resolve) => {
    let settled = false;
    const finish = (v) => {
      if (settled) return;
      settled = true;
      clearInterval(watch);
      window.$("modal").style.display = "none";
      resolve(v);
    };
    window.openModal("Download voice model?", (body) => {
      body.appendChild(window.el("p", "",
        "Kokoro (~86 MB) will be downloaded from huggingface.co once, then " +
        "cached in the browser. Network access is set to \"ask first\"."));
      const row = window.el("div", "actions");
      const skip = window.el("button", "btn-secondary", "Not now");
      skip.onclick = () => finish(false);
      const dl = window.el("button", "btn-secondary", "Download");
      dl.onclick = () => finish(true);
      row.appendChild(skip);
      row.appendChild(dl);
      body.appendChild(row);
    });
    // Dismissing via the shared modal chrome (x / backdrop) sets display:none
    // without calling either handler above; poll for it and treat as "not
    // now" - same idiom promptText / _offerModelDownload use in helpers.js.
    const watch = setInterval(() => {
      if (window.$("modal").style.display === "none") finish(false);
    }, 200);
  });
}

export async function register(ctx) {
  let cfg;
  try {
    const r = await fetch("/api/tts/config", { headers: ctx.authHeaders() });
    cfg = r.ok ? await r.json() : {};
  } catch (e) {
    // Config endpoint unreachable: fall back to built-in defaults (benign), but
    // log at debug so a persistent failure is still discoverable (RULE 5).
    console.debug("[tts] /api/tts/config unavailable, using defaults:", e);
    cfg = {};
  }
  const model = cfg.model || "onnx-community/Kokoro-82M-v1.0-ONNX";
  const libraryURL = new URL(cfg.library || "vendor/kokoro.min.js", import.meta.url);
  // `let`, not `const`: a Settings save applies voice + speed to the running
  // provider through applyConfig() below, so the change is audible immediately
  // instead of only after a reload. model/device/dtype cannot follow suit - the
  // model is compiled once at load - and the Settings section says so.
  let speed = Number(cfg.speed) || 1;

  // Voice list for the picker, loaded statically (no model download needed).
  let voiceList = [];
  try {
    const vr = await fetch(VENDOR_VOICES);
    if (vr.ok) {
      const map = await vr.json();
      // Grouped by language so the picker lists each language together, with
      // English first: it holds the default voice and most of the catalogue.
      // The grade is upstream's own and is omitted for the voices upstream
      // does not grade, rather than shown as an invented value.
      voiceList = Object.entries(map)
        .sort(([, a], [, b]) => {
          const rank = (v) => (v.language.startsWith("en") ? 0 : 1);
          return rank(a) - rank(b) ||
            a.language.localeCompare(b.language) ||
            a.name.localeCompare(b.name);
        })
        .map(([id, v]) => ({
          id,
          language: v.language,
          label: `${v.name} (${v.language}, ${v.gender}${v.grade ? ", " + v.grade : ""})`,
        }));
    }
  } catch { /* picker simply shows the default */ }

  // "af_heart" by name, not by list position: it is the shipped default and the
  // list above is ordered for the picker. See test_falls_back_to_af_heart.
  let currentVoice =
    (voiceList.find((v) => v.id === cfg.voice) && cfg.voice) ||
    (voiceList.find((v) => v.id === "af_heart") && "af_heart") ||
    (voiceList[0] && voiceList[0].id) ||
    cfg.voice || "af_heart";

  // ---- lazy model load (with WebGPU -> WASM fallback) -------------------- //
  let kokoro = null;
  let Splitter = null;       // vendored TextSplitterStream; see speak()
  let kokoroMod = null;      // the vendored bundle namespace; see speak()
  let loadPromise = null;
  let announced = false;
  let activeDevice = null;   // the backend that actually built the model (see build())
  let repairWarned = false;      // warn once per page, not once per sentence
  let corruptionWarned = false;  // ditto, for the harder webgpu abort case
  let splitterWarned = false;    // ditto (see speak())

  // R08: is this Kokoro model already in the transformers.js browser cache?
  // Used to avoid a misleading "first run downloads" toast on a hard reload, and
  // to tell when the Cache API is unavailable (an insecure / plain-HTTP context)
  // so we can explain why the model will not persist.
  async function modelCached() {
    if (typeof caches === "undefined") return false;   // insecure context: no Cache API
    try {
      const c = await caches.open("transformers-cache");
      const keys = await c.keys();
      return keys.some((req) => req.url && req.url.includes(model));
    } catch {
      return false;
    }
  }

  // R-NET: a fresh read, not the page-load `cfg` snapshot - net_mode is a
  // real kill switch and a stale value would let a mid-session net_mode=off
  // change go unhonoured until the tab reloads. Same-origin call to localm's
  // own local API, never to huggingface.co, so it is not itself net_mode-gated.
  async function currentNetPolicy() {
    try {
      const r = await fetch("/api/tts/config", { headers: ctx.authHeaders() });
      if (r.ok) {
        const fresh = await r.json();
        if (typeof fresh.net_mode === "string") {
          return { mode: fresh.net_mode,
                   allowDownloadsWhenOff: !!fresh.net_allow_model_downloads };
        }
      }
    } catch { /* fall through to the page-load value */ }
    return { mode: cfg.net_mode, allowDownloadsWhenOff: !!cfg.net_allow_model_downloads };
  }

  async function load() {
    const cached = await modelCached();
    const policy = await currentNetPolicy();
    const decision = planModelFetch(policy.mode, cached, policy.allowDownloadsWhenOff);
    if (decision === "refuse") {
      throw new NetGateError(
        "Voice model download is off. Turn it on, or allow downloads only, " +
        "in Settings → Network.");
    }
    if (decision === "confirm" && !(await requestDownloadConsent())) {
      throw new NetGateError(
        "Voice model download needs a one-time confirmation (net_mode=ask) " +
        "and was not granted.");
    }

    const mod = await import(libraryURL);
    Splitter = mod.TextSplitterStream || null;
    kokoroMod = mod;
    const onnx = mod.env && mod.env.backends && mod.env.backends.onnx;
    // The onnxruntime runtime is VENDORED (see vendor/NOTICE.md), and this
    // fallback is where the default has to live rather than only in
    // tts.example.json. The template default reaches us through
    // /api/tts/config, and every path that loses it lands here silently: the
    // config fetch above swallows a failure into `cfg = {}`, settings.py
    // returns {} when the template is unreadable, and an old install may still
    // carry a saved `wasm_paths: ""` override from before this shipped. In each
    // of those the bundle would fall back to its OWN default,
    // cdn.jsdelivr.net - which the CSP no longer admits, so neural TTS would
    // die with "no available backend found" for a reason nothing local caused.
    // A user value still wins (never silently override an explicit choice).
    // The trailing slash is not cosmetic: onnxruntime concatenates the filename
    // straight onto this prefix, so "vendor/onnxruntime" (no slash) would
    // request ".../vendor/onnxruntimeort-wasm-simd-threaded.jsep.wasm" and 404.
    // The settings validator accepts the value with or without it, so normalise
    // here rather than trusting the way it was typed.
    if (onnx) {
      const wasmDir = String(cfg.wasm_paths || "vendor/onnxruntime/");
      onnx.wasm.wasmPaths =
        new URL(wasmDir.endsWith("/") ? wasmDir : wasmDir + "/", import.meta.url).href;
    }
    if (!announced) {
      announced = true;
      // R08: best-effort request to keep the cached model (resists eviction
      // under storage pressure; the browser may deny it without a user gesture,
      // and it is auto-granted for an installed PWA). The model is actually
      // persisted by the transformers.js Cache API; this only hardens it.
      try {
        if (navigator.storage && navigator.storage.persist) navigator.storage.persist();
      } catch { /* storage manager unavailable: best effort */ }
      ctx.toast(loadToast({
        cached,
        secureContext: typeof caches !== "undefined",
      }));
    }
    const device = pickDevice(cfg, !!navigator.gpu);
    // R35: on the WASM path, run the heavy ONNX model compile + inference in
    // onnxruntime's proxy worker so the load no longer freezes the page ("a
    // script is slowing down" - the Firefox / no-WebGPU case in the report).
    // Verified: the main thread stays responsive through a real proxy-worker
    // load. The WebGPU path keeps its (light, largely async) compile on the main
    // thread. If the bundle cannot start the proxy worker the load throws and we
    // retry on the main thread, so this never regresses TTS.
    function build(dev, useProxy) {
      if (onnx && onnx.wasm) onnx.wasm.proxy = !!(useProxy && dev === "wasm");
      // Record the device that actually produced the model, not the one we asked
      // for: the fallbacks below can land on wasm after a webgpu failure, and a
      // repair warning naming the wrong backend would send the reader hunting a
      // fault on hardware that never ran.
      activeDevice = dev;
      return mod.KokoroTTS.from_pretrained(model, { dtype: pickDtype(cfg, dev), device: dev });
    }
    try {
      return await build(device, true);
    } catch (e) {
      if (device === "wasm" && onnx && onnx.wasm && onnx.wasm.proxy) {
        console.warn("[tts] Kokoro proxy-worker load failed, retrying on the main thread:", e);
        return await build("wasm", false);
      }
      if (device === "webgpu") {                 // GPU path unavailable: fall back to WASM
        console.warn("[tts] Kokoro WebGPU load failed, retrying on WASM:", e);
        try {
          return await build("wasm", true);
        } catch (e2) {
          console.warn("[tts] Kokoro WASM proxy load failed, retrying on the main thread:", e2);
          return await build("wasm", false);
        }
      }
      throw e;
    }
  }

  let passiveCheck = null;   // dedupes overlapping passive-warmup eligibility checks

  // opts.passive: called after a reply finishes, before the user has clicked
  // anything. Only proceeds when it needs no download-consent prompt; a bare
  // ensureLoaded() (a real click, or a passive check that already cleared this
  // gate) is unaffected and behaves exactly as before.
  function ensureLoaded(opts = {}) {
    if (kokoro) return Promise.resolve(kokoro);
    if (opts.passive) {
      if (loadPromise) return loadPromise;
      if (passiveCheck) return passiveCheck;
      passiveCheck = (async () => {
        try {
          const cached = await modelCached();
          const policy = await currentNetPolicy();
          if (!shouldWarmPassively(cached, policy.mode)) return null;
          return await ensureLoaded();
        } finally {
          passiveCheck = null;
        }
      })();
      return passiveCheck;
    }
    if (!loadPromise) {
      loadPromise = load().then(
        (k) => (kokoro = k),
        async (e) => {
          loadPromise = null;                    // allow a later retry
          // RULE 5 + R07: surface the REAL reason. A blocked huggingface.co
          // download gets an actionable "allow huggingface.co" message; any
          // other fault keeps its true error. This is the single source of truth
          // for load failures; the speak() catch below stays quiet because the
          // cause is already logged + toasted here.
          console.error("[tts] Kokoro voice model failed to load:", e);
          let message;
          if (e instanceof NetGateError) {
            message = e.message;               // already the real, actionable reason
          } else {
            let cached = false;
            try { cached = await modelCached(); } catch { /* probe best effort */ }
            const online = (typeof navigator === "undefined") || navigator.onLine !== false;
            message = classifyLoadError(e, { cached, online }).message;
          }
          ctx.toast(message + "; using the browser voice (see console for details)", true);
          ctx.registerTTS(null);                 // revert to the built-in fallback
          throw e;
        },
      );
    }
    return loadPromise;
  }

  // ---- sequential audio playback queue ---------------------------------- //
  const player = new Audio();
  let queue = [];          // pending object URLs
  let token = 0;           // bumped on stop/new utterance to cancel stale work
  let speaking = false;
  let endCallback = null;  // the current utterance's opts.onEnd, if any

  // Fires and clears whatever endCallback is currently registered. Called
  // whenever the active utterance stops, however it stops - queue drained,
  // interrupted by a new speak(), or an explicit stop(). See
  // chat-speak-indicator.test.mjs and chat.js's speakToggle.
  function fireEnd() {
    const cb = endCallback;
    endCallback = null;
    if (cb) {
      try { cb(); } catch (e) { console.error("[tts] onEnd callback failed:", e); }
    }
  }

  function clearQueue() {
    for (const u of queue) URL.revokeObjectURL(u);
    queue = [];
  }
  function playNext(myToken) {
    if (myToken !== token) return;
    const url = queue.shift();
    if (!url) { speaking = false; fireEnd(); return; }   // drained: done for now
    player.src = url;
    player.play().catch(() => {});
    player.onended = () => {
      URL.revokeObjectURL(url);
      playNext(myToken);
    };
  }

  function stop() {
    token++;
    speaking = false;
    try { player.pause(); } catch {}
    player.onended = null;
    player.removeAttribute("src");
    clearQueue();
    fireEnd();
  }

  async function speak(text, opts = {}) {
    stop();                                       // reset any prior utterance
    const myToken = ++token;
    speaking = true;
    endCallback = opts.onEnd || null;
    let k;
    try {
      k = await ensureLoaded();
    } catch {
      // The load failure was already surfaced (console.error + toast) inside
      // ensureLoaded's handler; just abandon this utterance (RULE 5: not hidden).
      speaking = false;
      if (myToken === token) fireEnd();
      return;
    }
    if (myToken !== token) return;                // superseded while loading
    try {
      // stream() splits into sentences and yields audio per sentence, so the
      // first words start playing without waiting for the whole reply.
      //
      // FEED IT A SPLITTER WE CLOSE OURSELVES. Handing stream() a plain STRING
      // looks equivalent and is not: kokoro-js then builds a TextSplitterStream
      // internally, pushes the text and NEVER closes it - and that splitter only
      // emits its trailing sentence from flush(), which only close() triggers.
      // So the LAST sentence of every reply was never synthesised and the loop
      // then awaited more input forever, leaving speaking() stuck true. Measured
      // against the vendored bundle: "One. Two. Three." yields exactly "One." and
      // "Two." and then hangs, and a ONE-sentence reply yields nothing at all,
      // which is silence with no error anywhere. Closing up front costs no
      // streaming - the sentences are already segmented, so audio still starts on
      // the first one.
      let source = text;
      if (Splitter) {
        source = new Splitter();
        source.push(text);
        source.close();
      } else if (!splitterWarned) {
        splitterWarned = true;
        // RULE 5: the degraded path still works for all but the final sentence,
        // so it is worth taking rather than failing the utterance - but it must
        // not be silent, because the symptom (a reply that stops one sentence
        // short) looks like a model problem rather than a missing export.
        console.warn(
          "[tts] the vendored bundle exports no TextSplitterStream, so the last " +
          "sentence of each reply cannot be flushed and will not be spoken.");
      }
      // The bundle's own stream() phonemizes as English and then applies
      // English-only phoneme fixups, so a non-English voice goes through
      // g2p.js, which drives the same engine with that voice's language.
      const stream = isNonEnglishVoice(currentVoice)
        ? streamNonEnglish(kokoroMod, k, source,
            { voice: currentVoice, speed, baseURL: import.meta.url })
        : k.stream(source, { voice: currentVoice, speed });
      for await (const chunk of stream) {
        if (myToken !== token) return;
        // Repair the WebGPU leading transient BEFORE encoding: toBlob() writes a
        // float32 wav, so an out-of-range sample is carried through verbatim and
        // clips at the output as a full-scale click on EVERY sentence.
        const rep = repairAudioTransient(chunk.audio.audio, chunk.audio.sampling_rate);
        if (rep.count && !repairWarned) {
          repairWarned = true;                   // once per page: this fires per sentence
          console.warn(
            "[tts] repaired " + rep.count + " out-of-range audio sample(s) (peak " +
            rep.peak.toFixed(2) + ", first at index " + rep.firstIndex + ") from the " +
            activeDevice + " backend. Audio must lie in [-1,1]; this is a backend fault, " +
            "not a repair we want to be needed.");
        }
        if (shouldAbortForCorruption(rep, activeDevice)) {
          console.warn("[tts] refusing further webgpu playback: " + rep.beyondHead +
            " sample(s) out of range beyond the repairable leading transient");
          if (!corruptionWarned) {
            corruptionWarned = true;
            ctx.toast(
              "Kokoro's WebGPU voice produced corrupted audio on this GPU (a " +
              "known issue) - stopped. Switch Compute device to wasm in " +
              "Settings → Text-to-speech.", true);
          }
          stop();
          return;
        }
        const url = URL.createObjectURL(chunk.audio.toBlob());
        const wasIdle = queue.length === 0 && player.paused;
        queue.push(url);
        if (wasIdle) playNext(myToken);
      }
    } catch (e) {
      // Surface the synthesis failure too (RULE 5): a generic toast alone hides
      // whether it was an OOM, a bad voice id, or a model-runtime fault.
      console.error("[tts] Kokoro synthesis failed:", e);
      if (myToken === token) {
        ctx.toast("Voice synthesis failed", true);
        fireEnd();
      }
      speaking = false;
    }
  }

  ctx.registerTTS({
    name: "Kokoro",
    voices: () => voiceList.slice(),
    getVoice: () => currentVoice,
    setVoice: (id) => { currentVoice = id; },
    speaking: () => speaking,
    ready: ensureLoaded,
    speak,
    stop,
    // Apply a freshly saved server-side config without a reload. Only the
    // fields present are touched, and an unknown voice id is ignored rather
    // than handed to the model (which would throw at synthesis time).
    applyConfig: ({ voice, speed: newSpeed } = {}) => {
      if (voice && (!voiceList.length || voiceList.some((v) => v.id === voice))) {
        currentVoice = voice;
      }
      const n = Number(newSpeed);
      if (Number.isFinite(n) && n > 0) speed = n;
    },
  });
}
