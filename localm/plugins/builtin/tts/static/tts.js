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
 */

const VENDOR_VOICES = new URL("vendor/voices.json", import.meta.url);

export async function register(ctx) {
  let cfg;
  try {
    const r = await fetch("/api/tts/config", { headers: ctx.authHeaders() });
    cfg = r.ok ? await r.json() : {};
  } catch {
    cfg = {};
  }
  const model = cfg.model || "onnx-community/Kokoro-82M-v1.0-ONNX";
  const libraryURL = new URL(cfg.library || "vendor/kokoro.min.js", import.meta.url);
  const speed = Number(cfg.speed) || 1;

  // Voice list for the picker, loaded statically (no model download needed).
  let voiceList = [];
  try {
    const vr = await fetch(VENDOR_VOICES);
    if (vr.ok) {
      const map = await vr.json();
      voiceList = Object.entries(map).map(([id, v]) => ({
        id,
        label: `${v.name} (${v.language}, ${v.gender}${v.grade ? ", " + v.grade : ""})`,
      }));
    }
  } catch { /* picker simply shows the default */ }

  let currentVoice =
    (voiceList.find((v) => v.id === cfg.voice) && cfg.voice) ||
    (voiceList[0] && voiceList[0].id) ||
    cfg.voice || "af_heart";

  // ---- lazy model load (with WebGPU -> WASM fallback) -------------------- //
  let kokoro = null;
  let loadPromise = null;
  let announced = false;

  function pickDevice() {
    if (cfg.device && cfg.device !== "auto") return cfg.device;
    return navigator.gpu ? "webgpu" : "wasm";
  }
  function pickDtype(device) {
    if (cfg.dtype && cfg.dtype !== "auto") return cfg.dtype;
    return device === "webgpu" ? "fp32" : "q8";
  }

  async function load() {
    const mod = await import(libraryURL);
    if (cfg.wasm_paths && mod.env && mod.env.backends && mod.env.backends.onnx) {
      mod.env.backends.onnx.wasm.wasmPaths =
        new URL(cfg.wasm_paths, import.meta.url).href;
    }
    if (!announced) {
      announced = true;
      ctx.toast("Loading Kokoro voice (first run downloads ~90 MB, then cached)");
    }
    let device = pickDevice();
    try {
      return await mod.KokoroTTS.from_pretrained(model, { dtype: pickDtype(device), device });
    } catch (e) {
      if (device === "webgpu") {                 // GPU path unavailable: fall back
        device = "wasm";
        return await mod.KokoroTTS.from_pretrained(model, { dtype: pickDtype(device), device });
      }
      throw e;
    }
  }

  function ensureLoaded() {
    if (kokoro) return Promise.resolve(kokoro);
    if (!loadPromise) {
      loadPromise = load().then(
        (k) => (kokoro = k),
        (e) => {
          loadPromise = null;                    // allow a later retry
          ctx.toast("Kokoro voice failed to load; using the browser voice", true);
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

  function clearQueue() {
    for (const u of queue) URL.revokeObjectURL(u);
    queue = [];
  }
  function playNext(myToken) {
    if (myToken !== token) return;
    const url = queue.shift();
    if (!url) { speaking = false; return; }      // drained: done for now
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
  }

  async function speak(text, _opts = {}) {
    stop();                                       // reset any prior utterance
    const myToken = ++token;
    speaking = true;
    let k;
    try {
      k = await ensureLoaded();
    } catch { speaking = false; return; }
    if (myToken !== token) return;                // superseded while loading
    try {
      // stream() splits into sentences and yields audio per sentence, so the
      // first words start playing without waiting for the whole reply.
      for await (const chunk of k.stream(text, { voice: currentVoice, speed })) {
        if (myToken !== token) return;
        const url = URL.createObjectURL(chunk.audio.toBlob());
        const wasIdle = queue.length === 0 && player.paused;
        queue.push(url);
        if (wasIdle) playNext(myToken);
      }
    } catch (e) {
      if (myToken === token) ctx.toast("Voice synthesis failed", true);
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
  });
}
