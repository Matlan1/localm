// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - settings: performance sliders + VRAM estimate (split from app.js). Classic script: it
   shares the one global lexical environment with the other app/* and
   pages/* scripts, so every cross-section reference resolves by bare
   name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { COMPACT_KEEP, addMessageRow, chat, chatParams, compactConversation, currentConv, maybeCompactConversation, msgImages, msgText, newConversation, renderAttachChips, renderChat, renderConvList, saveConversations, stripUserImages } from "./chat.js";
import { $, GIB, authHeaders, autoGrow, el, nearBottom, openModal, readSSE, renderMarkdown, stripThink, toast } from "./helpers.js";
import { modelCache, modelSelect } from "./models-sidebar.js";
import { execChatCommand, handleSlashSubmit } from "./slash.js";
import { CORE_VIEWS, VIEWS, _applyActiveClasses, closeNav, showView } from "./tabs.js";

/* ================================================================ */
/*  Settings: performance sliders (GPU layers + context) + VRAM est  */
/* ================================================================ */

export function _perfGiB(b) { return (Number(b) / GIB).toFixed(1); }

export let _perfEstTimer = null;
export async function refreshPerfEstimate() {
  const gl = $("perf-gpu-layers"), ctx = $("perf-ctx"), out = $("perf-estimate");
  if (!gl || !ctx || !out) return;
  try {
    const q = new URLSearchParams({ n_ctx: ctx.value, n_gpu_layers: gl.value });
    const r = await fetch("/api/vram-estimate?" + q.toString(), { headers: authHeaders() });
    if (!r.ok) { out.textContent = "estimate unavailable"; return; }
    const d = await r.json();
    let text = `~${_perfGiB(d.needed)} GB needed `
      + `(weights ${_perfGiB(d.weights)} · context ${_perfGiB(d.kv_cache)} `
      + `· overhead ${_perfGiB(d.overhead)})`;
    if (typeof d.free === "number")
      text += ` · ${_perfGiB(d.free)} GB free - ` + (d.fits ? "fits ✓" : "may not fit ⚠");
    else
      text += " · free VRAM unknown";
    out.textContent = text;
    out.classList.toggle("perf-warn", d.fits === false);
  } catch (e) {
    out.textContent = "estimate unavailable";
  }
}

export function setupPerfCard() {
  const gl = $("perf-gpu-layers"), ctx = $("perf-ctx");
  if (!gl || !ctx) return;
  const sync = () => {
    $("perf-ctx-val").textContent = ctx.value;
  };
  const onInput = () => {
    sync();
    clearTimeout(_perfEstTimer);
    _perfEstTimer = setTimeout(refreshPerfEstimate, 150);   // debounce while dragging
  };
  gl.addEventListener("input", onInput);
  ctx.addEventListener("input", onInput);
  const apply = $("perf-apply");
  if (apply) apply.onclick = async () => {
    const glVal = Number(gl.value);
    if (!Number.isInteger(glVal) || glVal < 0 || glVal > 999) {
      toast("GPU layers must be between 0 and 999", true);
      return;
    }
    try {
      const r = await fetch("/v1/config", {
        method: "PATCH", headers: authHeaders(),
        body: JSON.stringify({ n_gpu_layers: glVal, n_ctx: Number(ctx.value) }),
      });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
      toast("Saved - applies on the next model load");
    } catch (e) { toast("Could not save: " + e.message, true); }
  };
  // Seed slider positions from the current config, then estimate.
  fetch("/v1/config", { headers: authHeaders() })
    .then((r) => (r.ok ? r.json() : {}))
    .then((cfg) => {
      if (typeof cfg.n_gpu_layers === "number")
        gl.value = cfg.n_gpu_layers < 0 ? 999 : Math.min(999, cfg.n_gpu_layers);
      if (typeof cfg.n_ctx === "number")
        ctx.value = Math.min(Number(ctx.max), Math.max(Number(ctx.min), cfg.n_ctx));
      sync();
      refreshPerfEstimate();
    })
    .catch(() => { sync(); refreshPerfEstimate(); });
}

/* ---- web access (model-initiated, via the params-drawer toggle) ---- */

export const WEB_MAX_ROUNDS = 3;

// R27: a remembered "don't ask again this session" choice. null = ask each time;
// true = allow all this session; false = deny all this session. In-memory only
// (so it resets on reload = a new session) and leaves no persisted trace.
export let webAskSession = null;
// Setter so OTHER modules can reset the choice: webAskSession is an ES module
// import for them (read-only), and `webAskSession = null` from another module
// throws "Assignment to constant variable" in the real browser (the jsdom test
// harness strips imports into one shared scope, so it never catches this). This
// module reassigns its OWN local binding here, which is allowed.
export function setWebAskSession(v) { webAskSession = v; }

// net_mode = ask means the GUI must APPROVE each model-initiated web request
// before it runs (the settings promise: "ask = approve each request"). Read it
// fresh from /v1/config so a change in Settings takes effect without a reload;
// the cost is one small GET per model-initiated round (bounded by
// WEB_MAX_ROUNDS). Unknown / unreachable -> do not block (the per-conversation
// toggle is the standing consent; only "off", enforced server-side, blocks).
export async function webModeIsAsk() {
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (r.ok) {
      const cfg = await r.json();
      return !!(cfg && cfg.net_mode === "ask");
    }
  } catch (e) { /* server unreachable - fall through to "do not block" */ }
  return false;
}
window.webModeIsAsk = webModeIsAsk;

// Approval dialog for a model-initiated web request under net_mode=ask. Returns
// a promise<boolean>. Uses the in-page modal (window.confirm/prompt are
// suppressed in some PWA/mobile browsers, the NET-1 class of bug). Overridable
// in tests.
export function confirmWebRequest(call) {
  // R27: a remembered choice short-circuits the modal for the rest of the session.
  if (webAskSession !== null) return Promise.resolve(webAskSession);
  return new Promise((resolve) => {
    const args = (call && call.args) || {};
    const target = args.query || args.url || "";
    const verb = call && call.name === "fetch_url" ? "fetch a web page" : "search the web";
    openModal("Allow web access?", (body) => {
      body.appendChild(el("p", "", "The model wants to " + verb + " for:"));
      body.appendChild(el("p", "web-ask-target", target));
      // R27: "don't ask again this session" - remember Allow/Deny so the popup
      // does not fire on every model-initiated request for the rest of the session.
      const remember = el("label", "web-ask-remember");
      const cb = el("input");
      cb.type = "checkbox";
      remember.appendChild(cb);
      remember.appendChild(document.createTextNode(" Don't ask again this session"));
      body.appendChild(remember);
      const row = el("div", "actions");
      const deny = el("button", "btn-secondary", "Deny");
      deny.onclick = () => {
        if (cb.checked) webAskSession = false;
        $("modal").style.display = "none";
        resolve(false);
      };
      const allow = el("button", "btn-secondary btn-primary", "Allow");
      allow.onclick = () => {
        if (cb.checked) webAskSession = true;
        $("modal").style.display = "none";
        resolve(true);
      };
      row.appendChild(deny);
      row.appendChild(allow);
      body.appendChild(row);
    });
  });
}
window.confirmWebRequest = confirmWebRequest;

export const WEB_TOOL_PROMPT =
  "You can access the internet through tools. When the answer depends on " +
  "current, real-time, or external information you cannot be certain of " +
  "(news, prices, software versions, documentation, anything after your " +
  "training cutoff), get it from the web instead of guessing. Reply with " +
  "ONLY a tool call block and nothing else:\n" +
  '<tool_call>{"name": "web_search", "args": {"query": "..."}}</tool_call>\n' +
  "To read a specific page:\n" +
  '<tool_call>{"name": "fetch_url", "args": {"url": "https://..."}}</tool_call>\n' +
  "The results arrive in the next message; then answer and cite the source " +
  "URLs you used.\n" +
  "HONESTY: never invent search results, URLs, or page contents, and never " +
  "say you searched or read a page unless you actually emitted a tool call " +
  "and received its result. If a search fails or finds nothing useful, say " +
  "so plainly instead of making something up.";

export const NO_WEB_PROMPT =
  "You are offline with NO internet access in this conversation. Do not " +
  "present guessed or invented information as verified fact: current events, " +
  "news, prices, live data, software versions, or anything you cannot confirm " +
  "from this conversation. Never claim you looked something up, searched the " +
  "web, or read a page, because you cannot. If the user needs current or " +
  "external information, say plainly that you cannot verify it offline and " +
  "that they can enable \"Web access\" with the 🌐 toggle in the " +
  "parameters drawer (⚙). Saying \"I do not know\" is better than " +
  "stating something false.";

// Used when web results were just injected (the explicit /web command, or a
// model-initiated search) but the standing toggle is off: the model HAS fresh
// results in hand, so the offline-denial floor would contradict them. Tell it
// to use and cite the provided results, and not to fabricate beyond them.
export const WEB_GROUNDED_PROMPT =
  "Web search results have been provided to you in this conversation. Use them " +
  "to answer, and cite the source URLs you relied on. Stay within what the " +
  "results actually support: do not invent facts, URLs, or details beyond them, " +
  "and if they do not answer the question, say so plainly.";

/** True when the most recent message is freshly injected web grounding (search
 *  results or fetched page content), as opposed to a repair note or a failure
 *  note. Used so an explicit /web run is not told it is offline. */
export function lastTurnHasWebResults(conv) {
  const last = conv.messages[conv.messages.length - 1];
  if (!last || !last.web) return false;
  const text = typeof last.content === "string" ? last.content : "";
  return /Results of web_search|Content of /.test(text);
}

// Tool-call wrappers a local model may emit. We accept the canonical
// <tool_call> tags plus the mangled finetune dialects (<|tool_call|>, closing
// as <tool_call|>) and ```tool_call / ```json / bare ``` fences, mirroring the
// coder's lenient parser so a slightly-off call still runs instead of being
// silently dropped (which let the model's un-grounded answer through).
export const _WEB_TOOLS = new Set(["web_search", "fetch_url"]);

/** Lenient JSON parse for the mangles local finetunes produce (single-quoted
 *  keys, trailing commas). Returns the parsed object, or null. */
export function _lenientJSON(body) {
  const tries = [
    (s) => s,
    (s) => s.replace(/'([^']+)'\s*:/g, '"$1":'),       // single-quoted keys
    (s) => s.replace(/,(\s*[}\]])/g, "$1"),            // trailing commas
    (s) => s.replace(/'([^']+)'\s*:/g, '"$1":').replace(/,(\s*[}\]])/g, "$1"),
  ];
  for (const fix of tries) {
    try {
      const obj = JSON.parse(fix(body));
      if (obj && typeof obj === "object") return obj;
    } catch (e) { /* try the next recovery layer */ }
  }
  return null;
}

/** Yield every brace-balanced top-level {...} region in text. String literals
 *  are tracked so braces inside them do not confuse the depth count. */
export function* _topLevelObjects(text) {
  for (let i = 0; i < text.length; i++) {
    if (text[i] !== "{") continue;
    let depth = 0, inStr = false, esc = false;
    for (let j = i; j < text.length; j++) {
      const c = text[j];
      if (inStr) {
        if (esc) esc = false;
        else if (c === "\\") esc = true;
        else if (c === '"') inStr = false;
      } else if (c === '"') { inStr = true; }
      else if (c === "{") { depth++; }
      else if (c === "}") {
        depth--;
        if (depth === 0) { yield text.slice(i, j + 1); i = j; break; }
      }
    }
  }
}

/** Normalise a parsed object to a {name, args} web call, or null. Accepts the
 *  OpenAI "arguments" alias for "args". */
export function _asWebCall(obj) {
  if (!obj || typeof obj.name !== "string" || !_WEB_TOOLS.has(obj.name)) return null;
  const args = (obj.args && typeof obj.args === "object") ? obj.args
             : (obj.arguments && typeof obj.arguments === "object") ? obj.arguments
             : {};
  return { name: obj.name, args };
}

/** First web tool call in a reply, or null. Tolerates the wrapper and JSON
 *  mangles local models emit so a real attempt is not silently dropped. */
export function parseWebCall(text) {
  const clean = stripThink(text);
  // Candidate {prefixName, body} pairs, in priority order: explicit wrappers
  // first (the name may live in a "call:NAME" prefix, Gemma-style), then
  // fences, then any bare top-level JSON object naming a web tool.
  const bodies = [];
  const wrap = /<\|?\/?tool_call\|?>\s*(?:call:(\w+)\s*)?([\s\S]*?)\s*<\|?\/?tool_call\|?>/g;
  for (const mm of clean.matchAll(wrap)) bodies.push({ name: mm[1], body: mm[2] });
  const fence = /```[ \t]*[A-Za-z_]*[ \t]*\r?\n([\s\S]*?)\r?\n[ \t]*```/g;
  for (const mm of clean.matchAll(fence)) bodies.push({ name: undefined, body: mm[1] });
  for (const { name: prefixName, body } of bodies) {
    const trimmed = body.trim().replace(/<\|"\|>/g, '"');   // Gemma quote tokens
    const obj = _lenientJSON(trimmed);
    let call = _asWebCall(obj);
    // Args-only body with the tool named in the wrapper prefix (Gemma native).
    if (!call && obj && _WEB_TOOLS.has(prefixName)) {
      call = { name: prefixName, args: obj };
    }
    if (call) return call;
  }
  // Last resort: a bare {...} object anywhere in the reply naming a web tool.
  for (const chunk of _topLevelObjects(clean)) {
    const call = _asWebCall(_lenientJSON(chunk));
    if (call) return call;
  }
  return null;
}

/** True when a reply looks like a botched web tool call we could not parse: a
 *  tool-call wrapper/fence, or a JSON object that mentions a web tool by name.
 *  Lets the caller ask the model to re-emit it cleanly instead of accepting an
 *  un-grounded answer. */
export function looksLikeWebToolAttempt(text) {
  const clean = stripThink(text);
  if (/<\|?\/?tool_call\|?>/.test(clean) || /```[ \t]*tool_call\b/.test(clean)) return true;
  return /"name"\s*:/.test(clean) && /web_search|fetch_url/.test(clean);
}

/** Run a web tool call through the policy-enforced server endpoints. */
export async function requestWebTool(call) {
  const a = call.args || call.arguments || {};
  if (call.name === "web_search") {
    const r = await fetch("/api/web/search", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ query: a.query || "", max_results: 5 }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const lines = data.results.map((res, i) =>
      `${i + 1}. ${res.title}\n   ${res.url}` +
      (res.snippet ? `\n   ${res.snippet}` : ""));
    return `[Results of web_search "${a.query}"]\n` + lines.join("\n");
  }
  if (call.name === "fetch_url") {
    const r = await fetch("/api/web/fetch", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ url: a.url || "", max_chars: 6000 }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    return `[Content of ${data.url}]` +
      (data.truncated ? " (truncated)" : "") + `\n${data.text}`;
  }
  throw new Error("Unknown web tool: " + call.name);
}

/** Run one model-requested web call, injecting the result (or the failure,
 *  so the model can adapt) as a dimmed "Web" message. */
export async function runWebCall(conv, call) {
  let note;
  try {
    note = await requestWebTool(call);
  } catch (e) {
    note = `[Web request failed: ${e.message}] Answer without the web, ` +
           "and say that web access did not work.";
    toast("Web request failed: " + e.message, true);
  }
  conv.messages.push({ role: "user", content: note, web: true });
  saveConversations(conv);
  renderChat();
}

/* ---- voice: mic (Whisper STT) + read-aloud (browser TTS) ---- */

export const voice = { rec: null, chunks: [], available: true, reason: "",
                modelCached: true, model: "" };

/** Grey out the mic up front when the server lacks the [voice] extra,
 *  instead of letting the user record and only then failing. */
export async function refreshVoiceStatus() {
  try {
    const r = await fetch("/api/voice/status", { headers: authHeaders() });
    if (!r.ok) return;   // old server without the endpoint - leave enabled
    const data = await r.json();
    voice.available = data.available;
    voice.reason = data.reason || "";
    voice.modelCached = data.model_cached !== false;
    voice.model = data.model || "";
    const btn = $("chat-mic");
    btn.classList.toggle("unavailable", !data.available);
    if (!data.available) btn.title = data.reason;
  } catch (e) { /* server unreachable - status refreshes on next load */ }
}

export function blobToB64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(new Error("could not read recording"));
    reader.readAsDataURL(blob);
  });
}

export async function toggleMic() {
  const btn = $("chat-mic");
  if (voice.rec) {           // second click stops and transcribes
    voice.rec.stop();
    return;
  }
  if (!voice.available) {
    toast(voice.reason || "Speech-to-text is not installed on the server", true);
    return;
  }
  if (!voice.modelCached) {
    // Transcription is fully local, but the FIRST use fetches the Whisper
    // model from HuggingFace - make that one network access explicit.
    if (!confirm(
        `First use downloads the Whisper "${voice.model}" speech model ` +
        "from HuggingFace (one-time). Transcription itself runs fully " +
        "offline afterwards. Download now?")) return;
    voice.modelCached = true;   // consent given - don't re-ask this session
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    toast("This browser does not support audio recording", true);
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    toast("Microphone unavailable: " + e.message, true);
    return;
  }
  voice.chunks = [];
  voice.rec = new MediaRecorder(stream);
  voice.rec.ondataavailable = (e) => { if (e.data.size) voice.chunks.push(e.data); };
  voice.rec.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    btn.classList.remove("recording");
    const blob = new Blob(voice.chunks, { type: voice.rec.mimeType || "audio/webm" });
    voice.rec = null;
    btn.disabled = true;
    btn.textContent = "…";
    try {
      const r = await fetch("/api/voice/transcribe", {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ audio_b64: await blobToB64(blob) }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || r.statusText);
      const input = $("chat-input");
      input.value = (input.value ? input.value.trimEnd() + " " : "") + data.text;
      autoGrow(input);
      input.focus();
    } catch (e) {
      toast("Transcription failed: " + e.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "🎤";
    }
  };
  voice.rec.start();
  btn.classList.add("recording");
  toast("Recording - click 🎤 again to stop");
}

$("chat-mic").onclick = toggleMic;

/* ---- text-to-speech ----
 *  A client plugin (the `tts` plugin) may install a neural provider via
 *  registerTTS(); otherwise we fall back to the browser's built-in offline
 *  voices. The browser fallback can only reach robotic local voices on Windows
 *  (the good Win11 voices are Narrator-only or cloud), which is exactly why the
 *  tts plugin exists. */
export let ttsProvider = null;   // {name, voices(), getVoice(), setVoice(id),
                          //  speaking(), ready(), speak(text, opts), stop()}

/** Install (or clear, with null) the active TTS provider, then refresh the
 *  voice picker. Called by a client plugin's register(ctx). */
export function registerTTS(provider) {
  ttsProvider = provider;
  populateVoicePicker();
}

/** The browser SpeechSynthesisVoice the user picked for the fallback, if any. */
export function selectedBrowserVoice() {
  if (!window.speechSynthesis) return null;
  const want = localStorage.getItem("localm.ttsVoiceBrowser");
  if (!want) return null;
  return speechSynthesis.getVoices().find((v) => v.name === want) || null;
}

/** Read text aloud. With toggle: true (the 🔊 button) a second call stops
 *  instead; auto-speak replaces the current utterance. */
export function speak(text, opts = {}) {
  const clean = stripThink(text).replace(/[*_`#>\[\]()]/g, " ").trim();
  if (ttsProvider) {
    if (ttsProvider.speaking()) {
      ttsProvider.stop();
      if (opts.toggle) return;
    }
    if (clean) ttsProvider.speak(clean, opts);
    return;
  }
  if (!window.speechSynthesis) {
    toast("This browser has no speech synthesis", true);
    return;
  }
  if (speechSynthesis.speaking) {
    speechSynthesis.cancel();
    if (opts.toggle) return;
  }
  if (clean) {
    const u = new SpeechSynthesisUtterance(clean);
    const v = selectedBrowserVoice();
    if (v) u.voice = v;
    speechSynthesis.speak(u);
  }
}

/** Fill the voice picker from the active provider (or, with no provider, the
 *  browser's LOCAL voices) and remember the choice. Hidden when there is
 *  nothing to choose. */
export function populateVoicePicker() {
  const sel = $("p-voice");
  const row = $("voice-row");
  if (!sel) return;
  let opts = [];
  let current = "";
  if (ttsProvider) {
    opts = ttsProvider.voices();
    current = localStorage.getItem("localm.ttsVoice") || ttsProvider.getVoice();
    if (current) ttsProvider.setVoice(current);
  } else if (window.speechSynthesis) {
    // getVoices() is async-populated; filter to localService so we never offer
    // a cloud voice that would send text off the machine.
    opts = speechSynthesis
      .getVoices()
      .filter((v) => v.localService)
      .map((v) => ({ id: v.name, label: `${v.name} (${v.lang})` }));
    current = localStorage.getItem("localm.ttsVoiceBrowser") || "";
  }
  sel.replaceChildren();
  if (!opts.length) {
    if (row) row.style.display = "none";
    return;
  }
  if (row) row.style.display = "";
  for (const o of opts) {
    const el = document.createElement("option");
    el.value = o.id;
    el.textContent = o.label;
    sel.appendChild(el);
  }
  if (current) sel.value = current;
}

/** Persist the picked voice and apply it to the active provider. */
export function onVoicePick() {
  const id = $("p-voice").value;
  if (ttsProvider) {
    ttsProvider.setVoice(id);
    localStorage.setItem("localm.ttsVoice", id);
  } else {
    localStorage.setItem("localm.ttsVoiceBrowser", id);
  }
}

/** Load client-side plugin modules: for each ACTIVE plugin that ships a
 *  client_entry, import it and call register(ctx). Failures are isolated so a
 *  broken plugin module never breaks chat. */
export async function loadClientPlugins() {
  let plugins = [];
  try {
    // /api/capabilities (not /api/plugins) so client-entry modules load for a
    // scoped key too, and ONLY the plugins this key's scopes grant are imported.
    const r = await fetch("/api/capabilities", { headers: authHeaders() });
    if (r.ok) plugins = (await r.json()).plugins || [];
  } catch {
    return; // server unreachable; the built-in browser voice still works
  }
  const ctx = { registerTTS, toast, authHeaders, voicesChanged: populateVoicePicker };
  for (const p of plugins) {
    if (!p.active || !p.client_entry) continue;
    const base = p.assets_base || `/plugins/${p.name}`;
    try {
      const mod = await import(`${base}/${p.client_entry}`);
      if (mod && typeof mod.register === "function") await mod.register(ctx);
    } catch (e) {
      console.error(`client plugin ${p.name} failed to load`, e);
    }
  }
}

/* ---- first-party plugin command catalog ---- */
// Map of slash-command verb -> { plugin, active } across the first-party
// catalog, so a command that belongs to a known-but-inactive plugin (e.g.
// /generate-image with the image plugin off) gets a "needs the X plugin" hint
// instead of a confusing 404 or "unknown command". Populated from /api/plugins;
// stays empty (silent, current behaviour) until loaded or if the server is
// unreachable. `suggest` mirrors the suggest_plugins config toggle.
export const pluginCommands = { map: {}, suggest: true };

// R50: signal other same-origin tabs that the installed/enabled plugin set
// changed (a new value is required for the storage event to fire, so use the
// clock). The writing tab refreshes itself directly; other tabs react to the
// storage event wired near the focus listener.
export function bumpPluginsRev() {
  try { localStorage.setItem("localm.pluginsRev", String(Date.now())); }
  catch (e) { /* storage blocked / full - cross-tab sync degrades to focus only */ }
}
window.bumpPluginsRev = bumpPluginsRev;

export async function refreshPluginCommands() {
  try {
    // /api/capabilities returns ONLY what THIS key may use (scope-filtered) and
    // the core-tab flags, so the nav shows just the usable tabs without needing
    // plugins:read. The Plugins management page still uses /api/plugins.
    const r = await fetch("/api/capabilities", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    const map = {};
    for (const p of data.plugins || []) {
      for (const c of p.commands || []) {
        map[c] = { plugin: p.name, active: !!p.active };
      }
    }
    pluginCommands.map = map;
    pluginCommands.suggest = data.suggest_plugins !== false;
    pluginState = data.plugins || [];
    if (data.core) applyCoreTabVisibility(data.core);
    // Reveal the bug-report "Send to maintainer" button only when an upload
    // endpoint is configured (otherwise the report is saved-to-file + emailed).
    const bugUp = $("bug-upload");
    if (bugUp) bugUp.hidden = !data.bugreport_upload;
    // Reveal the Updates + Issues settings cards only when their proxy surfaces are
    // configured. On startup, a single throttled, quiet update check surfaces a
    // banner - it NEVER applies anything (apply is always an explicit click).
    const upSec = $("sec-updates");
    if (upSec) {
      upSec.hidden = !data.update_available;
      if (data.update_available) maybeAutoUpdateCheck();
    }
    const isSec = $("sec-issues");
    if (isSec) isSec.hidden = !data.issues_available;
    renderNav();
  } catch { /* server unreachable; fall back to plain unknown-command */ }
}

// One quiet update check per ~6h (a startup auto-surface). Calls the check only -
// never applies. Defined here; the actual fetch lives in pages.js (window hook).
export function maybeAutoUpdateCheck() {
  try {
    const last = +(localStorage.getItem("localm.updateCheckAt") || 0);
    if (Date.now() - last < 6 * 3600 * 1000) return;
    localStorage.setItem("localm.updateCheckAt", String(Date.now()));
  } catch (e) { /* storage blocked: just check */ }
  if (typeof window.__localmUpdateCheck === "function") window.__localmUpdateCheck();
}

/** A "/cmd needs the X plugin" hint when *cmd* belongs to a known first-party
 *  plugin that is not active, else null (handle it normally). */
export function pluginSuggestion(cmd) {
  if (!pluginCommands.suggest) return null;
  const hit = pluginCommands.map[cmd];
  if (!hit || hit.active) return null;
  return `/${cmd} needs the ${hit.plugin} plugin - install or enable it on the Plugins page.`;
}

/* ---- dynamic nav rail (tabs follow the active plugins) ---- */
// The most recent /api/plugins entries, refreshed alongside the command cache.
export let pluginState = [];

// Each plugin's manifest icon name -> the nav emoji. Kernel buttons keep their
// own emoji in index.html; "studio" is the media parent.
export const NAV_ICON = { chat: "💬", code: "⚙️", image: "🖼️", music: "🎵", video: "🎬", book: "📚", clock: "⏰" };
// Canonical rail order of first-party plugin tabs (stable so the rail does not
// reshuffle as plugins toggle); "studio" is the media slot (image/music/video).
export const NAV_TAB_ORDER = ["coder", "studio", "knowledge"];

export function _navButton(id, icon, label, onClick, cls) {
  const b = el("button", cls || "", `${icon} ${label}`);
  b.id = id;
  b.onclick = onClick;
  return b;
}

/** Rebuild the plugin portion of the nav rail from the active-with-a-tab
 *  plugins, then re-derive VIEWS and re-assert the active tab. */
/* Show only the core tabs the current key's scopes grant. chat is the baseline
 * anchor and is NEVER hidden (chatting needs no scope); models/plugins/settings
 * render only when the key holds models:read / plugins:read / config:read. A tab
 * the key lacks is not shown at all (no show-then-"no access"). Driven by
 * /api/capabilities .core. If the active view becomes hidden (e.g. a remembered
 * Settings tab on a key without config:read), fall back to chat so the user is
 * never parked on an inaccessible view. */
export function applyCoreTabVisibility(core) {
  if (!core) return;
  const activeView = (document.querySelector(".view.active") || {}).id;
  let activeHidden = false;
  for (const view of ["models", "plugins", "settings"]) {
    const nav = $("nav-" + view);
    if (!nav) continue;
    const show = core[view] !== false;        // default to showing if unknown
    nav.style.display = show ? "" : "none";
    if (!show && activeView === "view-" + view) activeHidden = true;
  }
  if (activeHidden) showView("chat");
}
window.applyCoreTabVisibility = applyCoreTabVisibility;

export function renderNav() {
  const slot = $("nav-plugin-slot");
  if (!slot) return;
  slot.replaceChildren();
  // chat owns a tab but is the static kernel anchor; never render it (or any
  // plugin claiming a kernel view) as a dynamic tab.
  const active = pluginState.filter(
    (p) => p.active && p.tab && !CORE_VIEWS.includes(p.tab));
  const studio = active.filter((p) => p.group === "studio");
  const flat = active.filter((p) => p.group !== "studio");
  const byTab = {};
  for (const p of flat) byTab[p.tab] = p;

  const renderFlat = (p) => slot.appendChild(_navButton(
    "nav-" + p.tab, NAV_ICON[p.icon] || "•", p.label || p.name, () => showView(p.tab)));

  const done = new Set();
  for (const key of NAV_TAB_ORDER) {
    if (key === "studio") { renderStudioGroup(slot, studio); continue; }
    if (byTab[key]) { renderFlat(byTab[key]); done.add(key); }
  }
  // any other active plugin tab not in the canonical order, in catalog order.
  // Iterate the tab-deduped map (not raw `flat`) so two plugins claiming the
  // same tab cannot emit duplicate id="nav-<tab>" nodes (LBUG-1).
  for (const p of Object.values(byTab)) if (!done.has(p.tab)) renderFlat(p);

  rebuildViews();
  reconcileActiveView();
}

/** Studio hybrid grouping: nothing for 0 media plugins, a single flat tab for
 *  exactly 1, and one stable-position "Studio" parent expanding to the active
 *  children for 2+. */
export function renderStudioGroup(slot, studio) {
  if (studio.length === 0) return;
  if (studio.length === 1) {
    const p = studio[0];
    slot.appendChild(_navButton(
      "nav-" + p.tab, NAV_ICON[p.icon] || "•", p.label || p.name, () => showView(p.tab)));
    return;
  }
  const order = ["images", "music", "video"];
  const known = order.map((t) => studio.find((p) => p.tab === t)).filter(Boolean);
  // include any studio plugin with a non-canonical tab so a third-party media
  // plugin is not counted toward the group yet silently never rendered (LGAP-1)
  const extra = studio.filter((p) => !order.includes(p.tab));
  const kids = [...known, ...extra];
  const activeView = (document.querySelector(".view.active") || { id: "view-chat" })
    .id.replace("view-", "");
  const hasActiveKid = kids.some((p) => p.tab === activeView);
  const open = hasActiveKid ||
    (chat.privacy ? true : localStorage.getItem("localm.studioOpen") !== "0");

  const wrap = el("div", "nav-group");
  const parent = el("button", "nav-group-parent" + (open ? " open" : ""), "🎨 Studio");
  const children = el("div", "nav-children");
  children.style.display = open ? "block" : "none";
  parent.onclick = () => {
    const nowOpen = children.style.display === "none";
    children.style.display = nowOpen ? "block" : "none";
    parent.classList.toggle("open", nowOpen);
    if (!chat.privacy) localStorage.setItem("localm.studioOpen", nowOpen ? "1" : "0");
  };
  for (const p of kids) {
    children.appendChild(_navButton(
      "nav-" + p.tab, NAV_ICON[p.icon] || "•", p.label || p.name,
      () => showView(p.tab), "nav-child"));
  }
  wrap.appendChild(parent);
  wrap.appendChild(children);
  slot.appendChild(wrap);
}

export function rebuildViews() {
  const tabs = pluginState
    .filter((p) => p.active && p.tab && !CORE_VIEWS.includes(p.tab))
    .map((p) => p.tab);
  // MUTATE the shared VIEWS array in place - do NOT reassign it. VIEWS is an ES
  // module IMPORT from tabs.js (a read-only binding), so `VIEWS = [...]` throws
  // "TypeError: Assignment to constant variable" in the real browser, which
  // aborted renderNav() right after the nav buttons were appended: VIEWS stayed
  // at CORE_VIEWS, so _applyActiveClasses (which iterates VIEWS) could never
  // toggle a plugin view (coder/images/music/video/knowledge/jobs) active -
  // clicking those tabs silently did nothing. (The jsdom test harness STRIPS
  // import/export into one shared scope, where the reassignment worked, so the
  // suite never caught this - only the real ESM browser did.) Mutating the array
  // contents is allowed on an imported binding and is seen by every importer.
  VIEWS.length = 0;
  VIEWS.push("chat", ...tabs, "models", "plugins", "settings");
}

// After the rail is rebuilt, keep the shown view reachable: if its plugin was
// just disabled/uninstalled, fall back to chat; otherwise re-assert the active
// highlight on the (possibly freshly created) nav button.
export function reconcileActiveView() {
  const cur = document.querySelector(".view.active");
  const name = cur ? cur.id.replace("view-", "") : "chat";
  const ok = CORE_VIEWS.includes(name) ||
             pluginState.some((p) => p.active && p.tab === name);
  if (ok) {
    // The shown view is still valid: just re-assert the highlight on the
    // (possibly freshly-created) nav button. Do NOT call showView(name) here -
    // it re-fires onViewShown, and for chat/coder onViewShown re-enters
    // refreshPluginCommands -> renderNav -> reconcileActiveView, a runaway
    // /api/plugins loop (and it double-renders whatever page is open).
    _applyActiveClasses(name);
  } else {
    // The shown view's plugin was uninstalled - fall back to chat (a real
    // view switch, page refresh included).
    showView("chat");
  }
}

/* ---- assistant memory ---- */

export const memory = { text: "", writable: false };

export async function refreshMemory() {
  try {
    const r = await fetch("/api/memory", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    memory.text = data.text || "";
    memory.writable = !!data.writable;
  } catch (e) { /* server unreachable */ }
}

export async function rememberFact(fact) {
  if (!fact) { toast("Usage: /remember <fact>", true); return; }
  try {
    const r = await fetch("/api/memory/append", {
      method: "POST", headers: authHeaders(),
      body: JSON.stringify({ text: fact }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    await refreshMemory();
    toast("Remembered ✓");
  } catch (e) {
    toast("Could not save: " + e.message, true);
  }
}

export async function synthesizeMemoryNow(statusEl) {
  // Manual trigger of the same consolidation the background pass runs, for
  // immediate feedback (a tester can chat, then click this and watch facts
  // appear). Needs a loaded model; the route 503s otherwise.
  if (statusEl) statusEl.textContent = "Distilling facts from recent chats...";
  try {
    const r = await fetch("/api/memory/consolidate", {
      method: "POST", headers: authHeaders(),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    await refreshMemory();
    const n = data.added || 0;
    const msg = data.status === "skipped"
      ? (data.reason === "privacy"
          ? "Memory writes are off in privacy mode."
          : "Nothing new to remember yet.")
      : (n ? `Added ${n} new fact${n === 1 ? "" : "s"}.`
           : "No new durable facts found.");
    if (statusEl) statusEl.textContent = msg;
    toast(msg);
    return data;
  } catch (e) {
    const msg = "Synthesize failed: " + e.message;
    if (statusEl) statusEl.textContent = msg;
    toast(msg, true);
  }
}

export function openMemoryModal() {
  openModal("Memory - what the model knows about you", (body) => {
    body.appendChild(el("div", "sub", memory.writable
      ? "Durable facts localm remembers about you, added to the prompt while the " +
        "🧠 toggle is on. Memory grows automatically as you chat; edit freely - " +
        "one fact per line; Save replaces the list."
      : "Read-only: privacy mode blocks memory writes (no new traces). " +
        "Existing memory is still recalled while the 🧠 toggle is on."));
    const ta = document.createElement("textarea");
    ta.value = memory.text;
    ta.rows = 14;
    ta.style.width = "100%";
    ta.readOnly = !memory.writable;
    body.appendChild(ta);
    if (memory.writable) {
      const status = el("div", "sub", "");
      status.style.marginTop = "8px";
      const row = el("div");
      row.style.cssText = "margin-top:10px;display:flex;gap:8px;align-items:center";
      const save = el("button", "btn-primary", "Save");
      save.onclick = async () => {
        try {
          const r = await fetch("/api/memory", {
            method: "PUT", headers: authHeaders(),
            body: JSON.stringify({ text: ta.value }),
          });
          const data = await r.json();
          if (!r.ok) throw new Error(data.detail || r.statusText);
          await refreshMemory();
          toast("Memory saved");
          $("modal").style.display = "none";
        } catch (e) {
          toast("Save failed: " + e.message, true);
        }
      };
      const synth = el("button", "btn", "Synthesize now");
      synth.title = "Distil durable facts from your recent chats into memory now";
      synth.onclick = async () => {
        synth.disabled = true;
        await synthesizeMemoryNow(status);
        ta.value = memory.text;              // reflect any newly-added facts
        synth.disabled = false;
      };
      row.appendChild(save);
      row.appendChild(synth);
      body.appendChild(row);
      body.appendChild(status);
    }
  });
}

/* ---- prompt library (personas) ---- */

export const PERSONA_PARAM_IDS = {
  temperature: "p-temperature",
  top_p: "p-top-p",
  top_k: "p-top-k",
  repeat_penalty: "p-repeat-penalty",
  max_tokens: "p-max-tokens",
};

export let personaCache = [];

export async function refreshPersonas() {
  try {
    const r = await fetch("/api/prompts", { headers: authHeaders() });
    if (!r.ok) return;
    personaCache = (await r.json()).prompts;
    const sel = $("p-persona");
    const current = sel.value;
    sel.replaceChildren();
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "(none)";
    sel.appendChild(none);
    for (const p of personaCache) {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.name;
      sel.appendChild(opt);
    }
    if ([...sel.options].some((o) => o.value === current)) sel.value = current;
  } catch (e) { /* server unreachable */ }
}

export function applyPersona(name) {
  const p = personaCache.find((x) => x.name === name);
  if (!p) { toast("No such persona: " + name, true); return false; }
  $("p-system").value = p.system || "";
  for (const [key, id] of Object.entries(PERSONA_PARAM_IDS)) {
    $(id).value = p.params?.[key] ?? "";
  }
  $("p-persona").value = name;
  toast(`Persona '${name}' applied`);
  return true;
}

$("p-persona").onchange = () => {
  const name = $("p-persona").value;
  if (name) applyPersona(name);
};

$("persona-save").onclick = async () => {
  const name = prompt("Persona name:", $("p-persona").value || "");
  if (!name || !name.trim()) return;
  const params = {};
  for (const [key, id] of Object.entries(PERSONA_PARAM_IDS)) {
    const v = $(id).value.trim();
    if (v !== "" && !Number.isNaN(Number(v))) params[key] = Number(v);
  }
  try {
    const r = await fetch("/api/prompts/" + encodeURIComponent(name.trim()), {
      method: "PUT", headers: authHeaders(),
      body: JSON.stringify({ system: $("p-system").value, params }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    await refreshPersonas();
    $("p-persona").value = name.trim();
    toast(`Persona '${name.trim()}' saved`);
  } catch (e) {
    toast("Save failed: " + e.message, true);
  }
};

$("persona-delete").onclick = async () => {
  const name = $("p-persona").value;
  if (!name) { toast("Select a persona first", true); return; }
  if (!confirm(`Delete persona '${name}'? The drawer values stay as they are.`)) return;
  try {
    const r = await fetch("/api/prompts/" + encodeURIComponent(name), {
      method: "DELETE", headers: authHeaders() });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    await refreshPersonas();
    $("p-persona").value = "";
    toast(`Persona '${name}' deleted`);
  } catch (e) {
    toast("Delete failed: " + e.message, true);
  }
};

/* sending */

export async function runCompletion(conv, webDepth = 0, web = null) {
  // R36: per-send web state. `seen` dedupes already-issued queries so the model
  // cannot loop on the same search; `ask` caches the net policy so a transient
  // /v1/config blip mid-loop cannot silently flip approval off; `forced` ensures
  // we only inject the "limit reached, answer now" nudge once per send.
  if (!web) web = { seen: new Set(), ask: null, forced: false };
  await maybeCompactConversation(conv);
  const params = chatParams();
  const webEnabled = $("p-web").checked;
  const messages = [];
  let sysText = params.system || "";
  // Long-term memory is now injected SERVER-SIDE by the chat plugin's inlet hook
  // (query-aware, for every client), gated on the memory_enabled config that the
  // brain toggle drives. We deliberately no longer prepend it here, so it is not
  // injected twice.
  // Always give the model an honesty floor:
  //  - web ON  -> teach the tools so it searches instead of guessing.
  //  - results just injected (explicit /web, toggle off) -> tell it to use and
  //    cite them; the offline-denial floor would contradict results in hand.
  //  - web OFF, no results -> tell it plainly it is offline and must not
  //    fabricate current facts or claim it looked anything up. This is what
  //    stops the model hallucinating instead of admitting it cannot reach the net.
  let webFloor;
  if (webEnabled) webFloor = WEB_TOOL_PROMPT;
  else if (lastTurnHasWebResults(conv)) webFloor = WEB_GROUNDED_PROMPT;
  else webFloor = NO_WEB_PROMPT;
  sysText = (sysText ? sysText + "\n\n" : "") + webFloor;
  if (sysText) messages.push({ role: "system", content: sysText });
  // Server-generated images (/api/ URLs from /generate-image) must not be sent
  // to the model as image parts - replace those messages with a text note.
  const mapped = conv.messages.map((m) => {
    if (Array.isArray(m.content) &&
        m.content.some((p) => p.type === "image_url" &&
                              p.image_url?.url?.startsWith("/api/"))) {
      return { role: m.role,
               content: msgText(m) + "\n[An image was generated and shown to the user.]" };
    }
    // Reasoning blocks are display-only - never resend them as context.
    if (m.role === "assistant" && typeof m.content === "string") {
      return { role: m.role, content: stripThink(m.content) };
    }
    return { role: m.role, content: m.content };
  });
  // Attached documents and knowledge excerpts are stored as separate user
  // rows; some chat templates require strict user/assistant alternation,
  // so consecutive plain-text same-role messages are merged before sending.
  for (const m of mapped) {
    const prev = messages[messages.length - 1];
    if (prev && prev.role === m.role && prev.role !== "system" &&
        typeof prev.content === "string" && typeof m.content === "string") {
      prev.content += "\n\n" + m.content;
    } else {
      messages.push({ role: m.role, content: m.content });
    }
  }

  const body = { model: modelSelect.value, messages, stream: true };
  for (const k of ["temperature", "top_p", "top_k", "repeat_penalty",
                   "max_tokens", "seed"]) {
    if (params[k] !== null && !Number.isNaN(params[k])) body[k] = params[k];
  }
  if (params.grammar) body.grammar = params.grammar;

  const box = $("chat-messages");
  const { body: liveBody } = addMessageRow(box, "assistant", "");
  chat.stick = true;   // R31: a fresh send re-arms autoscroll (follow the reply)
  box.scrollTop = box.scrollHeight;

  const sendBtn = $("chat-send");
  const input = $("chat-input");
  sendBtn.classList.add("stop");
  sendBtn.textContent = "■";
  chat.abort = new AbortController();
  input.disabled = true;
  document.querySelectorAll(".message-actions button").forEach(b => b.disabled = true);

  // VIS-1: did this request carry a user-attached image? If a text-only model
  // rejects it (400), we must drop the image so the chat is not wedged.
  const sentImage = messages.some((m) => Array.isArray(m.content) &&
    m.content.some((p) => p.type === "image_url"));

  let full = "";
  let reasoning = "";   // H4: <think> reasoning now streams in delta.reasoning_content
  let usage = null;
  let finishReason = null;
  let aborted = false;
  let visionRejected = false;
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(body),
      signal: chat.abort.signal,
    });
    if (!r.ok) {
      const detail = await r.text();
      const err = new Error(`${r.status}: ${detail.slice(0, 300)}`);
      err.status = r.status;   // so the catch can recover an image-reject 400
      throw err;
    }
    await readSSE(r, (payload) => {
      if (payload === "[DONE]") return;
      let chunk;
      try { chunk = JSON.parse(payload); } catch { return; }
      if (chunk.usage) usage = chunk.usage;
      if (chunk.choices?.[0]?.finish_reason) finishReason = chunk.choices[0].finish_reason;
      const d = chunk.choices?.[0]?.delta || {};
      const cDelta = d.content || "";
      const rDelta = d.reasoning_content || "";   // H4: reasoning streamed apart
      if (cDelta || rDelta) {
        full += cDelta;
        reasoning += rDelta;
        // Rebuild <think> from the reasoning stream so splitThink renders the
        // collapsible block exactly as before. Back-compat: an older server that
        // still inlines <think> in content also renders (reasoning stays empty).
        renderMarkdown(liveBody,
          reasoning ? "<think>\n" + reasoning + "\n</think>\n" + full : full);
        // R31: follow the stream ONLY while the user is at the bottom. chat.stick
        // is latched by the scroll listener, so scrolling up reliably pauses
        // autoscroll (recomputing nearBottom per token fought the user instead).
        if (chat.stick) box.scrollTop = box.scrollHeight;
      }
    });
  } catch (e) {
    if (e.name === "AbortError") {
      aborted = true;
    } else if (sentImage && e.status === 400 && !full.trim()) {
      // VIS-1: a text-only model rejected the image. Drop it from history so the
      // next turn is text-only and the chat stays usable, instead of re-sending
      // the image every turn (which 400s forever -> all-blank assistant replies).
      visionRejected = true;
      const n = stripUserImages(conv);
      saveConversations(conv);
      toast(n
        ? "This model cannot read images - removed the image. You can keep " +
          "chatting (text only)."
        : "Chat request failed: " + e.message, true);
    } else {
      renderMarkdown(liveBody, full + "\n\n*[error: " + e.message + "]*");
      toast("Chat request failed: " + e.message, true);
    }
  } finally {
    chat.abort = null;
    sendBtn.classList.remove("stop");
    sendBtn.textContent = "➤";
    input.disabled = false;
    document.querySelectorAll(".message-actions button").forEach(b => b.disabled = false);
  }

  // User pressed Stop: leave the partial text on screen but do NOT persist it,
  // read it aloud, or fire the web loop / recurse on a partial reply (BUG-13).
  // U-STOP: make the stop unmistakable - mark the partial as stopped and halt any
  // speech already playing - so a stopped reply is never silently treated as live.
  if (aborted) {
    renderMarkdown(liveBody,
      (reasoning ? "<think>\n" + reasoning + "\n</think>\n" + full : full) +
      (full || reasoning ? "\n\n" : "") + "*[stopped]*");
    try { window.speechSynthesis && window.speechSynthesis.cancel(); } catch (e) { /* no TTS */ }
    return;
  }

  // VIS-1: a vision reject (or any failure that streamed nothing) must NOT
  // persist an empty assistant turn - a blank reply saved every send is the
  // "every turn after the image is empty" wedge. Re-render from real history
  // (which now has the image stripped) and stop here so the chat recovers.
  if (visionRejected || (!full.trim() && !reasoning.trim())) {
    renderChat();
    return;
  }

  // Persist content with <think> rebuilt (same shape as before this change), so
  // reload + splitThink re-render the collapsible block and TTS/visibleText are
  // unaffected.
  const reply = {
    role: "assistant",
    content: reasoning ? "<think>\n" + reasoning + "\n</think>\n" + full : full,
    // NEW-1: record which model produced this turn so the transcript can show a
    // divider when the active model changes between turns (model-switch-indication).
    model: modelSelect.value || undefined,
  };
  if (finishReason === "length") {
    // The reply was cut by the max-tokens budget, not finished by the model.
    reply.truncated = true;
    toast("Reply hit the max-tokens limit - raise “Max tokens” in ⚙ parameters, or reply “continue”", true);
  }
  conv.messages.push(reply);
  saveConversations(conv);
  if (usage) {
    const bits = [`${usage.total_tokens} tok`];
    if (usage.ttft_ms != null) bits.push(`TTFT ${usage.ttft_ms} ms`);
    if (usage.tokens_per_sec != null) bits.push(`${usage.tokens_per_sec} tok/s`);
    $("chat-usage").textContent = bits.join(" · ");
    
    // Update context gauge
    const gaugeContainer = $("context-gauge-container");
    const gaugeBar = $("context-gauge-bar");
    if (gaugeContainer && gaugeBar && usage.context_capacity) {
      const pct = Math.min(100, Math.max(0, (usage.total_tokens / usage.context_capacity) * 100));
      gaugeBar.style.width = pct + "%";
      gaugeBar.className = "context-gauge-bar" + (pct > 90 ? " danger" : (pct > 75 ? " warning" : ""));
      gaugeContainer.classList.add("visible");
    } else if (gaugeContainer) {
      gaugeContainer.classList.remove("visible");
    }
  }
  renderChat();

  // Web-access loop: when the model requested a search/page and the toggle
  // is on, run it and let the model continue - bounded rounds per send.
  const canWeb = webEnabled && webDepth < WEB_MAX_ROUNDS;
  const nextCall = canWeb ? parseWebCall(full) : null;
  if (nextCall) {
    // R36: dedupe - the model re-issuing a search it already ran this send is the
    // loop. Do not repeat it; tell the model the results are already in hand and
    // to answer from them, and end the web rounds for this send.
    const key = nextCall.name + ":" + String(
      (nextCall.args && (nextCall.args.query || nextCall.args.url)) || "")
      .trim().toLowerCase();
    if (web.seen.has(key)) {
      conv.messages.push({
        role: "user", web: true,
        content:
          "[duplicate web request] You already ran that exact search this turn; " +
          "its results are above. Do not search again - answer now from those " +
          "results and cite the sources, or say plainly if they are insufficient.",
      });
      saveConversations(conv);
      renderChat();
      await runCompletion(conv, WEB_MAX_ROUNDS, web);   // stop web; force an answer
      return;
    }
    // net_mode=ask: approve each MODEL-INITIATED request before it runs (WEB-ask).
    // The explicit /web command is direct consent and is NOT routed through here.
    // Cache the policy for this send so a mid-loop /v1/config blip cannot flip it.
    if (web.ask === null) web.ask = await webModeIsAsk();
    const approved = web.ask ? await confirmWebRequest(nextCall) : true;
    if (!approved) {
      conv.messages.push({
        role: "user", web: true,
        content:
          "[web access denied] The user declined this web request. Do not claim " +
          "you searched or browsed; answer from what you already know, or say " +
          "plainly that you could not look it up.",
      });
      saveConversations(conv);
      renderChat();
      await runCompletion(conv, WEB_MAX_ROUNDS, web);   // no further web rounds this send
      return;
    }
    web.seen.add(key);
    await runWebCall(conv, nextCall);
    await runCompletion(conv, webDepth + 1, web);
  } else if (canWeb && looksLikeWebToolAttempt(full)) {
    // The model tried to call a web tool but emitted a block we could not
    // parse. Re-prompt for the exact format instead of letting the un-grounded
    // reply stand (it would otherwise read as a confident, un-searched answer).
    conv.messages.push({
      role: "user", web: true,
      content:
        "[tool-call format] That looked like a web tool call, but I could not " +
        "parse it. Re-emit it EXACTLY like this and nothing else:\n" +
        '<tool_call>{"name": "web_search", "args": {"query": "..."}}</tool_call>\n' +
        "If you did not mean to search, answer in plain text and do not claim " +
        "you accessed the web.",
    });
    saveConversations(conv);
    renderChat();
    await runCompletion(conv, webDepth + 1, web);
  } else if (webEnabled && webDepth === WEB_MAX_ROUNDS && !web.forced &&
             (parseWebCall(full) || looksLikeWebToolAttempt(full))) {
    // R36: web rounds are used up but the model is STILL trying to search instead
    // of answering (the "never synthesizes an answer" symptom). Force exactly one
    // synthesizing turn from the results already gathered, then accept its answer.
    web.forced = true;
    conv.messages.push({
      role: "user", web: true,
      content:
        "[web search limit reached] You have used the maximum web lookups for " +
        "this turn. Stop searching and answer the question now using the results " +
        "already provided above, citing the sources; if they are insufficient, " +
        "say so plainly.",
    });
    saveConversations(conv);
    renderChat();
    await runCompletion(conv, WEB_MAX_ROUNDS + 1, web);
    return;
  } else if ($("p-speak").checked && full) {
    speak(full);   // read the finished reply aloud (offline browser voices)
  }
}

/** Query the selected knowledge collection and inject cited excerpts. */
export async function retrieveKnowledge(conv, query) {
  const kb = $("p-kb").value;
  if (!kb || !query) return;
  try {
    const r = await fetch(
      `/api/rag/collections/${encodeURIComponent(kb)}/query`, {
        method: "POST", headers: authHeaders(),
        body: JSON.stringify({ query, k: 4 }),
      });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    if (!data.hits.length) return;
    const basename = (p) => p.split(/[\\/]/).pop();
    const lines = data.hits.map((h, i) =>
      `[${i + 1}] ${basename(h.source)}:${h.pos}\n${h.text.slice(0, 900)}`);
    conv.messages.push({
      role: "user", tag: "kb",
      content:
        `[Excerpts from the "${kb}" collection relevant to: ` +
        `${query.slice(0, 120)}]\n\n` + lines.join("\n\n") +
        "\n\nUse these excerpts where relevant and cite them as [1], [2]… " +
        "If they don't answer the question, say so before answering from " +
        "general knowledge.",
    });
    saveConversations(conv);
    renderChat();
  } catch (e) {
    toast("Knowledge retrieval failed: " + e.message, true);
  }
}

/** Populate the params-drawer knowledge selector. pages.js calls this after
 *  collections change on the Knowledge page. */
export async function refreshKbSelect() {
  try {
    const r = await fetch("/api/rag/collections", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    const sel = $("p-kb");
    const current = sel.value;
    sel.replaceChildren();
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "(none)";
    sel.appendChild(none);
    for (const c of data.collections) {
      const opt = document.createElement("option");
      opt.value = c.name;
      opt.textContent = `${c.name} (${c.n_docs} docs, ${c.n_chunks} chunks)`;
      sel.appendChild(opt);
    }
    if ([...sel.options].some((o) => o.value === current)) sel.value = current;
  } catch (e) { /* server unreachable - selector stays as-is */ }
}

export async function sendChat() {
  const input = $("chat-input");
  const text = input.value.trim();
  if (!text && chat.attachments.length === 0 && chat.docs.length === 0) return;
  if (chat.abort) {
    // A reply is still streaming. Tell the user how to act instead of silently
    // swallowing the send (the send button is a Stop control while streaming).
    toast("Reply still streaming - press the stop button to interrupt", true);
    return;
  }

  if (text.startsWith("/")) {
    input.value = "";
    autoGrow(input);
    handleSlashSubmit(text, execChatCommand);
    return;
  }

  if (!currentConv()) newConversation();
  const conv = currentConv();
  const isFirstMessage = conv.messages.length === 0;

  // Attached documents come first so the model reads them before the question
  for (const doc of chat.docs) {
    conv.messages.push({
      role: "user", tag: "doc",
      content: `[Attached document: ${doc.name}` +
        (doc.truncated ? " (truncated)" : "") + `]\n${doc.text}`,
    });
  }
  chat.docs = [];

  let content;
  if (chat.attachments.length) {
    content = [{ type: "text", text }];
    for (const att of chat.attachments) {
      content.push({ type: "image_url", image_url: { url: att.dataUri } });
    }
  } else {
    content = text || "Please read the attached document(s).";
  }
  conv.messages.push({ role: "user", content });
  chat.attachments = [];
  renderAttachChips();

  if (isFirstMessage) {
    conv.title = text.slice(0, 42) + (text.length > 42 ? "…" : "") || "Document chat";
    renderConvList();
  }
  saveConversations(conv);   // user message persists even if the reply dies
  input.value = "";
  autoGrow(input);
  renderChat();
  await retrieveKnowledge(conv, text);
  await runCompletion(conv);
}

export function exportConversation() {
  const conv = currentConv();
  if (!conv || !conv.messages.length) { toast("Nothing to export", true); return; }
  const lines = [`# ${conv.title}`, ""];
  for (const m of conv.messages) {
    lines.push(`**${m.role === "user" ? "You" : (modelCache.active || "Model")}:**`, "", msgText(m), "");
    if (msgImages(m).length) lines.push(`*[${msgImages(m).length} image(s) attached]*`, "");
  }
  // Include alternative branches that compaction summarised away and archived
  // (chat.js pruneBranches -> conv.droppedBranches). This is what makes those
  // branches genuinely RECOVERABLE rather than only retained-but-unreachable:
  // the export is their one read path (memory-audit 2026-07-02 F5 follow-up).
  const dropped = conv.droppedBranches || [];
  if (dropped.length) {
    lines.push("---", "",
      `## Archived alternative branches (${dropped.length})`,
      "*These alternative timelines were summarised away by context compaction "
      + "and preserved here so they are not lost.*", "");
    dropped.forEach((tail, i) => {
      lines.push(`### Branch ${i + 1}`, "");
      for (const m of (tail || [])) {
        lines.push(`**${m.role === "user" ? "You" : (modelCache.active || "Model")}:**`,
          "", msgText(m), "");
      }
    });
  }
  const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = conv.title.replace(/[^\w\- ]+/g, "").trim().replace(/\s+/g, "_") + ".md";
  a.click();
  URL.revokeObjectURL(a.href);
}

$("chat-send").onclick = () => {
  if (chat.abort) { chat.abort.abort(); return; }
  sendChat();
};
/** Enter sends, Shift+Enter inserts a newline, Ctrl/Cmd+Enter also sends.
 *  Skipped while an IME composition or the slash-command menu is active
 *  (the menu's own keydown handler picks the highlighted command).
 *  U1: Also blocked (preventDefault, no send) while a reply is actively
 *  streaming - the input is disabled at that point, but we guard here too
 *  so a race or accessibility path cannot slip a second message through. */
export function composerEnterToSend(e, send) {
  if (e.key !== "Enter" || e.isComposing) return;
  if (e.shiftKey) return;   // newline - the textarea's default behaviour
  const menu = e.target.closest(".composer-wrap")?.querySelector(".slash-menu");
  if (menu && menu.style.display !== "none") return;
  // U1: block the form-submit path while streaming (not just visually).
  // preventDefault here means the Enter never becomes a newline and the send
  // function is never called - the chat.abort check in sendChat() is the
  // second line of defence, but preventing dispatch is the correct first one.
  if (chat.abort) { e.preventDefault(); return; }
  e.preventDefault();
  send();
}

$("chat-input").addEventListener("keydown", (e) => composerEnterToSend(e, sendChat));
$("chat-input").addEventListener("input", (e) => autoGrow(e.target));
// R31: latch autoscroll on the user's actual scroll position. Scrolling up sets
// chat.stick=false (the streaming loop then leaves the viewport alone); returning
// to the bottom re-arms it. A programmatic scroll-to-bottom lands near the bottom
// and so keeps it armed, so this never fights itself.
$("chat-messages").addEventListener("scroll", () => {
  chat.stick = nearBottom($("chat-messages"));
});
// R34: persist the Web-access and Speak-aloud toggles so they survive a reload
// (privacy mode leaves no trace). hydrateChatToggles restores them on boot.
$("p-web").addEventListener("change", () => {
  if (!chat.privacy) { try { localStorage.setItem("localm.webAccess", $("p-web").checked ? "1" : "0"); } catch (e) { /* storage full/blocked */ } }
});
$("p-speak").addEventListener("change", () => {
  if (!chat.privacy) { try { localStorage.setItem("localm.speakAloud", $("p-speak").checked ? "1" : "0"); } catch (e) { /* storage full/blocked */ } }
});
// The brain toggle drives the server-side memory_enabled config (single-user), so
// injection is decided server-side and applies to every client. Persisted via
// PATCH /v1/config (needs config:write; the owner GUI has it). hydrateChatToggles
// restores the checkbox from config on boot.
$("p-memory").addEventListener("change", () => {
  fetch("/v1/config", {
    method: "PATCH", headers: authHeaders(),
    body: JSON.stringify({ memory_enabled: $("p-memory").checked }),
  }).catch(() => { /* best-effort; a failed save just keeps the old value */ });
});
$("toggle-params").onclick = () => $("params").classList.toggle("open");
$("export-conv").onclick = exportConversation;
$("compact-conv").onclick = async () => {
  const conv = currentConv();
  if (!conv || conv.messages.length <= COMPACT_KEEP) {
    toast("Nothing to compact yet", true);
    return;
  }
  if (chat.abort) { toast("Wait for the current reply to finish", true); return; }
  await compactConversation(conv);
};
$("new-conv").onclick = () => { newConversation(); showView("chat"); };
// Mobile top-bar new-chat button mirrors the sidebar +; also closes the drawer
// if it happened to be open. Guarded for the headless/jsdom DOM.
if ($("mtb-new")) {
  $("mtb-new").onclick = () => { newConversation(); showView("chat"); closeNav(); };
}

