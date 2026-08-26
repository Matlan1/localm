// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - chat (split from app.js). Classic script: it
   shares the one global lexical environment with the other app/* and
   pages/* scripts, so every cross-section reference resolves by bare
   name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { $, authHeaders, autoGrow, confirmDanger, el, fetchImageURL, INSTANCE_SCOPED_KEYS, promptText, readStoredJSON, reconcileInstanceId, renderMarkdown, scrubMarkers, stripThink, toast } from "./helpers.js";
import { emptyState, iconEl } from "./icons.js";
import { modelCache, modelSelect } from "./models-sidebar.js";
import { openMemoryModal, runCompletion, speak, setWebAskSession } from "./settings-perf.js";
import { showView } from "./tabs.js";
import { applyCoderRailSide } from "./coder.js";

/* ================================================================ */
/*  Chat                                                             */
/* ================================================================ */

export const chat = {
  conversations: readStoredJSON("localm.conversations", []),
  activeId: null,
  abort: null,
  attachments: [],   // image attachments: {name, dataUri}
  docs: [],          // document attachments: {name, text, chars, truncated}
  ctxMax: 16384,     // context ceiling - refreshed from /v1/config
  systemDefault: "", // default system prompt from Settings; a blank drawer inherits it
  toolGrammar: true, // chat_tool_grammar from /v1/config - grammar-constrain web-tool calls
  // Set once a backend REFUSES a web-tool grammar request; never cleared until
  // reload. Kept separate from toolGrammar (the config preference, refreshed
  // by refreshCtxLimit's 30s poll) - a config refresh must not resurrect a
  // grammar THIS backend has already demonstrated it cannot honour.
  toolGrammarUnsupported: false,
  privacy: false,    // server in privacy mode → conversations not persisted
  persist: false,    // non-privacy: conversations sync to the server store
  stick: true,       // R31: follow the stream to the bottom until the user scrolls up
  // AUD-INSTANCEID residual 2: true ONLY once reconcileInstanceId returns
  // "confirmed" - a real, positive match against the connected backend's
  // instance id. Defaults to true so the FIRST paint (before refreshCtxLimit's
  // round trip lands) still renders optimistically, but refreshCtxLimit
  // overwrites it on EVERY path - including the two failure paths, which used
  // to leave this default standing (below) - so by the time anything acts on
  // it the value reflects the last real answer, never a guess. Deliberately
  // NOT true for "unknown" (an old server with no instance_id field, or a
  // /v1/config round trip that failed or was rejected) - unknown state may
  // still RENDER whatever is cached (see refreshCtxLimit below), but must
  // never be read as permission to upload. See initServerConversations, which
  // gates re-uploading a not-yet-synced local conversation on this flag.
  instanceMatch: true,
  // The raw tri-state itself ("confirmed" | "mismatched" | "unknown"), kept
  // alongside instanceMatch so a caller that needs to tell "unknown" and
  // "mismatched" apart (initServerConversations' render gate, below) does not
  // have to re-derive it - only instanceMatch (confirmed-only) is the right
  // gate for anything that WRITES to the backend.
  instanceState: "confirmed",
  // True once the in-memory privacy-mode wipe below has run for this page
  // load. refreshCtxLimit() also runs on a 30s poll for the tab's whole
  // lifetime (init.js), and the wipe must fire only the FIRST time privacy
  // mode is confirmed - never again while a conversation is live, or a slow
  // enough reply (or just enough elapsed time) makes the poll erase the
  // user's own in-flight or just-landed message, misreporting a real answer
  // as if nothing was said.
  privacyWiped: false,
};

// System-injected messages (retrieved web/kb/doc content) are pushed with
// role:"user" so the MODEL reads them as conversation context - changing that
// would be a model-behaviour change, not a labelling fix. noteLabel() is the
// single source of truth for the human-facing override instead: renderChat()
// below AND exportConversation() (settings-perf.js) both call it, so a message
// flagged here can never render as "You:" in the live UI but slip through
// unlabelled in an exported transcript, or the reverse. Extracted after
// NEW-WEBSEARCH-UX-EXPORT-ATTRIBUTES-TOOL-RESULTS-TO-USER - the export path
// used to check only m.role, attributing raw web-search/knowledge-base/
// document text to the user as though they had typed it.
const NOTE_LABELS = { web: "Web", doc: "Doc", kb: "Sources" };

export function noteLabel(m) {
  const tag = m.tag || (m.web ? "web" : null);
  return tag ? NOTE_LABELS[tag] : null;
}

// Conversation compaction mirrors localm/inference/compact.py: once the estimate
// passes 70% of the ceiling, summarise the OLDEST turns and keep the recent ones
// verbatim. R44: keep recent turns by token budget (not a flat last-4), feed the
// summariser whole messages truncated at word boundaries with reasoning stripped,
// and sanitise the returned summary so half-words / <think> never re-enter
// context. Never blocks chat.
export const COMPACT_RATIO = 0.7;
export const COMPACT_KEEP = 4;            // floor: always keep at least this many recent turns
export const COMPACT_TARGET = 0.5;        // R44: keep recent turns verbatim up to ~50% of the ceiling

/** R44: rough token estimate - ~4 chars/token plus a small per-message overhead
 *  for the role and delimiters. Coarse, but less skewed than a bare length/4. */
export function estimateTokens(text) {
  return Math.ceil((text || "").length / 4) + 4;
}

export function estimateConvTokens(conv) {
  let total = estimateTokens($("p-system").value || "");
  for (const m of conv.messages) {
    total += estimateTokens(msgText(m)) + msgImages(m).length * 750;
  }
  return total;
}

/** R44: truncate at a word boundary with an explicit marker, instead of a raw
 *  mid-word slice that fed half-words (e.g. "REA") into the summariser. */
export function truncateAtWord(text, max) {
  if (!text || text.length <= max) return text || "";
  const cut = text.slice(0, max);
  const sp = cut.lastIndexOf(" ");
  return (sp > max * 0.6 ? cut.slice(0, sp) : cut).trimEnd() + " ...[truncated]";
}

export async function compactConversation(conv) {
  if (conv.messages.length <= COMPACT_KEEP) return false;
  // R44: keep as many of the most-recent turns verbatim as fit in COMPACT_TARGET
  // of the ceiling (at least COMPACT_KEEP), summarising only what is older -
  // instead of always discarding everything but the last 4 turns.
  const budget = (chat.ctxMax && chat.ctxMax > 0) ? COMPACT_TARGET * chat.ctxMax : 0;
  let keepCount = 0, used = 0;
  for (let i = conv.messages.length - 1; i >= 0; i--) {
    const t = estimateTokens(msgText(conv.messages[i])) +
              msgImages(conv.messages[i]).length * 750;
    const overBudget = budget > 0 ? (used + t > budget) : true;
    if (keepCount >= COMPACT_KEEP && overBudget) break;
    used += t;
    keepCount++;
  }
  keepCount = Math.max(COMPACT_KEEP, Math.min(keepCount, conv.messages.length - 1));
  const older = conv.messages.slice(0, -keepCount);
  const recent = conv.messages.slice(-keepCount);
  if (!older.length) return false;

  // R44: feed whole messages truncated at a word boundary, with reasoning blocks
  // stripped, so the summariser never sees half-words or display-only <think>.
  const excerpt = older.map((m) =>
    `${m.role.toUpperCase()}: ${truncateAtWord(stripThink(msgText(m)), 1200)}`).join("\n\n");

  let summary = "";
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model: modelSelect.value,
        messages: [{
          role: "user",
          content: "Summarise the following conversation in under 200 words. " +
            "Keep facts, names, decisions, and anything the user asked to " +
            "remember. Reply with the summary only.\n\n" + excerpt,
        }],
        max_tokens: 400,
        temperature: 0.3,
        stream: false,
      }),
    });
    if (r.ok) {
      const data = await r.json();
      summary = (data.choices?.[0]?.message?.content || "").trim();
    }
  } catch (e) { /* summarisation unavailable - fall back to a note below */ }
  // R44: sanitise the summary so leaked <think>/markers never re-enter context.
  summary = stripThink(scrubMarkers(summary)).trim();

  const bridge = summary
    ? [{ role: "user", content: "[Conversation summary]\n" + summary },
       { role: "assistant", content: "Understood. Continuing from this summary." }]
    : [{ role: "user", content:
         "[Earlier conversation was trimmed to fit the context window; " +
         "the recent messages below are intact.]" },
       { role: "assistant", content: "Understood." }];

  conv.messages = [...bridge, ...recent];
  // Forks anchored in the summarised-away region can no longer be reached by
  // the ‹ › navigation; archive their content and DISCLOSE the count instead
  // of deleting them silently (memory-audit 2026-07-02).
  const lostBranches = pruneBranches(conv);
  saveConversations(conv);
  renderChat();
  const base = summary ? "Older messages summarised to free context"
                       : "Older messages trimmed (summarisation unavailable)";
  toast(lostBranches
    ? `${base} (${lostBranches} alternative branch${lostBranches > 1 ? "es" : ""} `
      + `from the older messages ${lostBranches > 1 ? "were" : "was"} archived)`
    : base);
  return true;
}

export async function maybeCompactConversation(conv) {
  if (!chat.ctxMax || chat.ctxMax <= 0) return;
  if (estimateConvTokens(conv) >= COMPACT_RATIO * chat.ctxMax) {
    await compactConversation(conv);
  }
}

let _instanceUnknownWarned = false;   // one warning per breakage, re-armed on success

/** AUD-INSTANCEID residual 2: /v1/config could not be read, so this load has NO
 *  answer about which backend is behind this origin. That is not a mismatch (we
 *  wipe nothing) and it is emphatically not a match either, so it lands on the
 *  same "unknown" state an older server without an instance_id produces: keep
 *  showing whatever is cached, never authorise a write into a store we have not
 *  identified. Loud rather than silent (AGENTS.md rule 5) - the round trip that
 *  produced this used to be swallowed with no trace at all. */
function markInstanceUnknown(why) {
  chat.instanceState = "unknown";
  chat.instanceMatch = false;
  // The STATE is set every time; only the LINE is deduped. refreshCtxLimit is
  // polled every 30s for the tab's whole lifetime (init.js), so an unreachable
  // server would otherwise print two lines a minute for as long as the tab is
  // open, and a flood is how a real warning gets ignored. Same one-per-breakage
  // shape as _convPushWarned below, re-armed the moment a verdict lands.
  if (_instanceUnknownWarned) return;
  _instanceUnknownWarned = true;
  console.warn("localm: /v1/config could not be read (" + why + ") - the " +
    "backend instance is unconfirmed, so cached conversations are still shown " +
    "but will not be uploaded until a future check confirms this backend");
}

export async function refreshCtxLimit() {
  // Set once reconcileInstanceId has returned a real verdict for THIS call, so
  // a throw from the rendering work further down (which runs only AFTER the
  // verdict) cannot be mistaken for "we never got an answer" and downgrade a
  // confirmed "mismatched" to "unknown" - that would re-permit rendering the
  // foreign cache the mismatch branch exists to remove.
  let reconciled = false;
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (!r.ok) {
      // A resolved-but-failed round trip is NOT the "missing information" the
      // permissive defaults were chosen for: the server answered and we still
      // do not know who it is.
      markInstanceUnknown("HTTP " + r.status);
    } else {
      const cfg = await r.json();
      // Prefer the resolved ceiling (VRAM-derived under ctx_auto) over the
      // static config value - compaction should track what the model can
      // actually hold.
      chat.ctxMax = cfg.effective_ctx_max ?? cfg.n_ctx_max ?? 16384;
      // The Settings "Default system prompt": a chat with a blank System prompt
      // field inherits this (the per-chat drawer overrides it).
      chat.systemDefault = (cfg.chat_system_prompt || "").trim();
      chat.toolGrammar = cfg.chat_tool_grammar !== false;
      // The coder session rail's side. Applied here because this is the one boot
      // round trip that already has the config in hand - a second fetch just for a
      // panel side would be a request per page load for a value that never changes
      // between them. Unknown/absent falls through to the CSS default (right), so
      // an older server or a partial payload lays out correctly rather than blank.
      applyCoderRailSide(cfg.coder_rail_side);

      // AUD-INSTANCEID: confirm this browser origin is actually talking to the
      // SAME backend data directory whose cache it holds, BEFORE any of that
      // cache is trusted for merging/uploading. Three states now (residual 2):
      // "confirmed" is the only one that may upload; "mismatched" wipes the
      // instance-scoped localStorage keys AND the in-memory state a synchronous
      // boot-time read may already have loaded, then repaints so nothing
      // foreign is ever shown or re-uploaded; "unknown" (no server info - an
      // old server, or a round trip that failed, which markInstanceUnknown
      // below routes here rather than leaving the boot defaults standing)
      // keeps whatever is already rendered - permissive about DISPLAY, never
      // about upload.
      const instanceState = reconcileInstanceId(cfg.instance_id);
      reconciled = true;
      _instanceUnknownWarned = false;   // a verdict landed: re-arm the warning
      chat.instanceState = instanceState;
      chat.instanceMatch = instanceState === "confirmed";
      if (instanceState === "mismatched") {
        chat.conversations = [];
        chat.activeId = null;
        convUI.collapsed = new Set();
        renderConvList();
        renderChat();
        // AUD-INSTANCEID: reconcileInstanceId already wiped "localm.coderCwd"
        // from localStorage, but init.js reads that key into the Coder tab's
        // "Project directory" INPUT VALUE synchronously at boot (before this
        // round trip can resolve) whenever ANY instance id was previously
        // cached - so on a confirmed MISMATCH the input itself still holds a
        // different install's host filesystem path until something corrects
        // it here. Nothing else re-reads that key into the field afterwards.
        const cwdInput = $("setup-cwd");
        if (cwdInput) cwdInput.value = "";
        // Residual 1: the landing PAGE is the same kind of leftover as the
        // input above - a savedView restored synchronously at boot (init.js,
        // gated only on instanceCacheTrusted()'s PRESENCE check, not a
        // confirmed match) can already have switched to a foreign install's
        // last-open page before this round trip lands. Nothing else re-asserts
        // the view afterwards, so a confirmed mismatch must correct it here,
        // the same way it already corrects the cache and the cwd input.
        showView("chat");
      }

      // Privacy mode: conversations live in memory only - wipe anything a
      // previous non-privacy session left behind (in-memory AND on disk) and
      // show the hint.
      chat.privacy = cfg.effective_mode === "privacy";
      if (chat.privacy) {
        for (const key of INSTANCE_SCOPED_KEYS) localStorage.removeItem(key);
        // GUI-LIVE-WIPE: the destructive in-memory reset below must run only
        // the FIRST time privacy mode is confirmed for this page load, not on
        // every refreshCtxLimit() call - this function is also polled every
        // 30s for the tab's whole lifetime (init.js). Re-running it against a
        // LIVE conversation (the user's own current chat, not a leftover from
        // a previous non-privacy session) wipes an in-flight or just-answered
        // message out of the transcript with no error, before the user ever
        // gets to read the reply.
        if (!chat.privacyWiped) {
          chat.privacyWiped = true;
          // The localStorage wipe above used to leave the already-populated
          // in-memory chat.conversations (and the sidebar list already painted
          // from it) untouched, so the stale list stayed on screen for the
          // whole session even though the on-disk wipe "succeeded" (AUD-PRIV-2).
          chat.conversations = [];
          chat.activeId = null;
          convUI.collapsed = new Set();
          setWebAskSession(null);                        // R27: forget the session choice
          renderConvList();
          renderChat();
          // AUD-INSTANCEID/AUD-PRIV-2: same DOM-field gap as the mismatch branch
          // above - the Coder tab's "Project directory" input was already
          // populated from "localm.coderCwd" at boot before this could run, and
          // nothing else re-reads that key into the field, so the wipe above
          // needs a matching correction here too.
          const cwdInputPriv = $("setup-cwd");
          if (cwdInputPriv) cwdInputPriv.value = "";
        }
        const h = document.querySelector("#conversations h3");
        if (h && !document.getElementById("privacy-hint")) {
          const hint = document.createElement("div");
          hint.id = "privacy-hint";
          hint.className = "privacy-hint";
          hint.textContent = "privacy mode - this session only";
          hint.title = "The server runs in privacy mode: conversations are " +
            "not saved (here or on disk) and vanish on reload. Export still works.";
          h.after(hint);
        }
      } else {
        // GUI-LIVE-WIPE re-arm: privacyWiped only guards against re-wiping a LIVE
        // conversation while privacy mode stays on for this page load (the 30s
        // poll). Once the server leaves privacy mode, any leftovers painted here
        // are non-privacy content the next privacy-mode confirmation is supposed
        // to wipe (AUD-PRIV-2) - without this reset, a later restart back into
        // privacy mode would suppress that wipe and leave them on screen for the
        // rest of the tab's life.
        chat.privacyWiped = false;
      }
      hydrateChatToggles(cfg);   // R34: reflect saved choice / global net policy
    }
  } catch (e) {
    // The rest of this function stays best-effort (a repaint that throws must
    // not break the 30s poll), but the instance guard is not: a rejected fetch
    // - server down, connection dropped, TLS refused - is the loudest possible
    // "no answer", and leaving the boot defaults standing made it read as a
    // confirmed match. Only claim it if the verdict itself never landed.
    if (!reconciled) markInstanceUnknown(e && e.message ? e.message : String(e));
  }
}

// LM-DA-047: privacy mode's "no traces, not even localStorage" guarantee used
// to rest on two independently hand-maintained mechanisms - a per-call-site
// `if (!chat.privacy) localStorage.setItem(...)` guard at every write site,
// and helpers.js's INSTANCE_SCOPED_KEYS wipe list - with nothing checking they
// agreed, and two keys had drifted (write-gated but never wiped). Every
// instance-scoped write now goes through this one function instead, so the
// write-gate decision lives in exactly one place.
export function lsSetScoped(key, value) {
  if (chat.privacy) return;
  if (!INSTANCE_SCOPED_KEYS.includes(key)) {
    // Loud, not silent (AGENTS.md rule 5): a key reaching here that is not in
    // the wipe list is LM-DA-047's exact failure mode. Warn rather than throw
    // so a real write still succeeds; tests-js/privacy-scoped-keys.test.mjs
    // source-scans every lsSetScoped call site and fails before this ever
    // ships, so this branch is a live backstop, not the primary defense.
    console.warn(`localm: "${key}" written via lsSetScoped but missing from ` +
      "INSTANCE_SCOPED_KEYS (helpers.js) - add it so a privacy-mode or " +
      "instance-mismatch wipe covers it too.");
  }
  try { localStorage.setItem(key, value); }
  catch (e) { /* storage full/blocked - callers already tolerate a miss */ }
}

// R34: the per-chat Web-access and Speak-aloud toggles used to reset to OFF on
// every load, so the user had to re-enable them in every session. Reflect the
// user's saved choice; for web, when there is no saved choice fall back to the
// global net policy (net_mode=allow auto-enables web; ask/off leave it off so
// consent still applies). Writes are gated on privacy mode (no traces there).
export function hydrateChatToggles(cfg) {
  const webEl = $("p-web"), speakEl = $("p-speak");
  if (!webEl || !speakEl) return;
  const lsGet = (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } };
  const savedSpeak = chat.privacy ? null : lsGet("localm.speakAloud");
  if (savedSpeak !== null) speakEl.checked = savedSpeak === "1";
  const savedWeb = chat.privacy ? null : lsGet("localm.webAccess");
  if (savedWeb !== null) webEl.checked = savedWeb === "1";
  else if (cfg && cfg.net_mode === "allow") webEl.checked = true;
  // The brain toggle mirrors the server-side memory_enabled config (default on).
  const memEl = $("p-memory");
  if (memEl && cfg && typeof cfg.memory_enabled === "boolean")
    memEl.checked = cfg.memory_enabled;
}
window.hydrateChatToggles = hydrateChatToggles;

export function saveConversations(changed) {
  if (chat.privacy) return;   // privacy mode: no traces, not even localStorage
  // R40: do NOT cache server index-only rows (_meta) - they carry no messages, so
  // caching them would shadow a real local copy and show empty conversations
  // offline. Only full conversations are cached locally.
  const cacheable = chat.conversations.filter((c) => !c._meta);
  try {
    localStorage.setItem("localm.conversations",
      JSON.stringify(cacheable.slice(0, 50)));
  } catch (e) {
    // Quota: drop image-heavy older conversations and retry once
    const slim = cacheable.slice(0, 10);
    try { localStorage.setItem("localm.conversations", JSON.stringify(slim)); } catch {}
  }
  if (changed) pushConversation(changed);
}

/* ---- server-side conversation store (non-privacy modes only) ---- */

export const _convPushTimers = new Map();
let _convPushWarned = false;   // one warning per breakage, re-armed on success

/** Debounced upsert of one conversation to the server store. */
export function pushConversation(conv) {
  if (!chat.persist) return;
  // Brand-new conversations with nothing in them yet aren't worth a file.
  if (!conv.messages.length && conv.title === "New chat") return;
  conv.updated_at = Date.now();
  clearTimeout(_convPushTimers.get(conv.id));
  _convPushTimers.set(conv.id, setTimeout(async () => {
    _convPushTimers.delete(conv.id);
    try {
      const r = await fetch("/api/conversations/" + encodeURIComponent(conv.id), {
        method: "PUT",
        headers: authHeaders(),
        body: JSON.stringify({ title: conv.title,
                               updated_at: conv.updated_at,
                               pinned: !!conv.pinned,
                               folder: conv.folder || null,
                               branches: conv.branches || [],
                               messages: conv.messages }),
      });
      // A resolved-but-failed save (500, 413 on a huge conversation) means
      // server-side persistence quietly stopped; localStorage still holds the
      // copy, so surface it ONCE per breakage (this runs on every debounce
      // tick), re-arming after the next successful save.
      if (!r.ok && !_convPushWarned) {
        _convPushWarned = true;
        console.error("conversation save failed (HTTP " + r.status +
                      ") - server-side persistence may be stale; the local copy is intact");
      } else if (r.ok) {
        _convPushWarned = false;
      }
    } catch (e) { /* offline - localStorage still has the copy; a reachable
                     server that ANSWERS with an error is the r.ok branch
                     above, not this one */ }
  }, 600));
}

export function deleteConversationRemote(convId) {
  if (!chat.persist) return;
  clearTimeout(_convPushTimers.get(convId));
  _convPushTimers.delete(convId);
  fetch("/api/conversations/" + encodeURIComponent(convId), {
    method: "DELETE", headers: authHeaders(),
  }).then((r) => {
    if (!r.ok) throw new Error("HTTP " + r.status);
  }).catch((e) => {
    // Deleting is privacy-relevant and the UI copy is already gone: a failed
    // server delete must not pass silently - the server copy would resurrect
    // on the next load while the user believes it was deleted.
    toast("Could not delete the conversation on the server - it may reappear", true);
    console.error("conversation delete failed: " + (e && e.message ? e.message : e));
  });
}

/** R40: fetch a full conversation's body on demand and replace its index-only
 *  ("_meta") placeholder in chat.conversations. Returns true on success. */
export async function hydrateConversation(conv) {
  if (!conv || !conv._meta) return true;
  try {
    const r = await fetch("/api/conversations/" + encodeURIComponent(conv.id),
                          { headers: authHeaders() });
    if (!r.ok) return false;
    const data = await r.json();
    conv.messages = data.messages || [];
    conv.branches = data.branches || [];
    if (data.title != null) conv.title = data.title;
    conv.pinned = !!data.pinned;
    conv.folder = data.folder || null;
    delete conv._meta;
    saveConversations();   // cache the now-full conversation locally
    return true;
  } catch (e) {
    return false;   // offline - keep the placeholder; renderChat shows the hint
  }
}
window.hydrateConversation = hydrateConversation;

/** Load the server store and merge with the localStorage cache: the newer copy
 *  of each conversation wins; local-only ones are uploaded. R40: the server
 *  list is a lightweight index (no message bodies); each conversation's messages
 *  load lazily on open via hydrateConversation. */
export async function initServerConversations() {
  if (chat.privacy) return;
  try {
    // R40: lightweight, paginated index - no message bodies or data-URI images.
    const r = await fetch("/api/conversations?meta=true&limit=200&offset=0",
                          { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    if (!data.enabled) return;
    chat.persist = true;
    // Remote rows are index-only placeholders until opened.
    const byId = new Map((data.conversations || []).map(
      (c) => [c.id, { ...c, _meta: true, messages: [] }]));
    for (const local of chat.conversations) {
      const remote = byId.get(local.id);
      if (!remote) {
        // AUD-INSTANCEID residual 2: a cached conversation THIS backend's own
        // index does not know about is either (a) a real local-only chat not
        // yet synced back to the SAME instance after a restart (legitimate -
        // keep + push), or (b) a conversation that belongs to an entirely
        // DIFFERENT backend data directory that happens to share this browser
        // origin. Keeping (b) would show foreign chat history in the sidebar,
        // and pushConversation would PERMANENTLY WRITE it into this install's
        // own data directory - the worst part of the cross-instance leak.
        //
        // RENDERING and UPLOADING are different risk levels, so they are
        // gated separately. A CONFIRMED "mismatched" state (chat.instanceState,
        // set by refreshCtxLimit's reconcileInstanceId) already wiped
        // chat.conversations entirely before this ever runs (see refreshCtxLimit),
        // so there is nothing foreign left in this loop to render by the time
        // we get here - the check below is a defensive belt for that same
        // property, not a coincidence this code relies on silently. An
        // "unknown" state (an old server with no instance_id field, or a
        // failed /v1/config round trip) is NOT a confirmed mismatch, so it may
        // keep rendering whatever was already cached (an old server, or an
        // offline boot, must still work) - but only a CONFIRMED match
        // (chat.instanceMatch) may write it into the backend's own store.
        if (chat.instanceState !== "mismatched") byId.set(local.id, local);
        if (chat.instanceMatch) pushConversation(local);
        continue;
      }
      // Keep the local FULL copy unless the server has a strictly newer version
      // (>= keeps it on a tie, so an already-cached conversation is not demoted to
      // an index-only placeholder that would need a needless re-fetch / break offline).
      if ((local.updated_at || 0) >= (remote.updated_at || 0)) {
        byId.set(local.id, local);
        pushConversation(local);
      }
    }
    chat.conversations = [...byId.values()]
      .sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
    if (!chat.activeId && chat.conversations.length) {
      chat.activeId = chat.conversations[0].id;
    }
    await hydrateConversation(currentConv());   // load the active conversation's body
    saveConversations();
    renderConvList();
    renderChat();
    const h = document.querySelector("#conversations h3");
    if (h && !document.getElementById("persist-hint")) {
      const hint = document.createElement("div");
      hint.id = "persist-hint";
      hint.className = "privacy-hint";
      hint.textContent = "history saved on this machine";
      hint.title = "Conversations are stored in the localm data directory " +
        "(chats/) because the server runs in log or full mode. They survive " +
        "browser reloads and profile wipes.";
      h.after(hint);
    }
  } catch (e) { /* store unavailable - localStorage keeps working */ }
}

export function currentConv() {
  return chat.conversations.find((c) => c.id === chat.activeId) || null;
}

export function newConversation() {
  const conv = { id: Date.now().toString(36), title: "New chat", messages: [] };
  chat.conversations.unshift(conv);
  chat.activeId = conv.id;
  saveConversations(conv);
  renderConvList();
  renderChat();
}

/* message content helpers - content is a string or OpenAI multipart list */
export function msgText(m) {
  if (typeof m.content === "string") return m.content;
  return (m.content || []).filter((p) => p.type === "text").map((p) => p.text).join("");
}
export function msgImages(m) {
  if (typeof m.content === "string") return [];
  return (m.content || []).filter((p) => p.type === "image_url")
    .map((p) => p.image_url?.url).filter(Boolean);
}

/** VIS-1: replace every user-attached image (a data: URI) in the conversation
 *  with a short text note, so a model that rejected the image is never asked for
 *  it again. Re-sending a rejected image 400s on every turn and wedges the chat
 *  in endless empty assistant replies; dropping it keeps the chat usable.
 *  Server-generated /api/ images are display-only (already mapped to text before
 *  sending), so they are left alone. Returns the number of images dropped. */
export function stripUserImages(conv) {
  let dropped = 0;
  for (const m of conv.messages) {
    if (m.role !== "user" || !Array.isArray(m.content)) continue;
    const imgs = m.content.filter(
      (p) => p.type === "image_url" && !p.image_url?.url?.startsWith("/api/"));
    if (!imgs.length) continue;
    dropped += imgs.length;
    const text = m.content.filter((p) => p.type === "text")
      .map((p) => p.text).join("");
    m.content = (text ? text + "\n" : "") +
      "[Image removed: this model cannot read images.]";
  }
  return dropped;
}

/* ---- conversation list: search, pin, folders ---- */

export const convUI = {
  search: "",
  collapsed: new Set(readStoredJSON("localm.convCollapsed", [])),
};

export function saveCollapsed() {
  // folder names are conversation-derived, so this is instance-scoped too
  lsSetScoped("localm.convCollapsed", JSON.stringify([...convUI.collapsed]));
}

/** Short excerpt around the first content match, or null (no match). */
export function searchSnippet(conv, term) {
  for (const m of conv.messages) {
    const text = msgText(m);
    const idx = text.toLowerCase().indexOf(term);
    if (idx !== -1) {
      const start = Math.max(0, idx - 18);
      return (start > 0 ? "…" : "") +
        text.slice(start, idx + 50).replace(/\s+/g, " ");
    }
  }
  return null;
}

export function buildConvItem(conv, snippet) {
  const item = el("div", "conv-item" + (conv.id === chat.activeId ? " active" : ""));
  const title = el("span", "title");
  title.appendChild(document.createTextNode(conv.title));
  if (snippet) title.appendChild(el("span", "snippet", snippet));
  title.ondblclick = (e) => {
    e.stopPropagation();
    const input = document.createElement("input");
    input.value = conv.title;
    const commit = () => {
      conv.title = input.value.trim() || conv.title;
      saveConversations(conv);
      renderConvList();
    };
    input.onblur = commit;
    input.onkeydown = (ke) => { if (ke.key === "Enter") input.blur(); };
    item.replaceChild(input, title);
    input.focus();
    input.select();
  };
  item.appendChild(title);

  const pin = el("button", "del" + (conv.pinned ? " pinned-btn" : ""));
  pin.appendChild(iconEl("pin", "ic"));
  pin.title = conv.pinned ? "Unpin" : "Pin to top";
  pin.onclick = (e) => {
    e.stopPropagation();
    conv.pinned = !conv.pinned;
    if (!conv.pinned) delete conv.pinned;
    saveConversations(conv);
    renderConvList();
  };
  item.appendChild(pin);

  const fold = el("button", "del");
  fold.appendChild(iconEl("folder", "ic"));
  fold.title = conv.folder ? `Folder: ${conv.folder} (click to change)`
                           : "Move to a folder";
  fold.onclick = async (e) => {
    e.stopPropagation();
    const name = await promptText("Folder name (empty removes it from the folder):",
                        conv.folder || "");
    if (name === null) return;
    if (name.trim()) conv.folder = name.trim();
    else delete conv.folder;
    saveConversations(conv);
    renderConvList();
  };
  item.appendChild(fold);

  const del = el("button", "del", "×");
  del.title = "Delete conversation";
  del.onclick = (e) => {
    e.stopPropagation();
    chat.conversations = chat.conversations.filter((c) => c.id !== conv.id);
    if (chat.activeId === conv.id) chat.activeId = chat.conversations[0]?.id || null;
    deleteConversationRemote(conv.id);
    saveConversations();
    renderConvList();
    renderChat();
  };
  item.appendChild(del);

  item.onclick = async () => {
    chat.activeId = conv.id;
    renderConvList();
    if (conv._meta) await hydrateConversation(conv);   // R40: load the body on open
    renderChat();
    showView("chat");
  };
  return item;
}

export function renderConvList() {
  const list = $("conv-list");
  list.replaceChildren();
  const term = convUI.search.trim().toLowerCase();

  // Filter: title match shows plain; content match shows a snippet
  const visible = [];
  for (const conv of chat.conversations) {
    if (!term) {
      visible.push({ conv, snippet: null });
    } else if ((conv.title || "").toLowerCase().includes(term)) {
      visible.push({ conv, snippet: null });
    } else {
      const snippet = searchSnippet(conv, term);
      if (snippet !== null) visible.push({ conv, snippet });
    }
  }

  // Group: pinned on top, then folders (A-Z), then the rest
  const pinned = visible.filter((v) => v.conv.pinned);
  const folders = new Map();
  const loose = [];
  for (const v of visible) {
    if (v.conv.pinned) continue;
    if (v.conv.folder) {
      if (!folders.has(v.conv.folder)) folders.set(v.conv.folder, []);
      folders.get(v.conv.folder).push(v);
    } else {
      loose.push(v);
    }
  }

  const addGroup = (icon, label, key, items) => {
    if (!items.length) return;
    if (label) {
      // While searching, groups stay expanded so matches are never hidden
      const collapsed = !term && convUI.collapsed.has(key);
      const head = el("div", "conv-group");
      head.appendChild(document.createTextNode(collapsed ? "▸ " : "▾ "));
      if (icon) head.appendChild(iconEl(icon, "btn-ic"));
      head.appendChild(document.createTextNode(label));
      head.onclick = () => {
        if (convUI.collapsed.has(key)) convUI.collapsed.delete(key);
        else convUI.collapsed.add(key);
        saveCollapsed();
        renderConvList();
      };
      list.appendChild(head);
      if (collapsed) return;
    }
    for (const v of items) list.appendChild(buildConvItem(v.conv, v.snippet));
  };

  addGroup("pin", pinned.length ? "pinned" : "", "::pinned", pinned);
  for (const name of [...folders.keys()].sort()) {
    addGroup("folder", name, "f:" + name, folders.get(name));
  }
  addGroup(null, (pinned.length || folders.size) && loose.length ? "chats" : "",
           "::chats", loose);

  if (term && !visible.length) {
    list.appendChild(el("div", "privacy-hint", "no matching chats"));
  } else if (!term && !chat.conversations.length) {
    list.appendChild(emptyState("chat", "No conversations yet",
      "Start one with the + above."));
  }
}

$("conv-search").addEventListener("input", (e) => {
  convUI.search = e.target.value;
  renderConvList();
});

// A sensible download name for a chat image (VIS-2): from the data: URI's mime,
// or the /api path's basename, falling back to localm-image.png.
export function imageFilename(url) {
  try {
    if (url.startsWith("data:")) {
      const m = url.match(/^data:image\/([a-z0-9.+-]+)/i);
      return "localm-image." + (m ? m[1].replace("jpeg", "jpg") : "png");
    }
    const base = new URL(url, location.origin).pathname.split("/").pop();
    return base && base.includes(".") ? base : "localm-image.png";
  } catch (e) { return "localm-image.png"; }
}

// Copy an image (by its resolved src) to the clipboard. Returns true on success.
// Not every browser/context can write an image to the clipboard, so the caller
// surfaces a fallback instead of silently failing (RULE 5).
export async function copyImageSrc(src) {
  try {
    if (!window.ClipboardItem || !navigator.clipboard || !navigator.clipboard.write)
      return false;
    const blob = await (await fetch(src)).blob();
    await navigator.clipboard.write([new window.ClipboardItem({ [blob.type]: blob })]);
    return true;
  } catch (e) { return false; }
}
window.copyImageSrc = copyImageSrc;

// Full-view image lightbox (VIS-2): a click on a chat image opens it large with
// Save (download to disk) and Copy controls. Closes on the backdrop, the Close
// button, or Escape.
export function openImageLightbox(src, name) {
  if (!src) return;
  const back = el("div", "img-lightbox");
  const panel = el("div", "img-lightbox-panel");
  const full = document.createElement("img");
  full.className = "img-lightbox-img";
  full.src = src;
  full.alt = name || "image";
  const bar = el("div", "img-lightbox-bar");
  const dismiss = () => {
    back.remove();
    document.removeEventListener("keydown", onKey);
  };
  function onKey(e) { if (e.key === "Escape") dismiss(); }
  const save = el("button", "btn-secondary", "Save");
  save.onclick = () => {
    const a = document.createElement("a");
    a.href = src;
    a.download = name || "localm-image.png";
    document.body.appendChild(a);
    a.click();
    a.remove();
  };
  const copy = el("button", "btn-secondary", "Copy image");
  copy.onclick = async () => {
    const ok = await copyImageSrc(src);
    toast(ok ? "Image copied" : "Could not copy the image - use Save instead", !ok);
  };
  const close = el("button", "btn-secondary", "Close");
  close.onclick = dismiss;
  bar.appendChild(save);
  bar.appendChild(copy);
  bar.appendChild(close);
  panel.appendChild(full);
  panel.appendChild(bar);
  back.appendChild(panel);
  back.addEventListener("click", (e) => { if (e.target === back) dismiss(); });
  document.addEventListener("keydown", onKey);
  document.body.appendChild(back);
}
window.openImageLightbox = openImageLightbox;

// F11 observability: human-readable labels for the recall degrade reasons the
// server reports (store._vector_status). Shown in the memory chip's tooltip so
// lexical-only / pending-embedding recall is visible, not invisible.
const MEMORY_DEGRADE_LABELS = {
  no_embedder: "keyword match only - no embedding model installed",
  no_vectors: "keyword match only - memories not embedded yet (run setup-embeddings)",
  low_coverage: "partial semantic match - some memories not embedded yet",
  dim_mismatch: "keyword match only - the embedding model changed",
  query_embed_failed: "keyword match only - could not embed this message",
};

/** F11: build the memory-icon "N" chip for an assistant turn - how many
 *  remembered facts the server injected into that reply. Click opens the memory
 *  modal so a wrong or stale fact steering an answer is correctable in place.
 *  The tooltip lists the facts used and any recall degrade reason. */
export function buildMemoryChip(mem) {
  const n = mem.n | 0;
  const chip = el("button", "mem-chip");
  chip.appendChild(iconEl("memory", "btn-ic"));
  chip.appendChild(document.createTextNode(String(n)));
  const facts = (mem.items || []).map((it) => it && it.text).filter(Boolean);
  const reason = MEMORY_DEGRADE_LABELS[mem.degrade];
  chip.title = [
    `${n} remembered ${n === 1 ? "fact" : "facts"} used for this reply`,
    facts.length ? facts.map((t) => "- " + t).join("\n") : "",
    reason ? "(" + reason + ")" : "",
  ].filter(Boolean).join("\n");
  chip.setAttribute("aria-label", "Memory used: " + n + " - open memory");
  chip.onclick = () => openMemoryModal();
  return chip;
}

export function addMessageRow(container, role, text, opts = {}) {
  const row = el("div", "msg-row " + role + (opts.cls ? " " + opts.cls : ""));
  const mName = opts.model && opts.model !== "MODEL" ? opts.model : (modelCache.active || "Model");
  row.appendChild(el("div", "msg-role",
    opts.label || (role === "user" ? "You" : mName)));
  const body = el("div", "msg-body");
  if (role === "user") {
    // CHAT-1: a user's OWN message renders LITERALLY (exactly as typed). Markdown is
    // the model's output format, not the user's input - so typed *asterisks*, # hashes,
    // or pasted code/URLs are never silently reformatted. pre-wrap (.msg-literal)
    // preserves their line breaks; content is set via textContent so it is inert.
    body.classList.add("msg-literal");
    body.textContent = text;
  } else {
    // opts.final marks a SETTLED message (a reload / renderChat rebuild), not a
    // fresh streaming shell - only then does renderMarkdown show its "(no reply
    // text)" note for a body that rendered to nothing, so a live shell that is
    // briefly empty before its first token never flashes it.
    renderMarkdown(body, text, { final: opts.final });
  }
  for (const url of opts.images || []) {
    const img = document.createElement("img");
    img.className = "msg-img";
    if (url.startsWith("/api/")) {
      // server-side generated image - fetch with auth headers
      fetchImageURL(url).then((u) => (img.src = u)).catch(() => img.remove());
    } else {
      img.src = url;   // data: URI from the user's own attachment
    }
    // Interactable (VIS-2): click to open a full-view lightbox with Save / Copy.
    // imageFilename uses the ORIGINAL url (a data: URI or /api path) for a sane
    // download name, while img.src is the resolved displayable source.
    img.style.cursor = "zoom-in";
    img.title = "Click to view, save, or copy";
    img.addEventListener("click", () => openImageLightbox(img.src, imageFilename(url)));
    body.appendChild(img);
  }
  for (const url of opts.audio || []) {
    const player = document.createElement("audio");
    player.controls = true;
    player.style.width = "100%";
    // bearer-protected file - fetch as a blob with auth headers
    fetchImageURL(url).then((u) => (player.src = u)).catch(() => player.remove());
    body.appendChild(player);
  }
  for (const url of opts.video || []) {
    const player = document.createElement("video");
    player.controls = true;
    player.style.width = "100%";
    player.style.borderRadius = "8px";
    fetchImageURL(url).then((u) => (player.src = u)).catch(() => player.remove());
    body.appendChild(player);
  }
  row.appendChild(body);
  const meta = el("div", "msg-meta");
  const copy = el("button", "copy-btn", "copy");
  copy.onclick = async () => {
    const plain = stripThink(text) || text;
    const firstImg = body.querySelector(".msg-img");
    // Image-only message (e.g. a bare attachment): copy the IMAGE, not empty
    // text - the "copy copied the prompt, not the image" report (VIS-2). For a
    // text+image message the text copy is kept; the image is in the lightbox.
    if (!plain && firstImg && firstImg.src) {
      const ok = await copyImageSrc(firstImg.src);
      copy.textContent = ok ? "copied" : "copy";
      if (!ok) toast("Could not copy the image - open it and use Save", true);
      else setTimeout(() => (copy.textContent = "copy"), 1200);
      return;
    }
    try {
      await navigator.clipboard.writeText(plain);
      copy.textContent = "copied";
      setTimeout(() => (copy.textContent = "copy"), 1200);
    } catch (e) {
      // Matches the image branch above: a real failure (permission denied,
      // insecure context) must never be reported as "copied" - that is
      // claiming a step happened that did not (AGENTS.md rule 5).
      toast("Could not copy - your browser blocked clipboard access", true);
    }
  };
  meta.appendChild(copy);
  if (opts.variant) {
    const nav = el("span", "variant");
    const prev = el("button", "action", "‹");
    prev.title = "Previous variant";
    prev.onclick = opts.variant.prev;
    const next = el("button", "action", "›");
    next.title = "Next variant";
    next.onclick = opts.variant.next;
    nav.appendChild(prev);
    nav.appendChild(el("span", "k", `${opts.variant.k}/${opts.variant.n}`));
    nav.appendChild(next);
    meta.appendChild(nav);
  }
  for (const [label, fn, icon] of opts.actions || []) {
    const btn = el("button", "action");
    if (icon) { btn.appendChild(iconEl(icon, "ic")); btn.title = label; }
    else btn.textContent = label;
    btn.onclick = fn;
    meta.appendChild(btn);
  }
  if (opts.memory && opts.memory.n > 0) meta.appendChild(buildMemoryChip(opts.memory));
  row.appendChild(meta);
  container.appendChild(row);
  return { row, body, meta };
}

export function buildEmptyHint() {
  const div = el("div", "empty-hint");
  const big = el("div", "big");
  big.appendChild(document.createTextNode("local"));
  const accent = el("span", "", "m");
  accent.style.color = "var(--accent)";
  big.appendChild(accent);
  div.appendChild(big);
  div.appendChild(document.createTextNode(
    "Chat with " + (modelCache.active || "your local model") + ". Everything stays on this machine."));
  const tip = el("div", "", "Type / for commands - /generate-image creates images locally.");
  tip.style.marginTop = "10px";
  tip.style.fontSize = "13px";
  div.appendChild(tip);
  return div;
}

// Render (or clear) the tok/s line and context gauge for *usage* - the same
// shape the server sends on a completed turn's final SSE chunk. Shared by the
// live-completion path (settings-perf.js) and renderChat() below, so a
// reloaded or switched-to conversation shows the same figures a live
// generation would have, instead of the last conversation's stats lingering
// on screen or reload wiping them entirely.
export function updateUsageDisplay(usage) {
  const gaugeContainer = $("context-gauge-container");
  const gaugeBar = $("context-gauge-bar");
  if (!usage) {
    $("chat-usage").textContent = "";
    if (gaugeContainer) gaugeContainer.classList.remove("visible");
    return;
  }
  const bits = [`${usage.total_tokens} tok`];
  if (usage.ttft_ms != null) bits.push(`TTFT ${usage.ttft_ms} ms`);
  if (usage.tokens_per_sec != null) bits.push(`${usage.tokens_per_sec} tok/s`);
  $("chat-usage").textContent = bits.join(" · ");
  if (gaugeContainer && gaugeBar && usage.context_capacity) {
    const pct = Math.min(100, Math.max(0, (usage.total_tokens / usage.context_capacity) * 100));
    gaugeBar.style.width = pct + "%";
    gaugeBar.className = "context-gauge-bar" + (pct > 90 ? " danger" : (pct > 75 ? " warning" : ""));
    gaugeContainer.classList.add("visible");
  } else if (gaugeContainer) {
    gaugeContainer.classList.remove("visible");
  }
}

export function renderChat() {
  const box = $("chat-messages");
  box.innerHTML = "";
  const conv = currentConv();
  // R40: a not-yet-loaded conversation (server index row) hydrates its body on
  // first render, then re-renders. Try once per row (a failed/offline load sets
  // _hydrateFailed so we never spin).
  if (conv && conv._meta && !conv._hydrating && !conv._hydrateFailed) {
    conv._hydrating = true;
    hydrateConversation(conv).then((ok) => {
      conv._hydrating = false;
      if (ok) renderChat();
      else conv._hydrateFailed = true;
    });
  }
  if (!conv || conv.messages.length === 0) {
    box.appendChild(buildEmptyHint());
    updateUsageDisplay(null);
    return;
  }
  // NEW-1 model-switch-indication: track the model of the previous assistant turn
  // so a small divider marks where the active model changed mid-conversation.
  let lastAssistantModel = null;
  conv.messages.forEach((m, i) => {
    if (m.role === "assistant" && m.model) {
      const currentModel = m.model === "MODEL" ? (modelCache.active || m.model) : m.model;
      if (lastAssistantModel && currentModel !== lastAssistantModel) {
        box.appendChild(el("div", "model-switch", "switched to " + currentModel));
      }
      lastAssistantModel = currentModel;
    }
    const tag = m.tag || (m.web ? "web" : null);
    const actions = [];
    if (m.role === "user" && !tag && !chat.abort) {
      actions.push(["edit", () => editMessage(conv, i)]);
      actions.push(["revert", () => revertTo(conv, i)]);
    }
    if (m.role === "assistant" && !tag) {
      actions.push(["Speak this reply aloud", () => speak(msgText(m), { toggle: true }), "speak"]);
    }
    if (m.role === "assistant" && i === conv.messages.length - 1 && !chat.abort) {
      actions.push(["regenerate", () => regenerate(conv)]);
    }
    // ‹ k/N › on the first message of a fork point with siblings
    let variant = null;
    const pid = i > 0 ? conv.messages[i - 1].id : "root";
    const rec = (conv.branches || []).find((b) => b.parent === pid);
    if (rec && rec.tails.length > 1) {
      variant = {
        k: rec.current + 1,
        n: rec.tails.length,
        prev: () => switchBranch(conv, i, -1),
        next: () => switchBranch(conv, i, +1),
      };
    }
    const noteSuffix = m.truncated
      ? "\n\n*[stopped at the max-tokens limit - raise “Max tokens” in parameters, or reply “continue”]*"
      : m.stopped
      ? "\n\n*[stopped]*"
      : "";
    addMessageRow(box, m.role, msgText(m) + noteSuffix, {
      images: msgImages(m),
      audio: m.audio ? [m.audio] : [],
      video: m.video ? [m.video] : [],
      actions,
      variant,
      model: m.model,
      memory: m.memory,           // F11: "used N memories" chip (assistant turns)
      cls: tag ? "web-note" : "",
      label: noteLabel(m) || undefined,
      // A settled turn from history: let renderMarkdown surface a "(no reply
      // text)" note if this message rendered to a blank body (empty ```fence).
      final: true,
    });
  });
  // R31: only re-pin to the bottom when the user has not scrolled up. This tail
  // runs on every re-render (incl mid-stream web/finalize), so an unconditional
  // scroll here used to yank a reader back down while a reply was still streaming.
  if (chat.stick) box.scrollTop = box.scrollHeight;
  // Reflect the last turn's usage (or clear it) so the figures shown match
  // whatever conversation is actually on screen - a reload or a switch to a
  // different conversation must not leave a previous turn's stats lingering,
  // and a completed turn's stats must survive a reload instead of vanishing.
  const lastMsg = conv.messages[conv.messages.length - 1];
  updateUsageDisplay(lastMsg && lastMsg.role === "assistant" ? lastMsg.usage : null);
}

/* ---- message branching ----
   conv.messages is always the LIVE linear branch (so compaction, retrieval
   injection, export, and the API mapping stay untouched). Alternative
   timelines are parked at fork points:
     conv.branches = [{parent: <msg id or "root">, tails: [[msg…]…], current}]
   The slot at `current` belongs to the live tail and is only written back
   when switching away. Editing a message or regenerating a reply parks the
   old tail as a sibling instead of destroying it; ‹ k/N › in the message
   meta row navigates between siblings. */

export let _msgIdCounter = 0;

export function msgId(m) {
  if (!m.id) m.id = Date.now().toString(36) + "-" + (_msgIdCounter++);
  return m.id;
}

export function parentIdAt(conv, index) {
  return index > 0 ? msgId(conv.messages[index - 1]) : "root";
}

export function forkRecord(conv, parentId, create) {
  conv.branches = conv.branches || [];
  let rec = conv.branches.find((b) => b.parent === parentId);
  if (!rec && create) {
    rec = { parent: parentId, tails: [], current: 0 };
    conv.branches.push(rec);
  }
  return rec;
}

/** Park the live tail from *index* as a sibling and open a fresh timeline. */
export function forkAt(conv, index) {
  const rec = forkRecord(conv, parentIdAt(conv, index), true);
  if (!rec.tails.length) {
    rec.tails.push(null);          // slot for the pre-existing timeline
    rec.current = 0;
  }
  rec.tails[rec.current] = conv.messages.slice(index);
  rec.tails.push(null);            // slot owned by the new live timeline
  rec.current = rec.tails.length - 1;
  conv.messages = conv.messages.slice(0, index);
}

/** Switch the fork at *index* one sibling left/right (dir = ±1). */
export function switchBranch(conv, index, dir) {
  if (chat.abort) { toast("Wait for the current reply to finish", true); return; }
  const rec = forkRecord(conv, parentIdAt(conv, index), false);
  if (!rec || rec.tails.length < 2) return;
  const n = rec.tails.length;
  const next = (rec.current + dir + n) % n;
  if (next === rec.current) return;
  rec.tails[rec.current] = conv.messages.slice(index);   // park live tail
  conv.messages = conv.messages.slice(0, index).concat(rec.tails[next] || []);
  rec.current = next;
  saveConversations(conv);
  renderChat();
}

/** Drop fork records whose parent message no longer exists anywhere
 *  (active branch or any parked tail) - called after compaction rewrites
 *  old history. Returns the number of ALTERNATIVE (non-current) parked
 *  timelines that were dropped, and archives their content to
 *  conv.droppedBranches so a summarised-away fork is recoverable via export,
 *  never a silent hard delete (memory-audit 2026-07-02). */
export function pruneBranches(conv) {
  if (!conv.branches || !conv.branches.length) return 0;
  const ids = new Set(["root"]);
  for (const m of conv.messages) if (m.id) ids.add(m.id);
  for (const rec of conv.branches) {
    for (const tail of rec.tails) {
      for (const m of tail || []) if (m.id) ids.add(m.id);
    }
  }
  const kept = conv.branches.filter((b) => ids.has(b.parent));
  const dropped = conv.branches.filter((b) => !ids.has(b.parent));
  let lost = 0;
  for (const rec of dropped) {
    const alts = rec.tails.filter((t, ti) => ti !== rec.current && t && t.length);
    if (alts.length) {
      conv.droppedBranches = conv.droppedBranches || [];
      for (const tail of alts) conv.droppedBranches.push(tail);
      // Cap the archive so a long session cannot grow it without bound.
      if (conv.droppedBranches.length > 50) {
        conv.droppedBranches = conv.droppedBranches.slice(-50);
      }
      lost += alts.length;
    }
  }
  conv.branches = kept;
  return lost;
}

export function editMessage(conv, index) {
  // Editing forks the branch tree (forkAt); doing that mid-stream parks the
  // messages before the streaming reply has landed, corrupting the branch
  // state. Bail while a reply streams, like switchBranch / regenerate do.
  if (chat.abort) { toast("Wait for the current reply to finish", true); return; }
  const m = conv.messages[index];
  $("chat-input").value = msgText(m);
  autoGrow($("chat-input"));
  // Fork instead of destroy: the old timeline (this message and everything
  // after it) stays reachable via ‹ › once the edited version is sent.
  forkAt(conv, index);
  saveConversations(conv);
  renderChat();
  $("chat-input").focus();
}

export function regenerate(conv) {
  if (chat.abort) return;
  const last = conv.messages.length - 1;
  if (conv.messages[last]?.role !== "assistant") return;
  forkAt(conv, last);                // park the old reply as a sibling
  saveConversations(conv);
  renderChat();
  runCompletion(conv);
}

/** Count the sibling timelines a revert to *index* would permanently destroy:
 *  every fork point at or after the revert point keeps its alternatives in the
 *  region being removed. Used to warn before reverting *past* a branch point. */
export function branchesLostByRevert(conv, index) {
  let lost = 0;
  for (let i = index; i < conv.messages.length; i++) {
    const pid = i > 0 ? conv.messages[i - 1].id : "root";
    const rec = (conv.branches || []).find((b) => b.parent === pid);
    if (rec && rec.tails.length > 1) {
      lost += rec.tails.filter((t, ti) => ti !== rec.current && t && t.length).length;
    }
  }
  return lost;
}

/** Revert the conversation to *index*: drop this message and everything after it
 *  DESTRUCTIVELY (unlike editMessage, which forks a sibling), and drop the
 *  clicked message back into the composer to modify and resend. Stays in the
 *  SAME branch. Reverting past a fork point destroys the sibling branches in the
 *  removed region, so confirm first when that would happen (the safeguard). */
export function revertTo(conv, index) {
  if (chat.abort) { toast("Wait for the current reply to finish", true); return; }
  if (index < 0 || index >= conv.messages.length) return;
  const text = msgText(conv.messages[index]);
  const lost = branchesLostByRevert(conv, index);

  const apply = () => {
    const orig = conv.messages;
    // Keep only forks that diverge STRICTLY before the revert point. A fork at
    // or after it (parent inside the removed region, or the live tail being
    // reverted) is destroyed; a fork whose parent is not on the live branch
    // (nested in a surviving parked tail) is left for pruneBranches to judge.
    conv.branches = (conv.branches || []).filter((rec) => {
      if (rec.parent === "root") return index > 0;
      const p = orig.findIndex((x) => x.id === rec.parent);
      if (p === -1) return true;
      return p + 1 < index;
    });
    conv.messages = orig.slice(0, index);
    pruneBranches(conv);
    $("chat-input").value = text;
    autoGrow($("chat-input"));
    saveConversations(conv);
    renderChat();
    $("chat-input").focus();
  };

  if (lost > 0) {
    confirmDanger(
      "Revert conversation",
      `This permanently deletes ${lost} alternative branch` +
      `${lost === 1 ? "" : "es"} and everything after this message - it ` +
      "can't be undone. Revert anyway?",
      "Revert", apply);
  } else {
    apply();
  }
}

export function chatParams() {
  const num = (id) => {
    const v = $(id).value.trim();
    return v === "" ? null : Number(v);
  };
  return {
    temperature: num("p-temperature"),
    top_p: num("p-top-p"),
    top_k: num("p-top-k"),
    repeat_penalty: num("p-repeat-penalty"),
    max_tokens: num("p-max-tokens"),
    seed: num("p-seed"),
    system: $("p-system").value.trim() || null,
    grammar: $("p-grammar").value.trim() || null,
  };
}

/* attachments */

export function renderAttachChips() {
  const box = $("attach-chips");
  box.replaceChildren();
  chat.attachments.forEach((att, i) => {
    const chip = el("span", "chip");
    const img = document.createElement("img");
    img.src = att.dataUri;
    chip.appendChild(img);
    chip.appendChild(el("span", "", att.name));
    const rm = el("button", "", "×");
    rm.onclick = () => { chat.attachments.splice(i, 1); renderAttachChips(); };
    chip.appendChild(rm);
    box.appendChild(chip);
  });
  chat.docs.forEach((doc, i) => {
    const chip = el("span", "chip");
    chip.appendChild(iconEl("file", "ic"));
    chip.appendChild(el("span", "", doc.name +
      ` (${(doc.chars / 1000).toFixed(1)}k chars${doc.truncated ? ", trimmed" : ""})`));
    const rm = el("button", "", "×");
    rm.onclick = () => { chat.docs.splice(i, 1); renderAttachChips(); };
    chip.appendChild(rm);
    box.appendChild(chip);
  });
}

/** Document attachment → text via /api/rag/extract. The file is converted
 *  in memory on the server and never written to disk (privacy-clean). */
export async function attachDocument(file) {
  const b64 = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(new Error("could not read file"));
    reader.readAsDataURL(file);
  });
  const r = await fetch("/api/rag/extract", {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ filename: file.name, content_b64: b64 }),
  });
  // A plain-text 500 body is not JSON; fall back to {} so the error message
  // below is the clean statusText, not a JSON parse error.
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  chat.docs.push({ name: data.filename, text: data.text,
                   chars: data.chars, truncated: data.truncated });
  renderAttachChips();
}

/** Ingest files as chat attachments: images inline (data URI), every other
 *  type extracted to text server-side. Shared by the file picker and the
 *  drag-and-drop zone so both behave identically. */
export function addAttachedFiles(files) {
  for (const file of files) {
    if (file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = () => {
        chat.attachments.push({ name: file.name, dataUri: reader.result });
        renderAttachChips();
      };
      reader.readAsDataURL(file);
    } else {
      attachDocument(file).catch((err) =>
        toast(`${file.name}: ${err.message}`, true));
    }
  }
}

$("chat-attach").onclick = () => $("chat-file").click();
$("chat-file").addEventListener("change", (e) => {
  addAttachedFiles(e.target.files);
  e.target.value = "";
});

// Dedicated camera button (mobile): opens the camera directly to attach a photo.
// The OS file picker on the attach button already offers camera + gallery, but a
// one-tap "take a photo" makes the phone feel like a real input, not a website.
{
  const cam = $("chat-camera"), camFile = $("chat-camera-file");
  if (cam && camFile) {
    cam.onclick = () => camFile.click();
    camFile.addEventListener("change", (e) => {
      addAttachedFiles(e.target.files);
      e.target.value = "";
    });
  }
}

/** Ingest images shared INTO localm from the phone's share sheet (PWA share
 *  target). The server stashed them; we pull them as chat attachments and then
 *  clear the server inbox. Text/links shared in drop into the composer. */
export async function ingestSharedFiles() {
  let items;
  try {
    const r = await fetch("/api/share/pending", { headers: authHeaders() });
    if (!r.ok) return;
    items = (await r.json()).items || [];
  } catch (e) { return; }
  if (!items.length) return;
  const ids = [];
  let imgs = 0;
  for (const it of items) {
    ids.push(it.id);
    if ((it.type || "").startsWith("image/")) {
      chat.attachments.push({ name: it.name, dataUri: it.data_uri });
      imgs++;
    } else if (it.data_uri) {
      try {
        const txt = decodeURIComponent(escape(atob(it.data_uri.split(",", 2)[1] || "")));
        const ta = $("chat-input");
        if (ta) { ta.value = (ta.value ? ta.value + "\n" : "") + txt; autoGrow(ta); }
      } catch (e) { /* not decodable text */ }
    }
  }
  renderAttachChips();
  // We have the data client-side now; clear the server inbox. Best-effort: on
  // failure the items stay stashed server-side and the NEXT ingest re-clears
  // them (the ids are re-sent), so this self-heals - but leave a trace so a
  // persistent failure is diagnosable from the client log.
  fetch("/api/share/clear", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  }).then(async (r) => {
    if (!r.ok) {
      console.error("share-inbox clear failed (HTTP " + r.status +
                    ") - will retry on the next share ingest");
      return;
    }
    // A 200 is not the same as "the inbox is empty now": the endpoint deletes
    // best-effort and reports the entries it could NOT remove (locked file,
    // permission denied). Nothing rendered that count, so a partial failure on
    // a privacy-adjacent store looked identical to a clean sweep from here.
    // A server that does not report the field yields 0 and changes nothing.
    let failed = 0;
    try { failed = Number((await r.json()).failed) || 0; }
    catch (e) { /* no/!JSON body: nothing more to report than the 200 above */ }
    if (failed > 0) {
      console.error("share-inbox clear: " + failed + " item(s) could not be " +
                    "deleted server-side - retried on the next share ingest");
      toast(`${failed} shared item${failed > 1 ? "s" : ""} could not be ` +
            `cleared from the server`, true);
    }
  }).catch((e) => {
    console.error("share-inbox clear failed: " + (e && e.message ? e.message : e));
  });
  if (imgs) toast(`${imgs} image${imgs > 1 ? "s" : ""} shared into chat`);
}

/** Drag-and-drop files anywhere on the chat view to attach them, with a
 *  highlight while a file is hovering. Only reacts to file drags (not text
 *  selections), and preventDefault stops the browser opening the dropped file. */
export function setupChatDropZone() {
  const zone = $("view-chat");
  if (!zone) return;
  const isFileDrag = (e) =>
    e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
  zone.addEventListener("dragover", (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    zone.classList.add("drag-over");
  });
  zone.addEventListener("dragleave", (e) => {
    if (!zone.contains(e.relatedTarget)) zone.classList.remove("drag-over");
  });
  zone.addEventListener("drop", (e) => {
    if (!e.dataTransfer || !e.dataTransfer.files || !e.dataTransfer.files.length) return;
    e.preventDefault();
    zone.classList.remove("drag-over");
    addAttachedFiles(e.dataTransfer.files);
  });
}
setupChatDropZone();

