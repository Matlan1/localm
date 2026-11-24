// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - chat (split from app.js). Classic script: it
   shares the one global lexical environment with the other app/* and
   pages/* scripts, so every cross-section reference resolves by bare
   name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { $, applyChatBackground, authHeaders, autoGrow, CHAT_SURFACE_HEADERS, confirmDanger, el, fetchImageURL, INSTANCE_SCOPED_KEYS, promptText, readStoredJSON, reconcileInstanceId, renderMarkdown, revealFilledAdvanced, safeAvatarImageSrc, scrubMarkers, stripAsrPrefix, stripThink, toast, updateAdvancedCount } from "./helpers.js";
import { restoreRecentGuiErrors, setRecentGuiErrorStore } from "./client-log.js";
import { t, tn, tOr } from "./i18n.js";
import { emptyState, iconEl } from "./icons.js";
import { modelCache, modelSelect } from "./models-sidebar.js";
import { openMemoryModal, pluginState, processChatQueue, renderQueuedIndicator, runCompletion, speak, setWebAskSession, toolEventPrompt } from "./settings-perf.js";
import { showView } from "./tabs.js";
import { applyCoderRailSide } from "./coder.js";

/* ================================================================ */
/*  Tool events                                                      */
/* ================================================================ */

/* A tool event is an entry of conv.messages that is neither a user nor an
   assistant turn: a web search, a page read, an approval outcome or a control
   note the chat injected. It carries no `role` key. runCompletion renders it
   to fenced user-role text when it assembles a request (toolEventPrompt);
   renderChat draws a search or page read as a collapsed activity card and a
   control note as a one-line notice (addToolEventRow).

   Shape: { kind: "tool", id?, tool: "search"|"fetch"|"note",
            status: "running"|"done"|"failed"|"denied"|"duplicate"|"skipped",
            limit?: the per-message lookup ceiling a "skipped" call hit,
            query?, url?, started_at?, finished_at?, error?,
            search (done): provider, search_status, search_error, grounding,
              grounding_summary, sources[], chunks[], prompt_text
            fetch (done): page {url, text, truncated}
            note?: trusted model-directed prose appended after the result, or
              the whole text of a control note; reason? ("format"|"unparsed"|
              "pending", or "limit" on stored rows) names a control note, and
              "ignored" marks a result whose note says later calls did not run
            text?, untrusted_spans?: a row migrated from the legacy
              {role:"user", web:true} shape, rendered verbatim } */
export const TOOL_EVENT_KIND = "tool";

export function isToolEvent(m) {
  return !!m && m.kind === TOOL_EVENT_KIND;
}

/** A new tool event with *fields* (tool, status, query/url, ...). */
export function newToolEvent(fields) {
  return { kind: TOOL_EVENT_KIND, ...fields };
}

// Header patterns of a legacy web note, mapped to its tool, status and subject.
const _LEGACY_WEB_NOTE_SHAPES = [
  [/^\[Results of web_search "([^"]*)"\]/, (q) => ({ tool: "search", status: "done", query: q })],
  [/^\[web_search results(?: for "([^"]*)")?\]/, (q) => ({ tool: "search", status: "done", query: q || "" })],
  [/^\[Content of (\S+)\]/, (u) => ({ tool: "fetch", status: "done", url: u })],
  [/^\[Web (?:request|search) failed: /, () => ({ tool: "search", status: "failed" })],
  [/^\[duplicate web request\]/, () => ({ tool: "search", status: "duplicate" })],
  [/^\[web access denied\]/, () => ({ tool: "search", status: "denied" })],
  [/^\[tool-call format\]/, () => ({ tool: "note", status: "done", reason: "format" })],
  [/^\[web search limit reached\]/, () => ({ tool: "note", status: "done", reason: "limit" })],
  [/^\[pending action\]/, () => ({ tool: "note", status: "done", reason: "pending" })],
];

/** The tool event a legacy {role:"user", web:true} row becomes: its header
 *  decides tool/status/query/url, and the original text and untrusted spans
 *  are kept for rendering and prompt assembly. */
export function legacyWebNoteToToolEvent(m) {
  const text = typeof m.content === "string" ? m.content
    : (m.content || []).filter((p) => p.type === "text").map((p) => p.text).join("");
  let fields = { tool: "note", status: "done" };
  for (const [re, make] of _LEGACY_WEB_NOTE_SHAPES) {
    const hit = re.exec(text);
    if (hit) { fields = make(hit[1]); break; }
  }
  const ev = newToolEvent(fields);
  if (m.id !== undefined) ev.id = m.id;
  ev.text = text;
  if (Array.isArray(m.untrusted_spans) && m.untrusted_spans.length) {
    ev.untrusted_spans = m.untrusted_spans.map((s) => s.slice());
  }
  if (Array.isArray(m.compacted) && m.compacted.length) ev.compacted = m.compacted;
  return ev;
}

/** Migrate *list* (an array of messages) in place: every legacy
 *  {role:"user", web:true} row becomes a tool event, a tool event still
 *  `running` (a page load interrupted the call; it carries no result) is
 *  removed, and nested compaction archives are walked. Idempotent. Returns
 *  *list*. */
export function migrateToolEvents(list) {
  if (!Array.isArray(list)) return list;
  for (let i = 0; i < list.length; i++) {
    const m = list[i];
    if (!m || typeof m !== "object") continue;
    if (m.role === "user" && m.web === true) list[i] = legacyWebNoteToToolEvent(m);
    else if (isToolEvent(m) && m.status === "running") {
      list.splice(i, 1);
      i--;
      continue;
    }
    if (Array.isArray(list[i].compacted)) migrateToolEvents(list[i].compacted);
  }
  return list;
}

/** Migrate a conversation in place: the live messages, every parked branch
 *  tail and every archived dropped branch. Returns *conv*. */
export function migrateConversation(conv) {
  if (!conv || typeof conv !== "object") return conv;
  migrateToolEvents(conv.messages);
  for (const rec of conv.branches || []) {
    for (const tail of (rec && rec.tails) || []) migrateToolEvents(tail);
  }
  for (const tail of conv.droppedBranches || []) migrateToolEvents(tail);
  return conv;
}

export function migrateConversations(list) {
  if (Array.isArray(list)) for (const c of list) migrateConversation(c);
  return list;
}

/* ================================================================ */
/*  Chat                                                             */
/* ================================================================ */

export const chat = {
  conversations: migrateConversations(readStoredJSON("localm.conversations", [])),
  activeId: null,
  abort: null,
  busy: false,
  queue: [],
  webCall: null,     // the running tool event while a web call is in flight, else null
  attachments: [],   // image attachments: {name, dataUri}
  docs: [],          // document attachments: {name, text, chars, truncated}
  clips: [],         // audio attachments: {name, data (base64), format, size, seconds}
  ctxMax: 16384,     // context ceiling - refreshed from /v1/config
  systemDefault: "", // default system prompt from Settings; a blank drawer inherits it
  toolGrammar: true, // chat_tool_grammar from /v1/config - grammar-constrain web-tool calls
  webMaxLookups: null, // chat_web_max_lookups from /v1/config; null = the built-in default
  netOff: false,       // net_mode is "off": the Web access toggle is hidden
  userAvatar: "",           // user_avatar from /v1/config: "", an emoji/glyph, or a data: URI
  userName: "",             // user_name from /v1/config: "" falls back to "You"
  modelAvatarDefault: "",   // model_avatar_default from /v1/config, same shape
  modelAvatarOverrides: {}, // model_avatar_overrides from /v1/config: {model_id: icon}
  chatBackground: "",       // chat_background from /v1/config: "", or a data: URI
  // Set once a backend REFUSES a web-tool grammar request; never cleared until
  // reload. Kept separate from toolGrammar (the config preference, refreshed
  // by refreshCtxLimit's 30s poll) - a config refresh must not resurrect a
  // grammar THIS backend has already demonstrated it cannot honour.
  toolGrammarUnsupported: false,
  privacy: false,    // chat in privacy mode: conversations not persisted
  serverPrivacy: false,  // server in privacy mode: non-chat stored keys not persisted
  // True once /v1/config has answered and `privacy` reflects the server's real
  // mode. The two localStorage write funnels (lsSetScoped, saveConversations)
  // queue their writes while this is false - the boot default above is not an
  // answer - and settlePendingScopedWrites() replays or discards the queue
  // once it is. Never reset by a later failed poll: the last confirmed answer
  // stands.
  modeKnown: false,
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
  // The same latch for the server-mode wipe of the Coder tab's project field.
  serverPrivacyWiped: false,
};

// The human-facing label of a tagged row (kb/doc) or a tool event; used by
// renderChat() below and by exportConversation() (settings-perf.js).
const NOTE_LABEL_KEYS = { web: "chat.note.web", doc: "chat.note.doc", kb: "chat.note.kb" };

export function noteLabel(m) {
  if (isToolEvent(m)) return t(NOTE_LABEL_KEYS.web);
  return m.tag && NOTE_LABEL_KEYS[m.tag] ? t(NOTE_LABEL_KEYS[m.tag]) : null;
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
    total += estimateTokens(msgText(m)) + msgImages(m).length * 750 +
             msgAudioClips(m).length * 750;
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

/** A text-only copy of a message for the compaction archive: role, text,
 *  id, note tag and terminal flags are kept; images, audio and video are
 *  replaced by a count note; a nested compaction archive is kept as is. */
export function archiveCopy(m) {
  if (isToolEvent(m)) return { ...m };
  const out = { role: m.role, content: msgText(m) };
  for (const k of ["id", "tag", "model", "truncated", "stopped", "failed", "cut", "continues", "webUnfinished", "noWebEvidence", "bridge"]) {
    if (m[k] !== undefined) out[k] = m[k];
  }
  const media = msgImages(m).length + msgAudioClips(m).length +
                (m.audio ? 1 : 0) + (m.video ? 1 : 0);
  if (media) out.content += `\n\n*[${media} attachment(s) not archived]*`;
  if (Array.isArray(m.compacted) && m.compacted.length) out.compacted = m.compacted;
  return out;
}

/** Index where the kept tail of *messages* starts, at or before *from*: the
 *  nearest user turn that is not a tool event, else the next one after it.
 *  0 when there is none, meaning nothing can be compacted. */
export function compactionCut(messages, from) {
  const isTurn = (m) => m && m.role === "user" && !isToolEvent(m);
  for (let i = Math.min(from, messages.length - 1); i > 0; i--) {
    if (isTurn(messages[i])) return i;
  }
  for (let i = Math.max(from, 1); i < messages.length; i++) {
    if (isTurn(messages[i])) return i;
  }
  return 0;
}

/** The text of *m* a digest or summary reads: a reply without its reasoning
 *  blocks, any other message as it stands. */
function spokenText(m) {
  return m.role === "assistant" ? stripThink(msgText(m)) : msgText(m);
}

export const DIGEST_MAX_CHARS = 4096;
const DIGEST_MIN_EXCERPT = 160;

/** The fallback bridge text when no summary is available: a note plus one
 *  word-boundary excerpt per removed message (reasoning stripped), newest kept
 *  first within DIGEST_MAX_CHARS, in conversation order. */
export function compactionDigest(older) {
  const per = Math.max(DIGEST_MIN_EXCERPT, Math.floor(DIGEST_MAX_CHARS / Math.max(1, older.length)));
  const lines = [];
  let used = 0;
  for (let i = older.length - 1; i >= 0; i--) {
    const m = older[i];
    const line = `${isToolEvent(m) ? "WEB" : String(m.role || "user").toUpperCase()}: ` +
      truncateAtWord(spokenText(m), per);
    if (lines.length && used + line.length > DIGEST_MAX_CHARS) break;
    lines.unshift(line);
    used += line.length;
  }
  const parts = ["[Earlier conversation was condensed to fit the context window. " +
    "Summarisation was unavailable, so these are excerpts of the earlier turns:]"];
  const omitted = older.length - lines.length;
  if (omitted) parts.push(`(${omitted} earlier message(s) omitted)`);
  return [...parts, ...lines].join("\n\n");
}

// Largest reply asked of the summariser; a small window gets a quarter of the
// window, at least SUMMARY_MIN_TOKENS.
export const SUMMARY_MAX_TOKENS = 512;
const SUMMARY_MIN_TOKENS = 64;
// Most summariser requests one compaction makes.
export const MAX_SUMMARY_PASSES = 8;
// Smallest excerpt room, in tokens, a rolling summary pass is made for.
const MIN_CHUNK_TOKENS = 48;
// Smallest share of a rolling pass, in tokens, one excerpt is cut down to.
const MIN_ENTRY_TOKENS = 24;
// Tokens a rolling pass keeps free for the earlier summary's labels and an
// omitted-messages note, on top of the earlier summary itself.
const ROLLING_RESERVE_EXTRA = 32;
// Characters per token the summariser budget assumes.
const BUDGET_CHARS_PER_TOKEN = 3;

const SUMMARY_INSTRUCTION = "Summarise the following conversation in under 200 words. " +
  "Keep facts, names, decisions, and anything the user asked to " +
  "remember. Reply with the summary only.";
const ROLLING_SUMMARY_INSTRUCTION = "Summarise the following conversation in under 200 words. " +
  "It begins with a summary of its earlier part: keep the facts, names, decisions, " +
  "and anything the user asked to remember from that summary and from the later " +
  "turns. Reply with the summary only.";

/** The max_tokens of one summariser request in a window of *ctxMax* tokens:
 *  SUMMARY_MAX_TOKENS for an unknown window (0), else a quarter of the window
 *  capped at SUMMARY_MAX_TOKENS and never below SUMMARY_MIN_TOKENS. */
export function summaryReplyTokens(ctxMax) {
  if (!(ctxMax > 0)) return SUMMARY_MAX_TOKENS;
  return Math.max(SUMMARY_MIN_TOKENS, Math.min(SUMMARY_MAX_TOKENS, Math.floor(ctxMax / 4)));
}

/** Tokens a rolling pass gives the summary so far: four thirds of *reply*,
 *  the budget count of a reply of that many tokens at four characters per
 *  token. */
export function earlierSummaryTokens(reply) {
  return Math.floor(reply * 4 / 3);
}

/** Tokens of a window of *ctxMax* kept free for the chat template. */
export function templateMargin(ctxMax) {
  return 64 + (ctxMax > 0 ? Math.floor(ctxMax / 64) : 0);
}

/** Tokens the summariser budget counts for *text*: one per
 *  BUDGET_CHARS_PER_TOKEN ASCII characters, rounded up, plus one per
 *  non-ASCII character. */
export function budgetTokens(text) {
  const s = String(text || "");
  const wide = (s.match(/[\u0080-\uffff]/g) || []).length;
  return wide + Math.ceil((s.length - wide) / BUDGET_CHARS_PER_TOKEN);
}

/** *text* when it counts at most *tokens* budget tokens, else cut at a word
 *  boundary with a truncation marker so it does. */
export function fitTokens(text, tokens) {
  if (budgetTokens(text) <= tokens) return text;
  const marker = " ...[truncated]".length;
  let keep = Math.max(1, tokens * BUDGET_CHARS_PER_TOKEN - marker);
  let cut = truncateAtWord(text, keep);
  for (let i = 0; i < 40 && keep > 1 && budgetTokens(cut) > tokens; i++) {
    keep = Math.max(1, Math.floor(keep * 0.85));
    cut = truncateAtWord(text, keep);
  }
  return cut;
}

/** The excerpt groups of successive summariser requests for *entries* (one
 *  excerpt per removed message, oldest first) in a window of *ctxMax*.
 *  One group holding every entry when the window is unknown (0), when they
 *  fit one request (excerpt + *overhead* + *reply* + the template margin
 *  within *ctxMax*), or when the window leaves less than MIN_CHUNK_TOKENS for
 *  a rolling pass. Otherwise groups that each fit a rolling pass, whose
 *  prompt also carries an earlier summary of up to earlierSummaryTokens(*reply*)
 *  tokens; an entry
 *  larger than a group is cut to fit. With more than MAX_SUMMARY_PASSES
 *  groups, entries are cut to an equal share (at least MIN_ENTRY_TOKENS) and
 *  regrouped; groups past the bound are left out from the middle (the first
 *  group and the newest ones are kept) and the first kept group after the gap
 *  opens with an "(N message(s) omitted)" line. */
export function planSummaryPasses(entries, ctxMax, reply, overhead) {
  if (!entries.length || !(ctxMax > 0)) return [entries.slice()];
  const sep = budgetTokens("\n\n");
  const room = ctxMax - reply - templateMargin(ctxMax) - overhead;
  const sizes = entries.map(budgetTokens);
  const total = sizes.reduce((a, b) => a + b, 0) + sep * (entries.length - 1);
  if (total <= room) return [entries.slice()];
  const chunkRoom = room - earlierSummaryTokens(reply) - ROLLING_RESERVE_EXTRA;
  if (chunkRoom < MIN_CHUNK_TOKENS) return [entries.slice()];
  const group = (cap) => {
    const groups = [];
    let used = 0;
    entries.forEach((e, i) => {
      let entry = e, size = sizes[i];
      if (size > cap) { entry = fitTokens(entry, cap); size = budgetTokens(entry); }
      if (groups.length && used + sep + size <= chunkRoom) {
        groups[groups.length - 1].push(entry);
        used += sep + size;
      } else {
        groups.push([entry]);
        used = size;
      }
    });
    return groups;
  };
  let groups = group(chunkRoom);
  if (groups.length > MAX_SUMMARY_PASSES) {
    const share = Math.floor(MAX_SUMMARY_PASSES * chunkRoom / entries.length) - sep;
    groups = group(Math.max(MIN_ENTRY_TOKENS, Math.min(chunkRoom, share)));
  }
  if (groups.length > MAX_SUMMARY_PASSES) {
    const keepNew = MAX_SUMMARY_PASSES - 1;
    const dropped = groups.slice(1, groups.length - keepNew);
    const omitted = dropped.reduce((n, g) => n + g.length, 0);
    groups = [groups[0], ...groups.slice(groups.length - keepNew)];
    groups[1] = [`(${omitted} message(s) omitted)`, ...groups[1]];
  }
  return groups;
}

/** One summariser request for *content*: the visible reply text, or "" when
 *  the request fails, is aborted through *signal*, is refused, or ends with
 *  a finish reason other than "stop" or "length". Never rejects. */
async function summaryPass(content, maxTokens, signal, conv) {
  const pinnedModel = pinnedModelName(conv);
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: authHeaders(chatSurfaceHeaders(conv)),
      body: JSON.stringify({
        model: pinnedModel || modelSelect.value || modelCache.active,
        pin_model: !!pinnedModel,
        // origin "client": the server does not recall memories for this
        // prompt or log it as the user's line.
        messages: [{ role: "user", origin: "client", content }],
        max_tokens: maxTokens,
        temperature: 0.3,
        stream: false,
        chat_template_kwargs: { enable_thinking: false },
      }),
      ...(signal ? { signal } : {}),
    });
    if (!r.ok) return "";
    const data = await r.json();
    const choice = data.choices?.[0];
    // A summary is accepted from a generation that finished with "stop" or
    // "length"; an "error" choice is treated as summarisation unavailable.
    if (choice && (choice.finish_reason === "stop" || choice.finish_reason === "length")) {
      return stripThink(scrubMarkers(choice.message?.content || "")).trim();
    }
  } catch { /* summarisation unavailable - the caller falls back to a digest */ }
  return "";
}

const BRIDGE_SUMMARY_PREFIX = "[Conversation summary]\n";

/** The earlier summary carried by the leading bridge of *older*, and how many
 *  leading messages the bridge occupies: `{ summary, skip }`. A digest bridge,
 *  or no bridge, gives `{ summary: "", skip: 0 }`. */
export function leadingBridgeSummary(older) {
  const first = older[0];
  if (!first || !first.bridge || first.role !== "user" ||
      typeof first.content !== "string" ||
      !first.content.startsWith(BRIDGE_SUMMARY_PREFIX)) {
    return { summary: "", skip: 0 };
  }
  const skip = older[1] && older[1].bridge ? 2 : 1;
  return { summary: first.content.slice(BRIDGE_SUMMARY_PREFIX.length).trim(), skip };
}

/** The text the summariser reads for *m*: its text, then one note per image and
 *  per audio clip (the clip's file name when it has one). */
export function compactionText(m) {
  const notes = [];
  for (let i = 0; i < msgImages(m).length; i++) notes.push("[image attached]");
  for (const clip of msgAudioClips(m)) {
    notes.push(clip.name ? `[audio clip attached: ${clip.name}]` : "[audio clip attached]");
  }
  const text = spokenText(m);
  return notes.length ? (text ? text + "\n" : "") + notes.join("\n") : text;
}

/** Summarises or condenses the older part of *conv*. The kept tail starts at
 *  a user turn (see compactionCut). Each summary request is sized to fit
 *  chat.ctxMax and older turns that do not fit one request are summarised in
 *  rolling passes, oldest first; one unsized request carries every older turn
 *  when the window is unknown or too small for a rolling pass (see
 *  planSummaryPasses). A failed pass makes the
 *  bridge a digest of the older turns. An aborted *signal* cancels the
 *  summary requests and leaves the conversation unchanged, and so does a change
 *  of the older turns while the summary was written; messages added after them
 *  meanwhile are kept. */
export async function compactConversation(conv, signal = null) {
  if (conv.messages.length <= COMPACT_KEEP) return false;
  // R44: keep as many of the most-recent turns verbatim as fit in COMPACT_TARGET
  // of the ceiling (at least COMPACT_KEEP), summarising only what is older -
  // instead of always discarding everything but the last 4 turns.
  const budget = (chat.ctxMax && chat.ctxMax > 0) ? COMPACT_TARGET * chat.ctxMax : 0;
  let keepCount = 0, used = 0;
  for (let i = conv.messages.length - 1; i >= 0; i--) {
    const t = estimateTokens(msgText(conv.messages[i])) +
              msgImages(conv.messages[i]).length * 750 +
              msgAudioClips(conv.messages[i]).length * 750;
    const overBudget = budget > 0 ? (used + t > budget) : true;
    if (keepCount >= COMPACT_KEEP && overBudget) break;
    used += t;
    keepCount++;
  }
  keepCount = Math.max(COMPACT_KEEP, Math.min(keepCount, conv.messages.length - 1));
  const cut = compactionCut(conv.messages, conv.messages.length - keepCount);
  if (cut <= 0) return false;
  const older = conv.messages.slice(0, cut);

  // R44: feed whole messages truncated at a word boundary, with reasoning blocks
  // stripped, so the summariser never sees half-words or display-only <think>.
  // A summary left by an earlier compaction seeds the rolling passes instead of
  // being re-summarised as one more turn.
  const { summary: earlier, skip } = leadingBridgeSummary(older);
  const fresh = older.slice(skip);
  const entries = fresh.map((m) =>
    `${isToolEvent(m) ? "WEB" : m.role.toUpperCase()}: ` +
    truncateAtWord(compactionText(m), 1200));
  const ctxMax = chat.ctxMax > 0 ? chat.ctxMax : 0;
  const reply = summaryReplyTokens(ctxMax);
  const groups = fresh.length
    ? planSummaryPasses(entries, ctxMax, reply,
        budgetTokens(ROLLING_SUMMARY_INSTRUCTION + "\n\n") +
        (earlier ? earlierSummaryTokens(reply) : 0))
    : [];

  // The summary is written by the model that answers the conversation: its
  // pinned model, pinned, else the selected model as a preference.
  let summary = earlier;
  // A placeholder assistant row shows "Compacting conversation..." until the
  // summary requests settle.
  const box = $("chat-messages");
  const pending = box ? addMessageRow(box, "assistant", "") : null;
  if (pending) {
    mountStatusIndicator(pending.body, t("chat.status.compacting"));
    box.scrollTop = box.scrollHeight;
  }
  try {
    // Runs the passes oldest first; an empty reply ends them with no summary.
    for (const group of groups) {
      if (signal && signal.aborted) break;
      const excerpt = group.join("\n\n");
      const content = summary
        ? ROLLING_SUMMARY_INSTRUCTION + "\n\nSummary of the earlier part:\n" +
          fitTokens(summary, earlierSummaryTokens(reply)) + "\n\nLater turns:\n" + excerpt
        : SUMMARY_INSTRUCTION + "\n\n" + excerpt;
      summary = await summaryPass(content, reply, signal, conv);
      if (!summary) break;
    }
  } finally {
    if (pending) {
      removeStatusIndicator(pending.body);
      pending.row.remove();
    }
  }
  if (signal && signal.aborted) return false;
  if (older.some((m, i) => conv.messages[i] !== m)) return false;

  // The removed turns are archived on the bridge message (text only, with an
  // earlier bridge's own archive kept nested) and listed by exportConversation.
  const archived = older.map(archiveCopy);
  // Both halves of the bridge carry `bridge: true` so a later archive walk
  // can tell them from real turns.
  const bridge = summary
    ? [{ role: "user", content: BRIDGE_SUMMARY_PREFIX + summary,
         compacted: archived, bridge: true },
       { role: "assistant", content: "Understood. Continuing from this summary.", bridge: true }]
    : earlier
    ? [{ role: "user",
         content: BRIDGE_SUMMARY_PREFIX + earlier + "\n\n" + compactionDigest(fresh),
         compacted: archived, bridge: true },
       { role: "assistant", content: "Understood. Continuing from this summary.", bridge: true }]
    : [{ role: "user", content: compactionDigest(older), compacted: archived, bridge: true },
       { role: "assistant", content: "Understood. Continuing from these excerpts.", bridge: true }];

  conv.messages = [...bridge, ...conv.messages.slice(cut)];
  // Forks anchored in the summarised-away region can no longer be reached by
  // the ‹ › navigation; archive their content and DISCLOSE the count instead
  // of deleting them silently (memory-audit 2026-07-02).
  const lostBranches = pruneBranches(conv);
  saveConversations(conv);
  renderChat();
  const base = t(summary ? "chat.compact.summarised" : "chat.compact.trimmed");
  toast(lostBranches ? tn("chat.compact.archived", lostBranches, { base }) : base);
  return true;
}

/** Whether the chat request built from *conv* carries an image: a message
 *  with an image_url part and no server-generated /api/ image (a message with
 *  one is sent as text). */
function convSendsImage(conv) {
  return conv.messages.some((m) => {
    if (isToolEvent(m) || !Array.isArray(m.content)) return false;
    const images = m.content.filter((p) => p.type === "image_url");
    return images.length > 0 &&
      !images.some((p) => p.image_url?.url?.startsWith("/api/"));
  });
}

/** The trained context window a longer conversation needs, when the model
 *  that would answer it cannot hold it and an installed one can: returns
 *  { minContext } for the request's `min_context`, so the server answers with
 *  that roomier model instead of the conversation being compacted. A
 *  conversation that sends an image counts only a roomier model confirmed to
 *  read images (`vision: true`). Null when the chat is pinned, when the server
 *  has already compacted this conversation to fit the model that answered it
 *  (`conv.serverCompacted`, set from the X-Localm-Context-Compacted reply
 *  header), when the answering model's window is unknown or big enough, or when
 *  no installed model qualifies. */
export function contextRoutingNeed(conv, est) {
  if (!conv || conv.pinnedModel || conv.serverCompacted) return null;
  const current = modelSelect.value || modelCache.active;
  const models = (modelCache && modelCache.models) || [];
  const cur = models.find((m) => m.name === current);
  const trained = cur && cur.context_length;
  if (!trained) return null;
  if (est < COMPACT_RATIO * trained) return null;
  const need = Math.ceil(est / COMPACT_RATIO);
  const image = convSendsImage(conv);
  const roomier = models.some((m) => m.name !== current && !m.missing &&
    (m.model_type || "llm") === "llm" && (m.context_length || 0) >= need &&
    (!image || m.vision === true));
  return roomier ? { minContext: need } : null;
}

/** Compacts *conv* when it has grown past COMPACT_RATIO of the context
 *  window, unless an installed model can hold it (see contextRoutingNeed).
 *  Returns that routing need, or null. *signal* aborts the summary request. */
export async function maybeCompactConversation(conv, signal = null) {
  const est = estimateConvTokens(conv);
  const routing = contextRoutingNeed(conv, est);
  if (routing) return routing;
  if (!chat.ctxMax || chat.ctxMax <= 0) return null;
  if (est >= COMPACT_RATIO * chat.ctxMax) {
    await compactConversation(conv, signal);
  }
  return null;
}

let _instanceUnknownWarned = false;   // one warning per breakage, re-armed on success

/** AUD-INSTANCEID residual 2: /v1/config could not be read, so this load has NO
 *  answer about which backend is behind this origin. That is not a mismatch (we
 *  wipe nothing) and it is emphatically not a match either, so it lands on the
 *  same "unknown" state an older server without an instance_id produces: keep
 *  showing whatever is cached, never authorise a write into a store we have not
 *  identified. Loud rather than silent - the round trip that produced this
 *  used to be swallowed with no trace at all. */
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

/** Applies the web-related settings of the /v1/config payload *cfg*: the
 *  per-message lookup ceiling, and whether network access is off, in which case
 *  the chat's Web access toggle is hidden. */
export function applyWebPolicy(cfg) {
  if ("chat_web_max_lookups" in cfg) {
    chat.webMaxLookups = Number.isInteger(cfg.chat_web_max_lookups)
      ? cfg.chat_web_max_lookups : null;
  }
  if ("net_mode" in cfg) {
    chat.netOff = cfg.net_mode === "off";
    const row = $("p-web") && $("p-web").closest(".check-row");
    if (row) row.style.display = chat.netOff ? "none" : "";
  }
}

/** Re-reads the web settings from the server. Best effort: an unreachable
 *  server leaves the values already applied. */
export async function refreshWebPolicy() {
  try {
    const r = await fetch("/v1/config", { headers: authHeaders() });
    if (r.ok) applyWebPolicy(await r.json());
  } catch { /* server unreachable - the applied values stay */ }
}

export async function refreshCtxLimit() {
  // Set once reconcileInstanceId has returned a real verdict for THIS call, so
  // a throw from the rendering work further down (which runs only AFTER the
  // verdict) cannot be mistaken for "we never got an answer" and downgrade a
  // confirmed "mismatched" to "unknown" - that would re-permit rendering the
  // foreign cache the mismatch branch exists to remove.
  let reconciled = false;
  const upTo = _writeSeq;
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
      applyWebPolicy(cfg);
      chat.userAvatar = cfg.user_avatar || "";
      chat.userName = (cfg.user_name || "").trim();
      chat.modelAvatarDefault = cfg.model_avatar_default || "";
      chat.modelAvatarOverrides = cfg.model_avatar_overrides || {};
      chat.chatBackground = cfg.chat_background || "";
      applyChatBackground(chat.chatBackground);
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

      applySessionMode(cfg, upTo);
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

/** Applies the session mode of the /v1/config payload *cfg*, asked for when the
 *  newest queued write had sequence number *upTo*, to the page. In
 *  privacy mode conversations live in memory only: anything an earlier
 *  non-privacy session left in this browser is wiped and the hint is shown. The
 *  server applies a change of the mode at once, so this also handles a switch
 *  while the page is open. */
function applySessionMode(cfg, upTo) {
  const wasPrivacy = chat.modeKnown && chat.privacy;
  chat.privacy = (cfg.effective_chat_mode ?? cfg.effective_mode) === "privacy";
  chat.serverPrivacy = cfg.effective_mode === "privacy";
  chat.modeKnown = true;
  for (const key of INSTANCE_SCOPED_KEYS) {
    if (scopedKeyPrivate(key)) localStorage.removeItem(key);
  }
  if (chat.privacy) wipeStoredDrafts();
  settlePendingScopedWrites(upTo);
  _syncRecentErrorsStore();
  if (chat.serverPrivacy) {
    // The Coder tab's "Project directory" input was filled from
    // "localm.coderCwd" at boot; clear it once per privacy confirmation.
    if (!chat.serverPrivacyWiped) {
      chat.serverPrivacyWiped = true;
      const cwdInputPriv = $("setup-cwd");
      if (cwdInputPriv) cwdInputPriv.value = "";
    }
  } else {
    chat.serverPrivacyWiped = false;
  }
  if (chat.privacy) {
    chat.persist = false;
    // The in-memory reset runs only the first time privacy mode is confirmed;
    // the mode is read again for the tab's whole lifetime.
    if (!chat.privacyWiped) {
      chat.privacyWiped = true;
      enterPrivacyMode();
    }
    const h = document.querySelector("#conversations h3");
    if (h && !document.getElementById("privacy-hint")) {
      const hint = document.createElement("div");
      hint.id = "privacy-hint";
      hint.className = "privacy-hint";
      hint.dataset.i18n = "chat.privacyHint";
      hint.dataset.i18nTitle = "chat.privacyHint.title";
      hint.textContent = t("chat.privacyHint");
      hint.title = t("chat.privacyHint.title");
      h.after(hint);
      showPrivacyLeftovers();
    }
  } else {
    // The next time privacy mode is confirmed, what this page shows from a
    // non-privacy session is wiped again.
    chat.privacyWiped = false;
    document.getElementById("privacy-hint")?.remove();
    document.getElementById("privacy-leftovers")?.remove();
    if (wasPrivacy) leavePrivacyMode();
  }
  _updateUnsavedIndicator();
  _flushPendingWrites();
}

/** Privacy mode was confirmed: the conversations on screen leave it. One that
 *  holds something not saved yet, or whose reply is being written, stays in
 *  this tab only, and so does what is typed in the message box. */
function enterPrivacyMode() {
  const live = chatBusy() ? currentConv() : null;
  const held = chat.conversations.filter((c) => c.tabOnly || c === live ||
    (c.unsaved && !c._meta && (c.messages.length || (c.branches || []).length)));
  for (const c of held) keepInTabOnly(c);
  chat.conversations = held;
  if (!held.some((c) => c.id === chat.activeId)) {
    chat.activeId = null;
    releaseComposer();
  }
  convUI.collapsed = new Set();
  _localSaveOk = true;
  _localSavedIds.clear();
  _saveFailed.clear();
  setWebAskSession(null);                        // R27: forget the session choice
  renderConvList();
  if (!live || live.id !== chat.activeId) renderChat();
}

/** Privacy mode was left while this page was open: the conversations held under
 *  it and what is typed in a message box stay in this tab and are never
 *  written; a conversation started after this is saved as usual. */
function leavePrivacyMode() {
  let held = 0;
  for (const c of chat.conversations) {
    if (!(c.messages || []).length && !(c.branches || []).length && !c.params &&
        c.title === "New chat") continue;
    keepInTabOnly(c);
    held++;
  }
  holdDraftsInTab();
  if (held) toast(tn("chat.conv.privacyKept", held));
  renderConvList();
  initServerConversations();
}

/** Marks *conv* as held in this tab only: it is never written to this browser
 *  or the server store (convPersistable), and what is sent from it asks the
 *  server to handle it in privacy mode (chatSurfaceHeaders). It gets an id of
 *  its own, so a stored conversation of the same id is another one. */
function keepInTabOnly(conv) {
  if (conv.tabOnly) return;
  rekeyConversation(conv, newConvId());
  conv.tabOnly = true;
  delete conv.rev;
  delete conv.sync;
  delete conv.unsaved;
}

/** The headers that name the chat surface on a request made for *conv*. */
export function chatSurfaceHeaders(conv) {
  return chat.privacy || (conv && conv.tabOnly)
    ? { ...CHAT_SURFACE_HEADERS, "X-Localm-Session-Mode": "privacy" }
    : CHAT_SURFACE_HEADERS;
}

/** In privacy mode, asks the server how many conversations saved before are
 *  still on disk and, when there are any, shows a note under the privacy hint.
 *  Shows nothing when the count cannot be read. */
export async function showPrivacyLeftovers() {
  let stored = 0;
  try {
    const r = await fetch("/api/conversations?meta=true&limit=1&offset=0",
                          { headers: authHeaders() });
    if (r.ok) stored = Number((await r.json()).stored) || 0;
  } catch { /* no count: no note */ }
  document.getElementById("privacy-leftovers")?.remove();
  const anchor = document.getElementById("privacy-hint");
  if (!chat.privacy || stored <= 0 || !anchor) return;
  const note = el("div", "privacy-hint", tn("chat.privacyLeftovers", stored));
  note.id = "privacy-leftovers";
  note.title = t("chat.privacyLeftovers.title");
  anchor.after(note);
}
document.addEventListener("localm:language", () => {
  if (document.getElementById("privacy-leftovers")) showPrivacyLeftovers();
});

let _modeCheck = null;

/** Reads the session mode from the server and applies it. Calls made while a
 *  read is in flight share it. An unreachable server, a refused read or an
 *  answer that names no mode leaves the last confirmed mode standing. Never
 *  throws. */
export function confirmSessionMode() {
  if (!_modeCheck) {
    const upTo = _writeSeq;
    _modeCheck = (async () => {
      try {
        const r = await fetch("/v1/config", { headers: authHeaders() });
        if (!r.ok) return;
        const cfg = await r.json();
        if (cfg && (cfg.effective_chat_mode ?? cfg.effective_mode)) applySessionMode(cfg, upTo);
      } catch { /* the last confirmed mode stands */ }
    })().finally(() => { _modeCheck = null; });
  }
  return _modeCheck;
}

// Every instance-scoped browser write goes through lsSetScoped, saveConversations
// or the draft store, and all three queue here: a queued write lands only after
// the server confirmed, later than the write was queued, that the mode allows it.

// Queued lsSetScoped writes by key, as { value, seq }. Last write per key wins.
const _pendingScopedWrites = new Map();
// The sequence number of the queued conversation cache write, 0 for none.
let _pendingConversationSave = 0;
// The sequence number of the queued draft write, 0 for none.
let _pendingDraftSave = 0;
let _writeSeq = 0;
let _flushing = false;

/** Writes what was queued up to sequence number *upTo* and is allowed under the
 *  confirmed mode; what the mode forbids is discarded. */
export function settlePendingScopedWrites(upTo = Infinity) {
  if (!chat.modeKnown) return;
  for (const [key, w] of [..._pendingScopedWrites]) {
    if (w.seq > upTo) continue;
    _pendingScopedWrites.delete(key);
    _writeScoped(key, w.value);
  }
  if (_pendingConversationSave && _pendingConversationSave <= upTo) {
    _pendingConversationSave = 0;
    if (!chat.privacy) _writeConversationCache();
  }
  if (_pendingDraftSave && _pendingDraftSave <= upTo) {
    _pendingDraftSave = 0;
    if (!chat.privacy) _writeDrafts();
  }
}

/** Confirms the mode and settles what is queued, until nothing is. Before the
 *  first confirmation the queue waits for refreshCtxLimit. */
async function _flushPendingWrites() {
  if (_flushing || !chat.modeKnown) return;
  _flushing = true;
  try {
    while (_pendingScopedWrites.size || _pendingConversationSave || _pendingDraftSave) {
      if (_modeCheck) await _modeCheck;
      const upTo = _writeSeq;
      await confirmSessionMode();
      settlePendingScopedWrites(upTo);
    }
  } finally {
    _flushing = false;
  }
}

// Instance-scoped keys holding chat conversations or chat settings. They follow
// the chat mode (chat.privacy); every other instance-scoped key follows the
// server mode (chat.serverPrivacy).
const CHAT_SCOPED_KEYS = Object.freeze([
  "localm.conversations", "localm.convCollapsed", "localm.webAccess",
  "localm.speakAloud",
]);

/** True when *key* must not be kept in localStorage under the confirmed mode. */
export function scopedKeyPrivate(key) {
  return CHAT_SCOPED_KEYS.includes(key) ? chat.privacy : chat.serverPrivacy;
}

// The Settings "Recent errors" list follows the server mode: memory only in
// privacy mode, kept in localStorage otherwise (restored once per page load).
const RECENT_ERRORS_STORE_KEY = "localm.recentErrors";
let _recentErrorsRestored = false;
function _syncRecentErrorsStore() {
  if (chat.serverPrivacy) {
    setRecentGuiErrorStore(null);
    return;
  }
  setRecentGuiErrorStore({ save: (list) => lsSetScoped(RECENT_ERRORS_STORE_KEY, JSON.stringify(list)) });
  if (!_recentErrorsRestored) {
    _recentErrorsRestored = true;
    restoreRecentGuiErrors(readStoredJSON(RECENT_ERRORS_STORE_KEY, []));
  }
}

export function lsSetScoped(key, value) {
  if (chat.modeKnown && scopedKeyPrivate(key)) return;
  _pendingScopedWrites.set(key, { value, seq: ++_writeSeq });
  _flushPendingWrites();
}

function _writeScoped(key, value) {
  if (scopedKeyPrivate(key)) return;
  if (!INSTANCE_SCOPED_KEYS.includes(key)) {
    // Loud, not silent: a key reaching here that is not in
    // the wipe list is LM-DA-047's exact failure mode. Warn rather than throw
    // so a real write still succeeds; tests-js/privacy-scoped-keys.test.mjs
    // source-scans every lsSetScoped call site and fails before this ever
    // ships, so this branch is a live backstop, not the primary defense.
    console.warn(`localm: "${key}" written via lsSetScoped but missing from ` +
      "INSTANCE_SCOPED_KEYS (helpers.js) - add it so a privacy-mode or " +
      "instance-mismatch wipe covers it too.");
  }
  try { localStorage.setItem(key, value); }
  catch { /* storage full/blocked - callers already tolerate a miss */ }
}

// R34: the per-chat Web-access and Speak-aloud toggles used to reset to OFF on
// every load, so the user had to re-enable them in every session. Reflect the
// user's saved choice; for web, when there is no saved choice follow the
// global net policy: "allow" and "ask" both enable web (under "ask" the model
// knows the tools and every model-initiated request still goes through the
// approval card in settings-perf.js); "off" leaves it off. Writes are gated
// on privacy mode (no traces there).
export function hydrateChatToggles(cfg) {
  const webEl = $("p-web"), speakEl = $("p-speak");
  if (!webEl || !speakEl) return;
  const lsGet = (k) => { try { return localStorage.getItem(k); } catch { return null; } };
  const savedSpeak = chat.privacy ? null : lsGet("localm.speakAloud");
  if (savedSpeak !== null) speakEl.checked = savedSpeak === "1";
  const savedWeb = chat.privacy ? null : lsGet("localm.webAccess");
  if (savedWeb !== null) webEl.checked = savedWeb === "1";
  else if (cfg && (cfg.net_mode === "allow" || cfg.net_mode === "ask")) webEl.checked = true;
  // The brain toggle mirrors the server-side memory_enabled config (default on).
  const memEl = $("p-memory");
  if (memEl && cfg && typeof cfg.memory_enabled === "boolean")
    memEl.checked = cfg.memory_enabled;
}
window.hydrateChatToggles = hydrateChatToggles;

// Whether the last write of the conversation cache landed in localStorage, and
// the ids of the conversations it holds.
let _localSaveOk = true;
const _localSavedIds = new Set();
// The ids of the conversations whose last save the server did not take.
const _saveFailed = new Set();

/** One hint line under the conversations heading, shown when *on*. */
function _convHint(id, cls, key, on) {
  const existing = document.getElementById(id);
  if (!on) { existing?.remove(); return; }
  if (existing) return;
  const h = document.querySelector("#conversations h3");
  if (!h) return;
  const hint = document.createElement("div");
  hint.id = id;
  hint.className = cls;
  hint.dataset.i18n = key;
  hint.dataset.i18nTitle = key + ".title";
  hint.textContent = t(key);
  hint.title = t(key + ".title");
  h.after(hint);
}

/** Says under the conversations heading where the conversations are saved: on
 *  this machine, only in this browser (the server did not take a save), or
 *  nowhere but this tab (neither did). */
function _updateUnsavedIndicator() {
  const known = new Set(chat.conversations.map((c) => c.id));
  for (const id of [..._saveFailed]) if (!known.has(id)) _saveFailed.delete(id);
  const failed = _saveFailed.size > 0;
  const tabOnly = [..._saveFailed].some((id) => !_localSaveOk || !_localSavedIds.has(id));
  _convHint("conv-unsaved-warning", "unsaved-warning", "chat.conv.unsaved", failed && tabOnly);
  _convHint("conv-browser-only", "unsaved-warning", "chat.conv.browserOnly", failed && !tabOnly);
  _convHint("persist-hint", "privacy-hint", "chat.persistHint",
            chat.persist && !chat.privacy && !failed);
}

/** True when *conv* may be written to this browser or the server store. */
export function convPersistable(conv) {
  return !!conv && !conv.tabOnly && !conv.deleted;
}

// How often each conversation was changed in this page, so a save that ends
// after another change does not count that change as saved.
const _editCount = new WeakMap();

function _markChanged(conv) {
  conv.updated_at = Date.now();
  conv.unsaved = true;
  _editCount.set(conv, (_editCount.get(conv) || 0) + 1);
}

/** Saves the conversations to this browser and, when *changed* names one, that
 *  one to the server store. The browser write waits until the session mode is
 *  confirmed. */
export function saveConversations(changed) {
  if (chat.privacy || (changed && !convPersistable(changed))) return;
  if (changed) _markChanged(changed);
  _pendingConversationSave = ++_writeSeq;
  if (changed) pushConversation(changed);
  _flushPendingWrites();
}

// The ids this page deleted, dropped or gave up: never kept in the cache.
const _goneIds = new Set();

/** The conversations the browser cache gets: those this page holds, and a copy
 *  another tab cached with a change it has not saved yet. R40: server index-only
 *  rows (_meta) carry no messages and are not cached. */
function cacheableConversations() {
  const mine = chat.conversations.filter((c) => !c._meta && convPersistable(c));
  const unsavedHere = new Set(mine.filter((c) => c.unsaved).map((c) => c.id));
  const others = readStoredJSON("localm.conversations", []).filter((c) => c && c.unsaved &&
    (c.messages || []).length && !unsavedHere.has(c.id) && !_goneIds.has(c.id));
  const keep = new Set(others.map((c) => c.id));
  return [...others, ...mine.filter((c) => !keep.has(c.id))]
    .sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
}

function _writeConversationCache() {
  const cacheable = cacheableConversations();
  let localOk = true;
  let kept = cacheable.slice(0, 50);
  try {
    localStorage.setItem("localm.conversations", JSON.stringify(kept));
  } catch {
    // Quota: drop image-heavy older conversations and retry once
    kept = cacheable.slice(0, 10);
    try {
      localStorage.setItem("localm.conversations", JSON.stringify(kept));
    } catch (e2) {
      localOk = false;
      kept = [];
      console.error("localm: could not save conversations to localStorage:", e2);
    }
  }
  _localSaveOk = localOk;
  _localSavedIds.clear();
  for (const c of kept) _localSavedIds.add(c.id);
  _updateUnsavedIndicator();
}

/* ---- server-side conversation store (non-privacy modes only) ---- */

export const _convPushTimers = new Map();
let _convPushWarned = false;   // one warning per breakage, re-armed on success
const _convTooLargeWarned = new Set();   // conversation ids already toasted for a 413

// The save of each conversation in flight: the next one of the same conversation
// waits for it, so it names the revision that save returned.
const _convPushChain = new WeakMap();

/** Debounced upsert of one conversation to the server store. The save names the
 *  stored revision it replaces (`conv.rev`); when the store holds another one,
 *  resolveStaleSave decides. An index row's messages are loaded first. */
export function pushConversation(conv) {
  if (!chat.persist || !convPersistable(conv)) return;
  if (conv._meta) { saveIndexRowChange(conv); return; }
  // Brand-new conversations with nothing in them yet aren't worth a file.
  if (!conv.messages.length && !(conv.branches || []).length && conv.title === "New chat") return;
  if (!conv.updated_at) conv.updated_at = Date.now();
  clearTimeout(_convPushTimers.get(conv.id));
  _convPushTimers.set(conv.id, setTimeout(() => {
    _convPushTimers.delete(conv.id);
    const run = (_convPushChain.get(conv) || Promise.resolve()).then(() => _putConversation(conv));
    _convPushChain.set(conv, run);
  }, 600));
}

/** Saves a change of title, pin, folder, pinned model or drawer values made on
 *  *conv*, an index row whose messages are not loaded: loads the stored copy,
 *  puts the change back on it and saves it. Says so when it cannot be loaded. */
async function saveIndexRowChange(conv) {
  const mine = convMeta(conv);
  const ok = await hydrateConversation(conv, { keep: mine });
  if (ok && !conv._meta) {
    saveConversations(conv);
    renderConvList();
    if (conv === currentConv()) renderChat();
  } else {
    toast(t("chat.conv.saveFailed"), true);
  }
}

/** The title of *conv* as the list shows it. */
export function shownTitle(conv) {
  return !conv.title || conv.title === "New chat" ? t("chat.conv.newTitle") : conv.title;
}

/** The fields of *conv* a save carries beside its messages. */
function convMeta(conv) {
  return { title: conv.title, pinned: !!conv.pinned, folder: conv.folder || null,
           pinned_model: pinnedModelName(conv) || null, params: conv.params || {} };
}

function applyConvMeta(conv, meta) {
  if (meta.title != null) conv.title = meta.title;
  if (meta.pinned) conv.pinned = true; else delete conv.pinned;
  if (meta.folder) conv.folder = meta.folder; else delete conv.folder;
  if (meta.pinned_model !== conv.pinnedModel) {
    delete conv.pinnedModelId;
    if (meta.pinned_model) conv.pinnedModel = meta.pinned_model; else delete conv.pinnedModel;
  }
  if (meta.params && Object.keys(meta.params).length) conv.params = { ...meta.params };
  else delete conv.params;
  if (conv.id === _paramsOwner) showConvParams(conv.params);
}

const _sameJSON = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** A short fingerprint of the messages and branches of *data*. */
function bodyPrint(data) {
  const text = JSON.stringify([data.messages || [], data.branches || []]);
  let h1 = 0xdeadbeef, h2 = 0x41c6ce57;
  for (let i = 0; i < text.length; i++) {
    const ch = text.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  return `${text.length}.${(h1 >>> 0).toString(36)}.${(h2 >>> 0).toString(36)}`;
}

/** What *conv* looked like when it was last the same as the stored copy. */
function syncPoint(conv) {
  return { ...convMeta(conv), body: bodyPrint(conv) };
}

/** The PUT body of *conv*. `base_rev` is left out for a conversation whose
 *  stored revision this browser does not know. */
function putBody(conv) {
  const body = { ...convMeta(conv), updated_at: conv.updated_at,
                 branches: conv.branches || [], messages: conv.messages };
  if (conv.rev) body.base_rev = conv.rev;
  return JSON.stringify(body);
}

/** The PUT of one conversation (see pushConversation). Never throws. */
async function _putConversation(conv) {
  if (!chat.persist || !convPersistable(conv) || conv._meta) return;   // privacy mode, or the chat was deleted, while this waited
  const edits = _editCount.get(conv) || 0;
  const sent = syncPoint(conv);
  const failed = () => { if (convPersistable(conv)) _saveFailed.add(conv.id); };
  try {
    const r = await fetch("/api/conversations/" + encodeURIComponent(conv.id), {
      method: "PUT", headers: authHeaders(), body: putBody(conv),
    });
    if (r.ok) {
      _convPushWarned = false;
      _saveFailed.delete(conv.id);
      const saved = await r.json().catch(() => null);
      if (saved && typeof saved.rev === "string") {
        conv.rev = saved.rev;
        conv.sync = sent;
      }
      if ((_editCount.get(conv) || 0) === edits) delete conv.unsaved;
      saveConversations();
    } else if (r.status === 409) {
      const data = await r.json().catch(() => null);
      await resolveStaleSave(conv, (data && data.detail) || {});
    } else {
      failed();
      if (r.status === 403) refreshCtxLimit();   // the mode, or this key's rights, changed
      if (r.status === 413 && !_convTooLargeWarned.has(conv.id)) {
        _convTooLargeWarned.add(conv.id);
        toast(t(_localSaveOk && _localSavedIds.has(conv.id)
          ? "chat.conv.tooLarge" : "chat.conv.tooLargeTabOnly"), true);
      }
      if (!_convPushWarned) {
        _convPushWarned = true;
        console.error("conversation save failed (HTTP " + r.status + ") - " +
          (_localSaveOk
            ? "server-side persistence may be stale; the local copy is intact"
            : "server-side persistence may be stale, and the local copy " +
              "could not be saved either"));
      }
    }
  } catch (e) {
    failed();
    if (!_convPushWarned) {
      _convPushWarned = true;
      console.error("conversation save: could not reach the server" +
        (_localSaveOk ? " - the local copy is intact"
                      : " - and the local copy could not be saved either") + ":", e);
    }
  }
  _updateUnsavedIndicator();
}

/** A fresh conversation id. */
export function newConvId() {
  return Date.now().toString(36) + Math.floor(Math.random() * 1296).toString(36).padStart(2, "0");
}

/** Gives *conv* the id *id*; everything kept under its old id moves along. */
function rekeyConversation(conv, id) {
  const old = conv.id;
  clearTimeout(_convPushTimers.get(old));
  _convPushTimers.delete(old);
  _saveFailed.delete(old);
  _goneIds.add(old);
  conv.id = id;
  if (chat.activeId === old) chat.activeId = id;
  for (const q of chat.queue) if (q.convId === old) q.convId = id;
  if (_paramsOwner === old) _paramsOwner = id;
  rekeyDraft(old, id);
}

/** Settles a save of *conv* the store refused because it holds another
 *  revision. *detail* is the 409's detail: `rev` null when the conversation is
 *  not stored any more. A conversation deleted elsewhere stays in this tab only
 *  when it holds something not saved, else it goes here too. Changes of the
 *  title, pin, folder, pinned model and drawer values are merged. When the
 *  messages changed both here and there, the copy here becomes a conversation
 *  of its own and the stored one is listed beside it. */
async function resolveStaleSave(conv, detail) {
  if (!detail.rev) { dropDeletedElsewhere(conv); return; }
  let data = null;
  try {
    const r = await fetch("/api/conversations/" + encodeURIComponent(conv.id),
                          { headers: authHeaders() });
    if (r.status === 404) { dropDeletedElsewhere(conv); return; }
    if (r.ok) data = await r.json();
  } catch { /* kept as a copy below */ }
  if (!convPersistable(conv)) return;
  if (data && typeof data === "object") {
    migrateConversation(data);
    const base = conv.sync;
    const mine = bodyPrint(conv);
    const theirs = bodyPrint(data);
    const changedHere = mine !== theirs && (!base || mine !== base.body);
    const changedThere = mine !== theirs && (!base || theirs !== base.body);
    if (!(changedHere && changedThere)) {
      if (changedThere && chatBusy() && conv === currentConv()) {
        setTimeout(() => pushConversation(conv), 1000);
        return;
      }
      takeStoredCopy(conv, data, { body: changedThere });
      return;
    }
  }
  keepAsCopy(conv, detail.row);
}

/** Merges the stored copy *data* into *conv*: the messages when *body*, and
 *  each other field this page did not change since the last sync. What then
 *  differs from the stored copy is saved on top of it. */
function takeStoredCopy(conv, data, { body }) {
  const base = conv.sync;
  const mine = convMeta(conv);
  const stored = { title: data.title ?? mine.title, pinned: !!data.pinned,
                   folder: data.folder || null, pinned_model: data.pinned_model || null,
                   params: data.params || {} };
  const merged = {};
  for (const key of Object.keys(stored)) {
    const changedHere = base && !_sameJSON(mine[key], base[key]);
    merged[key] = changedHere ? mine[key] : stored[key];
  }
  if (body) {
    conv.messages = data.messages || [];
    conv.branches = data.branches || [];
  }
  applyConvMeta(conv, merged);
  conv.rev = data.rev;
  conv.sync = { ...stored, body: bodyPrint(data) };
  delete conv._meta;
  if (!_sameJSON(merged, stored) || (!body && conv.sync.body !== bodyPrint(conv))) {
    saveConversations(conv);
  } else {
    delete conv.unsaved;
    if (Number.isFinite(data.updated_at)) conv.updated_at = data.updated_at;
    _saveFailed.delete(conv.id);
    saveConversations();
  }
  renderConvList();
  if (body && conv === currentConv()) renderChat();
}

/** Makes *conv*, changed both here and in the store, a conversation of its own
 *  and lists the stored one (its index row *row*) beside it. */
function keepAsCopy(conv, row) {
  const name = shownTitle(conv);
  const oldId = conv.id;
  rekeyConversation(conv, newConvId());
  conv.title = t("chat.conv.copyTitle", { name });
  delete conv.rev;
  delete conv.sync;
  const at = chat.conversations.indexOf(conv);
  if (row && at !== -1 && !chat.conversations.some((c) => c.id === oldId)) {
    chat.conversations.splice(at + 1, 0, { title: name, ...row, id: oldId, _meta: true, messages: [] });
  }
  saveConversations(conv);
  renderConvList();
  toast(t("chat.conv.keptAsCopy"), true);
}

/** Takes *conv* out of the list here: it was deleted in another tab or browser.
 *  With something in it that was never saved, it stays in this tab only. */
function dropDeletedElsewhere(conv, { quiet = false } = {}) {
  const name = shownTitle(conv);
  if (conv.unsaved && !conv._meta) {
    keepInTabOnly(conv);
    toast(t("chat.conv.deletedElsewhereKept"), true);
  } else {
    conv.deleted = true;
    _goneIds.add(conv.id);
    clearTimeout(_convPushTimers.get(conv.id));
    _convPushTimers.delete(conv.id);
    chat.conversations = chat.conversations.filter((c) => c !== conv);
    if (chat.activeId === conv.id) chat.activeId = chat.conversations[0]?.id || null;
    chat.queue = chat.queue.filter((q) => q.convId !== conv.id);
    // What is typed for it stays in the message box.
    if (conv.id === _composerOwner && !draftIsEmpty(liveDraft())) _composerOwner = null;
    forgetDrafts([conv.id]);
    if (!quiet) toast(t("chat.conv.deletedElsewhere", { name }));
  }
  saveConversations();
  renderConvList();
  if (!chatBusy()) renderChat();
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
    toast(t("chat.conv.deleteFailed"), true);
    console.error("conversation delete failed: " + (e && e.message ? e.message : e));
  });
}

/** Delete every conversation in the server store, page by page, until the list
 *  holds nothing left to try, including files the list reports as unreadable.
 *  Returns `{ deleted, failed, listFailed }`: the ids deleted, the ids still
 *  stored (a DELETE refused, or the row still listed after an OK), and whether a
 *  list request failed, in which case more may remain. Never throws. */
export async function deleteAllServerConversations() {
  for (const timer of _convPushTimers.values()) clearTimeout(timer);
  _convPushTimers.clear();
  const deleted = new Set();
  const failed = new Set();
  const result = (listFailed) => ({ deleted: [...deleted], failed: [...failed], listFailed });
  for (;;) {
    let data;
    try {
      const r = await fetch(`/api/conversations?meta=true&limit=${200 + failed.size}&offset=0`,
                            { headers: authHeaders() });
      if (!r.ok) return result(true);
      data = await r.json();
    } catch {
      return result(true);
    }
    if (!data.enabled) return result(true);
    const listed = [...(data.conversations || []).map((c) => c.id), ...(data.unreadable || [])];
    for (const id of listed) {
      if (deleted.delete(id)) failed.add(id);
    }
    const todo = listed.filter((id) => !failed.has(id));
    if (!todo.length) return result(false);
    const outcomes = await Promise.all(todo.map(async (id) => {
      try {
        const r = await fetch("/api/conversations/" + encodeURIComponent(id), {
          method: "DELETE", headers: authHeaders(),
        });
        return r.ok;
      } catch {
        return false;
      }
    }));
    todo.forEach((id, i) => (outcomes[i] ? deleted : failed).add(id));
  }
}

/** Keep only the conversations *keep* accepts, here and in this browser's cache,
 *  and repaint the list and the chat. */
export function retainConversations(keep) {
  for (const c of chat.conversations) if (!keep(c)) _goneIds.add(c.id);
  chat.conversations = chat.conversations.filter(keep);
  if (!chat.conversations.some((c) => c.id === chat.activeId)) {
    chat.activeId = chat.conversations[0]?.id || null;
  }
  chat.queue = chat.queue.filter((q) => chat.conversations.some((c) => c.id === q.convId));
  forgetDrafts(null, (id) => !chat.conversations.some((c) => c.id === id));
  saveConversations();
  renderConvList();
  renderChat();
}

/** R40: fetch a full conversation's body on demand and replace its index-only
 *  ("_meta") placeholder in chat.conversations. *keep* (a convMeta) puts those
 *  values back over the stored ones. Returns true on success. */
export async function hydrateConversation(conv, { keep = null } = {}) {
  if (!conv || !conv._meta) return true;
  try {
    const r = await fetch("/api/conversations/" + encodeURIComponent(conv.id),
                          { headers: authHeaders() });
    if (!r.ok) return false;
    applyStoredConversation(conv, await r.json());
    if (keep) {
      const stored = convMeta(conv);
      for (const key of ["pinned_model", "params"]) {
        if (!keep[key] || !Object.keys(keep[key]).length) keep[key] = stored[key];
      }
      applyConvMeta(conv, keep);
    }
    delete conv._meta;
    saveConversations();   // cache the now-full conversation locally
    return true;
  } catch {
    return false;   // offline - keep the placeholder; renderChat shows the hint
  }
}

/** Takes the stored copy of *conv*, whose messages this page holds, unless it
 *  was changed here meanwhile. */
async function refreshStored(conv) {
  try {
    const r = await fetch("/api/conversations/" + encodeURIComponent(conv.id),
                          { headers: authHeaders() });
    if (r.ok) {
      const data = await r.json();
      if (!conv.unsaved && convPersistable(conv)) applyStoredConversation(conv, data);
    }
  } catch { /* the next sync tries again */ }
}

/** Copies the stored conversation *data* into *conv*. A file saved without
 *  drawer values leaves the ones held here. */
function applyStoredConversation(conv, data) {
  conv.messages = data.messages || [];
  conv.branches = data.branches || [];
  migrateConversation(conv);
  applyConvMeta(conv, { title: data.title, pinned: data.pinned, folder: data.folder,
                        pinned_model: data.pinned_model || null,
                        params: data.params || conv.params });
  if (Number.isFinite(data.updated_at)) conv.updated_at = data.updated_at;
  if (typeof data.rev === "string") {
    conv.rev = data.rev;
    conv.sync = syncPoint(conv);
  }
  delete conv.unsaved;
}
window.hydrateConversation = hydrateConversation;

/** The server store's index: the rows of its first page (`rows`, what the list
 *  shows), every stored row by id (`byId`), whether that is all of them
 *  (`complete`) and the ids of unreadable files. Null when it cannot be read. */
async function fetchServerIndex() {
  const page = async (offset) => {
    const r = await fetch(`/api/conversations?meta=true&limit=200&offset=${offset}`,
                          { headers: authHeaders() });
    if (!r.ok) return null;
    const data = await r.json();
    return data.enabled ? data : null;
  };
  const first = await page(0);
  if (!first) return null;
  const rows = first.conversations || [];
  const byId = new Map(rows.map((c) => [c.id, c]));
  const total = Number.isInteger(first.total) ? first.total : rows.length;
  let complete = true;
  for (let offset = rows.length; offset < total;) {
    const more = await page(offset);
    const got = (more && more.conversations) || [];
    if (!got.length) { complete = false; break; }
    for (const c of got) byId.set(c.id, c);
    offset += got.length;
  }
  return { rows, byId, complete, unreadable: new Set(first.unreadable || []) };
}

/** Loads the server store and merges it with what this page holds. R40: the
 *  server list is a lightweight index (no message bodies); each conversation's
 *  messages load when it is opened. A conversation this browser holds with the
 *  stored revision is kept, and saved when it holds a change; with another
 *  revision the stored copy is taken, unless it holds a change, in which case
 *  its save settles it (resolveStaleSave). One that was stored and is not
 *  stored any more was deleted elsewhere. One never stored is uploaded. */
export async function initServerConversations() {
  if (chat.privacy) return;
  try {
    const index = await fetchServerIndex();
    if (!index || chat.privacy) return;
    const boot = !chat.persist;
    chat.persist = true;
    if (boot && index.unreadable.size) {
      toast(tn("chat.conv.unreadable", index.unreadable.size), true);
    }
    const before = _listPrint();
    const byId = new Map(index.rows.map((c) => [c.id, { ...c, _meta: true, messages: [] }]));
    const stale = [];
    const gone = [];
    for (const local of chat.conversations) {
      const remote = index.byId.get(local.id);
      const keep = () => byId.set(local.id, local);
      const keepRow = () => byId.set(local.id, { ...remote, _meta: true, messages: [] });
      if (local.tabOnly) keep();
      else if (!remote) {
        // A conversation this backend's index does not know may belong to
        // another backend at the same origin. It is shown unless the backend is
        // a confirmed mismatch, and uploaded only for a confirmed match.
        if (chat.instanceState === "mismatched") continue;
        keep();
        if (local.rev && index.complete && chat.instanceMatch &&
            !index.unreadable.has(local.id)) gone.push(local);
        if (!local.rev && chat.instanceMatch && !index.unreadable.has(local.id)) {
          pushConversation(local);
        }
      } else if (local._meta) {
        Object.assign(local, remote);
        keep();
      } else if (!local.rev || typeof remote.rev !== "string") {
        // Held since before conversations had a revision: the newer copy wins,
        // and the server refuses a save that would lose messages.
        const localAt = local.updated_at || 0;
        const remoteAt = remote.updated_at || 0;
        if (localAt < remoteAt) { keepRow(); continue; }
        keep();
        if (localAt > remoteAt) pushConversation(local);
        else if (typeof remote.rev === "string") {
          local.rev = remote.rev;
          local.sync = syncPoint(local);
        }
      } else if (local.rev === remote.rev) {
        keep();
        if (local.unsaved) pushConversation(local);
      } else if (local.unsaved) {
        keep();
        pushConversation(local);
      } else if (boot) {
        keepRow();
      } else {
        keep();
        stale.push(local);
      }
    }
    chat.conversations = [...byId.values()]
      .sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
    // One saved since the index was read is stored after all.
    for (const conv of gone) {
      if (await isNotStored(conv)) dropDeletedElsewhere(conv, { quiet: boot });
    }
    if (!chat.conversations.some((c) => c.id === chat.activeId)) {
      chat.activeId = chat.conversations[0]?.id || null;
    }
    for (const conv of stale) await refreshStored(conv);
    await hydrateConversation(currentConv());   // load the active conversation's body
    if (boot || stale.length || before !== _listPrint()) {
      saveConversations();
      renderConvList();
      if (!chatBusy()) renderChat();
    }
    _updateUnsavedIndicator();
  } catch { /* store unavailable - localStorage keeps working */ }
}

/** Whether the server says that *conv* is not stored. */
async function isNotStored(conv) {
  try {
    const r = await fetch("/api/conversations/" + encodeURIComponent(conv.id),
                          { headers: authHeaders() });
    return r.status === 404 && convPersistable(conv);
  } catch { return false; }
}

/** A fingerprint of the listed conversations. */
function _listPrint() {
  return JSON.stringify(chat.conversations.map(
    (c) => [c.id, c.title, !!c.pinned, c.folder || "", c.rev || "", !!c.tabOnly]));
}

let _syncing = false;

/** Brings this page up to date with the server: the session mode first, then
 *  the conversations changed or deleted in another tab or browser. Does nothing
 *  while a reply is being written. */
export async function syncConversations() {
  if (_syncing || !chat.modeKnown || chatBusy()) return;
  _syncing = true;
  try {
    await confirmSessionMode();
    if (chat.persist && !chat.privacy && !chatBusy()) await initServerConversations();
  } finally {
    _syncing = false;
  }
}

// A page that goes away sends the saves it was about to make.
window.addEventListener("pagehide", () => {
  for (const id of _convPushTimers.keys()) {
    const conv = chat.conversations.find((c) => c.id === id);
    if (!conv || conv._meta || !chat.persist || !convPersistable(conv)) continue;
    fetch("/api/conversations/" + encodeURIComponent(id), {
      method: "PUT", headers: authHeaders(), body: putBody(conv), keepalive: true,
    }).catch(() => { /* too large for a closing page: the copy in this browser is saved when it opens again */ });
  }
});

export function currentConv() {
  return chat.conversations.find((c) => c.id === chat.activeId) || null;
}

/** The current name of the model *conv* is pinned to, "" when it is not pinned.
 *  Finds the model by its stable id, then by the stored name, then by a name the
 *  model used to have, and records what it found on *conv*. A pin no installed
 *  model matches keeps its stored name, and the server refuses it by that name. */
export function pinnedModelName(conv) {
  const pinned = conv && conv.pinnedModel;
  if (!pinned) return "";
  const models = (modelCache && modelCache.models) || [];
  const row = (conv.pinnedModelId && models.find((m) => m.model_id === conv.pinnedModelId))
    || models.find((m) => m.name === pinned)
    || models.find((m) => (m.former_names || []).includes(pinned));
  if (!row) return pinned;
  conv.pinnedModel = row.name;
  if (row.model_id) conv.pinnedModelId = row.model_id;
  return row.name;
}

/** Shows the drawer's Knowledge selector only while the rag plugin is active,
 *  since without it there is no Knowledge page to manage collections on. */
function syncKnowledgeRow() {
  const row = $("p-kb-row");
  if (row) row.hidden = !pluginState.some((p) => p && p.name === "rag" && p.active === true);
}
const _paramsDrawer = document.getElementById("params");
if (_paramsDrawer && typeof MutationObserver === "function") {
  new MutationObserver(syncKnowledgeRow).observe(_paramsDrawer, { attributes: true, attributeFilter: ["class"] });
}

/** Reflects *conv*'s pinned model in the drawer's pin checkbox and its label:
 *  checked and naming the pinned model when there is one, otherwise unchecked
 *  and naming the model selected in the sidebar. */
export function syncPinModelToggle(conv) {
  const box = $("p-pin-model");
  const label = $("p-pin-model-name");
  if (!box) return;
  const pinned = pinnedModelName(conv);
  box.checked = !!pinned;
  if (label) label.textContent = pinned || modelSelect.value || modelCache.active || t("chat.pinModel.none");
  syncKnowledgeRow();
  const note = $("pin-mismatch");
  if (note) {
    const active = modelCache.active;
    const gone = !!pinned && isModelGone(pinned);
    const mismatch = !gone && !!pinned && !!active && pinned !== active;
    note.hidden = !(gone || mismatch);
    note.textContent = gone ? t("chat.pinModel.gone", { pinned })
      : mismatch ? t("chat.pinModel.mismatch", { pinned, active }) : "";
  }
}

/** Pins the conversation on screen to the model selected in the sidebar, or
 *  unpins it. A pinned conversation is always answered by that model, even
 *  when a request needs something it lacks. Pinning with no chat open starts
 *  a chat for the pin. */
export function setConversationPin(on) {
  if (!currentConv()) {
    if (!on) { syncPinModelToggle(null); return; }
    newConversation();
  }
  const conv = currentConv();
  const model = modelSelect.value || modelCache.active;
  delete conv.pinnedModelId;
  if (on && model) conv.pinnedModel = model;
  else delete conv.pinnedModel;
  pinnedModelName(conv);
  saveConversations(conv);
  syncPinModelToggle(conv);
}
window.setConversationPin = setConversationPin;

/** True when *name* is certainly not a registered model: the list the page
 *  holds is known (not empty) and neither lists nor serves it. */
export function isModelGone(name) {
  const listed = (modelCache.models || []).map((m) => m.name);
  return !!name && listed.length > 0 && !listed.includes(name) && name !== modelCache.active;
}

/** Whether *conv*'s pinned model can answer: true when it is not pinned or
 *  its model is registered. A model renamed elsewhere is found by its stable
 *  id or a name it used to have, else picked up from the server's copy of the
 *  chat when that copy names a registered model. False
 *  means the pinned model no longer exists. Asks the server for the registry;
 *  when it cannot be asked, the model list already held decides. */
export async function resolvePinnedModel(conv) {
  const name = conv && conv.pinnedModel;
  if (!name) return true;
  let registered = null;
  let rows = [];
  try {
    const r = await fetch("/api/models", { headers: authHeaders() });
    const data = r.ok ? await r.json() : null;
    if (data && Array.isArray(data.models) && data.models.length) {
      rows = data.models;
      registered = new Set(rows.map((m) => m.name));
    }
  } catch { /* the held list decides */ }
  if (registered === null) return !isModelGone(pinnedModelName(conv));
  if (registered.has(name)) return true;
  const row = (conv.pinnedModelId && rows.find((m) => m.model_id === conv.pinnedModelId))
    || rows.find((m) => (m.former_names || []).includes(name));
  if (row) {
    conv.pinnedModel = row.name;
    if (row.model_id) conv.pinnedModelId = row.model_id;
    saveConversations();
    syncPinModelToggle(conv);
    return true;
  }
  if (chat.persist && !conv._meta) {
    try {
      const r = await fetch("/api/conversations/" + encodeURIComponent(conv.id),
                            { headers: authHeaders() });
      const stored = r.ok ? (await r.json()).pinned_model : null;
      if (stored && registered.has(stored)) {
        conv.pinnedModel = stored;
        saveConversations();
        syncPinModelToggle(conv);
        return true;
      }
    } catch { /* the stored copy cannot be read; the pin is gone */ }
  }
  return false;
}

/** Removes the pin of *conv* and says in the chat that the model it named no
 *  longer exists, so the person can pick another. The row is not saved with
 *  the chat. Returns the row, or null when there is no chat on screen. */
export function showPinnedGone(conv) {
  const gone = conv.pinnedModel;
  delete conv.pinnedModel;
  saveConversations(conv);
  syncPinModelToggle(conv);
  const box = $("chat-messages");
  if (!box) return null;
  const row = el("div", "msg-row tool-event tool-notice-row");
  row.appendChild(el("div", "msg-role", t("chat.pinGone.label")));
  row.appendChild(el("div", "tool-notice", t("chat.pinGone.notice", { model: gone })));
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
  return row;
}

/** The name a model called *name* when it answered has now. */
export function modelNameNow(name) {
  if (!name || name === "MODEL") return name;
  const models = (modelCache && modelCache.models) || [];
  if (models.some((m) => m.name === name)) return name;
  const row = models.find((m) => (m.former_names || []).includes(name));
  return row ? row.name : name;
}

/** After a model was renamed: the chats pinned to it (in this page, its
 *  browser copy and the server) and its avatar follow the new name. */
export function applyModelRename(from, to) {
  const moved = chat.conversations.filter((c) => c.pinnedModel === from);
  for (const c of moved) c.pinnedModel = to;
  if (moved.length) {
    saveConversations();
    for (const c of moved) pushConversation(c);
  }
  const avatars = chat.modelAvatarOverrides;
  if (avatars && Object.prototype.hasOwnProperty.call(avatars, from)) {
    avatars[to] = avatars[from];
    delete avatars[from];
  }
  syncPinModelToggle(currentConv());
}
/** The title a conversation gets from its first message *text*. */
export function titleFromText(text) {
  return text.slice(0, 42) + (text.length > 42 ? "\u2026" : "");
}

/** Whether the title of *conv* came from a first message (or it has none yet),
 *  so a new first message may replace it; a title the user typed may not. */
export function titleIsAutomatic(conv) {
  if (conv.title === "New chat") return true;
  const root = (conv.branches || []).find((b) => b.parent === "root");
  return ((root && root.tails) || []).some((tail) => {
    const first = (tail || []).find((m) => m.role === "user" && !m.tag);
    if (!first) return false;
    const text = msgText(first);
    const clip = msgAudioClips(first)[0];
    return conv.title === (titleFromText(text) || (clip ? clip.name.slice(0, 42) : "Document chat"));
  });
}

window.addEventListener("localm:model-renamed", (e) => applyModelRename(e.detail.from, e.detail.to));
window.addEventListener("localm:models-refreshed", () => syncPinModelToggle(currentConv()));

/* ---- per-conversation parameter fields ---- */

/** The parameters-drawer fields that belong to one conversation, by the key
 *  they are stored under in conv.params. Web access, memory, speak-aloud, voice
 *  and the knowledge collection are not here: they are global or handled
 *  elsewhere. */
export const CONV_PARAM_FIELDS = {
  persona: "p-persona",
  system: "p-system",
  temperature: "p-temperature",
  top_p: "p-top-p",
  top_k: "p-top-k",
  repeat_penalty: "p-repeat-penalty",
  max_tokens: "p-max-tokens",
  seed: "p-seed",
  grammar: "p-grammar",
};

// The id of the conversation whose values the drawer fields currently show;
// null when the fields belong to no conversation.
let _paramsOwner = null;

/** Writes the drawer's per-conversation fields into conv.params (only the
 *  non-blank ones; the key is removed when all are blank). Returns true when
 *  the stored value changed. */
export function captureConvParams(conv) {
  if (!conv) return false;
  const next = {};
  for (const [key, id] of Object.entries(CONV_PARAM_FIELDS)) {
    const f = $(id);
    if (f && String(f.value ?? "").trim() !== "") next[key] = f.value;
  }
  const had = conv.params ? JSON.stringify(conv.params) : "";
  const has = Object.keys(next).length ? JSON.stringify(next) : "";
  if (had === has) return false;
  if (has) conv.params = next;
  else delete conv.params;
  return true;
}

/** Shows *params* (a conv.params object, or none for a blank panel) in the
 *  drawer fields and refreshes the Advanced fold's badge; a fold holding a
 *  value opens. */
function showConvParams(params) {
  const p = params || {};
  for (const [key, id] of Object.entries(CONV_PARAM_FIELDS)) {
    const f = $(id);
    if (!f) continue;
    const v = p[key] == null ? "" : String(p[key]);
    f.value = v;
    if (f.tagName === "SELECT" && f.value !== v) f.value = "";
  }
  const drawer = document.getElementById("params");
  if (drawer) revealFilledAdvanced(drawer);
  const fold = document.getElementById("p-advanced");
  if (fold) updateAdvancedCount(fold);
}

/** Stores the drawer's current per-conversation values on the conversation
 *  they are showing. Call after a programmatic write into the fields. */
export function persistChatParams() {
  const conv = currentConv();
  if (!conv || conv.id !== _paramsOwner) return;
  if (captureConvParams(conv)) saveConversations(conv);
}

/** Makes the drawer show the active conversation's own values: the outgoing
 *  conversation keeps what the fields held, the incoming one gets its stored
 *  values (blank defaults when it has none). Fields holding values typed with
 *  no conversation open go to the first conversation that appears. */
function syncChatParams(conv) {
  const id = conv ? conv.id : null;
  if (id === _paramsOwner) return;
  const prev = _paramsOwner
    ? chat.conversations.find((c) => c.id === _paramsOwner) : null;
  if (prev && captureConvParams(prev)) saveConversations(prev);
  if (conv && _paramsOwner === null && !conv.params && !conv._meta) {
    if (captureConvParams(conv)) saveConversations(conv);
  } else {
    showConvParams(conv ? conv.params : null);
  }
  _paramsOwner = id;
}

/* ---- the message box of each conversation ---- */

// The id of the conversation the message box and the attachment lists belong
// to; null when no conversation is open.
let _composerOwner = null;
// What each other conversation's message box held when it was left, by id:
// { text, attachments, docs, clips, held }. *held*: typed under privacy mode, so
// it is never written to this browser.
const _drafts = new Map();
let _composerHeld = false;

function emptyDraft() {
  return { text: "", attachments: [], docs: [], clips: [], held: false };
}

function draftIsEmpty(d) {
  return !d.text.trim() && !d.attachments.length && !d.docs.length && !d.clips.length;
}

/** What the message box and the attachment lists hold. */
function liveDraft() {
  const input = $("chat-input");
  return { text: input ? input.value : "", attachments: chat.attachments,
           docs: chat.docs, clips: chat.clips, held: _composerHeld };
}

function showDraft(d) {
  const input = $("chat-input");
  if (input) {
    input.value = d.text;
    autoGrow(input);
  }
  chat.attachments = d.attachments;
  chat.docs = d.docs;
  chat.clips = d.clips;
  _composerHeld = !!d.held;
  renderAttachChips();
}

/** The draft of the conversation with id *owner*, for adding to it: the live
 *  lists when the message box belongs to it, else the one kept for it. */
function draftOf(owner) {
  if (owner === _composerOwner || owner === null) return chat;
  if (!_drafts.has(owner)) _drafts.set(owner, emptyDraft());
  return _drafts.get(owner);
}

/** Makes the message box belong to *conv*: what the outgoing conversation held
 *  is kept for it, the incoming one gets back what it held. What was typed with
 *  no conversation open stays for the first one that appears. */
export function syncComposer(conv) {
  const id = conv ? conv.id : null;
  if (id === _composerOwner) return;
  if (_composerOwner !== null) {
    const d = liveDraft();
    if (draftIsEmpty(d)) _drafts.delete(_composerOwner);
    else _drafts.set(_composerOwner, d);
    showDraft((id !== null && _drafts.get(id)) || emptyDraft());
  } else if (_drafts.has(id)) {
    showDraft(joinDrafts(_drafts.get(id), liveDraft()));
  }
  _drafts.delete(id);
  _composerOwner = id;
  for (const key of [..._drafts.keys()]) {
    if (!chat.conversations.some((c) => c.id === key)) _drafts.delete(key);
  }
  saveDraftsSoon();
}

function joinDrafts(a, b) {
  return { text: [a.text, b.text].filter((x) => x.trim()).join("\n"),
           attachments: [...a.attachments, ...b.attachments],
           docs: [...a.docs, ...b.docs], clips: [...a.clips, ...b.clips],
           held: !!(a.held || b.held) };
}

/** Takes what the message box of *conv* holds, leaving it empty. */
export function takeComposer(conv) {
  let d = _drafts.get(conv.id) || emptyDraft();
  if (conv.id === _composerOwner || _composerOwner === null) {
    d = liveDraft();
    showDraft(emptyDraft());
  }
  _drafts.delete(conv.id);
  saveDraftsSoon();
  return { text: d.text.trim(), attachments: d.attachments, docs: d.docs, clips: d.clips };
}

/** The message box keeps what it holds and belongs to no conversation. */
function releaseComposer() {
  _composerOwner = null;
  _drafts.clear();
}

function rekeyDraft(old, id) {
  if (_composerOwner === old) _composerOwner = id;
  if (_drafts.has(old)) {
    _drafts.set(id, _drafts.get(old));
    _drafts.delete(old);
  }
}

/** What is typed now is never written to this browser. */
function holdDraftsInTab() {
  if (!draftIsEmpty(liveDraft())) _composerHeld = true;
  for (const d of _drafts.values()) d.held = true;
}

/* Drafts in this browser. Outside privacy mode what is typed, attached or
   queued is kept in IndexedDB, so a reload or a server restart does not lose it.
   A row's key is "<conversation id>|<tab id>|text", "...|media" or "|<tab id>|queue":
   each tab writes its own rows, and a conversation opened in a new tab takes the
   newest rows a closed tab left. Writes wait for the session mode like every
   other (see _flushPendingWrites); privacy mode empties the store. */

const DRAFT_DB = "localm-drafts";
const TAB_ID_KEY = "localm.tabId";
const DRAFT_KEEP_MS = 30 * 24 * 3600 * 1000;

let _draftDb = null;

function draftDb() {
  if (!_draftDb) {
    _draftDb = new Promise((resolve, reject) => {
      const req = window.indexedDB.open(DRAFT_DB, 1);
      req.onupgradeneeded = () => req.result.createObjectStore("drafts");
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    _draftDb.catch(() => { _draftDb = null; });
  }
  return _draftDb;
}

/** The IndexedDB rows of the drafts. */
export const draftStore = {
  available() { return !!window.indexedDB; },
  async all() {
    const db = await draftDb();
    return new Promise((resolve, reject) => {
      const rows = new Map();
      const req = db.transaction("drafts").objectStore("drafts").openCursor();
      req.onerror = () => reject(req.error);
      req.onsuccess = () => {
        const cur = req.result;
        if (!cur) { resolve(rows); return; }
        rows.set(cur.key, cur.value);
        cur.continue();
      };
    });
  },
  async write(puts, dels) {
    const db = await draftDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction("drafts", "readwrite");
      const store = tx.objectStore("drafts");
      for (const [key, value] of puts) store.put(value, key);
      for (const key of dels) store.delete(key);
      tx.oncomplete = () => resolve();
      tx.onabort = tx.onerror = () => reject(tx.error);
    });
  },
  async clear() {
    const db = await draftDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction("drafts", "readwrite");
      tx.objectStore("drafts").clear();
      tx.oncomplete = () => resolve();
      tx.onabort = tx.onerror = () => reject(tx.error);
    });
  },
};

// What this tab's rows held when they were last written, by key; null for a row
// that was read back. A row this tab's drafts do not make any more is removed by
// the write that stores those they do.
const _draftsWritten = new Map();
let _draftWarned = false;
let _draftsWiped = false;
let _draftsRestored = false;
let _draftTimer = 0;
let _tabId = "";
// The draft store's calls run one after another.
let _draftChain = Promise.resolve();

function draftCall(work) {
  _draftChain = _draftChain.then(work).catch((e) => {
    if (_draftWarned) return;
    _draftWarned = true;
    console.warn("localm: unsent text and attachments cannot be kept in this browser:", e);
  });
  return _draftChain;
}

function tabId() {
  if (_tabId) return _tabId;
  try {
    _tabId = sessionStorage.getItem(TAB_ID_KEY) || "";
    if (!_tabId) {
      _tabId = newConvId() + newConvId();
      sessionStorage.setItem(TAB_ID_KEY, _tabId);
    }
  } catch { _tabId = _tabId || newConvId() + newConvId(); }
  return _tabId;
}

// While this tab is open it holds a lock of its name, so another tab can tell
// its rows from those a closed tab left.
const TAB_LOCK = "localm-tab:";
let _tabLocked = false;
function holdTabLock() {
  if (_tabLocked || !navigator.locks) return;
  _tabLocked = true;
  navigator.locks.request(TAB_LOCK + tabId(), () => new Promise(() => {})).catch(() => {});
}
async function openTabs() {
  try {
    const held = (await navigator.locks.query()).held || [];
    return new Set(held.filter((l) => l.name.startsWith(TAB_LOCK)).map((l) => l.name.slice(TAB_LOCK.length)));
  } catch { return new Set(); }
}

/** Queues a write of the drafts; text typed in quick succession is one write. */
export function saveDraftsSoon() {
  if (_composerHeld && draftIsEmpty(liveDraft())) _composerHeld = false;
  if (chat.privacy || !_draftsRestored || _draftTimer) return;
  _draftTimer = setTimeout(saveDraftsNow, 250);
}

function saveDraftsNow() {
  clearTimeout(_draftTimer);
  _draftTimer = 0;
  if (chat.privacy || !_draftsRestored) return;
  _pendingDraftSave = ++_writeSeq;
  _flushPendingWrites();
}

/** The rows this tab's drafts make, by key, each with a short *print* that
 *  differs when its value does. */
function draftRows() {
  const rows = new Map();
  const media = (list, size) => list.map((x) => x.name + ":" + size(x)).join("/");
  const add = (id, d) => {
    if (d.held || draftIsEmpty(d)) return;
    const conv = chat.conversations.find((c) => c.id === id);
    if (conv ? !convPersistable(conv) : id !== null) return;
    const key = `${id || ""}|${tabId()}|`;
    if (d.text) rows.set(key + "text", { print: d.text, value: { text: d.text } });
    if (d.attachments.length || d.docs.length || d.clips.length) {
      const end = (x) => x.length + x.slice(-24);
      rows.set(key + "media", {
        print: [media(d.attachments, (x) => end(x.dataUri)), media(d.docs, (x) => end(x.text)),
                media(d.clips, (x) => end(x.data) + x.seconds)].join("|"),
        value: { attachments: d.attachments, docs: d.docs, clips: d.clips } });
    }
  };
  add(_composerOwner, liveDraft());
  for (const [id, d] of _drafts) add(id, d);
  const queued = chat.queue.filter((q) =>
    convPersistable(chat.conversations.find((c) => c.id === q.convId)));
  if (queued.length) {
    rows.set(`|${tabId()}|queue`, {
      print: queued.map((q) => [q.convId, q.text, q.attachments.length, q.docs.length, q.clips.length].join(":")).join("|"),
      value: { items: queued.map(({ convId, text, attachments, docs, clips }) =>
        ({ convId, text, attachments, docs, clips })) } });
  }
  return rows;
}

function _writeDrafts() {
  if (!draftStore.available()) return;
  const rows = draftRows();
  const puts = [];
  const dels = [];
  for (const [key, row] of rows) {
    if (_draftsWritten.get(key) === row.print) continue;
    puts.push([key, { ...row.value, at: Date.now() }]);
    _draftsWritten.set(key, row.print);
  }
  for (const key of [..._draftsWritten.keys()]) {
    if (rows.has(key)) continue;
    dels.push(key);
    _draftsWritten.delete(key);
  }
  if (!puts.length && !dels.length) return;
  _draftsWiped = false;
  holdTabLock();
  draftCall(() => draftStore.write(puts, dels));
}

/** Empties the draft store of this browser (privacy mode). */
function wipeStoredDrafts() {
  if (_draftsWiped) return;
  _draftsWiped = true;
  _draftsWritten.clear();
  _tabId = "";
  _tabLocked = false;
  try { sessionStorage.removeItem(TAB_ID_KEY); } catch { /* nothing is stored there */ }
  if (draftStore.available()) draftCall(() => draftStore.clear());
}

/** Removes the stored drafts, of every tab, of the conversations with the ids
 *  *ids*, or of those *gone* answers true for. */
function forgetDrafts(ids, gone = (id) => ids.includes(id)) {
  for (const id of [..._drafts.keys()]) if (gone(id)) _drafts.delete(id);
  if (chat.privacy || !draftStore.available()) return;
  draftCall(async () => {
    const keys = [...(await draftStore.all()).keys()].filter((k) => {
      const id = k.split("|")[0];
      return id && gone(id);
    });
    for (const k of keys) _draftsWritten.delete(k);
    if (keys.length) await draftStore.write([], keys);
  });
}

/** Puts the drafts and queued messages this browser kept back: those of this
 *  tab, else the newest a closed tab left for a conversation. Those of a
 *  conversation that is not listed any more are removed. A queued message
 *  stays queued until something is sent. Call once the mode is known. */
export async function restoreDrafts() {
  if (_draftsRestored || !chat.modeKnown) return;
  _draftsRestored = true;
  if (chat.privacy || !draftStore.available()) return;
  await draftCall(async () => {
    const rows = await draftStore.all();
    if (chat.privacy) return;
    const own = tabId();
    holdTabLock();
    const open = await openTabs();
    const found = new Map();
    const drop = [];
    const entries = [...rows].map(([key, value]) => {
      const [id, tab, kind] = key.split("|");
      return { key, id, tab, kind, value };
    }).filter((r) => r.tab === own || !open.has(r.tab))
      .sort((a, b) => (b.tab === own) - (a.tab === own) || (b.value.at || 0) - (a.value.at || 0));
    const from = new Map();
    for (const r of entries) {
      const age = Date.now() - (r.value.at || 0);
      if (age > DRAFT_KEEP_MS) { drop.push(r.key); continue; }
      if (r.kind === "queue") {
        for (const item of r.value.items || []) {
          if (chat.conversations.some((c) => c.id === item.convId)) chat.queue.push(item);
        }
        _draftsWritten.set(r.key, null);
        continue;
      }
      if (r.id && !chat.conversations.some((c) => c.id === r.id)) {
        if (chat.persist) drop.push(r.key);
        continue;
      }
      if (!from.has(r.id)) from.set(r.id, r.tab);
      if (from.get(r.id) !== r.tab) continue;
      const d = found.get(r.id) || emptyDraft();
      if (r.kind === "text") d.text = r.value.text || "";
      else Object.assign(d, { attachments: r.value.attachments || [], docs: r.value.docs || [],
                              clips: r.value.clips || [] });
      found.set(r.id, d);
      _draftsWritten.set(r.key, null);
    }
    if (drop.length) await draftStore.write([], drop);
    for (const [id, d] of found) {
      if (id === (_composerOwner || "") || !id) showDraft(joinDrafts(d, liveDraft()));
      else _drafts.set(id, _drafts.has(id) ? joinDrafts(d, _drafts.get(id)) : d);
    }
  });
  renderQueuedIndicator();
  saveDraftsNow();
}

$("chat-input").addEventListener("input", saveDraftsSoon);
window.addEventListener("pagehide", saveDraftsNow);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden" && _draftTimer) saveDraftsNow();
});


for (const evt of ["input", "change"]) {
  $("params").addEventListener(evt, (e) => {
    if (Object.values(CONV_PARAM_FIELDS).includes(e.target.id)) persistChatParams();
  });
}

/** Open a new, empty chat. An untouched empty "New chat" already in the list
 *  (no messages, default title, no folder, pin or model pin) is reused and
 *  moved to the top instead of adding another. */
export function newConversation() {
  const idle = chat.conversations.find((c) => c.title === "New chat" && !c.tabOnly &&
    !(c.messages || []).length && !(c.branches || []).length && !c.folder && !c.pinned && !c.pinnedModel && !c.params);
  if (idle) {
    chat.conversations.splice(chat.conversations.indexOf(idle), 1);
    chat.conversations.unshift(idle);
    chat.activeId = idle.id;
    renderConvList();
    renderChat();
    return;
  }
  const conv = { id: newConvId(), title: "New chat", messages: [] };
  chat.conversations.unshift(conv);
  chat.activeId = conv.id;
  saveConversations(conv);
  renderConvList();
  renderChat();
}

/* message content helpers - content is a string or OpenAI multipart list.
   A tool event's text is the fenced prompt text it renders to. */
export function msgText(m) {
  if (isToolEvent(m)) return toolEventPrompt(m).content;
  if (typeof m.content === "string") return m.content;
  return (m.content || []).filter((p) => p.type === "text").map((p) => p.text).join("");
}
export function msgImages(m) {
  if (typeof m.content === "string") return [];
  return (m.content || []).filter((p) => p.type === "image_url")
    .map((p) => p.image_url?.url).filter(Boolean);
}

/** The user-attached audio clips of message *m*: one {data, format, name,
 *  seconds} per input_audio part that carries data. */
export function msgAudioClips(m) {
  if (!m || typeof m.content === "string") return [];
  return (m.content || [])
    .filter((p) => p.type === "input_audio" && p.input_audio?.data)
    .map((p) => ({ data: p.input_audio.data, format: p.input_audio.format || "wav",
                   name: p.name || "", seconds: p.seconds || 0 }));
}

/** The input_audio part for an audio attachment. The wire shape is
 *  {type, input_audio: {data, format}}; name and seconds are display
 *  metadata that runCompletion drops before sending. */
export function audioPart(clip) {
  return { type: "input_audio", input_audio: { data: clip.data, format: clip.format },
           name: clip.name, seconds: clip.seconds || 0 };
}

const IMAGE_LEFT_OUT = "[An image from an earlier message is not included in this request.]";
const AUDIO_LEFT_OUT = "[An audio clip from an earlier message is not included in this request.]";
const IMAGE_NOT_SENT = "[An attached image is not included: the server did not accept it.]";
const AUDIO_NOT_SENT = "[An attached audio clip is not included: the server did not accept it.]";

/** True for an image the user attached (a data: URI), not one the server
 *  generated (an /api/ URL). */
function isUserImage(p) {
  return p.type === "image_url" && !p.image_url?.url?.startsWith("/api/");
}

/** The part list sent over the wire for a message's *content* when *model*
 *  answers. An image or clip the server refused (`notSent`) is sent as a text
 *  note. With *latest* false (an earlier turn), an image or clip that *model*
 *  refused (`refusedBy`) is sent as a text note too. Display-only fields
 *  (`refusedBy`, `notSent`, a clip's name and seconds) are never sent; a clip
 *  without data is dropped. */
export function wireParts(content, { model = null, latest = true } = {}) {
  if (!Array.isArray(content)) return content;
  if (model !== null) model = model || "";
  const out = [];
  for (const p of content) {
    if (p.type === "input_audio") {
      if (!p.input_audio?.data) continue;
      if (p.notSent) { out.push({ type: "text", text: AUDIO_NOT_SENT }); continue; }
      if (!latest && model !== null && (p.refusedBy || []).includes(model)) {
        out.push({ type: "text", text: AUDIO_LEFT_OUT });
        continue;
      }
      out.push({ type: "input_audio",
                 input_audio: { data: p.input_audio.data, format: p.input_audio.format } });
    } else if (p.type === "image_url") {
      if (p.notSent) { out.push({ type: "text", text: IMAGE_NOT_SENT }); continue; }
      if (!latest && model !== null && (p.refusedBy || []).includes(model)) {
        out.push({ type: "text", text: IMAGE_LEFT_OUT });
        continue;
      }
      out.push({ type: "image_url", image_url: p.image_url });
    } else {
      out.push(p);
    }
  }
  return out;
}

/** The parts of type *type* ("image_url" or "input_audio") the user attached
 *  in *conv*: in every user message, or with *latestOnly* in the newest user
 *  message that has one (the turn just sent). */
function attachedParts(conv, type, latestOnly) {
  const carries = (p) => (type === "image_url" ? isUserImage(p) : p.type === type);
  const parts = [];
  for (let i = conv.messages.length - 1; i >= 0; i--) {
    const m = conv.messages[i];
    if (m.role !== "user" || !Array.isArray(m.content)) continue;
    const own = m.content.filter(carries);
    if (!own.length) continue;
    parts.push(...own);
    if (latestOnly) break;
  }
  return parts;
}

/** Record that *model* refused the images (*type* "image_url") or audio clips
 *  (*type* "input_audio") of *conv*, or with *latestOnly* those of the turn
 *  just sent: later requests to *model* leave them out once they belong to an
 *  earlier turn (wireParts). The conversation keeps every part. Returns
 *  the number of parts marked. */
export function markRefusedBy(conv, type, model, latestOnly = false) {
  const parts = attachedParts(conv, type, latestOnly);
  for (const p of parts) {
    const by = p.refusedBy || (p.refusedBy = []);
    const name = model || "";
    if (!by.includes(name)) by.push(name);
  }
  return parts.length;
}

/** Mark the images or audio clips (*type* as for markRefusedBy) of *conv*
 *  that the server could not use, so no later request sends them
 *  (wireParts): all of them, or with *latestOnly* those of the turn just sent.
 *  The conversation keeps every part. Returns the number marked. */
export function markNotSent(conv, type, latestOnly = false) {
  const parts = attachedParts(conv, type, latestOnly);
  for (const p of parts) p.notSent = true;
  return parts.length;
}

/** Replace, in place, every image part of the wire *messages* with a text
 *  note, so one request leaves the images out while the conversation keeps
 *  them. */
export function elideImagesForWire(messages) {
  for (const m of messages) {
    if (!Array.isArray(m.content)) continue;
    m.content = m.content.map((p) => p.type === "image_url"
      ? { type: "text", text: IMAGE_LEFT_OUT }
      : p);
  }
  return messages;
}

/** Replace, in place, every audio clip part of the wire *messages* with a text
 *  note, so one request leaves the clips out while the conversation keeps
 *  them. */
export function elideAudioForWire(messages) {
  for (const m of messages) {
    if (!Array.isArray(m.content)) continue;
    m.content = m.content.map((p) => p.type === "input_audio"
      ? { type: "text", text: AUDIO_LEFT_OUT }
      : p);
  }
  return messages;
}

/** True when the registry row of the model *name* records that it cannot take
 *  audio input (`audio_input: false`). An unknown model, or one whose files
 *  could not be inspected (no `audio_input` key), is not reported as deaf. */
export function modelCannotHearAudio(name) {
  const row = ((modelCache && modelCache.models) || []).find((m) => m.name === name);
  return !!row && row.audio_input === false;
}

/** True when the registry row of the model *name* says it reads images. */
export function modelReadsImages(name) {
  const row = ((modelCache && modelCache.models) || []).find((m) => m.name === name);
  return !!row && row.vision === true;
}

/** True when the wire *messages* carry both an image part and an audio clip. */
export function wireHasImageAndAudio(messages) {
  const has = (type) => messages.some((m) => Array.isArray(m.content) &&
    m.content.some((p) => p.type === type));
  return has("image_url") && has("input_audio");
}

/** True when the newest user message of *conv* carries an audio clip. */
export function latestTurnHasAudio(conv) {
  for (let i = conv.messages.length - 1; i >= 0; i--) {
    const m = conv.messages[i];
    if (m.role !== "user" || isToolEvent(m)) continue;
    return msgAudioClips(m).length > 0;
  }
  return false;
}

/** True when the newest user message of *conv* carries an audio clip and no
 *  image, so the images still in the conversation belong to earlier turns. */
export function latestTurnIsAudioOnly(conv) {
  for (let i = conv.messages.length - 1; i >= 0; i--) {
    const m = conv.messages[i];
    if (m.role !== "user" || isToolEvent(m)) continue;
    return msgAudioClips(m).length > 0 && msgImages(m).length === 0;
  }
  return false;
}

/* ---- conversation list: search, pin, folders ---- */

export const convUI = {
  search: "",
  remoteHits: new Map(),   // id -> snippet: server-side content matches for remoteTerm
  remoteTerm: "",
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

// The title field open for a rename; the list is not rebuilt while it is.
let _renameInput = null;

export function buildConvItem(conv, snippet) {
  const item = el("div", "conv-item" + (conv.id === chat.activeId ? " active" : ""));
  if (conv.tabOnly) {
    item.classList.add("conv-tab-only");
    item.title = t("chat.conv.tabOnly.title");
  }
  const title = el("span", "title");
  title.appendChild(document.createTextNode(
    conv.title === "New chat" ? t("chat.conv.newTitle") : conv.title));
  if (snippet) title.appendChild(el("span", "snippet", snippet));
  title.ondblclick = (e) => {
    e.stopPropagation();
    const input = document.createElement("input");
    input.value = conv.title;
    let cancelled = false;
    const commit = () => {
      _renameInput = null;
      const typed = input.value.trim();
      if (!cancelled && typed && typed !== conv.title) {
        conv.title = typed;
        saveConversations(conv);
      }
      renderConvList();
    };
    input.onblur = commit;
    // Enter keeps the typed name; Escape leaves the old one.
    input.onkeydown = (ke) => {
      if (ke.key === "Enter") input.blur();
      else if (ke.key === "Escape") { ke.preventDefault(); ke.stopPropagation(); cancelled = true; input.blur(); }
    };
    item.replaceChild(input, title);
    _renameInput = input;
    input.focus();
    input.select();
  };
  item.appendChild(title);

  const pin = el("button", "del" + (conv.pinned ? " pinned-btn" : ""));
  pin.appendChild(iconEl("pin", "ic"));
  pin.title = t(conv.pinned ? "chat.conv.unpin" : "chat.conv.pin");
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
  fold.title = conv.folder ? t("chat.conv.folderTitle", { folder: conv.folder })
                           : t("chat.conv.moveToFolder");
  fold.onclick = async (e) => {
    e.stopPropagation();
    const name = await promptText(t("chat.conv.folderPrompt"), conv.folder || "");
    if (name === null) return;
    if (name.trim()) conv.folder = name.trim();
    else delete conv.folder;
    saveConversations(conv);
    renderConvList();
  };
  item.appendChild(fold);

  const del = el("button", "del", "×");
  del.title = t("chat.conv.delete");
  const removeConversation = () => {
    // Marked deleted: pushConversation never writes it again, also not at the
    // end of a reply still streaming into it.
    conv.deleted = true;
    _goneIds.add(conv.id);
    clearTimeout(_convPushTimers.get(conv.id));
    _convPushTimers.delete(conv.id);
    chat.conversations = chat.conversations.filter((c) => c.id !== conv.id);
    if (chat.activeId === conv.id) chat.activeId = chat.conversations[0]?.id || null;
    chat.queue = chat.queue.filter((q) => q.convId !== conv.id);
    if (!conv.tabOnly) deleteConversationRemote(conv.id);
    forgetDrafts([conv.id]);
    saveConversations();
    renderConvList();
    renderChat();
    processChatQueue();
  };
  del.onclick = (e) => {
    e.stopPropagation();
    const shown = conv.title && conv.title !== "New chat" ? conv.title : t("chat.conv.newTitle");
    confirmDanger(
      t("chat.conv.deleteConfirm.title", { name: shown }),
      t("chat.conv.deleteConfirm.body"),
      t("chat.conv.deleteConfirm.confirm"), removeConversation);
  };
  item.appendChild(del);

  item.onclick = async () => {
    chat.activeId = conv.id;
    // Marks this row active without rebuilding the list.
    for (const row of document.querySelectorAll("#conv-list .conv-item.active")) {
      if (row !== item) row.classList.remove("active");
    }
    item.classList.add("active");
    if (conv._meta) await hydrateConversation(conv);   // R40: load the body on open
    renderChat();
    showView("chat");
    processChatQueue();
  };
  return item;
}

export function renderConvList() {
  if (_renameInput && _renameInput.isConnected) return;
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
      const snippet = searchSnippet(conv, term) ??
        (conv._meta && convUI.remoteTerm === term ? convUI.remoteHits.get(conv.id) ?? null : null);
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
    list.appendChild(el("div", "privacy-hint", t("chat.conv.noMatches")));
  } else if (!term && !chat.conversations.length) {
    list.appendChild(emptyState("chat", t("chat.conv.empty"),
      t("chat.conv.empty.hint")));
  }
}

let _remoteSearchTimer = 0;

/** Ask the server which chats mention the term in their messages: the browser
 *  holds only index rows for chats it has not opened. */
export async function searchRemoteConversations(term) {
  if (!chat.persist || !term) return;
  try {
    const r = await fetch("/api/conversations?meta=true&limit=200&offset=0&q=" +
                          encodeURIComponent(term), { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    if (convUI.search.trim().toLowerCase() !== term) return;   // the term moved on
    convUI.remoteHits = new Map((data.conversations || [])
      .filter((c) => c.snippet).map((c) => [c.id, c.snippet]));
    convUI.remoteTerm = term;
    renderConvList();
  } catch { /* offline: the local search still answers */ }
}

$("conv-search").addEventListener("input", (e) => {
  convUI.search = e.target.value;
  renderConvList();
  clearTimeout(_remoteSearchTimer);
  const term = convUI.search.trim().toLowerCase();
  if (term) _remoteSearchTimer = setTimeout(() => searchRemoteConversations(term), 250);
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
  } catch { return "localm-image.png"; }
}

// Copy an image (by its resolved src) to the clipboard. Returns true on success.
// Not every browser/context can write an image to the clipboard, so the caller
// surfaces a fallback instead of silently failing (RULE 5).
export function dataUrlToBlob(src) {
  const m = /^data:([^;,]*)(?:;[^;,]*)*?(;base64)?,(.*)$/s.exec(src);
  if (!m) return null;
  const raw = m[2] ? atob(m[3]) : decodeURIComponent(m[3]);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return new Blob([bytes], { type: m[1] || "application/octet-stream" });
}

export async function copyImageSrc(src) {
  try {
    if (!window.ClipboardItem || !navigator.clipboard || !navigator.clipboard.write)
      return false;
    const blob = src.startsWith("data:") ? dataUrlToBlob(src) : await (await fetch(src)).blob();
    if (!blob) return false;
    await navigator.clipboard.write([new window.ClipboardItem({ [blob.type]: blob })]);
    return true;
  } catch { return false; }
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
  const save = el("button", "btn-secondary", t("chat.image.save"));
  save.onclick = () => {
    const a = document.createElement("a");
    a.href = src;
    a.download = name || "localm-image.png";
    document.body.appendChild(a);
    a.click();
    a.remove();
  };
  const copy = el("button", "btn-secondary", t("chat.image.copy"));
  copy.onclick = async () => {
    const ok = await copyImageSrc(src);
    toast(t(ok ? "chat.image.copied" : "chat.image.copyFailedSave"), !ok);
  };
  const close = el("button", "btn-secondary", t("chat.image.close"));
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
  no_embedder: "chat.memoryChip.noEmbedder",
  no_vectors: "chat.memoryChip.noVectors",
  low_coverage: "chat.memoryChip.lowCoverage",
  dim_mismatch: "chat.memoryChip.dimMismatch",
  query_embed_failed: "chat.memoryChip.queryEmbedFailed",
};

/** A chip on an assistant turn that was answered by a different model than
 *  the one selected, because the selected one lacked something the request
 *  needed. *routed* is { from, gaps: [capability, ...], placement? }, or
 *  { fallback } when the selected model answered because a model that could
 *  have answered was skipped or failed to load: `fallback` is the server's
 *  explanation. `placement` ({ gpu, total, moe? }) marks an answering model
 *  that runs partly on the CPU; `moe` counts the layers whose routed experts
 *  stayed in system RAM. */
export function buildRoutedChip(routed) {
  const chip = routed.suggest ? el("button", "routed-chip routed-suggest")
    : el("span", "routed-chip");
  chip.appendChild(iconEl("models", "btn-ic"));
  if (routed.suggest) {
    const needs = (routed.gaps || []).map((g) => t("chat.routed.cap." + g)).join(", ") || "?";
    chip.appendChild(document.createTextNode(t("chat.routed.suggestChip", { model: routed.suggest })));
    chip.title = t("chat.routed.suggestTitle", {
      from: routed.from || "?", needs, model: routed.suggest });
    if (routed.askReason) {
      chip.title += " " + t("chat.routed.askReason", { reason: routed.askReason });
    }
    chip.onclick = () => {
      if ([...modelSelect.options].some((o) => o.value === routed.suggest)) {
        modelSelect.value = routed.suggest;
        modelSelect.dispatchEvent(new Event("change"));
      }
    };
    return chip;
  }
  if (routed.fallback) {
    chip.appendChild(document.createTextNode(t("chat.routed.fallbackChip")));
    chip.title = routed.fallback;
    return chip;
  }
  const gaps = (routed.gaps || []).map((g) => t("chat.routed.cap." + g));
  chip.appendChild(document.createTextNode(t("chat.routed.chip", { from: routed.from || "?" })));
  chip.title = t("chat.routed.title", {
    from: routed.from || "?", needs: gaps.join(", ") || "?" });
  if (routed.placement) {
    const counts = { gpu: routed.placement.gpu, total: routed.placement.total,
      moe: routed.placement.moe };
    chip.appendChild(document.createTextNode(", " + t("chat.routed.partial", counts)));
    chip.title += " " + t(routed.placement.moe ? "chat.routed.partialMoeTitle"
      : "chat.routed.partialTitle", counts);
  }
  return chip;
}

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
  const reason = MEMORY_DEGRADE_LABELS[mem.degrade] ? t(MEMORY_DEGRADE_LABELS[mem.degrade]) : "";
  chip.title = [
    tn("chat.memoryChip.used", n),
    facts.length ? facts.map((t) => "- " + t).join("\n") : "",
    reason ? "(" + reason + ")" : "",
  ].filter(Boolean).join("\n");
  chip.setAttribute("aria-label", t("chat.memoryChip.aria", { count: n }));
  chip.onclick = () => openMemoryModal();
  return chip;
}

/** The sentence for the outcome of a "remember ..." request the server handled
 *  for a message: `saved` is `{ text, status }`, status "saved" | "exists" |
 *  "full" | "failed". */
export function memorySavedText(saved) {
  const key = { saved: "chat.memory.saved", exists: "chat.memory.exists",
                full: "chat.memory.full", off: "chat.memory.off" }[saved && saved.status] || "chat.memory.saveFailed";
  return t(key, { text: (saved && saved.text) || "" });
}

/** The confirmation line under a reply whose message asked to remember
 *  something, so the user sees it was stored. */
export function buildMemorySavedNote(saved) {
  const note = el("span", "mem-saved", memorySavedText(saved));
  note.setAttribute("role", "status");
  return note;
}

// A stable per-label mark with no external asset and no network fetch: the
// same label always derives the same letters and the same hue. Two or more
// hyphen/underscore/space/dot-separated words give one initial each (e.g.
// "Qwen3-Coder-30B" -> "QC"); a single word gives its own first two
// characters (e.g. "phi3" -> "PH"). See monogramFor's tests.
function monogramFor(label) {
  const s = String(label || "");
  let hash = 0;
  for (let i = 0; i < s.length; i++) hash = (hash * 31 + s.charCodeAt(i)) | 0;
  const words = s.split(/[^A-Za-z0-9]+/).filter(Boolean);
  let letters = "";
  if (words.length >= 2) letters = words.slice(0, 2).map((w) => w[0]).join("");
  else if (words.length === 1) letters = words[0].slice(0, 2);
  return { letters: letters.toUpperCase() || "?", hue: Math.abs(hash) % 360 };
}

// Resolves what to show for one turn's avatar mark: null (nothing), or
// {kind: "image"|"glyph"|"mono", value, hue}. Never a URL - a configured
// value is already either a short glyph or a data: URI (see
// settings_schema.py's _validate_avatar_value).
function avatarInfoFor(role, mName) {
  if (role === "user") {
    const v = (chat.userAvatar || "").trim();
    if (!v) return null;
    const safeSrc = safeAvatarImageSrc(v);
    return safeSrc ? { kind: "image", value: safeSrc } : { kind: "glyph", value: v };
  }
  const override = (chat.modelAvatarOverrides || {})[mName];
  const v = (override || chat.modelAvatarDefault || "").trim();
  if (v) {
    const safeSrc = safeAvatarImageSrc(v);
    return safeSrc ? { kind: "image", value: safeSrc } : { kind: "glyph", value: v };
  }
  const mono = monogramFor(mName);
  return { kind: "mono", letters: mono.letters, hue: mono.hue };
}

function buildAvatarEl(info) {
  if (!info) return null;
  const av = el("div", "msg-avatar");
  if (info.kind === "image") {
    const img = document.createElement("img");
    img.src = info.value;
    img.alt = "";
    img.draggable = false;
    av.appendChild(img);
  } else if (info.kind === "glyph") {
    av.textContent = info.value;
  } else {
    av.textContent = info.letters;
    av.style.background = `hsl(${info.hue} 45% 30%)`;
  }
  return av;
}

// The one message-row speak button currently marked "playing", if any - only
// one utterance ever plays at a time, so at most one button is ever active.
let activeSpeakBtn = null;

function setSpeakButtonActive(btn, active) {
  btn.classList.toggle("speaking", active);
  btn.replaceChildren(iconEl(active ? "stop" : "speak", "ic"));
  btn.title = t(active ? "chat.action.stopSpeaking" : "chat.action.speak");
}

/** The "Speak this reply aloud" button's click handler: starts, or stops,
 *  speaking this message, and keeps the clicked button's own active/playing
 *  indicator in sync with what actually happens (including the utterance
 *  ending on its own, via speak()'s onEnd). See chat-speak-indicator.test.mjs. */
function speakToggle(btn, text) {
  const started = speak(text, {
    toggle: true,
    onEnd: () => {
      if (activeSpeakBtn === btn) {
        setSpeakButtonActive(btn, false);
        activeSpeakBtn = null;
      }
    },
  });
  if (activeSpeakBtn && activeSpeakBtn !== btn) setSpeakButtonActive(activeSpeakBtn, false);
  activeSpeakBtn = started ? btn : null;
  setSpeakButtonActive(btn, started);
}

/** The ‹ k/N › control of a fork point. *variant* is { k, n, prev, next }. */
function buildVariantNav(variant) {
  const nav = el("span", "variant");
  const prev = el("button", "action", "‹");
  prev.title = t("chat.variant.previous");
  prev.onclick = variant.prev;
  const next = el("button", "action", "›");
  next.title = t("chat.variant.next");
  next.onclick = variant.next;
  nav.appendChild(prev);
  nav.appendChild(el("span", "k", `${variant.k}/${variant.n}`));
  nav.appendChild(next);
  return nav;
}

export function addMessageRow(container, role, text, opts = {}) {
  // A note row (opts.cls set: injected web/kb/doc results, a queued-task
  // marker) is not something the human typed, so it must not carry the role
  // class and inherit the user turn's right-aligned bubble layout.
  const row = el("div", "msg-row " + (opts.cls || role));
  const mName = opts.model && opts.model !== "MODEL" ? opts.model
    : (modelCache.active || modelCache.resumable || t("chat.row.model"));
  const roleEl = el("div", "msg-role", opts.label || (role === "user" ? (chat.userName || t("chat.row.you")) : mName));
  const avatarEl = buildAvatarEl(avatarInfoFor(role, mName));
  if (avatarEl) {
    // Wrap only when there is an avatar. See
    // test_addMessageRow_user_turn_with_no_avatar_has_no_head_wrapper.
    const head = el("div", "msg-head");
    head.appendChild(avatarEl);
    head.appendChild(roleEl);
    row.appendChild(head);
  } else {
    row.appendChild(roleEl);
  }
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
    //
    // imageScope is the ACTIVE conversation, which is the only one the chat view
    // ever renders - a streaming shell appends to it and renderChat() rebuilds
    // it. It scopes the reader's "load images from this site" answers, so an
    // answer given in one conversation is not visible in another. Read here
    // rather than threaded from the caller so a future caller cannot forget it
    // and silently share one conversation's answers with the next.
    renderMarkdown(body, text,
                   { final: opts.final, imageScope: "chat:" + (chat.activeId || "") });
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
    img.title = t("chat.image.viewTitle");
    img.addEventListener("click", () => openImageLightbox(img.src, imageFilename(url)));
    body.appendChild(img);
  }
  for (const clip of opts.clips || []) {
    body.classList.add("has-player");
    const wrap = el("div", "msg-clip");
    if (clip.name) wrap.appendChild(el("div", "msg-clip-name", clip.name));
    const player = document.createElement("audio");
    player.controls = true;
    player.style.width = "100%";
    try {
      player.src = audioBlobUrl(clip);
      wrap.appendChild(player);
    } catch (err) {
      console.error("localm: could not prepare an audio clip for playback:", err);
      wrap.appendChild(el("div", "msg-clip-name", t("chat.audio.unplayable")));
    }
    body.appendChild(wrap);
  }
  for (const url of opts.audio || []) {
    body.classList.add("has-player");
    const player = document.createElement("audio");
    player.controls = true;
    player.style.width = "100%";
    // bearer-protected file - fetch as a blob with auth headers
    fetchImageURL(url).then((u) => (player.src = u)).catch(() => player.remove());
    body.appendChild(player);
  }
  for (const url of opts.video || []) {
    body.classList.add("has-player");
    const player = document.createElement("video");
    player.controls = true;
    player.style.width = "100%";
    player.style.borderRadius = "8px";
    fetchImageURL(url).then((u) => (player.src = u)).catch(() => player.remove());
    body.appendChild(player);
  }
  row.appendChild(body);
  const meta = el("div", "msg-meta");
  const copy = el("button", "copy-btn", t("chat.copy"));
  copy.onclick = async () => {
    // Reasoning blocks are stripped from a reply only; a user message is copied as typed.
    const plain = role === "user" ? text : stripAsrPrefix(stripThink(text) || text);
    const firstImg = body.querySelector(".msg-img");
    // Image-only message (e.g. a bare attachment): copy the IMAGE, not empty
    // text - the "copy copied the prompt, not the image" report (VIS-2). For a
    // text+image message the text copy is kept; the image is in the lightbox.
    if (!plain && firstImg && firstImg.src) {
      const ok = await copyImageSrc(firstImg.src);
      copy.textContent = t(ok ? "chat.copied" : "chat.copy");
      if (!ok) toast(t("chat.image.copyFailedOpen"), true);
      else setTimeout(() => (copy.textContent = t("chat.copy")), 1200);
      return;
    }
    try {
      await navigator.clipboard.writeText(plain);
      copy.textContent = t("chat.copied");
      setTimeout(() => (copy.textContent = t("chat.copy")), 1200);
    } catch {
      // Matches the image branch above: a real failure (permission denied,
      // insecure context) must never be reported as "copied" - that is
      // claiming a step happened that did not.
      toast(t("chat.copyBlocked"), true);
    }
  };
  meta.appendChild(copy);
  if (opts.variant) meta.appendChild(buildVariantNav(opts.variant));
  for (const [label, fn, icon] of opts.actions || []) {
    const btn = el("button", "action");
    if (icon) { btn.appendChild(iconEl(icon, "ic")); btn.title = label; }
    else btn.textContent = label;
    btn.onclick = () => fn(btn);
    meta.appendChild(btn);
  }
  if (opts.memory && opts.memory.saved) meta.appendChild(buildMemorySavedNote(opts.memory.saved));
  if (opts.memory && opts.memory.n > 0) meta.appendChild(buildMemoryChip(opts.memory));
  if (opts.routed) meta.appendChild(buildRoutedChip(opts.routed));
  row.appendChild(meta);
  container.appendChild(row);
  return { row, body, meta };
}

// Catalog keys for a tool event's card, looked up by field value.
const TOOL_LABEL_KEYS = {
  search: "chat.tool.search", fetch: "chat.tool.fetch", note: "chat.tool.note",
};
const TOOL_STATUS_KEYS = {
  running: "chat.tool.status.running", done: "chat.tool.status.done",
  failed: "chat.tool.status.failed", denied: "chat.tool.status.denied",
  duplicate: "chat.tool.status.duplicate", skipped: "chat.tool.status.skipped",
};
// Catalog keys of the one-line notice shown for a control note, by reason.
const TOOL_NOTE_NOTICE_KEYS = {
  format: "chat.tool.notice.format", unparsed: "chat.tool.notice.unparsed",
  pending: "chat.tool.notice.pending", limit: "chat.tool.notice.limit",
};
const TOOL_GROUNDING_KEYS = {
  "page-backed": "chat.tool.grounding.pageBacked",
  "snippet-only": "chat.tool.grounding.snippetOnly",
  failed: "chat.tool.grounding.failed",
};
const TOOL_READ_KEYS = {
  fetched: "chat.tool.read.fetched", failed: "chat.tool.read.failed",
  skipped: "chat.tool.read.skipped", duplicate: "chat.tool.read.duplicate",
};

/** The card's one-line title: the tool's name plus its query or URL. */
export function toolEventTitle(ev) {
  const what = t(TOOL_LABEL_KEYS[ev.tool] || TOOL_LABEL_KEYS.note);
  const subject = ev.tool === "search" ? (ev.query || "")
    : ev.tool === "fetch" ? ((ev.page && ev.page.url) || ev.url || "") : "";
  return subject ? `${what}: ${subject}` : what;
}

/** A link for a source URL when it is a web URL, else the text alone. */
function toolSourceLink(url, label) {
  if (!/^https?:\/\//i.test(url || "")) return el("span", "tool-link", label);
  const a = el("a", "tool-link", label);
  a.href = url;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  return a;
}

/** Append a section heading and a block of verbatim text to *body*. */
function toolTextSection(body, heading, text) {
  if (heading) body.appendChild(el("div", "tool-section", heading));
  body.appendChild(el("div", "tool-evidence", text));
}

/** The user-facing one-line notice for a tool event, in the UI language:
 *  what happened, never the text the model reads. "" when there is nothing to
 *  say (a result's own note, such as /web's answering instruction, has none). */
export function toolEventNotice(ev) {
  if (ev.tool === "note") {
    return t(TOOL_NOTE_NOTICE_KEYS[ev.reason] || "chat.tool.notice.note");
  }
  if (ev.status === "duplicate") return t("chat.tool.notice.duplicate");
  if (ev.status === "denied") return t("chat.tool.notice.denied");
  if (ev.status === "skipped") return t("chat.tool.notice.skipped", { limit: ev.limit ?? "" });
  return ev.reason === "ignored" ? t("chat.tool.notice.ignored") : "";
}

/** Render a tool event: a control note as a one-line notice, anything else as
 *  a collapsed activity-and-sources card whose summary line (tool, query or
 *  URL, status, grounding, source count, elapsed time) expands to the source
 *  list and the evidence, page text, error or notice it carries. The text the
 *  model reads for a note is never shown. Every remote string is set as text,
 *  never as markup. */
export function addToolEventRow(container, ev, opts = {}) {
  const row = el("div", "msg-row tool-event");
  row.appendChild(el("div", "msg-role", noteLabel(ev)));
  if (ev.tool === "note") {
    row.classList.add("tool-notice-row");
    row.appendChild(el("div", "tool-notice", toolEventNotice(ev)));
    appendToolEventMeta(row, ev, opts);
    container.appendChild(row);
    return { row, card: null, body: null };
  }
  const card = el("details", "tool-card");
  card.dataset.status = ev.status || "done";
  const summary = el("summary", "tool-summary");
  summary.appendChild(iconEl(ev.tool === "fetch" ? "file" : "web", "ic"));
  summary.appendChild(el("span", "tool-title", toolEventTitle(ev)));
  summary.appendChild(el("span", "tool-status",
    t(TOOL_STATUS_KEYS[ev.status] || TOOL_STATUS_KEYS.done)));
  if (ev.status === "done" && TOOL_GROUNDING_KEYS[ev.grounding]) {
    summary.appendChild(el("span", "tool-grounding", t(TOOL_GROUNDING_KEYS[ev.grounding])));
  }
  const sources = Array.isArray(ev.sources) ? ev.sources : [];
  if (sources.length) {
    summary.appendChild(el("span", "tool-count", tn("chat.tool.sources", sources.length)));
  }
  if (ev.started_at && ev.finished_at && ev.finished_at >= ev.started_at) {
    const seconds = ((ev.finished_at - ev.started_at) / 1000).toFixed(1);
    summary.appendChild(el("span", "tool-elapsed", t("chat.tool.elapsed", { seconds })));
  }
  card.appendChild(summary);

  const body = el("div", "tool-body");
  if (sources.length) {
    body.appendChild(el("div", "tool-section", t("chat.tool.sourcesTitle")));
    const list = el("ol", "tool-sources");
    for (const s of sources) {
      const li = el("li");
      li.appendChild(el("span", "tool-sid", s.id || ""));
      const url = s.final_url || s.url || "";
      const title = (s.title || "").trim() || url;
      li.appendChild(toolSourceLink(url, title));
      const meta = [];
      // A source whose read failed but whose search snippet is in the evidence
      // says so instead of "no evidence".
      const snippetUsed = s.grounding === "failed" && (ev.chunks || []).some(
        (c) => c && c.source_id === s.id && c.kind === "snippet");
      if (snippetUsed) {
        meta.push(t("chat.tool.read.failedSnippetUsed"));
      } else {
        if (TOOL_GROUNDING_KEYS[s.grounding]) meta.push(t(TOOL_GROUNDING_KEYS[s.grounding]));
        if (TOOL_READ_KEYS[s.retrieval_status]) meta.push(t(TOOL_READ_KEYS[s.retrieval_status]));
      }
      if (s.error) meta.push(s.error);
      if (meta.length) li.appendChild(el("span", "tool-source-meta", meta.join(" · ")));
      list.appendChild(li);
    }
    body.appendChild(list);
  }
  const chunks = Array.isArray(ev.chunks) ? ev.chunks : [];
  if (chunks.length) {
    body.appendChild(el("div", "tool-section", t("chat.tool.evidence")));
    for (const c of chunks) {
      const p = el("div", "tool-evidence");
      p.appendChild(el("span", "tool-sid",
        (c.source_id || "") + (c.kind === "snippet" ? " snippet" : "")));
      p.appendChild(document.createTextNode(" " + (c.text || "")));
      body.appendChild(p);
    }
  }
  if (ev.page && typeof ev.page.text === "string") {
    toolTextSection(body,
      t("chat.tool.page") + (ev.page.truncated ? " " + t("chat.tool.truncated") : ""),
      ev.page.text);
  }
  if (ev.error && typeof ev.text !== "string") toolTextSection(body, t("chat.tool.error"), ev.error);
  if (typeof ev.text === "string") toolTextSection(body, "", ev.text);
  const notice = toolEventNotice(ev);
  if (notice) body.appendChild(el("div", "tool-notice", notice));
  card.appendChild(body);
  row.appendChild(card);
  appendToolEventMeta(row, ev, opts);
  container.appendChild(row);
  return { row, card, body };
}

/** Append a tool event row's meta line: the copy button (a control note
 *  copies its notice, anything else its text) and, when *opts.variant* is set,
 *  the branch navigation. */
function appendToolEventMeta(row, ev, opts) {
  const meta = el("div", "msg-meta");
  const copy = el("button", "copy-btn", t("chat.copy"));
  copy.onclick = async () => {
    try {
      await navigator.clipboard.writeText(ev.tool === "note" ? toolEventNotice(ev) : msgText(ev));
      copy.textContent = t("chat.copied");
      setTimeout(() => (copy.textContent = t("chat.copy")), 1200);
    } catch {
      toast(t("chat.copyBlocked"), true);
    }
  };
  meta.appendChild(copy);
  if (opts.variant) meta.appendChild(buildVariantNav(opts.variant));
  row.appendChild(meta);
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
    t("chat.empty.blurb", { model: modelCache.active || t("chat.empty.yourModel") })));
  const tip = el("div", "", t("chat.empty.tip"));
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
// The MTP part of the usage line for a reply's usage.mtp, as {text, title}:
// the acceptance rate when the reply speculated, that MTP paused itself for
// being slower, that it was off for this reply (a turn with an image), or that
// it stopped or is unavailable, with the reason as the
// tooltip. Empty for no MTP figures or an idle turn.
export function mtpUsageText(mtp) {
  if (!mtp || typeof mtp !== "object") return { text: "", title: "" };
  if (mtp.state === "on" && mtp.drafted > 0) {
    const pct = Math.round((100 * mtp.accepted) / mtp.drafted);
    return { text: t("chat.usage.mtpOn", { pct }),
             title: t("chat.usage.mtpOn.title", { accepted: mtp.accepted, drafted: mtp.drafted }) };
  }
  if (mtp.state === "paused") {
    return { text: t("chat.usage.mtpPaused"),
             title: t("chat.usage.mtpPaused.title", { steps: mtp.paused_steps || 0 }) };
  }
  if (mtp.state === "off") {
    const why = { image: "chat.usage.mtpOff.image", audio: "chat.usage.mtpOff.audio" }[mtp.reason];
    return { text: t("chat.usage.mtpOff"),
             title: why ? t(why) : t("chat.usage.mtpReason", { reason: mtp.reason || "" }) };
  }
  if (mtp.state === "stopped") {
    return { text: t("chat.usage.mtpStopped"),
             title: t("chat.usage.mtpReason", { reason: mtp.reason || "" }) };
  }
  if (mtp.state === "unavailable") {
    return { text: t("chat.usage.mtpUnavailable"),
             title: t("chat.usage.mtpReason", { reason: mtp.reason || "" }) };
  }
  return { text: "", title: "" };
}

// The n-gram or draft-model drafting part of the usage line for a reply's
// usage.speculation, as {text, title}, in the same states as mtpUsageText. Empty
// when the source is mtp (usage.mtp already covers it), for no figures, or for
// an idle turn.
const SPECULATION_USAGE_KEYS = {
  ngram: {
    on: "chat.usage.ngramOn", onTitle: "chat.usage.ngramOn.title",
    paused: "chat.usage.ngramPaused", pausedTitle: "chat.usage.ngramPaused.title",
    off: "chat.usage.ngramOff", offImage: "chat.usage.ngramOff.image",
    stopped: "chat.usage.ngramStopped", unavailable: "chat.usage.ngramUnavailable",
    reason: "chat.usage.ngramReason",
  },
  draft: {
    on: "chat.usage.draftOn", onTitle: "chat.usage.draftOn.title",
    onCpuTitle: "chat.usage.draftOnCpu.title",
    paused: "chat.usage.draftPaused", pausedTitle: "chat.usage.draftPaused.title",
    off: "chat.usage.draftOff", offImage: "chat.usage.draftOff.image",
    stopped: "chat.usage.draftStopped", unavailable: "chat.usage.draftUnavailable",
    reason: "chat.usage.draftReason",
  },
};

export function speculationUsageText(spec) {
  const keys = spec && typeof spec === "object"
    && Object.prototype.hasOwnProperty.call(SPECULATION_USAGE_KEYS, spec.source)
    ? SPECULATION_USAGE_KEYS[spec.source] : null;
  if (!keys) {
    return { text: "", title: "" };
  }
  if (spec.state === "on" && spec.drafted > 0) {
    const pct = Math.round((100 * spec.accepted) / spec.drafted);
    const onTitle = spec.reason === "draft-on-cpu" && keys.onCpuTitle ? keys.onCpuTitle : keys.onTitle;
    return { text: t(keys.on, { pct }),
             title: t(onTitle, { accepted: spec.accepted, drafted: spec.drafted }) };
  }
  if (spec.state === "paused") {
    return { text: t(keys.paused),
             title: t(keys.pausedTitle, { steps: spec.paused_steps || 0 }) };
  }
  if (spec.state === "off") {
    return { text: t(keys.off),
             title: spec.reason === "image" ? t(keys.offImage) : t(keys.reason, { reason: spec.reason || "" }) };
  }
  if (spec.state === "stopped") {
    return { text: t(keys.stopped), title: t(keys.reason, { reason: spec.reason || "" }) };
  }
  if (spec.state === "unavailable") {
    return { text: t(keys.unavailable), title: t(keys.reason, { reason: spec.reason || "" }) };
  }
  return { text: "", title: "" };
}

export function updateUsageDisplay(usage) {
  const gaugeContainer = $("context-gauge-container");
  const gaugeBar = $("context-gauge-bar");
  if (!usage) {
    $("chat-usage").textContent = "";
    $("chat-usage").title = "";
    if (gaugeContainer) gaugeContainer.classList.remove("visible");
    return;
  }
  const bits = [`${usage.total_tokens} tok`];
  if (usage.ttft_ms != null) bits.push(t("chat.usage.firstToken", { ms: Math.round(usage.ttft_ms) }));
  if (usage.tokens_per_sec != null) {
    const rate = Number(usage.tokens_per_sec);
    bits.push(`${rate >= 10 ? Math.round(rate) : Math.round(rate * 10) / 10} tok/s`);
  }
  const mtp = mtpUsageText(usage.mtp);
  if (mtp.text) bits.push(mtp.text);
  const spec = speculationUsageText(usage.speculation);
  if (spec.text) bits.push(spec.text);
  $("chat-usage").textContent = bits.join(" · ");
  $("chat-usage").title = mtp.title || spec.title;
  if (gaugeContainer && gaugeBar && usage.context_capacity) {
    const pct = Math.min(100, Math.max(0, (usage.total_tokens / usage.context_capacity) * 100));
    gaugeBar.style.width = pct + "%";
    gaugeBar.className = "context-gauge-bar" + (pct > 90 ? " danger" : (pct > 75 ? " warning" : ""));
    gaugeContainer.classList.add("visible");
  } else if (gaugeContainer) {
    gaugeContainer.classList.remove("visible");
  }
}

const liveRows = new Set();

/** Keeps *row*, an unsaved status bubble of *conv* (a running media
 *  generation), on screen across re-renders of the chat. Returns the function
 *  that stops keeping it. */
export function trackLiveRow(conv, row) {
  const entry = { conv, row };
  liveRows.add(entry);
  return () => { liveRows.delete(entry); };
}

export function renderChat() {
  const box = $("chat-messages");
  box.innerHTML = "";
  const conv = currentConv();
  pruneClipUrls(conv);
  syncPinModelToggle(conv);
  syncChatParams(conv);
  syncComposer(conv);
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
  const pending = conv && !chatBusy() ? pendingBranch(conv) : null;
  if (!conv || (conv.messages.length === 0 && !pending)) {
    box.appendChild(buildEmptyHint());
    updateUsageDisplay(null);
    if (typeof renderQueuedIndicator === "function") renderQueuedIndicator();
    return;
  }
  // NEW-1 model-switch-indication: track the model of the previous assistant turn
  // so a small divider marks where the active model changed mid-conversation.
  let lastAssistantModel = null;
  conv.messages.forEach((m, i) => {
    if (m.role === "assistant" && m.model) {
      const currentModel = m.model === "MODEL" ? (modelCache.active || m.model) : modelNameNow(m.model);
      if (lastAssistantModel && currentModel !== lastAssistantModel) {
        box.appendChild(el("div", "model-switch", t("chat.switchedTo", { model: currentModel })));
      }
      lastAssistantModel = currentModel;
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
    if (isToolEvent(m)) {
      addToolEventRow(box, m, { variant });
      return;
    }
    const tag = m.tag;
    const actions = [];
    if (m.role === "user" && !tag && !chatBusy()) {
      actions.push([t("chat.action.edit"), () => editMessage(conv, i)]);
      actions.push([t("chat.action.revert"), () => revertTo(conv, i)]);
    }
    if (m.role === "assistant" && !tag) {
      actions.push([t("chat.action.speak"), (btn) => speakToggle(btn, msgText(m)), "speak"]);
    }
    if (m.role === "assistant" && i === conv.messages.length - 1 && !chatBusy()) {
      actions.push([t("chat.action.regenerate"), () => regenerate(conv)]);
    }
    const noteSuffix = m.failed
      ? "\n\n*[" + t("chat.reply.failedNote") + "]*"
      : m.truncated
      ? "\n\n*[" + t("chat.reply.maxTokensNote") + "]*"
      : m.stopped
      ? "\n\n*[" + t("chat.reply.stoppedNote") + "]*"
      : m.cut
      ? "\n\n*[" + t("chat.reply.cut") + "]*"
      : m.webUnfinished
      ? "\n\n*[" + t("chat.web.unfinished") + "]*"
      : "";
    const evidenceSuffix = m.noWebEvidence
      ? "\n\n*[" + t("chat.web.noEvidence") + "]*"
      : "";
    addMessageRow(box, m.role, msgText(m) + noteSuffix + evidenceSuffix, {
      images: msgImages(m),
      clips: msgAudioClips(m),
      audio: m.audio ? [m.audio] : [],
      video: m.video ? [m.video] : [],
      actions,
      variant,
      model: modelNameNow(m.model),
      memory: m.memory,           // F11: "used N memories" chip (assistant turns)
      routed: m.routed,
      cls: tag ? "web-note" : "",
      label: noteLabel(m) || undefined,
      // A settled turn from history: let renderMarkdown surface a "(no reply
      // text)" note if this message rendered to a blank body (empty ```fence).
      final: true,
    });
  });
  if (pending) box.appendChild(buildPendingBranchRow(pending));
  for (const entry of liveRows) if (entry.conv === conv) box.appendChild(entry.row);
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
  if (typeof renderQueuedIndicator === "function") renderQueuedIndicator();
}

/** The fork at the end of the live timeline of *conv* (an edit that parked the
 *  old messages and is not sent yet) as { k, n, prev, next }, or null. */
export function pendingBranch(conv) {
  const n = conv.messages.length;
  const pid = n ? conv.messages[n - 1].id : "root";
  const rec = pid && (conv.branches || []).find((b) => b.parent === pid);
  if (!rec || rec.tails.length < 2) return null;
  return { k: rec.current + 1, n: rec.tails.length,
           prev: () => switchBranch(conv, n, -1), next: () => switchBranch(conv, n, +1) };
}

/** The row shown where an edited message will go until it is sent: it says the
 *  earlier version is kept and carries the arrows back to it. */
function buildPendingBranchRow(variant) {
  const row = el("div", "msg-row tool-event tool-notice-row branch-pending");
  row.appendChild(el("div", "tool-notice", t("chat.branch.pending")));
  const meta = el("div", "msg-meta");
  meta.appendChild(buildVariantNav(variant));
  row.appendChild(meta);
  return row;
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
  if (chat.abort) { toast(t("chat.waitForReply"), true); return; }
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

/** True while a reply is streaming or a web call is in flight: the
 *  conversation must not be edited, forked or sent to until it settles. */
export function chatBusy() {
  return !!(chat.abort || chat.webCall || chat.busy);
}

export function editMessage(conv, index) {
  // Editing forks the branch tree (forkAt); doing that mid-stream parks the
  // messages before the streaming reply has landed, corrupting the branch
  // state. Bail while a reply streams, like switchBranch / regenerate do.
  if (chatBusy()) { toast(t("chat.waitForReply"), true); return; }
  const m = conv.messages[index];
  const shown = chat.activeId;
  confirmReplaceComposer(m, () => {
    if (chatBusy() || conv.messages[index] !== m || chat.activeId !== shown) return;
    fillComposerFrom(m);
    // Fork instead of destroy: the old timeline (this message and everything
    // after it) stays reachable via ‹ ›, also before the edit is sent.
    forkAt(conv, index);
    saveConversations(conv);
    renderChat();
    $("chat-input").focus();
  });
}

/** Puts the text, pictures and audio clips of message *m* into the message box
 *  and the attachment lists, in place of what they held. */
function fillComposerFrom(m) {
  showDraft({
    text: msgText(m),
    attachments: msgImages(m).filter((url) => url.startsWith("data:"))
      .map((dataUri) => ({ name: imageFilename(dataUri), dataUri })),
    docs: [],
    clips: msgAudioClips(m).map((c) => ({ name: c.name, data: c.data, format: c.format,
      size: Math.floor(c.data.length * 3 / 4), seconds: c.seconds })),
    held: _composerHeld,
  });
  saveDraftsSoon();
}

/** Runs *apply* when the message box holds nothing *m* would not put back, else
 *  asks first whether to replace what it holds. */
function confirmReplaceComposer(m, apply) {
  const d = liveDraft();
  if ((!d.text.trim() || d.text === msgText(m)) &&
      !d.attachments.length && !d.docs.length && !d.clips.length) {
    apply();
    return;
  }
  confirmDanger(t("chat.edit.replaceDraft.title"), t("chat.edit.replaceDraft.body"),
    t("chat.edit.replaceDraft.confirm"), apply);
}

export function regenerate(conv) {
  if (chatBusy()) { toast(t("chat.waitForReply"), true); return; }
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
  if (chatBusy()) { toast(t("chat.waitForReply"), true); return; }
  if (index < 0 || index >= conv.messages.length) return;
  const m = conv.messages[index];
  const shown = chat.activeId;
  const lost = branchesLostByRevert(conv, index);

  const apply = () => {
    if (chatBusy() || conv.messages[index] !== m || chat.activeId !== shown) return;
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
    fillComposerFrom(m);
    saveConversations(conv);
    renderChat();
    $("chat-input").focus();
  };

  confirmReplaceComposer(m, () => {
    if (lost > 0) {
      confirmDanger(
        t("chat.revert.title"),
        tn("chat.revert.body", lost),
        t("chat.revert.confirm"), apply);
    } else {
      apply();
    }
  });
}

export function chatParams() {
  // A value above the field's max is lowered to the max, and the field shows
  // the value that is sent.
  const num = (id) => {
    const box = $(id);
    const v = box.value.trim();
    if (v === "") return null;
    const n = Number(v);
    const hi = box.max === "" ? NaN : Number(box.max);
    if (Number.isFinite(n) && Number.isFinite(hi) && n > hi) {
      box.value = String(hi);
      return hi;
    }
    return n;
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
  saveDraftsSoon();
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
  chat.clips.forEach((clip, i) => {
    const chip = el("span", "chip");
    chip.appendChild(iconEl("music", "ic"));
    chip.appendChild(el("span", "", clip.name +
      (clip.seconds ? ` (${formatClipTime(clip.seconds)})` : "")));
    const rm = el("button", "", "×");
    rm.onclick = () => { chat.clips.splice(i, 1); renderAttachChips(); };
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

/** The longest and largest audio clip the server accepts
 *  (AUDIO_MAX_SECONDS / AUDIO_MAX_BYTES in localm/inference/media.py). */
export const AUDIO_MAX_SECONDS = 600;
export const AUDIO_MAX_BYTES = 50_000_000;
const AUDIO_EXTS = ["wav", "mp3", "flac", "ogg", "oga", "opus", "m4a", "aac"];

const fileExt = (name) => {
  const dot = String(name || "").lastIndexOf(".");
  return dot < 0 ? "" : name.slice(dot + 1).toLowerCase();
};

export function isAudioFile(file) {
  return (file.type || "").startsWith("audio/") || AUDIO_EXTS.includes(fileExt(file.name));
}

/** The `format` of the input_audio part for *file*: its extension when it is
 *  a known audio one, else the MIME subtype, else "mp3". */
export function audioFormat(file) {
  const ext = fileExt(file.name);
  if (AUDIO_EXTS.includes(ext)) return ext;
  const sub = (file.type || "").split(";")[0].split("/")[1] || "";
  if (sub === "mpeg") return "mp3";
  if (/^(x-)?(wav|wave|vnd\.wave)$/.test(sub)) return "wav";
  return sub.replace(/^x-/, "") || "mp3";
}

/** The MIME type a browser audio element plays an input_audio *format* as. */
export function audioMime(format) {
  const f = String(format || "").toLowerCase();
  if (f === "mp3") return "audio/mpeg";
  if (f === "m4a") return "audio/mp4";
  if (f === "ogg" || f === "oga" || f === "opus") return "audio/ogg";
  return "audio/" + (f || "wav");
}

const _clipUrls = new Map();

/** A blob: URL playing *clip*, created once per distinct clip and reused by
 *  every render. The page CSP allows media only from 'self' and blob:, so a
 *  data: URL is refused by the browser. */
export function audioBlobUrl(clip) {
  let url = _clipUrls.get(clip.data);
  if (url) return url;
  const bin = atob(clip.data);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  url = URL.createObjectURL(new Blob([bytes], { type: audioMime(clip.format) }));
  _clipUrls.set(clip.data, url);
  return url;
}

/** Revoke the blob: URL of every clip that is not in *conv*, so the URLs alive
 *  are exactly those of the conversation on screen. */
export function pruneClipUrls(conv) {
  const keep = new Set();
  for (const m of (conv && conv.messages) || []) {
    for (const c of msgAudioClips(m)) keep.add(c.data);
  }
  for (const [data, url] of _clipUrls) {
    if (keep.has(data)) continue;
    URL.revokeObjectURL(url);
    _clipUrls.delete(data);
  }
}

/** *seconds* as m:ss. */
export function formatClipTime(seconds) {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** Whether *blob* starts with a RIFF/WAVE header. */
export function hasWavHeader(blob) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onerror = () => resolve(false);
    reader.onload = () => {
      const b = new Uint8Array(reader.result);
      const at = (o, s) => [...s].every((ch, i) => b[o + i] === ch.charCodeAt(0));
      resolve(b.length >= 12 && at(0, "RIFF") && at(8, "WAVE"));
    };
    reader.readAsArrayBuffer(blob.slice(0, 12));
  });
}

/** The duration in seconds of a RIFF/WAVE file, read from its header (the first
 *  MiB is enough to reach the fmt and data chunks), or 0 when *blob* is not a
 *  WAV file it can read. */
export function wavSeconds(blob) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onerror = () => resolve(0);
    reader.onload = () => {
      const buf = reader.result;
      if (!(buf instanceof ArrayBuffer) || buf.byteLength < 12) { resolve(0); return; }
      const v = new DataView(buf);
      const tag = (o) => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1),
                                             v.getUint8(o + 2), v.getUint8(o + 3));
      if (tag(0) !== "RIFF" || tag(8) !== "WAVE") { resolve(0); return; }
      let byteRate = 0;
      let off = 12;
      while (off + 8 <= buf.byteLength) {
        const id = tag(off);
        const size = v.getUint32(off + 4, true);
        if (id === "fmt " && off + 20 <= buf.byteLength) byteRate = v.getUint32(off + 16, true);
        if (id === "data") {
          const room = blob.size - (off + 8);
          const bytes = size === 0 || size === 0xFFFFFFFF || size > room ? room : size;
          resolve(byteRate > 0 && bytes > 0 ? bytes / byteRate : 0);
          return;
        }
        off += 8 + size + (size % 2);
      }
      resolve(0);
    };
    reader.readAsArrayBuffer(blob.slice(0, 1 << 20));
  });
}

/** Audio attachment: kept in memory as base64 and sent as an input_audio part.
 *  A file over the server's size limit is refused before it is read. A WAV's
 *  length is read from its header after the chip appears; a WAV longer than the
 *  server's limit is removed again. Other formats show no length and the server
 *  judges them. */
export async function attachAudio(file) {
  const owner = _composerOwner;
  if (file.size > AUDIO_MAX_BYTES) {
    throw new Error(t("chat.audio.tooLarge", {
      name: file.name, size: (file.size / 1e6).toFixed(1),
      max: String(AUDIO_MAX_BYTES / 1e6) }));
  }
  const format = audioFormat(file);
  if (format === "wav" && !(await hasWavHeader(file))) {
    throw new Error(t("chat.audio.notWav", { name: file.name }));
  }
  const uri = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error(t("chat.audio.readError", { name: file.name })));
    reader.readAsDataURL(file);
  });
  const data = uri.split(",", 2)[1] || "";
  if (!data) throw new Error(t("chat.audio.readError", { name: file.name }));
  const clip = { name: file.name, data, format, size: file.size, seconds: 0 };
  draftOf(owner).clips.push(clip);
  renderAttachChips();
  const seconds = await wavSeconds(file);
  const { clips } = draftOf(owner);
  if (clips.indexOf(clip) === -1) return;
  if (seconds > AUDIO_MAX_SECONDS) {
    clips.splice(clips.indexOf(clip), 1);
    renderAttachChips();
    throw new Error(t("chat.audio.tooLong", {
      name: file.name, minutes: String(Math.ceil(seconds / 60)),
      max: String(AUDIO_MAX_SECONDS / 60) }));
  }
  if (seconds > 0) { clip.seconds = seconds; renderAttachChips(); }
}

/** Document attachment → text via /api/chat/extract. The file is converted
 *  in memory on the server and never written to disk (privacy-clean). */
export async function attachDocument(file) {
  const owner = _composerOwner;
  const b64 = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(new Error("could not read file"));
    reader.readAsDataURL(file);
  });
  const r = await fetch("/api/chat/extract", {
    method: "POST", headers: authHeaders(),
    body: JSON.stringify({ filename: file.name, content_b64: b64 }),
  });
  // A plain-text 500 body is not JSON; fall back to {} so the error message
  // below is the clean statusText, not a JSON parse error.
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  draftOf(owner).docs.push({ name: data.filename, text: data.text,
                             chars: data.chars, truncated: data.truncated });
  renderAttachChips();
}

/** Ingest files as chat attachments: images inline (data URI), audio as
 *  input_audio clips, every other type extracted to text server-side. Shared by the file picker and the
 *  drag-and-drop zone so both behave identically. */
export function addAttachedFiles(files) {
  const owner = _composerOwner;
  for (const file of files) {
    if (file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = () => {
        draftOf(owner).attachments.push({ name: file.name, dataUri: reader.result });
        renderAttachChips();
      };
      reader.readAsDataURL(file);
    } else if (isAudioFile(file)) {
      attachAudio(file).catch((err) => toast(err.message, true));
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
    const data = await r.json();
    items = data.items || [];
    const skipped = data.skipped || [];
    if (skipped.length) {
      toast(tn("chat.share.skipped", skipped.length, { names: skipped.join(", ") }));
    }
  } catch { return; }
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
      } catch { /* not decodable text */ }
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
    catch { /* no/!JSON body: nothing more to report than the 200 above */ }
    if (failed > 0) {
      console.error("share-inbox clear: " + failed + " item(s) could not be " +
                    "deleted server-side - retried on the next share ingest");
      toast(tn("chat.share.clearFailed", failed), true);
    }
  }).catch((e) => {
    console.error("share-inbox clear failed: " + (e && e.message ? e.message : e));
  });
  if (imgs) toast(tn("chat.share.imagesIn", imgs));
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

// The conversation list and the message area hold text built here rather than
// marked up in index.html, so they are rebuilt when the interface language
// changes. Skipped while a reply is streaming: renderChat() replaces the
// element the stream is writing into.
document.addEventListener("localm:language", () => {
  if (chat.abort) return;
  renderConvList();
  renderChat();
});

/** Format elapsed seconds into "m:ss" (e.g. 0:01, 1:23). */
export function formatStatusElapsed(sec) {
  const s = Math.max(0, Math.floor(sec || 0));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}:${rem < 10 ? "0" : ""}${rem}`;
}

/** Create a status pill element with a pulsing dot, status text and timer. */
export function createStatusIndicator(text, isWarn = false) {
  const el = document.createElement("div");
  el.className = "msg-status-indicator" + (isWarn ? " st-warn" : "");
  const pulse = document.createElement("span");
  pulse.className = "status-pulse";
  const label = document.createElement("span");
  label.className = "status-text";
  label.textContent = text || "";
  const timer = document.createElement("span");
  timer.className = "status-timer";
  timer.textContent = "0:00";
  el.appendChild(pulse);
  el.appendChild(label);
  el.appendChild(timer);
  return el;
}

/** Mount a status indicator into a message body and start its 1s elapsed timer. */
export function mountStatusIndicator(bodyEl, text, isWarn = false) {
  if (!bodyEl) return null;
  removeStatusIndicator(bodyEl);
  const ind = createStatusIndicator(text, isWarn);
  bodyEl.appendChild(ind);
  const t0 = Date.now();
  const timerEl = ind.querySelector(".status-timer");
  const intervalId = setInterval(() => {
    if (!ind.isConnected) {
      clearInterval(intervalId);
      return;
    }
    if (timerEl) {
      const elapsedSec = (Date.now() - t0) / 1000;
      timerEl.textContent = formatStatusElapsed(elapsedSec);
    }
  }, 1000);
  bodyEl._statusTimer = intervalId;
  return ind;
}

// Status codes (delta.status_code) that call for the warning style, independent
// of which language the accompanying text is in.
const WARN_STATUS_CODES = new Set(["vision_cpu_retry", "audio_cpu_retry", "loading_beside"]);

/** The chat.status.* catalog key for a snake_case server status code. */
function statusIndicatorI18nKey(code) {
  return "chat.status." + code.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
}

/** The label for a server status: the localized text for its status `code`
 *  filled from `args` (the server's delta.status_args) when the catalog has
 *  one, otherwise `text` as given. A localized text with a placeholder `args`
 *  does not fill is not used. */
export function statusLabel(text, code = null, args = null) {
  if (!code) return text;
  const label = tOr(statusIndicatorI18nKey(code), text, args || undefined);
  return /\{\w+\}/.test(label) ? text : label;
}

/** Whether a server status takes the warning style: by its `code` when it has
 *  one, otherwise by the English wording of `text`. */
export function isWarnStatus(text, code = null) {
  return code
    ? WARN_STATUS_CODES.has(code)
    : (typeof text === "string" && (
        text.toLowerCase().includes("failed") || text.toLowerCase().includes("retrying")
      ));
}

/** Update the text and warning state of an existing status indicator in bodyEl,
 *  or mount a new one if not present. `code` is the server's delta.status_code:
 *  when it resolves to a catalog entry, that localized text and the code's
 *  warning style are used; otherwise `text` is shown as given and the warning
 *  style is matched from its English wording. */
export function updateStatusIndicator(bodyEl, text, code = null, args = null) {
  if (!bodyEl) return null;
  const label = statusLabel(text, code, args);
  const shouldWarn = isWarnStatus(text, code);
  const ind = bodyEl.querySelector(".msg-status-indicator");
  if (!ind) {
    return mountStatusIndicator(bodyEl, label, shouldWarn);
  }
  const labelEl = ind.querySelector(".status-text");
  if (labelEl && label) labelEl.textContent = label;
  if (shouldWarn) ind.classList.add("st-warn");
  else ind.classList.remove("st-warn");
  return ind;
}

/** Stop the timer and remove any status indicator from bodyEl. */
export function removeStatusIndicator(bodyEl) {
  if (!bodyEl) return;
  if (bodyEl._statusTimer) {
    clearInterval(bodyEl._statusTimer);
    delete bodyEl._statusTimer;
  }
  const ind = bodyEl.querySelector(".msg-status-indicator");
  if (ind) ind.remove();
}

