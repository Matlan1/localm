// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - coder - multi-session (split from app.js). Classic script: it
   shares the one global lexical environment with the other app/* and
   pages/* scripts, so every cross-section reference resolves by bare
   name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { addMessageRow, lsSetScoped } from "./chat.js";
import { $, authHeaders, autoGrow, confirmDanger, el, nearBottom, openModal, readSSE, renderMarkdown, toast } from "./helpers.js";
import { emptyState, iconEl } from "./icons.js";
import { modelCache, refreshModels } from "./models-sidebar.js";
import { pickDirectory } from "./picker.js";
import { composerEnterToSend } from "./settings-perf.js";
import { execCoderCommand, handleSlashSubmit } from "./slash.js";

/* ================================================================ */
/*  Coder - multi-session                                            */
/* ================================================================ */

export const coder = {
  sessions: new Map(),   // id → {info, feedEl, busy, liveBody, liveText, pendingCards, gen}
  activeId: null,
  lastActiveId: null,    // session to return to when leaving setup mode
  docs: [],              // file attachments: {name, text, chars, truncated}
};

export function activeSession() {
  return coder.activeId ? coder.sessions.get(coder.activeId) : null;
}

export function sessionLabel(info) {
  const dir = info.cwd.split(/[\\/]/).filter(Boolean).pop() || info.cwd;
  return `${dir} (${info.id.slice(0, 6)})`;
}

// docs/gui-design.md rule 6: state renders as a .job-state pill. "working…"
// maps to st-running, "idle" to st-pending (existing neutral variant), "error"
// to st-error; the empty string (setup mode, no active session) gets no
// variant, matching the base pill's dim/no-background look.
function setCoderState(text) {
  const node = $("coder-state");
  node.textContent = text;
  node.className = "job-state" + (
    text === "working…" ? " st-running" :
    text === "idle" ? " st-pending" :
    text === "error" ? " st-error" : "");
}

export function renderSessionSelect() {
  const sel = $("session-select");
  sel.replaceChildren();
  // In setup mode (no active session) a placeholder holds the selection, so
  // picking any real session fires onchange - even when only one exists.
  if (!coder.activeId && coder.sessions.size) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(new session)";
    opt.selected = true;
    sel.appendChild(opt);
  }
  for (const [id, s] of coder.sessions) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = sessionLabel(s.info) + (s.busy ? " ⏳" : "");
    if (id === coder.activeId) opt.selected = true;
    sel.appendChild(opt);
  }
  renderCoderSessionList();   // R17: keep the right-side rail in lockstep with the dropdown
}

// R17: the coder's right-side open-sessions rail (mirrors the chat conversation
// list). The #session-select dropdown stays as the mobile fallback.
/** Put the session rail on the configured side.
 *
 * Drives a `data-rail` attribute rather than toggling a class per side: one
 * attribute with one value cannot end up in the both-classes state that two
 * independent toggles can reach, and CSS reads it directly.
 *
 * An unknown or absent value is left alone deliberately - the CSS default is the
 * right-hand rail, so an older server, a partial config payload or a typo lays the
 * page out correctly instead of producing a rail on neither side.
 */
export function applyCoderRailSide(side) {
  const view = $("view-coder");
  if (!view) return;
  if (side === "left") view.dataset.rail = "left";
  else delete view.dataset.rail;
}

// Past sessions, grouped by project, as the server last reported them. Kept
// separate from coder.sessions (which is LIVE, in-memory, authoritative) so a
// failed or slow dormant fetch can never blank the list of sessions the user
// currently has open.
export const dormant = { projects: [], note: "", loaded: false };

function _when(iso) {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const mins = Math.floor((Date.now() - t) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return mins + "m ago";
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return hrs + "h ago";
  return Math.floor(hrs / 24) + "d ago";
}

function _dormantRow(projectPath, sess, available) {
  const item = el("div", "coder-session-item dormant");
  item.appendChild(el("span", "title", sess.title || "(untitled session)"));
  const meta = [_when(sess.interrupted_at), (sess.turns || 0) + " turns"]
    .filter(Boolean).join(" · ");
  item.appendChild(el("span", "coder-session-meta", meta));
  if (available) {
    item.title = "Continue this session";
    item.onclick = () => startCoderSession(
      { resume: true, cwd: projectPath, checkpointId: sess.id });
  } else {
    // The conversation is still here, the FOLDER is not. Offering a click that
    // then fails at the server is worse than saying so up front.
    item.classList.add("unavailable");
    item.title = "The project folder is missing, so this session cannot be continued";
  }
  return item;
}

export function renderCoderSessionList() {
  const list = $("coder-session-list");
  if (!list) return;
  list.replaceChildren();

  // 1. LIVE. Always from memory, always first: these are running right now.
  if (coder.sessions.size) {
    list.appendChild(el("div", "coder-rail-head", "Open"));
    for (const [id, s] of coder.sessions) {
      const item = el("div", "coder-session-item" + (id === coder.activeId ? " active" : ""));
      item.appendChild(el("span", "title", sessionLabel(s.info)));
      if (s.busy) {
        const badge = el("span", "badge");
        badge.appendChild(iconEl("clock", "ic"));
        badge.title = "Busy";
        item.appendChild(badge);
      }
      item.onclick = () => activateSession(id);
      list.appendChild(item);
    }
  }

  // 2. PAST, for the project the form is pointing at.
  const current = dormant.projects.find((p) => p.current);
  if (current && current.sessions.length) {
    list.appendChild(el("div", "coder-rail-head", "Past sessions here"));
    for (const sess of current.sessions) {
      list.appendChild(_dormantRow(current.path, sess, current.available));
    }
  }

  // 3. OTHER PROJECTS, collapsed. This is the half that did not exist: a past
  // session is reachable without first typing its project path into the form.
  const others = dormant.projects.filter((p) => !p.current && p.sessions.length);
  if (others.length) {
    list.appendChild(el("div", "coder-rail-head", "Other projects"));
    for (const proj of others) {
      const group = el("details", "coder-rail-project");
      const sum = el("summary");
      sum.appendChild(el("span", "title", proj.name));
      sum.appendChild(el("span", "coder-session-meta",
        proj.sessions.length + (proj.available ? "" : " · folder missing")));
      group.appendChild(sum);
      for (const sess of proj.sessions) {
        group.appendChild(_dormantRow(proj.path, sess, proj.available));
      }
      list.appendChild(group);
    }
  }

  if (!list.childNodes.length) {
    list.appendChild(el("div", "coder-session-empty", "No sessions yet"));
  }

  // The note is PERMANENT, not an empty state, and its text comes from the
  // server rather than a string here - one wording, which cannot drift from
  // what the endpoint actually guarantees.
  const note = $("coder-rail-note");
  if (note) note.textContent = dormant.loaded ? (dormant.note || "") : "";
}

// Past sessions across every remembered project. Never throws and never clears
// what is already shown: a listing that blanks on a transient error looks
// exactly like "you have no past work".
export async function refreshDormant() {
  const cwdEl = $("setup-cwd");
  const cwd = cwdEl ? cwdEl.value.trim() : "";
  try {
    const r = await fetch("/api/coder/dormant?cwd=" + encodeURIComponent(cwd),
                          { headers: authHeaders() });
    if (!r.ok) return;
    const d = await r.json();
    if (!d || !Array.isArray(d.projects)) return;
    dormant.projects = d.projects;
    dormant.note = d.privacy_note || "";
    dormant.loaded = true;
  } catch {
    return;
  }
  renderCoderSessionList();
}

export function showCoderUI(hasSession) {
  $("coder-setup").style.display = hasSession ? "none" : "block";
  $("coder-composer").style.display = hasSession ? "block" : "none";
  // Keep the bar while other sessions exist so they stay reachable
  $("coder-bar").classList.toggle("open", hasSession || coder.sessions.size > 0);
  if (!hasSession) {
    // Setup mode: park every session feed and clear the session labels -
    // the form must not render on top of a previous session's transcript.
    // Remember where we came from so "back to session" can return there.
    if (coder.activeId && coder.sessions.has(coder.activeId)) {
      coder.lastActiveId = coder.activeId;
    }
    for (const [, s] of coder.sessions) s.feedEl.classList.remove("active");
    coder.activeId = null;
    $("coder-cwd").textContent = "";
    setCoderState("");
    $("coder-usage").textContent = "";
    renderSessionSelect();
    refreshResumable();   // reveal "Continue last session" if the cwd has one (CODER-2)
    refreshDormant();     // and list past sessions across every remembered project
  }
  $("setup-cancel").style.display =
    !hasSession && coder.sessions.size > 0 ? "" : "none";
}

export function activateSession(id) {
  coder.activeId = id;
  for (const [sid, s] of coder.sessions) {
    s.feedEl.classList.toggle("active", sid === id);
  }
  const s = coder.sessions.get(id);
  if (s) {
    $("coder-cwd").textContent = s.info.cwd;
    setCoderState(s.busy ? "working…" : "idle");
    $("coder-usage").textContent = s.info.total_tokens
      ? `${s.info.total_tokens} tok · turn ${s.info.turns}` : "";
  }
  // "patch" only exists for a patch-mode session: in any other session the
  // writes went to disk, so the button would download an empty file and read
  // as "the agent changed nothing".
  $("coder-patch").style.display = s && s.info.patch_mode ? "" : "none";
  renderSessionSelect();
  showCoderUI(!!s);
}

export function registerSession(info, { replay }) {
  const feedEl = el("div", "coder-feed");
  $("coder-feeds").appendChild(feedEl);
  const s = {
    info,
    feedEl,
    busy: info.busy || false,
    liveBody: null,
    liveText: "",
    liveReasoning: "",   // H4: thinking model's reasoning, streamed via "reasoning" events
    pendingCards: [],
    confirmCards: new Map(),   // confirm_id → {card, title, buttons, tool}
    closed: false,
  };
  coder.sessions.set(info.id, s);
  streamSession(s, replay);
  return s;
}

/* per-session feed helpers */

export function feedAppend(s, node) {
  const stick = nearBottom(s.feedEl);
  s.feedEl.appendChild(node);
  if (stick) s.feedEl.scrollTop = s.feedEl.scrollHeight;
}

export function startAssistantBlock(s) {
  if (s.liveBody) return;
  const { body } = addMessageRow(s.feedEl, "assistant", "");
  s.liveBody = body;
  s.liveText = "";
  s.liveReasoning = "";
}

// H4: rebuild <think>reasoning</think>content from the two separately-streamed
// accumulators and hand it to renderMarkdown, which already knows how to split
// that back into a collapsible .think-block + the main body (same trick the
// regular chat GUI uses in settings-perf.js's runCompletion). Both the "token"
// and "reasoning" event handlers call this so a mid-stream re-render of one
// channel never clobbers the other.
function renderLiveBlock(s) {
  const stick = nearBottom(s.feedEl);
  renderMarkdown(s.liveBody,
    s.liveReasoning ? `<think>\n${s.liveReasoning}\n</think>\n${s.liveText}` : s.liveText);
  if (stick) s.feedEl.scrollTop = s.feedEl.scrollHeight;
}

export function flushAssistantBlock(s) {
  // CODER-EMPTY-MODEL: when the assistant turn produced no VISIBLE text (it emitted
  // only a tool call, or its text scrubbed to nothing), drop the empty "Model" row
  // instead of leaving a blank bubble stacked above the tool card.
  if (s.liveBody && !s.liveBody.textContent.trim()) {
    const row = s.liveBody.closest(".msg-row");
    if (row) row.remove();
  }
  s.liveBody = null;
  s.liveText = "";
  s.liveReasoning = "";
}

export function renderDiff(text) {
  const pre = el("pre", "diff");
  for (const line of (text || "").split("\n")) {
    let cls = "";
    if (line.startsWith("+") && !line.startsWith("+++")) cls = "add";
    else if (line.startsWith("-") && !line.startsWith("---")) cls = "del";
    else if (line.startsWith("@@")) cls = "hunk";
    pre.appendChild(el("span", cls, line + "\n"));
  }
  return pre;
}

/** Args worth showing next to a diff - the bulky text fields ARE the diff. */
export function slimArgs(args) {
  const slim = {};
  for (const [k, v] of Object.entries(args || {})) {
    if (k === "content" || k === "old" || k === "new" || k === "diff") continue;
    slim[k] = v;
  }
  return slim;
}

/** Head-line hint for a set_todos call: progress plus the item being worked on,
 *  so the user sees the model's plan move without opening the card. Mirrors
 *  todos_summary() in coder/tools/tasks.py; "" for anything that is not a todo
 *  list. Markers: [x] done, [>] in progress, anything else pending. */
export function todoHint(items) {
  if (!Array.isArray(items) || !items.length) return "";
  const lines = items.map((t) =>
    String(t && typeof t === "object" ? (t.text ?? "") : (t ?? "")).trim());
  const done = lines.filter((l) => /^\[[xXvV✓✔]\]/.test(l)).length;
  const active = lines.find((l) => /^\[[>*~@]\]/.test(l)) || "";
  const label = `${done}/${lines.length} done`;
  return active ? `${label} · ${active.replace(/^\[.\]\s*/, "")}` : label;
}

export function buildToolCard(ev) {
  const card = el("div", "tool-card");
  const inner = el("div", "inner");
  const head = el("div", "head");
  head.appendChild(el("span", "name", ev.tool));
  const hintVal = ev.args?.path || ev.args?.command || ev.args?.pattern
    || ev.args?.url || todoHint(ev.args?.items) || "";
  head.appendChild(el("span", "hint", String(hintVal).slice(0, 120)));
  // docs/gui-design.md rule 6: state renders as a .job-state pill. A just-created
  // card is actively executing (result not back yet), so st-running until it lands.
  head.appendChild(el("span", "state job-state st-running", "…"));
  const body = el("div", "body");
  if (ev.diff) {
    const rest = slimArgs(ev.args);
    if (Object.keys(rest).length) {
      body.appendChild(el("pre", "args", JSON.stringify(rest, null, 2)));
    }
    body.appendChild(renderDiff(ev.diff));
  } else {
    body.textContent = JSON.stringify(ev.args, null, 2);
  }
  head.onclick = () => card.classList.toggle("open");
  inner.appendChild(head);
  inner.appendChild(body);
  card.appendChild(inner);
  return card;
}

/** Mark a confirm card as resolved. Idempotent - fed both by the local
 *  button click and by the confirm_resolved event from the server (which is
 *  also what replay sends for already-answered confirmations). */
export function resolveConfirmCard(s, confirmId, approved, timedOut) {
  const entry = s.confirmCards.get(confirmId);
  if (!entry || entry.card.classList.contains("answered")) return;
  entry.card.classList.add("answered");
  entry.title.replaceChildren();
  entry.title.appendChild(iconEl(!timedOut && approved ? "check" : "close", "btn-ic"));
  entry.title.appendChild(document.createTextNode(timedOut
    ? "Timed out - rejected " + entry.tool
    : (approved ? "Approved " : "Rejected ") + entry.tool));
}

export function buildConfirmCard(s, ev) {
  const card = el("div", "confirm-card");
  const inner = el("div", "inner");
  const title = el("div", "title");
  title.appendChild(document.createTextNode("Approve "));
  title.appendChild(el("span", "name", ev.tool));
  title.appendChild(document.createTextNode("?"));
  inner.appendChild(title);
  // Which sub-agent is asking. Parallel dispatch serialises several children onto
  // this one channel, so without it two identical cards arrive with nothing to
  // tell them apart. Absent for the session's own agent - its card is unchanged.
  if (ev.agent) {
    const who = el("div", "asker");
    who.appendChild(document.createTextNode("sub-agent "));
    who.appendChild(el("span", "name", ev.agent));
    who.appendChild(document.createTextNode(" is asking"));
    inner.appendChild(who);
  }
  if (ev.diff) {
    inner.appendChild(renderDiff(ev.diff));
  } else {
    inner.appendChild(el("pre", "diff", JSON.stringify(ev.args, null, 2)));
  }
  const buttons = el("div", "buttons");
  const yes = el("button", "btn-primary", "Approve");
  const no = el("button", "btn-danger", "Reject");
  // "always allow" lives inside .buttons so the answered-state CSS hides it
  const allowCb = document.createElement("input");
  allowCb.type = "checkbox";
  const allowLabel = el("label", "always-allow");
  allowLabel.appendChild(allowCb);
  allowLabel.appendChild(document.createTextNode(
    ` always allow ${ev.tool} this session`));
  const answer = async (approved) => {
    try {
      const r = await fetch(`/api/coder/sessions/${s.info.id}/confirm`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          confirm_id: ev.confirm_id, approved,
          always_allow: approved && allowCb.checked,
        }),
      });
      if (!r.ok) {
        // Already answered elsewhere (another tab) or timed out server-side -
        // the confirm_resolved event carries the real outcome.
        toast("Confirmation was no longer pending", true);
        return;
      }
      if (approved && allowCb.checked) {
        toast(`${ev.tool} auto-approved for the rest of this session`);
      }
      resolveConfirmCard(s, ev.confirm_id, approved, false);
    } catch (e) {
      toast("Failed to answer confirmation: " + e.message, true);
    }
  };
  yes.onclick = () => answer(true);
  no.onclick = () => answer(false);
  buttons.appendChild(yes);
  buttons.appendChild(no);
  buttons.appendChild(allowLabel);
  inner.appendChild(buttons);
  card.appendChild(inner);
  s.confirmCards.set(ev.confirm_id, { card, title, tool: ev.tool });
  return card;
}

export function handleCoderEvent(s, ev) {
  // Keep a light event log (no token/reasoning spam) so "export" can rebuild
  // the session as markdown without another server round-trip.
  if (ev.type !== "token" && ev.type !== "reasoning") {
    (s.eventLog = s.eventLog || []).push(ev);
  }
  switch (ev.type) {
    case "token": {
      startAssistantBlock(s);
      s.liveText += ev.text;
      renderLiveBlock(s);
      break;
    }
    case "reasoning": {
      // H4: a thinking model's reasoning (AUD-HIGH-17-3), kept in its own
      // collapsible block via renderLiveBlock - never mixed into the answer.
      startAssistantBlock(s);
      s.liveReasoning += ev.text;
      renderLiveBlock(s);
      break;
    }
    case "assistant_text": {
      // Authoritative fix-up, sent once the harness knows which spans of the
      // just-streamed response were REAL tool calls (loop.py, right before
      // dispatching them): replaces whatever streamed live with the actual
      // leftover text, so a call written in a shape the live "token" stream
      // does not know how to hide (e.g. a ```json fence) does not linger in
      // the bubble as visible raw JSON once it has been executed. Usually a
      // no-op - the live stream already got the common <tool_call> shape
      // right, so this just re-sends the same text the bubble already shows.
      if (!s.liveBody && !ev.text) break;   // nothing streamed, nothing to fix
      if (!s.liveBody) startAssistantBlock(s);
      s.liveText = ev.text || "";
      renderLiveBlock(s);
      break;
    }
    case "turn": {
      flushAssistantBlock(s);
      s.busy = true;
      s.info.turns = ev.turn;
      s.info.total_tokens = ev.total_tokens;
      if (s.info.id === coder.activeId) {
        setCoderState("working…");
        const ctx = ev.ctx_ratio ? ` · ctx ${Math.round(ev.ctx_ratio * 100)}%` : "";
        if (ev.total_tokens)
          $("coder-usage").textContent = `${ev.total_tokens} tok · turn ${ev.turn}${ctx}`;
      }
      break;
    }
    case "tool_call": {
      flushAssistantBlock(s);
      const card = buildToolCard(ev);
      feedAppend(s, card);
      s.pendingCards.push(card);
      break;
    }
    case "tool_result": {
      const card = s.pendingCards.shift();
      if (card) {
        const state = card.querySelector(".state");
        // Real server-side timing (execution.py) around the tool invocation
        // itself - not a client-side guess between two render events, which
        // read ~0.0s whenever both arrived in the same tick. Absent on events
        // replayed from a session recorded before this field existed; show
        // nothing rather than a fabricated number.
        const took = typeof ev.duration_s === "number"
          ? ` · ${ev.duration_s.toFixed(1)}s` : "";
        state.textContent = (ev.summary || (ev.ok ? "ok" : "failed")) + took;
        state.className = "state job-state " + (ev.ok ? "st-ok" : "st-error");
        if (ev.output && !card.querySelector(".body .diff")) {
          card.querySelector(".body").textContent = ev.output;
        }
      }
      break;
    }
    case "confirm_request": {
      flushAssistantBlock(s);
      feedAppend(s, buildConfirmCard(s, ev));
      break;
    }
    case "confirm_resolved": {
      resolveConfirmCard(s, ev.confirm_id, ev.approved, ev.timed_out);
      break;
    }
    case "user": {
      // replayed user message (emitted client-side on send; replay rebuilds it)
      flushAssistantBlock(s);
      addMessageRow(s.feedEl, "user", ev.text,
        ev.queued ? { cls: "web-note", label: "Queued" } : {});
      break;
    }
    case "info": {
      flushAssistantBlock(s);
      feedAppend(s, el("div", "feed-info", ev.text));
      break;
    }
    case "episodes_recalled": {
      // Which past lessons this run pulled in, with the id `localcoder
      // --forget-episode <id>` takes. Recall used to be invisible, so a lesson
      // that steered a run badly could not be traced back or removed.
      const eps = ev.episodes || [];
      if (!eps.length) break;
      flushAssistantBlock(s);
      const label = `Recalled ${eps.length} past lesson${eps.length > 1 ? "s" : ""}: ` +
        eps.map((e) => `${e.lesson || ""} (${e.id})`).join(" · ");
      feedAppend(s, el("div", "feed-info", label));
      break;
    }
    case "estimate": {
      // A plan that was never executed (the CLI's --estimate). Rendered as an
      // assistant row because it is multi-paragraph prose, but labelled, so it
      // cannot be mistaken on replay for a turn that actually ran.
      flushAssistantBlock(s);
      feedAppend(s, el("div", "feed-info", `Estimate for: ${ev.task || ""}`));
      addMessageRow(s.feedEl, "assistant", ev.text || "");
      if (ev.total_tokens) {
        feedAppend(s, el("div", "feed-info",
          `Planning turn used ${ev.total_tokens} tokens ` +
          `(${ev.prompt_tokens ?? "?"} prompt). Nothing was run or written.`));
      }
      break;
    }
    case "history": {
      // A recap row replayed when a past session is resumed (CODER-2): plain,
      // role-styled text, no streaming.
      flushAssistantBlock(s);
      addMessageRow(s.feedEl, ev.role === "assistant" ? "assistant" : "user",
                    ev.text || "");
      break;
    }
    case "replay_done": {
      flushAssistantBlock(s);
      s.feedEl.scrollTop = s.feedEl.scrollHeight;
      break;
    }
    case "final": {
      flushAssistantBlock(s);
      s.busy = false;
      s.info.turns = ev.turns;
      s.info.total_tokens = ev.total_tokens;
      if (s.info.id === coder.activeId) setCoderState("idle");
      renderSessionSelect();
      // "inconclusive" means the check could not run or collected nothing, so
      // an unqualified "Task finished" would claim a verification that never
      // happened. ok stays the run's own outcome; this names the gate's.
      const verifyNote = ev.verify_state === "inconclusive"
        ? " (not verified)" : "";
      let finalLine = (ev.ok ? "Task finished" : "Task ended") + verifyNote +
        ` - ${ev.turns} turns, ${ev.total_tokens} tokens`;
      if (ev.changed_files?.length) {
        finalLine += ` · ${ev.changed_files.length} file(s) changed (see "files")`;
      }
      feedAppend(s, el("div", "feed-final", finalLine));
      break;
    }
    case "error": {
      flushAssistantBlock(s);
      s.busy = false;
      if (s.info.id === coder.activeId) setCoderState("error");
      toast("Agent error: " + ev.text, true);
      break;
    }
    case "closed": {
      s.busy = false;
      s.closed = true;
      break;
    }
  }
}

export async function streamSession(s, replay) {
  while (coder.sessions.has(s.info.id) && !s.closed) {
    try {
      const r = await fetch(
        `/api/coder/sessions/${s.info.id}/events${replay ? "?replay=true" : ""}`,
        { headers: authHeaders() });
      if (r.status === 404) { s.closed = true; break; }
      if (!r.ok) throw new Error(r.statusText);
      replay = false;   // only the first connection replays
      await readSSE(r, (payload) => {
        let ev;
        try { ev = JSON.parse(payload); } catch { return; }
        if (coder.sessions.has(s.info.id)) handleCoderEvent(s, ev);
      });
    } catch (e) {
      if (!coder.sessions.has(s.info.id) || s.closed) return;
      await new Promise((res) => setTimeout(res, 1500));
    }
  }
}

/* session lifecycle */

export function populateSetupModels() {
  const sel = $("setup-model");
  sel.innerHTML = "";
  const current = document.createElement("option");
  current.value = "";
  current.textContent = "active model (" + (modelCache.active || "?") + ")";
  sel.appendChild(current);
  for (const m of modelCache.models || []) {
    if (m.active) continue;
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = m.name;
    sel.appendChild(opt);
  }
}

export async function startCoderSession(opts = {}) {
  const resume = !!opts.resume;
  // A rail row names its OWN project, which is usually not the one in the form -
  // that is the whole point of listing other projects. Taking the form's value
  // there would start a session in the wrong folder while looking correct.
  if (opts.cwd) $("setup-cwd").value = opts.cwd;
  const cwd = $("setup-cwd").value.trim();
  if (!cwd) { toast("Enter a project directory", true); return; }
  $("setup-start").disabled = true;
  try {
    const body = {
      cwd,
      auto_approve: $("setup-auto").checked,
      dry_run: $("setup-dry").checked,
      // Writes become a unified diff instead of landing on disk (the CLI's
      // --patch-mode); download it from the "patch" button in the session bar.
      patch_mode: $("setup-patch").checked,
      // Only meaningful WITH auto-approve, which is exactly what it carves an
      // exception out of - sent unconditionally anyway so the session reports
      // back what it was actually given rather than what we guessed it meant.
      interactive_confirm: $("setup-interactive-confirm").checked,
      // The CLI's --native-tools. The server reports back whether it could
      // actually be honoured - see info.notes below - rather than us guessing
      // here, so this stays a plain request.
      native_tools: $("setup-native-tools").checked,
      mode: $("setup-mode").value,
      resume,
      // WHICH past conversation, when the rail offered a specific one. Absent
      // for the plain "continue last session" button, which still means "the
      // most recent here".
      resume_checkpoint_id: opts.checkpointId || null,
    };
    // Blank = the server's own default (sessions.py), matching temperature two
    // lines below - a hardcoded "|| 40" here duplicated that default instead
    // of leaving it the single source of truth (NEW-DEFAULT-VALUE-PLACEHOLDER).
    const maxTurns = $("setup-max-turns").value.trim();
    if (maxTurns !== "") body.max_turns = Number(maxTurns);
    const model = $("setup-model").value;
    if (model) body.model = model;
    const temp = $("setup-temperature").value.trim();
    if (temp !== "") body.temperature = Number(temp);
    // Blank = no seed at all (a fresh random one per run), which is NOT the same
    // as seed 0 - a real and reproducible value. Omit rather than coerce.
    const seed = $("setup-seed").value.trim();
    if (seed !== "") body.seed = Number(seed);
    const scope = $("setup-scope").value.trim();
    if (scope) body.scope = scope;
    const system = $("setup-system").value.trim();
    if (system) body.custom_instructions = system;
    // The exit-code oracle (the CLI's --until / --verify / --no-verify). Blank
    // + auto_verify = the project's detected check; "skip verification" is the
    // --no-verify half, and it wins over a typed command the same way the CLI's
    // flag does.
    const verify = $("setup-verify").value.trim();
    if ($("setup-no-verify").checked) body.auto_verify = false;
    else if (verify) body.verify = verify;
    const verifyRetries = $("setup-verify-retries").value.trim();
    if (verifyRetries !== "") body.verify_max_retries = Number(verifyRetries);

    const r = await fetch("/api/coder/sessions", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const info = await r.json();
    lsSetScoped("localm.coderCwd", cwd);
    // A resumed session replays its restored recap from the server; a fresh one
    // has no history to replay (CODER-2).
    registerSession(info, { replay: !!info.resumed });
    activateSession(info.id);
    // An option the server could not honour is SAID, not silently swallowed -
    // otherwise a ticked box and an ignored one look identical (AGENTS.md
    // rule 5). The server decides; this only relays.
    for (const note of info.notes || []) toast(note, true);
    if (info.resumed) toast("Resumed your last session in this folder");
    else if (resume) toast("No saved session to resume - started fresh");
    refreshModels();
  } catch (e) {
    toast("Failed to start session: " + e.message, true);
  } finally {
    $("setup-start").disabled = false;
  }
}

/* Resume (CODER-2): a dynamically-created "Continue last session" button in the
 * setup panel, revealed when the chosen directory has a saved conversation. Built
 * in JS so it needs no index.html change. */
export let _coderContinueBtn = null;
export function coderContinueButton() {
  if (_coderContinueBtn) return _coderContinueBtn;
  const btn = el("button", "btn-secondary coder-continue", "Continue last session");
  btn.style.display = "none";
  btn.onclick = () => startCoderSession({ resume: true });
  const start = $("setup-start");
  if (start && start.parentNode) start.parentNode.insertBefore(btn, start.nextSibling);
  _coderContinueBtn = btn;
  return btn;
}

export async function refreshResumable() {
  const btn = coderContinueButton();
  const cwd = ($("setup-cwd")?.value || "").trim();
  if (!cwd) { btn.style.display = "none"; return; }
  try {
    const r = await fetch("/api/coder/resumable?cwd=" + encodeURIComponent(cwd),
                          { headers: authHeaders() });
    const d = await r.json();
    if (r.ok && d.resumable) {
      const when = d.interrupted_at
        ? new Date(d.interrupted_at).toLocaleString() : "earlier";
      btn.textContent = `Continue last session (${d.turns} turns, ${when})`;
      btn.style.display = "";
    } else {
      btn.style.display = "none";
    }
  } catch { btn.style.display = "none"; }
}

export async function reattachSessions() {
  try {
    const r = await fetch("/api/coder/sessions", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    for (const info of data.sessions) {
      if (!coder.sessions.has(info.id)) {
        registerSession(info, { replay: true });
      }
    }
    if (!coder.activeId && data.sessions.length) {
      activateSession(data.sessions[data.sessions.length - 1].id);
      toast("Reattached to a running coder session");
    } else {
      // Sessions may exist without one being activated (e.g. the host did not
      // auto-open a session). Still surface them in the selector + bar so the
      // host sees the same session list the mobile view does, without having to
      // enter a session first (CODER-3).
      renderSessionSelect();
      if (coder.sessions.size > 0) $("coder-bar").classList.add("open");
    }
  } catch (e) { /* server unreachable; startup poller will retry models anyway */ }
}

/* coder file attachments - extracted to text server-side (same in-memory
 * /api/rag/extract path as chat docs) and prepended to the task message,
 * so the agent sees the content without needing the file inside cwd. */

export function renderCoderAttachChips() {
  const box = $("coder-attach-chips");
  box.replaceChildren();
  coder.docs.forEach((doc, i) => {
    const chip = el("span", "chip");
    chip.appendChild(iconEl("file", "ic"));
    chip.appendChild(el("span", "", doc.name +
      ` (${(doc.chars / 1000).toFixed(1)}k chars${doc.truncated ? ", trimmed" : ""})`));
    const rm = el("button", "", "×");
    rm.onclick = () => { coder.docs.splice(i, 1); renderCoderAttachChips(); };
    chip.appendChild(rm);
    box.appendChild(chip);
  });
}

export async function attachCoderDocument(file) {
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
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || r.statusText);
  coder.docs.push({ name: data.filename, text: data.text,
                    chars: data.chars, truncated: data.truncated });
  renderCoderAttachChips();
}

$("coder-attach").onclick = () => $("coder-file").click();
$("coder-file").addEventListener("change", (e) => {
  for (const file of e.target.files) {
    attachCoderDocument(file).catch((err) =>
      toast(`${file.name}: ${err.message}`, true));
  }
  e.target.value = "";
});

export async function sendCoderTask() {
  const s = activeSession();
  const input = $("coder-input");
  const text = input.value.trim();
  if ((!text && coder.docs.length === 0) || !s) return;

  if (text.startsWith("/")) {
    input.value = "";
    autoGrow(input);
    handleSlashSubmit(text, (c) => execCoderCommand(c));
    return;
  }

  // Attached file contents go first so the agent reads them before the task
  let payload = text || "Read the attached file(s).";
  if (coder.docs.length) {
    const blocks = coder.docs.map((d) =>
      `[Attached file: ${d.name}${d.truncated ? " (truncated)" : ""}]\n${d.text}`);
    payload = blocks.join("\n\n") + "\n\n" + payload;
  }

  try {
    const r = await fetch(`/api/coder/sessions/${s.info.id}/message`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ text: payload }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    if (data.status === "queued") {
      // Mid-task steering: the agent reads it at the next turn boundary
      toast("Queued - the agent picks it up at the next turn");
    } else {
      s.busy = true;
      setCoderState("working…");
      renderSessionSelect();
    }
    // The user message arrives back through the event stream (so replay
    // works after a page reload) - no client-side row here.
    input.value = "";
    autoGrow(input);
    coder.docs = [];
    renderCoderAttachChips();
  } catch (e) {
    toast("Failed to send task: " + e.message, true);
  }
}

export async function endCoderSession() {
  const s = activeSession();
  if (!s) return;
  try {
    await fetch(`/api/coder/sessions/${s.info.id}`, {
      method: "DELETE", headers: authHeaders() });
  } catch (e) { /* server may already be gone */ }
  s.closed = true;
  s.feedEl.remove();
  coder.sessions.delete(s.info.id);
  const remaining = [...coder.sessions.keys()];
  coder.activeId = remaining[remaining.length - 1] || null;
  activateSession(coder.activeId);
}

/* coder bar buttons */

$("session-select").onchange = () => activateSession($("session-select").value);
$("session-new").onclick = () => {
  populateSetupModels();
  showCoderUI(false);
  $("coder-bar").classList.add("open");   // keep the bar so sessions stay reachable
};
// R17: the open-sessions rail's "+" mirrors the bar's "+ new".
if ($("coder-new-session")) $("coder-new-session").onclick = () => $("session-new").click();
renderCoderSessionList();   // R17: show the empty-state rail on first load

const _railFlip = $("coder-rail-flip");
if (_railFlip) _railFlip.onclick = async () => {
  const view = $("view-coder");
  const next = view && view.dataset.rail === "left" ? "right" : "left";
  applyCoderRailSide(next);            // instant, so the click never feels lost
  try {
    const r = await fetch("/v1/config", {
      method: "PATCH", headers: authHeaders(),
      body: JSON.stringify({ coder_rail_side: next }),
    });
    if (!r.ok) throw new Error("save failed");
  } catch {
    // Put it back rather than leaving the screen disagreeing with what was
    // saved - a side that silently reverts on the next load is worse than one
    // that refuses now and says why.
    applyCoderRailSide(next === "left" ? "right" : "left");
    toast("Could not save which side the session list sits on", true);
  }
};
// Arrow wrapper: a bare `.onclick = startCoderSession` would pass the click
// Event as opts, making opts.resume truthy and always resuming (CODER-2).
$("setup-start").onclick = () => startCoderSession();
// Probe for a resumable checkpoint as the directory changes (debounced).
export let _resumeProbeTimer = null;
$("setup-cwd").addEventListener("input", () => {
  clearTimeout(_resumeProbeTimer);
  _resumeProbeTimer = setTimeout(() => { refreshResumable(); refreshDormant(); }, 350);
});
$("setup-cancel").onclick = () => {
  // Return to the session we left (or any remaining one) without starting
  const id = coder.sessions.has(coder.lastActiveId)
    ? coder.lastActiveId
    : [...coder.sessions.keys()].pop();
  if (id) activateSession(id);
};

/* ---- directory picker (browse… on the setup form) ----
   pickDirectory / pickFile / pickPath now live in app/picker.js; the coder setup
   form imports pickDirectory from there. */

$("setup-browse").onclick = async () => {
  const dir = await pickDirectory("Pick a project directory",
                                  $("setup-cwd").value.trim());
  if (dir) {
    $("setup-cwd").value = dir;
    // Gate on privacy like the other coderCwd write (REC-CODER-CWD-LEAK): do not
    // persist the absolute project path to localStorage in privacy mode.
    lsSetScoped("localm.coderCwd", dir);
    refreshResumable();   // setting .value does not fire 'input' (CODER-2)
    refreshDormant();
  }
};
$("coder-send").onclick = sendCoderTask;
$("coder-input").addEventListener("keydown", (e) => composerEnterToSend(e, sendCoderTask));
$("coder-input").addEventListener("input", (e) => autoGrow(e.target));

$("coder-stop").onclick = async () => {
  const s = activeSession();
  if (!s) return;
  await fetch(`/api/coder/sessions/${s.info.id}/stop`, {
    method: "POST", headers: authHeaders() });
  toast("Stop requested - agent halts at the next safe point");
};
$("coder-end").onclick = endCoderSession;

$("coder-undo").onclick = async () => {
  const s = activeSession();
  if (!s) return;
  const r = await fetch(`/api/coder/sessions/${s.info.id}/undo`, {
    method: "POST", headers: authHeaders() });
  const data = await r.json();
  if (r.ok) {
    toast(data.summary);
    feedAppend(s, el("div", "feed-info", data.summary));
  } else {
    toast(data.detail || "Nothing to undo", true);
  }
};

$("coder-compact").onclick = async () => {
  const s = activeSession();
  if (!s) return;
  const r = await fetch(`/api/coder/sessions/${s.info.id}/compact`, {
    method: "POST", headers: authHeaders() });
  const data = await r.json();
  if (r.ok) {
    toast("History compacted");
    feedAppend(s, el("div", "feed-info", "Conversation history compacted."));
  } else {
    toast(data.detail || "Nothing to compact", true);
  }
};

/** Audit-entry modal shared by the live-session log and past-session history.
 *  A filter box narrows entries by substring (type, turn, or payload). */
export function showAuditModal(title, data) {
  openModal(title, (body) => {
    body.appendChild(el("div", "sub", data.path));
    const filter = document.createElement("input");
    filter.type = "text";
    filter.placeholder = "filter entries… (tool name, text, type)";
    filter.className = "log-filter";
    filter.spellcheck = false;
    body.appendChild(filter);
    const rows = [];
    for (const entry of data.entries) {
      const row = el("div", "log-entry");
      const ts = new Date(entry.t).toLocaleTimeString();
      const label = `${ts} #${entry.turn} ${entry.type}`;
      const payload = JSON.stringify(entry.data);
      row.appendChild(el("span", "t", label));
      row.appendChild(document.createTextNode(payload));
      body.appendChild(row);
      rows.push({ row, text: (label + " " + payload).toLowerCase() });
    }
    filter.addEventListener("input", () => {
      const q = filter.value.trim().toLowerCase();
      for (const r of rows) {
        r.row.style.display = !q || r.text.includes(q) ? "" : "none";
      }
    });
    if (!data.entries.length) {
      body.appendChild(emptyState("clock", "No entries yet",
        "Tool calls and events for this session appear here as they happen."));
    }
  });
}

/** Files the agent changed this session, with per-file and full-session diffs. */
export async function openFilesModal() {
  const s = activeSession();
  if (!s) return;
  let data;
  try {
    const r = await fetch(`/api/coder/sessions/${s.info.id}/files`,
                          { headers: authHeaders() });
    data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
  } catch (e) {
    toast("Could not load changed files: " + e.message, true);
    return;
  }
  openModal("Files changed - " + sessionLabel(s.info), (body) => {
    if (!data.files.length) {
      body.appendChild(emptyState("file", "No files changed",
        "Edits the agent makes this session appear here."));
      return;
    }
    const diffBox = el("div", "files-diff");
    const showDiff = async (path) => {
      diffBox.replaceChildren(el("div", "sub", "loading diff…"));
      try {
        const r = await fetch(
          `/api/coder/sessions/${s.info.id}/files/diff?path=` +
          encodeURIComponent(path || ""), { headers: authHeaders() });
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || r.statusText);
        diffBox.replaceChildren(
          d.diff ? renderDiff(d.diff) : el("div", "sub", "(no difference)"));
      } catch (e) {
        diffBox.replaceChildren(el("div", "sub", "diff failed: " + e.message));
      }
    };
    for (const f of data.files) {
      const row = el("div", "log-entry clickable");
      row.appendChild(el("span", "t", f.created ? "new" : "edit"));
      row.appendChild(document.createTextNode(
        `${f.path} - ${f.writes} write(s)` + (f.exists ? "" : " (deleted since)")));
      row.onclick = () => showDiff(f.path);
      // Download the file itself (pull coder output onto this device / phone).
      // Only for files that still exist on disk.
      if (f.exists) {
        const dl = el("button", "btn-secondary file-dl", "download");
        dl.title = "Download this file to your device";
        dl.onclick = (ev) => { ev.stopPropagation(); downloadCoderFile(s, f.path); };
        row.appendChild(dl);
      }
      body.appendChild(row);
    }
    const all = el("button", "btn-secondary", "full session diff");
    all.onclick = () => showDiff("");
    body.appendChild(all);
    body.appendChild(diffBox);
  });
}

/** Download one coder-created/changed file to this device (a phone, say). Fetched
 *  with auth so it works behind a key; saved via a blob so the OS "save file"
 *  flow runs. The server confines the download to tracked, in-root files. */
export async function downloadCoderFile(s, path) {
  try {
    const r = await fetch(
      `/api/coder/sessions/${s.info.id}/files/download?path=` +
      encodeURIComponent(path), { headers: authHeaders() });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || r.statusText);
    }
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = path.split(/[\\/]/).pop() || "file";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
  } catch (e) {
    toast("Download failed: " + e.message, true);
  }
}

/** Download the active session's feed as markdown (explicit user action -
 *  works in privacy mode too, same contract as chat /export). */
export function exportCoderSession() {
  const s = activeSession();
  const log = s?.eventLog || [];
  if (!log.length) { toast("Nothing to export yet", true); return; }
  const lines = [`# Coder session - ${sessionLabel(s.info)}`, ""];
  for (const ev of log) {
    if (ev.type === "user") {
      lines.push(`**You${ev.queued ? " (queued)" : ""}**: ${ev.text}`, "");
    } else if (ev.type === "tool_call") {
      lines.push(`- \`${ev.tool}\` ` +
        JSON.stringify(slimArgs(ev.args)).slice(0, 200));
    } else if (ev.type === "tool_result") {
      lines.push(`  - ${ev.ok ? "ok" : "FAILED"}` +
        (ev.summary ? `: ${ev.summary}` : ""));
    } else if (ev.type === "info") {
      lines.push(`> ${ev.text}`, "");
    } else if (ev.type === "estimate") {
      lines.push("", `**Estimate (not run)** for: ${ev.task || ""}`, "",
        ev.text || "", "");
    } else if (ev.type === "final") {
      lines.push("", `**Agent**: ${ev.text}`, "");
    }
  }
  const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `coder-session-${s.info.id}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/** The last finished task as JSON: the web form of the CLI's
 *  `--output-format json`. Fetched from the server rather than rebuilt from the
 *  client's event log, so a tab that joined late (or reconnected mid-task, and
 *  therefore never saw the final event) still gets the real result instead of
 *  an empty object that looks like a task which produced nothing. */
export async function exportCoderResultJson() {
  const s = activeSession();
  if (!s) { toast("No active session", true); return; }
  try {
    const r = await fetch(`/api/coder/sessions/${s.info.id}/result`,
                          { headers: authHeaders() });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const blob = new Blob([JSON.stringify(data, null, 2)],
                          { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `coder-result-${s.info.id}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) {
    toast("No result to export: " + e.message, true);
  }
}

/** Download the patch a patch-mode session captured instead of writing - the
 *  web form of `--patch-mode FILE`, where the file lands on THIS device because
 *  that is the only disk a browser can write to. Reading it server-side does not
 *  consume it, so this can be pressed as often as you like. */
export async function downloadCoderPatch() {
  const s = activeSession();
  if (!s) { toast("No active session", true); return; }
  try {
    const r = await fetch(`/api/coder/sessions/${s.info.id}/patch/download`,
                          { headers: authHeaders() });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || r.statusText);
    }
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `coder-session-${s.info.id}.patch`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
  } catch (e) {
    toast("Patch download failed: " + e.message, true);
  }
}

/** The episodic-memory lessons stored for the project in the setup form - the
 *  web form of `localcoder --episodes`. Read-only: forgetting and restoring a
 *  lesson stay CLI flags with their own destructive semantics. The id is shown
 *  because it is what `--forget-episode` / `--restore-episode` address. */
/** POST/DELETE helper for the episode write routes. Every one of them is
 *  owner-only and state-changing, so they are unsafe methods carrying the cwd in
 *  the body rather than a URL a user could be walked into following. */
async function episodeOp(method, path, cwd) {
  const r = await fetch(path, {
    method,
    headers: authHeaders(),
    body: JSON.stringify({ cwd }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

/** Manage the episodic-memory lessons stored for the project in the setup form:
 *  the web form of `localcoder --episodes` and its write siblings
 *  (`--episodes-archive`, `--forget-episode`, `--restore-episode`,
 *  `--forget-episodes`, `--consolidate-episodes`).
 *
 *  Ids are shown because they are what the CLI flags address, so what you read
 *  here stays actionable from a terminal. */
export async function openEpisodesModal(view = "live") {
  const cwd = ($("setup-cwd")?.value || "").trim();
  if (!cwd) { toast("Enter a project directory first", true); return; }
  const url = view === "archive"
    ? "/api/coder/episodes/archive?cwd=" : "/api/coder/episodes?cwd=";
  try {
    const r = await fetch(url + encodeURIComponent(cwd), { headers: authHeaders() });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    const rows = (view === "archive" ? data.archived : data.episodes) || [];
    openModal(view === "archive" ? "Dropped lessons" : "Stored lessons", (body) => {
      const tabs = el("div", "actions");
      const liveBtn = el("button", "btn-secondary", "stored");
      liveBtn.onclick = () => openEpisodesModal("live");
      const arcBtn = el("button", "btn-secondary", "dropped");
      arcBtn.onclick = () => openEpisodesModal("archive");
      (view === "archive" ? arcBtn : liveBtn).classList.add("active");
      tabs.appendChild(liveBtn);
      tabs.appendChild(arcBtn);
      body.appendChild(tabs);

      if (!rows.length) {
        body.appendChild(emptyState(view === "archive"
          ? "Nothing has been dropped for this project. Lessons land here when "
            + "they are merged, evicted at the cap, or forgotten - so they can be "
            + "brought back."
          : "No lessons stored for this project yet. The coder writes one when a "
            + "session that changed files finishes, outside privacy mode."));
      } else {
        body.appendChild(el("p", "sub",
          `${rows.length} lesson(s) for ${data.cwd}. The id is what `
          + "localcoder --forget-episode / --restore-episode take."));
        for (const ep of rows) {
          const card = el("div", "card");
          card.appendChild(el("div", "row",
            `${ep.lesson || ep.summary || ep.task || "(no lesson text)"}`));
          const meta = view === "archive"
            ? `${ep.id} · dropped: ${ep.reason || "?"}`
            : `${ep.id} · ${ep.outcome} · ${ep.turns} turn(s)`
              + (ep.merged ? ` · merged ${ep.merged}` : "");
          card.appendChild(el("div", "sub", meta));
          const act = el("div", "actions");
          if (view === "archive") {
            const b = el("button", "btn-secondary", "restore");
            b.onclick = async () => {
              try {
                const out = await episodeOp(
                  "POST", `/api/coder/episodes/${encodeURIComponent(ep.id)}/restore`, cwd);
                // The caveats are surfaced, not swallowed: both describe a
                // restore that SUCCEEDED and is still not what you pictured.
                for (const n of out.notes || []) toast(n, true);
                if (!(out.notes || []).length) toast("Lesson restored");
                openEpisodesModal("archive");
              } catch (e) { toast("Restore failed: " + e.message, true); }
            };
            act.appendChild(b);
          } else {
            const b = el("button", "btn-secondary", "forget");
            b.title = "Drops it from recall. Archived, so it can be restored.";
            b.onclick = async () => {
              try {
                const out = await episodeOp(
                  "POST", `/api/coder/episodes/${encodeURIComponent(ep.id)}/forget`, cwd);
                if (out.warning) toast(out.warning, true);
                else toast("Forgotten - restore it from the dropped tab");
                openEpisodesModal("live");
              } catch (e) { toast("Forget failed: " + e.message, true); }
            };
            act.appendChild(b);
          }
          card.appendChild(act);
          body.appendChild(card);
        }
      }

      const foot = el("div", "actions");
      const cons = el("button", "btn-secondary", "consolidate");
      cons.title = "Ask the model to merge related lessons into one. Opt-in and "
        + "manual; every original is archived, so a bad merge is reversible.";
      cons.onclick = async () => {
        cons.disabled = true;
        try {
          const rep = await episodeOp("POST", "/api/coder/episodes/consolidate", cwd);
          if (!rep.groups) toast("Nothing to consolidate: no related lessons found");
          else {
            toast(`Consolidated ${rep.groups} group(s): ${rep.replaced} merged `
              + `into ${rep.merged}, ${rep.archived} archived`);
            // A group the model returned nothing usable for is COUNTED, not
            // hidden - it was left untouched rather than dropped.
            if (rep.skipped) {
              toast(`${rep.skipped} group(s) left untouched (no usable merge)`, true);
            }
          }
          if (rep.warning) toast(rep.warning, true);
          openEpisodesModal("live");
        } catch (e) {
          toast("Consolidate failed: " + e.message, true);
        } finally { cons.disabled = false; }
      };
      foot.appendChild(cons);

      const wipe = el("button", "btn-secondary btn-danger", "erase all");
      wipe.title = "Erases every lesson for this project, dropped ones included. "
        + "Not reversible.";
      wipe.onclick = () => confirmDanger(
        "Erase all lessons?",
        "This erases every lesson this project remembers, including the dropped "
        + "ones you could otherwise restore. The archive goes too, so nothing is "
        + "recoverable afterwards. This cannot be undone.",
        "Erase everything",
        async () => {
          try {
            const out = await episodeOp("DELETE", "/api/coder/episodes", cwd);
            toast(`Erased ${out.erased} lesson(s) and ${out.erased_archived} `
              + "dropped one(s)");
            openEpisodesModal("live");
          } catch (e) { toast("Erase failed: " + e.message, true); }
        });
      foot.appendChild(wipe);
      body.appendChild(foot);
    });
  } catch (e) {
    toast("Could not read lessons: " + e.message, true);
  }
}

/** Plan the composer's task without running it (the CLI's `--estimate`): one
 *  turn, no tools, nothing written. The plan arrives on the event stream, so
 *  every open tab sees it; this only has to fire the request. */
export async function estimateCoderTask() {
  const s = activeSession();
  if (!s) { toast("No active session", true); return; }
  const text = $("coder-input").value.trim();
  if (!text) { toast("Describe a task to estimate", true); return; }
  const btn = $("coder-estimate");
  btn.disabled = true;
  try {
    const r = await fetch(`/api/coder/sessions/${s.info.id}/estimate`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ text }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    // The task stays in the composer: an estimate is a pre-flight, and the
    // usual next action is to send the very same text for real.
  } catch (e) {
    toast("Estimate failed: " + e.message, true);
  } finally {
    btn.disabled = false;
  }
}

$("coder-files").onclick = openFilesModal;
$("coder-patch").onclick = downloadCoderPatch;
$("coder-estimate").onclick = estimateCoderTask;
$("setup-episodes").onclick = openEpisodesModal;
$("coder-export").onclick = () => openModal("Export session", (body) => {
  body.appendChild(el("p", "sub",
    "Markdown is the readable transcript; JSON is the last finished task's "
    + "result, the same payload localcoder --output-format json prints."));
  const row = el("div", "actions");
  const md = el("button", "btn-secondary", "markdown transcript");
  md.onclick = () => { $("modal").style.display = "none"; exportCoderSession(); };
  const js = el("button", "btn-secondary", "result JSON");
  js.onclick = () => { $("modal").style.display = "none"; exportCoderResultJson(); };
  row.appendChild(md);
  row.appendChild(js);
  body.appendChild(row);
});

$("coder-log").onclick = async () => {
  const s = activeSession();
  if (!s) return;
  const r = await fetch(`/api/coder/sessions/${s.info.id}/log`, {
    headers: authHeaders() });
  const data = await r.json();
  if (!r.ok) { toast(data.detail || "No log available", true); return; }
  showAuditModal("Audit log - " + sessionLabel(s.info), data);
};

/** Past coder sessions: audit logs left behind by log/full-mode sessions,
 *  including ones from before a server restart. */
export async function openSessionHistory() {
  let data = null;
  try {
    const r = await fetch("/api/coder/history", { headers: authHeaders() });
    if (r.ok) data = await r.json();
  } catch (e) { /* handled below */ }
  if (!data) { toast("Could not load session history", true); return; }
  openModal("Past coder sessions", (body) => {
    if (data.authorized === false) {
      body.appendChild(el("div", "sub",
        "Past coder sessions are private to the server owner. Sign in with the " +
        "owner API key on this device to view them."));
      return;
    }
    if (!data.enabled) {
      body.appendChild(el("div", "sub",
        "New sessions are not being recorded (privacy mode). Anything below " +
        "is from earlier log/full-mode sessions."));
    }
    if (!data.logs.length) {
      body.appendChild(emptyState("book", "No session logs yet",
        "Start a session with persistence set to log or full to keep an audit trail here."));
      return;
    }
    for (const item of data.logs) {
      const row = el("div", "log-entry clickable");
      const when = new Date(item.mtime * 1000).toLocaleString();
      const kb = (item.size_bytes / 1024).toFixed(1);
      row.appendChild(el("span", "t", when));
      row.appendChild(document.createTextNode(`${item.name} (${kb} KB)`));
      row.onclick = async () => {
        try {
          const r = await fetch(
            "/api/coder/history/" + encodeURIComponent(item.name),
            { headers: authHeaders() });
          const entries = await r.json();
          if (!r.ok) throw new Error(entries.detail || r.statusText);
          showAuditModal("Session - " + item.name, entries);
        } catch (e) {
          toast("Could not open log: " + e.message, true);
        }
      };
      body.appendChild(row);
    }
  });
}

$("coder-history").onclick = openSessionHistory;
$("setup-history").onclick = openSessionHistory;

