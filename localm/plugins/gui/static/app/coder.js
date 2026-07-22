// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - coder - multi-session (split from app.js). Classic script: it
   shares the one global lexical environment with the other app/* and
   pages/* scripts, so every cross-section reference resolves by bare
   name exactly as before. */
"use strict";

// --- ES module imports (auto-generated boundary; bodies unchanged) ---
import { addMessageRow, chat } from "./chat.js";
import { $, authHeaders, autoGrow, el, nearBottom, openModal, readSSE, renderMarkdown, toast } from "./helpers.js";
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
export function renderCoderSessionList() {
  const list = $("coder-session-list");
  if (!list) return;
  list.replaceChildren();
  if (!coder.sessions.size) {
    list.appendChild(el("div", "coder-session-empty", "No open sessions"));
    return;
  }
  for (const [id, s] of coder.sessions) {
    const item = el("div", "coder-session-item" + (id === coder.activeId ? " active" : ""));
    item.appendChild(el("span", "title", sessionLabel(s.info)));
    if (s.busy) item.appendChild(el("span", "badge", "⏳"));
    item.onclick = () => activateSession(id);
    list.appendChild(item);
  }
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
    $("coder-state").textContent = "";
    $("coder-usage").textContent = "";
    renderSessionSelect();
    refreshResumable();   // reveal "Continue last session" if the cwd has one (CODER-2)
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
    $("coder-state").textContent = s.busy ? "working…" : "idle";
    $("coder-usage").textContent = s.info.total_tokens
      ? `${s.info.total_tokens} tok · turn ${s.info.turns}` : "";
  }
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

export function buildToolCard(ev) {
  const card = el("div", "tool-card");
  card.dataset.t0 = String(Date.now());
  const inner = el("div", "inner");
  const head = el("div", "head");
  head.appendChild(el("span", "name", ev.tool));
  const hintVal = ev.args?.path || ev.args?.command || ev.args?.pattern || ev.args?.url || "";
  head.appendChild(el("span", "hint", String(hintVal).slice(0, 120)));
  head.appendChild(el("span", "state", "…"));
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
  entry.title.textContent = timedOut
    ? "✗ Timed out - rejected " + entry.tool
    : (approved ? "✓ Approved " : "✗ Rejected ") + entry.tool;
}

export function buildConfirmCard(s, ev) {
  const card = el("div", "confirm-card");
  const inner = el("div", "inner");
  const title = el("div", "title");
  title.appendChild(document.createTextNode("Approve "));
  title.appendChild(el("span", "name", ev.tool));
  title.appendChild(document.createTextNode("?"));
  inner.appendChild(title);
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
    case "turn": {
      flushAssistantBlock(s);
      s.busy = true;
      s.info.turns = ev.turn;
      s.info.total_tokens = ev.total_tokens;
      if (s.info.id === coder.activeId) {
        $("coder-state").textContent = "working…";
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
        const t0 = Number(card.dataset.t0 || 0);
        const took = t0 ? ` · ${((Date.now() - t0) / 1000).toFixed(1)}s` : "";
        state.textContent = (ev.summary || (ev.ok ? "ok" : "failed")) + took;
        state.className = "state " + (ev.ok ? "ok" : "fail");
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
      if (s.info.id === coder.activeId) $("coder-state").textContent = "idle";
      renderSessionSelect();
      let finalLine = (ev.ok ? "Task finished" : "Task ended") +
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
      if (s.info.id === coder.activeId) $("coder-state").textContent = "error";
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
  const cwd = $("setup-cwd").value.trim();
  if (!cwd) { toast("Enter a project directory", true); return; }
  $("setup-start").disabled = true;
  try {
    const body = {
      cwd,
      auto_approve: $("setup-auto").checked,
      dry_run: $("setup-dry").checked,
      mode: $("setup-mode").value,
      max_turns: Number($("setup-max-turns").value) || 40,
      resume,
    };
    const model = $("setup-model").value;
    if (model) body.model = model;
    const temp = $("setup-temperature").value.trim();
    if (temp !== "") body.temperature = Number(temp);
    const scope = $("setup-scope").value.trim();
    if (scope) body.scope = scope;
    const system = $("setup-system").value.trim();
    if (system) body.custom_instructions = system;

    const r = await fetch("/api/coder/sessions", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const info = await r.json();
    if (!chat.privacy) localStorage.setItem("localm.coderCwd", cwd);
    // A resumed session replays its restored recap from the server; a fresh one
    // has no history to replay (CODER-2).
    registerSession(info, { replay: !!info.resumed });
    activateSession(info.id);
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
    chip.appendChild(el("span", "", "📄 " + doc.name +
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
      $("coder-state").textContent = "working…";
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
// Arrow wrapper: a bare `.onclick = startCoderSession` would pass the click
// Event as opts, making opts.resume truthy and always resuming (CODER-2).
$("setup-start").onclick = () => startCoderSession();
// Probe for a resumable checkpoint as the directory changes (debounced).
export let _resumeProbeTimer = null;
$("setup-cwd").addEventListener("input", () => {
  clearTimeout(_resumeProbeTimer);
  _resumeProbeTimer = setTimeout(refreshResumable, 350);
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
    if (!chat.privacy) localStorage.setItem("localm.coderCwd", dir);
    refreshResumable();   // setting .value does not fire 'input' (CODER-2)
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
    if (!data.entries.length) body.appendChild(el("div", "sub", "(empty)"));
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
      body.appendChild(el("div", "sub", "No files changed this session."));
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

$("coder-files").onclick = openFilesModal;
$("coder-export").onclick = exportCoderSession;

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
      body.appendChild(el("div", "sub",
        "No session logs yet - start a session with persistence set to " +
        "log or full, and its audit trail will appear here."));
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

