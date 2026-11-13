// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - models sidebar selector. */
"use strict";

// --- ES module imports ---
import { chat, lsSetScoped } from "./chat.js";
import { recordGuiError } from "./client-log.js";
import { $, GIB, SHELL_TOKEN, authHeaders, cancelJob, chooseAsync, confirmDangerAsync, el, fmtDuration, instanceCacheTrusted, openModal, readSSE, streamJob, toast } from "./helpers.js";
import { applyI18n, t, tn } from "./i18n.js";
import { interruptedCount, loadInterrupted, onInterruptedChange, renderInterrupted } from "./interrupted-pulls.js";
import { refreshPerfEstimate } from "./settings-perf.js";

export const modelSelect = $("model-select");
export const sidebarUnloadBtn = $("sidebar-unload-btn");

// Whether the status pill is currently showing a caller's busy state (e.g.
// switchModel's "loading X…"). refreshModels()'s periodic "ok" write is gated on
// this so it does not clobber one.
let _statusBusy = false;

// True while the busy state on the pill was written by refreshModels() itself
// (the server reported a load in flight that this page did not start).
let _statusFromPoll = false;
// A failed load keeps its "load failed" pill through the model re-read that
// follows it, until the next load or unload starts or the active model changes.
let _holdLoadFailed = false;
// The model switchModel() is loading, "" when none; modelCache.loading carries
// it until the load ends, whatever the polled list says.
let _switchingTo = "";
let _loadingPollTimer = null;
const LOADING_POLL_MS = 2000;

// State renders as a .job-state pill; busy maps to st-running. "note" is an ok
// state the pill keeps visible (a chat routed to another instance).
const STATUS_STATE_CLASS = { ok: "st-ok", note: "st-ok", busy: "st-running", err: "st-error" };

export function setStatus(state, text) {
  _statusBusy = state === "busy";
  if (state === "busy") _holdLoadFailed = false;
  _statusFromPoll = false;
  const pill = $("status-text");
  // An error is recorded when it newly appears, not on every repeat of it.
  const repeat = pill.classList.contains(STATUS_STATE_CLASS.err) && pill.textContent === text;
  pill.className = "job-state " + (STATUS_STATE_CLASS[state] || "");
  pill.textContent = text;
  if (state === "err" && !repeat) recordGuiError(text, "status");
  // The pill is hidden for "ok" alone: the dropdown above already renders the
  // active model's name, or its disabled "No model loaded" placeholder. "busy"
  // and "err" stay visible - neither is expressible in the select, and "err" is
  // the only surface for "server unreachable", "load failed", "unload failed",
  // "models unavailable (HTTP n)" and "page out of date".
  //
  // The text and class are written first and unconditionally, so a hidden pill
  // still holds the last state for anything that reads it.
  const box = $("model-status");
  if (box) box.hidden = state === "ok";
}

// Live hardware monitor in the status bar (CPU/RAM/VRAM/GPU). Renders whatever
// /api/stats reports; any section the box cannot measure is simply absent (no
// psutil -> no CPU/RAM). VRAM shows used/total when the used figure is known,
// and an explicitly labelled "N GB total" when it is not.
export function renderHwStats(data) {
  const el = $("hw-stats");
  if (!el) return;
  const gib = (b) => (b / GIB).toFixed(1);
  // Each metric is its own <span> so the VRAM figure can carry a fullness
  // colour. That colour is applied only to a used/total reading (the backend
  // sends `used` only for a fresh, device-global one - see sysstats._vram); a
  // total-only reading gets none.
  const spans = [];
  // `metric` pins the span to a GRID COLUMN in CSS (load left, memory right).
  // See the .hw-stats rule.
  const add = (metric, text, cls) => {
    const s = document.createElement("span");
    s.textContent = text;
    s.dataset.metric = metric;
    if (cls) s.className = cls;
    spans.push(s);
  };
  // ORDER: CPU, RAM, GPU, VRAM - two pairs, each "load, then memory", laid out
  // as a 2x2 grid (system on row 1, graphics card on row 2). Each span is pinned
  // to its column by `data-metric`, so an absent metric leaves its cell empty
  // instead of pulling the next one into the wrong column.
  if (data && data.cpu && typeof data.cpu.percent === "number")
    add("cpu", `CPU ${Math.round(data.cpu.percent)}%`);
  // RAM as used/total GB, matching the VRAM figure below it. sysstats sends
  // used/total/percent together or omits `ram` entirely; the percent branch is
  // the fallback for a partial payload.
  if (data && data.ram) {
    const r = data.ram;
    if (r.used != null && r.total) add("ram", `RAM ${gib(r.used)}/${gib(r.total)} GB`);
    else if (typeof r.percent === "number") add("ram", `RAM ${Math.round(r.percent)}%`);
  }
  // The aggregate GPU% is emitted ONLY on a single-card board: /api/stats sends
  // one system-wide figure (nvidia-smi, NVIDIA only) with no card attribution.
  // On a multi-card board the per-card rows below stand in for it.
  const multi = !!(data && data.vram && Array.isArray(data.vram.devices)
                   && data.vram.devices.length > 1);
  if (!multi && data && data.gpu && typeof data.gpu.percent === "number")
    add("gpu", `GPU ${Math.round(data.gpu.percent)}%`);
  const band = (used, total) => {
    const frac = total ? used / total : 0;
    return frac >= 0.9 ? "vram-full" : frac >= 0.7 ? "vram-busy" : "vram-ok";
  };
  if (data && data.vram && data.vram.total) {
    const v = data.vram;
    // MULTI-GPU: one row PER CARD, never the combined figure alone. The backend
    // sends `devices` only when there is more than one card, so the single-card
    // path below is unaffected.
    if (multi) {
      v.devices.forEach((d, i) => {
        const label = d.index == null ? `GPU ${i}` : `GPU ${d.index}`;
        // Column 1 carries the card's IDENTITY, not a utilisation percent:
        // /api/stats reports gpu.percent as ONE system-wide figure (nvidia-smi,
        // NVIDIA only), which belongs to no single card.
        add("gpu", label);
        if (d.used != null) {
          add("vram", `${gib(d.used)}/${gib(d.total)} GB`, `vram-usage ${band(d.used, d.total)}`);
        } else {
          // The "total" word is required: the VRAM label means used/total
          // everywhere else, so a bare figure would read as a full card.
          add("vram", t("sidebar.hw.gbTotal", { gb: gib(d.total) }));
        }
      });
    } else if (v.used != null) {
      add("vram", `VRAM ${gib(v.used)}/${gib(v.total)} GB`, `vram-usage ${band(v.used, v.total)}`);
    } else {
      // The "total" word is required here for the same reason as the per-card
      // branch above.
      add("vram", t("sidebar.hw.vramGbTotal", { gb: gib(v.total) }));
    }
  }
  el.textContent = "";
  if (spans.length) {
    // No separator TEXT NODES between the spans: the row wraps now (it used to
    // ellipsise and cut the VRAM figure off), and a " · " text node is exactly
    // the thing that gets stranded at the end of a wrapped line. The separator
    // is drawn in CSS as `span + span::before`, so it lives inside the metric
    // it introduces and wraps with it.
    spans.forEach((s) => el.appendChild(s));
    el.hidden = false;
  } else {
    el.hidden = true;
  }
}

export async function pollHwStats() {
  if (typeof document !== "undefined" && document.hidden) return;
  if (isLiveOpen()) return;   // the open stream keeps the readout current
  try {
    const r = await fetch("/api/stats", { headers: authHeaders() });
    if (!r.ok) return;
    const data = await r.json();
    if (isLiveOpen()) return;   // the stream opened while this was in flight; its reading is newer
    renderHwStats(data);
  } catch { /* transient - keep the last reading */ }
}

// --------------------------------------------------------------------------- //
//  Live channel: GET /api/live pushes the hardware stats and the activity     //
//  list as they change, so the meter follows a model load or unload within    //
//  the second instead of waiting out a 2.5 s poll. Polling (below) is the     //
//  fallback whenever the stream is not open.                                  //
// --------------------------------------------------------------------------- //

export let LIVE_BACKOFF_MIN_MS = 1000;
export let LIVE_BACKOFF_MAX_MS = 30000;
// No frame at all, keepalive comments included, for this long means the
// connection is dead without having closed (the server sends one at least every
// 15 s).
export let LIVE_STALL_MS = 40000;

let _liveAbort = null;         // AbortController of the open or opening stream
let _liveRetryTimer = null;
let _liveBackoffMs = LIVE_BACKOFF_MIN_MS;
let _liveFellBackLogged = false;
let _liveWanted = false;        // startHwStats() has run and the tab is visible
let _liveIntervalMs = 2500;

export let _hwStatsTimer = null;

function _startPolling() {
  if (_hwStatsTimer) return;
  _hwStatsTimer = setInterval(() => { pollHwStats(); pollActivity(); }, _liveIntervalMs);
}

function _stopPolling() {
  if (_hwStatsTimer) { clearInterval(_hwStatsTimer); _hwStatsTimer = null; }
}

export function isLiveOpen() {
  return _liveAbort !== null && _liveAbort.opened === true;
}

function _applyLiveFrame(ev) {
  if (ev.type === "stats") renderHwStats(ev.data);
  else if (ev.type === "activity") _applyActivity(ev);
}

function _noteLiveFallback(why) {
  if (_liveFellBackLogged) return;
  _liveFellBackLogged = true;
  console.warn(`localm: the live stats stream is unavailable (${why}); polling /api/stats instead and retrying the stream`);
}

function _scheduleLiveRetry() {
  if (_liveRetryTimer || !_liveWanted) return;
  _liveRetryTimer = setTimeout(() => { _liveRetryTimer = null; _openLive(); }, _liveBackoffMs);
  _liveBackoffMs = Math.min(_liveBackoffMs * 2, LIVE_BACKOFF_MAX_MS);
}

// Open the stream and read it until it ends. Never throws. When it cannot be
// opened, or ends without this page having asked it to, polling takes over and
// a retry is scheduled with a growing delay; a stream that delivered frames
// resets that delay.
async function _openLive() {
  if (_liveAbort || !_liveWanted) return;
  if (typeof document !== "undefined" && document.hidden) return;
  const ac = new AbortController();
  ac.opened = false;
  _liveAbort = ac;
  let watchdog = null;
  const arm = () => {
    if (watchdog) clearTimeout(watchdog);
    watchdog = setTimeout(() => ac.abort(), LIVE_STALL_MS);
  };
  let why = "the stream ended";
  try {
    const r = await fetch("/api/live", { headers: authHeaders(), signal: ac.signal });
    if (!r.ok || !r.body) throw new Error(r.ok ? "no response body" : `HTTP ${r.status}`);
    arm();
    await readSSE(r, (payload) => {
      let ev;
      try { ev = JSON.parse(payload); } catch { return; }
      if (!ac.opened && ev.type === "stats") {
        ac.opened = true;
        _liveBackoffMs = LIVE_BACKOFF_MIN_MS;
        _stopPolling();
      }
      try {
        _applyLiveFrame(ev);
      } catch (e) {
        console.error("localm: could not render a live frame:", e);
      }
    }, arm);
  } catch (e) {
    why = ac.intentional ? "" : String((e && e.message) || e);
    if (!ac.intentional && ac.signal.aborted) why = `no data for ${LIVE_STALL_MS / 1000}s`;
  } finally {
    if (watchdog) clearTimeout(watchdog);
    if (_liveAbort === ac) _liveAbort = null;
  }
  if (ac.intentional || !_liveWanted) return;
  _noteLiveFallback(why);
  _startPolling();
  _scheduleLiveRetry();
}

function _closeLive() {
  _liveWanted = false;
  if (_liveRetryTimer) { clearTimeout(_liveRetryTimer); _liveRetryTimer = null; }
  if (_liveAbort) { _liveAbort.intentional = true; _liveAbort.abort(); }
}

let _hwStarted = false;

export function startHwStats(intervalMs = 2500) {
  if (_hwStarted) return;
  _hwStarted = true;
  _liveIntervalMs = intervalMs;
  pollHwStats();
  pollActivity();   // ADR-0008 U4: folded into this existing poll, not a new timer
  loadInterrupted();
  _liveWanted = typeof document === "undefined" || !document.hidden;
  _liveBackoffMs = LIVE_BACKOFF_MIN_MS;
  _startPolling();
  _openLive();
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      _closeLive();
      return;
    }
    pollHwStats(); pollActivity();   // refresh promptly on tab focus
    if (!_liveWanted) {
      _liveWanted = true;
      _liveBackoffMs = LIVE_BACKOFF_MIN_MS;
      _startPolling();
      _openLive();
    }
  });
}

// --------------------------------------------------------------------------- //
//  ADR-0008 U4: cross-session/cross-tab activity. A persistent status-bar     //
//  affordance for background operations this tab did not necessarily start   //
//  (another browser session, or this same tab before a reload) - the         //
//  maintainer's own reported case. Polled on the SAME tick as pollHwStats     //
//  above (no new timer) and reattached at boot, mirroring coder.js's         //
//  reattachSessions()/streamSession(). Deliberately does NOT drive the       //
//  status DOT (setStatus) - only the pill below - to avoid inventing new,    //
//  unrequested semantics about which kinds of background work should read as
//  "the model is busy"; the _statusBusy fix above is what lets a future
//  caller do that safely, not a decision this unit makes for every op kind.
// --------------------------------------------------------------------------- //

// job_id -> {lines: string[]} - lines accumulated from a background streamJob()
// reattach, so the details modal can show "the output it missed" even for an
// operation whose page-specific progress UI was never open in this tab.
const _activityLogs = new Map();
let _activityOps = [];          // the last successfully-read /api/activity list
// The SERVER's clock (epoch seconds) at that same read. Ages are rendered as
// `_activityServerNow - op.created_at`, NEVER against this browser's own
// clock - the /api/activity route's own docstring is explicit that a client
// clock (a phone, a drifted box) disagrees by real amounts, and durations
// are exactly what created_at exists to make renderable.
let _activityServerNow = null;
// True once at least one read has SUCCEEDED. Distinguishes "have not asked
// yet" (boot, before the first poll resolves) from "asked, and there is
// genuinely nothing running" (a real, confirmed answer) - collapsing the two
// into one rendering is the same defect class as loop_lag's fabricated 0.00,
// just in the app shell: an unasked question must never read as a "no".
let _activityKnown = false;

// Consecutive FAILED /api/activity reads since the last success. A single
// dropped poll is routine at a 2.5s tick and must not flap the pill; once
// reads have failed this many times in a row the last-known state has had
// time to go stale, so every surface must say so (#1078 post-merge review,
// AC4) instead of silently keeping (or silently dropping) a claim nobody
// has actually confirmed lately.
const _ACTIVITY_STALE_AFTER_READS = 3;
let _activityStaleReads = 0;
// Server stall records from the last /api/activity read, and the ids already
// toasted. On the first read, a stall that ended more than
// _STALL_TOAST_RECENT_S before the server's "now" counts as already seen.
let _activityStalls = [];
const _seenStallIds = new Set();
let _stallsPrimed = false;
const _STALL_TOAST_RECENT_S = 120;

// True while a stall that ended in the last _STALL_PILL_S seconds has not been
// seen in the Activity window yet.
const _STALL_PILL_S = 600;
const _stallsShown = new Set();
function _unseenStall() {
  if (typeof _activityServerNow !== "number") return false;
  return _activityStalls.some((s) => !s.ongoing && !_stallsShown.has(s.id)
    && _activityServerNow - ((s.at || 0) + (s.seconds || 0)) < _STALL_PILL_S);
}

function _stallText(s) {
  return t(s.kind === "loop" ? "activity.stall.loop" : "activity.stall.gil",
    { seconds: Math.round(s.seconds), where: s.where });
}

// Toast each stall that has ended and was not toasted before.
function _noteStalls(stalls, serverNow) {
  _activityStalls = stalls;
  for (const s of stalls) {
    if (s.ongoing || _seenStallIds.has(s.id)) continue;
    _seenStallIds.add(s.id);
    const ended = (s.at || 0) + (s.seconds || 0);
    if (!_stallsPrimed && typeof serverNow === "number"
        && serverNow - ended > _STALL_TOAST_RECENT_S) continue;
    toast(_stallText(s), true);
  }
  _stallsPrimed = true;
}

export function isActivityBusy() {
  return _activityOps.some((op) => op.status === "running");
}

function activityLabel(op) {
  return (op && (op.label || op.kind)) || "operation";
}

function activityState(op) {
  return op.status === "running" && op.phase === "queued" ? "queued" : op.status;
}

// A compact "12m" / "3m ago" string, or "" when age cannot be computed (no
// server clock yet, or a malformed created_at). No "running " prefix for a
// running op - the adjacent .job-state pill already says "running"; verified
// live that a prefix there reads as "runningrunning 21s" (#1078 follow-up).
function activityAge(op) {
  if (_activityServerNow == null || typeof op.created_at !== "number") return "";
  // A finished op's age is "how long since it finished", not "how long since
  // it was created" - the two can differ by hours for a long pull. Use
  // finished_at (server epoch seconds, jobs.py Job.summary()) when the op is
  // no longer running and actually has one; fall back to created_at only if
  // it does not, which still renders SOMETHING rather than nothing (#1078
  // post-merge review: this previously always used created_at for both
  // states, so a two-hour pull that finished 5s ago read as "2h 00m ago").
  const stamp = op.status !== "running" && typeof op.finished_at === "number"
    ? op.finished_at : op.created_at;
  const d = fmtDuration(_activityServerNow - stamp);
  if (!d) return "";
  return op.status === "running" ? d : `${d} ago`;
}

// "2 running", "1 running, 1 queued" or "2 queued" for the running-status
// *ops*, counting an op still waiting its turn as queued.
function activityCountText(ops) {
  const queued = ops.filter((op) => activityState(op) === "queued").length;
  const active = ops.length - queued;
  return [active ? `${active} running` : "", queued ? `${queued} queued` : ""]
    .filter(Boolean).join(", ");
}

export function renderActivityPill(ops) {
  const pill = $("activity-pill");
  if (!pill) return;
  const stale = _activityKnown && _activityStaleReads >= _ACTIVITY_STALE_AFTER_READS;
  const running = ops.filter((op) => op.status === "running");
  if (!running.length && !stale) {
    pill.classList.remove("st-unknown");
    pill.classList.add("st-running");
    const interrupted = interruptedCount();
    if (interrupted) {
      pill.style.display = "";
      pill.title = t("activity.clickForDetails");
      pill.textContent = tn("activity.interruptedPill", interrupted);
    } else if (_unseenStall()) {
      pill.style.display = "";
      pill.title = t("activity.clickForDetails");
      pill.textContent = t("activity.stallPill");
    } else {
      pill.style.display = "none";
    }
    return;
  }
  pill.style.display = "";
  if (stale) {
    // The last several reads have failed. Showing nothing here (or silently
    // leaving a running-op pill up forever) repeats the fabricated-answer
    // mistake this feature exists to remove, just staged behind staleness
    // instead of a first-ever failure (#1078 post-merge review, AC4: "when
    // the state cannot be read, every surface says so"). Reuses the existing
    // st-unknown pill styling (already used for "type could not be
    // determined") rather than inventing new CSS for the same meaning.
    pill.classList.add("st-unknown");
    pill.classList.remove("st-running");
    pill.title = t("activity.staleTitle");
    const label = running.length === 1 ? activityLabel(running[0])
      : running.length > 1 ? activityCountText(running) : null;
    pill.textContent = label ? t("activity.statusUnknownFor", { label }) : t("activity.statusUnknown");
    return;
  }
  pill.classList.remove("st-unknown");
  pill.classList.add("st-running");
  pill.title = t("activity.clickForDetails");
  if (running.length === 1) {
    const op = running[0];
    const pct = typeof op.pct === "number" ? ` ${Math.round(op.pct)}%` : "";
    pill.textContent = activityState(op) === "queued"
      ? t("activity.queuedFor", { label: activityLabel(op) })
      : activityLabel(op) + pct;
  } else {
    pill.textContent = activityCountText(running);
  }
}

// Shared with the live-refresh path below so the modal never diverges from
// what openModal() rendered at open time.
function _buildActivityBody(body) {
  if (!_activityKnown) {
    // Never render "Nothing running." for a question that was never
    // successfully asked (R1) - this only shows before the very first
    // poll/reattach resolves, or if every attempt so far has failed.
    body.appendChild(el("p", "sub", t("activity.checking")));
    return;
  }
  if (_activityStaleReads >= _ACTIVITY_STALE_AFTER_READS) {
    body.appendChild(el("p", "sub", t("activity.staleBody")));
  }
  if (!_activityOps.length) body.appendChild(el("p", "sub", t("activity.nothingRunning")));
  for (const op of _activityOps) {
    const row = el("div", "job-row");
    const head = el("div", "job-head");
    head.appendChild(el("span", "job-name", activityLabel(op)));
    head.appendChild(el("span", "job-state st-" + op.status, activityState(op)));
    const age = activityAge(op);
    if (age) head.appendChild(el("span", "sub", age));
    if (op.status === "running" && op.cancellable) {
      const stop = el("button", "btn-secondary", t("activity.stop"));
      stop.title = t("activity.stopTitle");
      stop.onclick = async () => {
        stop.disabled = true;
        await cancelJob(op.id);
        await pollActivity();
      };
      head.appendChild(stop);
    }
    row.appendChild(head);
    if (op.status === "running" && op.detail) row.appendChild(el("div", "sub", op.detail));
    if (typeof op.pct === "number") {
      const dl = el("div", "dl-progress");
      const bar = el("div", "dl-bar");
      const fill = el("div", "dl-fill");
      fill.style.width = Math.max(0, Math.min(100, op.pct)) + "%";
      bar.appendChild(fill);
      dl.appendChild(bar);
      row.appendChild(dl);
    }
    const entry = _activityLogs.get(op.id);
    if (entry && entry.lines.length) {
      const log = el("pre", "job-log");
      log.textContent = entry.lines.join("\n");
      row.appendChild(log);
    }
    body.appendChild(row);
  }
  const list = el("div", "interrupted-list");
  if (renderInterrupted(list) || list.childNodes.length) {
    body.appendChild(el("h4", "", t("activity.interruptedHead")));
    body.appendChild(list);
  }
  if (_activityStalls.length) {
    body.appendChild(el("h4", "", t("activity.stallsHead")));
    for (const s of _activityStalls.slice().reverse()) {
      const row = el("div", "job-row");
      row.appendChild(el("div", "", _stallText(s)));
      if (Array.isArray(s.frames) && s.frames.length) {
        const pre = el("pre", "job-log");
        pre.textContent = s.frames.join("\n");
        row.appendChild(pre);
      }
      body.appendChild(row);
    }
  }
}

onInterruptedChange(() => {
  renderActivityPill(_activityOps);
  _refreshActivityModalIfOpen();
});

export function showActivityDetails() {
  for (const s of _activityStalls) _stallsShown.add(s.id);
  renderActivityPill(_activityOps);
  openModal(t("activity.modalTitle"), _buildActivityBody);
}
if ($("activity-pill")) $("activity-pill").onclick = showActivityDetails;

// openModal() calls its bodyBuilder exactly ONCE, at open time (helpers.js) -
// without this, a poll that changes op state while the modal is open left it
// showing a frozen snapshot that silently disagreed with the pill, which is
// worse than either alone (#1078 post-merge review). Keyed on the modal's
// title text since openModal has no per-modal-kind tracking to hook into;
// a false match is not possible for any modal opened elsewhere in the app
// (rename/confirm/etc. all use their own distinct titles), and if a DIFFERENT
// modal is opened on top the title changes, so this naturally becomes a
// no-op instead of writing into a modal something else now owns.
function _refreshActivityModalIfOpen() {
  const modal = $("modal");
  const title = $("modal-title");
  if (!modal || modal.style.display !== "flex") return;
  if (!title || title.textContent !== t("activity.modalTitle")) return;
  const body = $("modal-body");
  if (!body) return;
  body.textContent = "";
  _buildActivityBody(body);
}

// Shared read: GET /api/activity, update the module's known state + pill,
// and return the parsed operations list - or null on an unreadable response
// (R1: the caller must keep whatever it last knew, never fabricate "nothing
// running"; _activityKnown/_activityOps/_activityServerNow are simply left
// untouched on failure, matching pollHwStats' own "transient - keep the last
// reading" precedent).
async function _readActivity() {
  try {
    const r = await fetch("/api/activity", { headers: authHeaders() });
    if (!r.ok) return _activityReadFailed();
    const data = await r.json();
    if (_activityKnown && isLiveOpen()) return _activityOps;   // the stream's list is the newer one
    return _applyActivity(data);
  } catch {
    return _activityReadFailed();
  }
}

// One /api/activity body, from a poll or a pushed frame: record it, redraw the
// pill and the open details modal, and return the operations list.
function _applyActivity(data) {
  const ops = Array.isArray(data.operations) ? data.operations : [];
  _activityOps = ops;
  _activityServerNow = typeof data.now === "number" ? data.now : null;
  _activityKnown = true;
  _activityStaleReads = 0;
  _noteStalls(Array.isArray(data.stalls) ? data.stalls : [], _activityServerNow);
  renderActivityPill(ops);
  _refreshActivityModalIfOpen();
  return ops;
}

// A failed read never overwrites _activityOps/_activityKnown (R1 - the last
// known state, once genuinely known, is never discarded just because one
// poll could not confirm it), but DOES count toward surfacing that the state
// may now be stale (AC4 - a caller must be TOLD reads stopped working, not
// just left staring at an answer that quietly stopped being current). Always
// returns null so call sites can `return _activityReadFailed();` in place of
// `return null;`.
function _activityReadFailed() {
  _activityStaleReads += 1;
  if (_activityKnown) {
    renderActivityPill(_activityOps);
    _refreshActivityModalIfOpen();
  }
  return null;
}

export async function pollActivity() {
  if (typeof document !== "undefined" && document.hidden) return;
  await _readActivity();
}

// Boot-time reattach, mirroring coder.js's reattachSessions(): ask the server
// what is running, reattach a live stream to anything found (fire-and-forget -
// an operation can run far longer than boot should wait), and toast once.
// streamJob() already handles reconnect + replay dedup internally - this
// needs no dedup logic of its own, it only needs to call streamJob() once
// per running operation.
export async function reattachActivity() {
  try {
    const ops = await _readActivity();
    if (!ops) return;   // server unreachable at boot - same as reattachSessions()
    const running = ops.filter((op) => op.status === "running");
    if (!running.length) return;
    for (const op of running) {
      _activityLogs.set(op.id, { lines: [] });
      streamJob(op.id, (line) => {
        const entry = _activityLogs.get(op.id);
        if (entry) entry.lines.push(line);
        _refreshActivityModalIfOpen();
      }, (ev) => {
        const idx = _activityOps.findIndex((o) => o.id === op.id);
        if (idx !== -1 && (typeof ev.pct === "number" || ev.detail)) {
          _activityOps[idx] = { ..._activityOps[idx], pct: ev.pct, detail: ev.detail };
          renderActivityPill(_activityOps);
          _refreshActivityModalIfOpen();
        }
      }).then((end) => {
        // "disconnected" is a CLIENT-only concept (streamJob gave up
        // reconnecting) - it is not a real job outcome, so it must never
        // overwrite the last known SERVER-reported status; the job may well
        // still be running, we have just lost our own update channel to it.
        if (end.status === "disconnected") return;
        const idx = _activityOps.findIndex((o) => o.id === op.id);
        if (idx !== -1) {
          _activityOps[idx] = { ..._activityOps[idx], status: end.status };
          renderActivityPill(_activityOps);
          _refreshActivityModalIfOpen();
        }
      });
    }
    toast(running.length === 1
      ? t("activity.reattachedOne", { label: activityLabel(running[0]) })
      : t("activity.reattachedMany", { count: running.length }));
  } catch { /* server unreachable at boot - same as reattachSessions() */ }
}

// In-page API-key gate. Shown when an authed boot returns 401 and this browser
// has no working key - the network/phone case, where the loopback key is never
// auto-seeded. Replaces window.prompt() (suppressed by mobile/PWA browsers, the
// NET-1 white-page cause). Idempotent.
export function showKeyGate(message) {
  const gate = $("key-gate");
  if (!gate) return;
  if (message) {
    const m = $("key-gate-msg");
    if (m) { m.textContent = message; delete m.dataset.i18n; }
  }
  gate.style.display = "flex";
  // Show the "Install certificate" step only when the local CA is genuinely NOT
  // trusted yet (see updateKeyGateCertStep), so a returning trusted device is
  // never told to reinstall it each time the gate appears.
  updateKeyGateCertStep();
  // Offer "Scan QR code" wherever the browser can open a camera (secure context).
  // Decoding uses the native BarcodeDetector when present, else the bundled jsQR
  // fallback, so it is not limited to Android Chrome.
  const scan = $("key-gate-scan");
  if (scan) scan.style.display = scanSupported() ? "inline-block" : "none";
  const input = $("key-gate-input");
  if (input) {
    input.value = "";   // HttpOnly key is unreadable; the gate only shows unauthed
    input.type = "password";
    const reveal = input.parentNode && input.parentNode.querySelector(".reveal-btn");
    if (reveal) reveal.textContent = t("keygate.show");
    input.focus();
  }
}

/** Show *message* as the key gate's error line, or clear it with "". */
export function setKeyGateError(message) {
  let line = $("key-gate-error");
  if (!line) {
    const input = $("key-gate-input");
    const anchor = input && (input.closest(".input-reveal") || input);
    if (!anchor || !anchor.parentNode) return;
    line = el("div", "key-gate-error");
    line.id = "key-gate-error";
    line.setAttribute("role", "alert");
    anchor.parentNode.insertBefore(line, anchor.nextSibling);
  }
  line.textContent = message || "";
  line.hidden = !message;
}

// Decide whether the key gate should offer "Install certificate".
export function updateKeyGateCertStep() {
  const cert = $("key-gate-cert");
  if (!cert) return;
  const onHttps = location.protocol === "https:";
  const untrusted = onHttps && window.__swFailed !== false;
  cert.style.display = untrusted ? "block" : "none";
  if (untrusted) {
    // Pin the download to an absolute https URL so it never resolves to http on
    // the TLS port, where portmux answers with a 308 + HTML catch page that an
    // <a download> would save as the "cert" (J2).
    const certLink = $("key-gate-cert-link");
    if (certLink) certLink.href = "https://" + location.host + "/localm-ca.crt";
    // Firefox keeps its OWN certificate store and ignores the OS one, so a system
    // install still warns here (the common "I installed it but it still warns"
    // case). Surface the Firefox-specific step prominently.
    const ff = $("key-gate-cert-ff");
    if (ff) {
      const isFirefox = /firefox\//i.test(navigator.userAgent || "");
      ff.style.display = isFirefox ? "block" : "none";
    }
  }
}
window.updateKeyGateCertStep = updateKeyGateCertStep;

// POST the entered key to /api/session so the server sets the HttpOnly session
// cookie (the key never lives in JS) and returns the session CSRF token. Stash
// the token so a write issued before the reload already has it (the boot
// re-fetches it via refreshCsrf too), then reload to re-run authenticated.
//
// Resolves {ok, status, detail}: status is the HTTP status (0 when the server
// could not be reached) and detail the server's refusal text ("" when it sent
// none). Never rejects.
export async function loginWithKeyResult(key) {
  try {
    const r = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    });
    let body = null;
    try { body = await r.json(); } catch { /* body optional */ }
    if (r.ok) window.__LOCALM_CSRF__ = (body && body.csrf) || "";
    const detail = body && typeof body.detail === "string" ? body.detail : "";
    return { ok: r.ok, status: r.status, detail };
  } catch { return { ok: false, status: 0, detail: "" }; }
}

/** True when the server accepted *key* (loginWithKeyResult(key).ok). */
export async function loginWithKey(key) {
  return (await loginWithKeyResult(key)).ok;
}

/** The key gate's error line for a failed loginWithKeyResult *res*. */
export function keyGateErrorText(res) {
  if (res.status === 0) return t("keygate.unreachable");
  if (res.status === 401) return t("keygate.wrongKey");
  return res.detail || t("keygate.refused", { status: res.status });
}

// Log in with *key*: on success mark the login and reload into the app; on a
// failure stay on the gate and show why. Resolves the login result.
async function _loginFromGate(key) {
  const res = await loginWithKeyResult(key);
  if (res.ok) {
    // Mark a SUCCESSFUL login so a still-401 boot after the reload self-heals a
    // stale shell instead of looping the gate (AUTH-1b).
    try { sessionStorage.setItem("localm.loginOk", "1"); } catch { /* private mode */ }
    location.reload();
    return res;
  }
  setKeyGateError(keyGateErrorText(res));
  const input = $("key-gate-input");
  if (input) { input.focus(); input.select(); }
  return res;
}

// Submit the gate: log in with the entered key (trimmed). An empty entry asks
// for a key and sends nothing. Resolves the login result, or null.
export function submitKeyGate() {
  const input = $("key-gate-input");
  const key = (input ? input.value : "").trim();
  if (!key) { setKeyGateError(t("keygate.enterKey")); return Promise.resolve(null); }
  setKeyGateError("");
  return _loginFromGate(key);
}

// Add a show/hide reveal toggle to a masked API-key input (AUTH-2), like the
// "show password" eye on a login form, so the user can verify what they typed.
// Idempotent: wraps the input in a flex row once and appends a small toggle.
export function addRevealToggle(input) {
  if (!input || input.dataset.revealWired) return;
  input.dataset.revealWired = "1";
  const wrap = el("div", "input-reveal");
  if (input.parentNode) input.parentNode.insertBefore(wrap, input);
  wrap.appendChild(input);
  const btn = el("button", "reveal-btn", t("keygate.show"));
  btn.type = "button";          // never submit a surrounding form
  btn.setAttribute("aria-label", t("keygate.revealAria"));
  btn.onclick = () => {
    const hidden = input.type === "password";
    input.type = hidden ? "text" : "password";
    btn.textContent = t(hidden ? "keygate.hide" : "keygate.show");
  };
  document.addEventListener("localm:language", () => {
    btn.textContent = t(input.type === "password" ? "keygate.show" : "keygate.hide");
    btn.setAttribute("aria-label", t("keygate.revealAria"));
  });
  wrap.appendChild(btn);
}
window.addRevealToggle = addRevealToggle;

// Pairing QR scanner (phone): read the key QR shown in the computer's Settings
// (Companion app) with the camera, saving the key without typing. Decoding
// prefers the native BarcodeDetector and falls back to the bundled jsQR, so it
// works on any browser that can open a camera in a secure context (Firefox,
// Brave, Opera, iOS Safari, ...), not just Android Chrome.
export function scanSupported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

// Lazily load the bundled jsQR decoder (only fetched when a scan starts on a
// browser without BarcodeDetector). Sets window.jsQR (UMD). Cached after first
// load; the service worker caches the file for later offline pairing.
export let _jsqrPromise = null;
export function loadJsQR() {
  if (window.jsQR) return Promise.resolve(window.jsQR);
  if (_jsqrPromise) return _jsqrPromise;
  _jsqrPromise = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "/vendor/jsQR.js";
    s.onload = () => resolve(window.jsQR);
    s.onerror = () => { _jsqrPromise = null; reject(new Error("jsQR failed to load")); };
    document.head.appendChild(s);
  });
  return _jsqrPromise;
}

// Decode a scanned payload -> save the key + reload. Returns true when it was a
// localm pairing QR (so the scanner can stop). Factored out for unit testing.
export function handleScannedKey(text) {
  const prefix = "localm-key:";
  if (typeof text !== "string" || !text.startsWith(prefix)) return false;
  const key = text.slice(prefix.length).trim();
  if (!key) return false;
  _loginFromGate(key);
  return true;
}

export let _qrStream = null;
export let _qrTimer = null;
export function stopQrScan() {
  if (_qrTimer) { clearInterval(_qrTimer); _qrTimer = null; }
  if (_qrStream) { _qrStream.getTracks().forEach((t) => t.stop()); _qrStream = null; }
  const v = $("qr-video"); if (v) v.srcObject = null;
  const s = $("qr-scanner"); if (s) s.style.display = "none";
}

export async function startQrScan() {
  const overlay = $("qr-scanner"), video = $("qr-video"), status = $("qr-scan-status");
  if (!overlay || !video) return;
  overlay.style.display = "flex";
  if (status) status.textContent = t("qr.starting");

  // Prefer the native BarcodeDetector; otherwise load the bundled jsQR decoder.
  let detector = null;
  if ("BarcodeDetector" in window) {
    try { detector = new window.BarcodeDetector({ formats: ["qr_code"] }); }
    catch { detector = null; }
  }
  let jsqr = null;
  if (!detector) {
    try { jsqr = await loadJsQR(); }
    catch { if (status) status.textContent = t("qr.unavailable"); return; }
  }

  try {
    _qrStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" }, audio: false });
  } catch {
    if (status) status.textContent = t("qr.cameraFailed");
    return;
  }
  video.srcObject = _qrStream;
  try { await video.play(); } catch { /* autoplay guard */ }
  if (status) status.textContent = t("qr.pointAt");

  // Decode one video frame -> the QR text, or null. BarcodeDetector reads the
  // <video> directly; jsQR needs the pixels sampled onto a canvas first.
  const canvas = document.createElement("canvas");
  const decodeFrame = async () => {
    if (detector) {
      const codes = await detector.detect(video);
      return codes.length ? codes[0].rawValue : null;
    }
    const w = video.videoWidth, h = video.videoHeight;
    if (!w || !h) return null;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(video, 0, 0, w, h);
    const img = ctx.getImageData(0, 0, w, h);
    const res = jsqr(img.data, w, h, { inversionAttempts: "dontInvert" });
    return res ? res.data : null;
  };

  let busy = false;
  _qrTimer = setInterval(async () => {
    if (busy || !_qrStream) return;
    busy = true;
    try {
      const value = await decodeFrame();
      if (value && handleScannedKey(value)) { stopQrScan(); return; }
      if (value && status) status.textContent = t("qr.notPairing");
    } catch { /* transient detect error - keep scanning */ }
    finally { busy = false; }
  }, detector ? 300 : 200);
}

// PWA install affordance (P2c): the "Install app" path differs per platform, so
// the Settings card shows the right thing - Android/desktop Chrome fire
// `beforeinstallprompt` and get a real Install button; iOS Safari fires nothing
// (only Share sheet -> Add to Home Screen) so it gets written steps; a standalone
// launch just confirms it. applyInstallUI() is the single decision point,
// unit-tested by branch.

export function pwaDisplayMode() {
  try {
    if (window.matchMedia && window.matchMedia("(display-mode: standalone)").matches) {
      return "standalone";
    }
  } catch { /* matchMedia absent (older/headless) */ }
  // iOS records an installed PWA on navigator.standalone, not display-mode.
  if (window.navigator && window.navigator.standalone === true) return "standalone";
  return "browser";
}

export function isIOSSafari() {
  const nav = window.navigator || {};
  const ua = nav.userAgent || "";
  // iPadOS 13+ reports itself as a Mac, so add the touch check for that case.
  return /iPad|iPhone|iPod/.test(ua)
    || (nav.platform === "MacIntel" && (nav.maxTouchPoints || 0) > 1);
}

// Show exactly one of: installed confirmation, the native Install button, the
// iOS Add-to-Home-Screen steps, or the generic hint. env = {standalone, ios,
// canPrompt, certNeeded, certUntrusted}; missing fields are treated as false.
let _installEnv = null;
export function applyInstallUI(env) {
  env = env || {};
  _installEnv = env;
  const btn = document.getElementById("install-app");
  const hint = document.getElementById("install-hint");
  const ios = document.getElementById("install-ios");
  if (btn) btn.style.display = "none";
  if (ios) ios.style.display = "none";
  if (!hint) return;
  hint.style.display = "none";
  if (env.standalone) {
    if (env.certUntrusted) {
      hint.textContent = t("settings.companion.hintCertUntrusted");
      hint.style.display = "";
      return;
    }
    hint.textContent = t("settings.companion.hintInstalled");
    hint.style.display = "";
    return;
  }
  if (env.canPrompt && btn) { btn.style.display = ""; return; }
  if (env.ios && ios) { ios.style.display = ""; return; }
  if (env.certNeeded) {
    // On HTTPS the service worker only registers behind a TRUSTED certificate;
    // without it the browser can only make a plain shortcut, not an installed
    // app. Point the user at the certificate step instead of the generic hint.
    hint.textContent = t("settings.companion.hintTrustCert");
    hint.style.display = "";
    return;
  }
  applyI18n(hint.parentNode || undefined);
  hint.style.display = "";   // generic browser-menu hint (default)
}

// Onboarding install gate (mobile, P-mobile): after auth, a not-yet-installed
// phone lands on a one-time install screen (Install on Android / Add-to-Home-
// Screen steps on iOS), then taps "Continue" to enter the app. Desktop, an
// already-installed launch, and a return visit skip it.
export function shouldShowInstallGate() {
  if (pwaDisplayMode() === "standalone") return false;       // already installed
  try {
    // AUD-INSTANCEID (see helpers.js reconcileInstanceId): only trust a cached
    // "already onboarded" flag once this origin confirmed pairing with the
    // connected backend, else a fresh install reusing a prior instance's origin
    // would skip its own onboarding because a DIFFERENT install dismissed it here.
    if (instanceCacheTrusted() && localStorage.getItem("localm.onboarded") === "1") return false;
  } catch { /* storage blocked - treat as not onboarded */ }
  return isPhoneOrTablet();
}

// A phone or tablet: the browser says it is mobile (`userAgentData.mobile`), its
// user agent names a phone/tablet OS, it is iPadOS posing as a Mac, or its
// PRIMARY pointer is coarse. A desktop with touch input and a mouse is not one.
// See install-gate.test.mjs "a Windows PC with touch input and a mouse".
export function isPhoneOrTablet() {
  const nav = window.navigator || {};
  if (nav.userAgentData && nav.userAgentData.mobile === true) return true;
  if (/Android|iPhone|iPad|iPod|Mobile/.test(nav.userAgent || "")) return true;
  if (isIOSSafari()) return true;
  try {
    return !!(window.matchMedia && window.matchMedia("(pointer: coarse)").matches);
  } catch { return false; }
}

// Render the gate's install affordance - mirror of applyInstallUI for the gate's
// own elements. env = {ios, canPrompt}; missing fields treated as false.
export function applyInstallGateUI(env) {
  env = env || {};
  const btn = $("install-gate-install");
  const ios = $("install-gate-ios");
  const hint = $("install-gate-hint");
  if (btn) btn.style.display = env.canPrompt ? "" : "none";
  if (ios) ios.style.display = (!env.canPrompt && env.ios) ? "" : "none";
  if (hint) hint.style.display = (!env.canPrompt && !env.ios) ? "" : "none";
}

export function showInstallGate() {
  const gate = $("install-gate");
  if (!gate) return;
  const app = $("app");
  if (app) app.style.display = "none";   // the landing fully replaces the app
  applyInstallGateUI({ ios: isIOSSafari(), canPrompt: !!window.__deferredInstall });
  gate.style.display = "flex";
}

// Leave the gate and enter the app; remember it so we do not nag on return -
// but never in privacy mode (no localStorage traces). refreshCtxLimit also wipes
// the flag if privacy is detected after this runs, so the contract holds even if
// Continue is tapped before the privacy state is known.
export function dismissInstallGate() {
  const gate = $("install-gate");
  if (gate) gate.style.display = "none";
  const app = $("app");
  if (app) app.style.display = "";
  lsSetScoped("localm.onboarded", "1");
}

// Let a late-arriving `beforeinstallprompt` (Chrome fires it after engagement)
// upgrade the gate's generic hint to a real Install button while it is open.
window.refreshInstallGateIfOpen = function () {
  const gate = $("install-gate");
  if (gate && gate.style.display !== "none") {
    applyInstallGateUI({ ios: isIOSSafari(), canPrompt: !!window.__deferredInstall });
  }
};

export let modelCache = { models: [], active: "" };

// Tracks whether refreshModels() has completed at least once, so the very first
// (boot-time) call can be told apart from a later call that legitimately observes
// NO active model (an external unload, or a poll landing before anything has ever
// loaded) - modelCache.active is "" in both cases, so the value alone can't
// distinguish them. See the truthiness note below.
let _modelsEverRefreshed = false;

/** Dispatches `localm:active-model-changed` (detail.model, possibly "") when
 *  modelCache.active differs from *previous*. Fires for every change, an
 *  unload to "" included, unlike `localm:model-switched`. */
function _publishActiveChange(previous) {
  if (modelCache.active === previous) return;
  window.dispatchEvent(new CustomEvent("localm:active-model-changed",
    { detail: { model: modelCache.active } }));
}

export async function refreshModels() {
  try {
    const r = await fetch("/api/models?type=llm", { headers: authHeaders() });
    if (r.status === 401) {
      // No working key (e.g. a network bind, where the loopback key is never
      // auto-seeded). Show the in-page key gate, not window.prompt() - mobile/PWA
      // browsers suppress prompt(), leaving a phone/LAN client on a blank page
      // (NET-1). After a first successful read the key had been accepted, so
      // the gate says the session ended.
      showKeyGate(t(_modelsEverRefreshed ? "keygate.sessionEnded" : "models.keyRequired"));
      return;
    }
    if (r.status === 403) {
      // Open mode, stale per-process shell token (NEW-RESTART-DEAD-BUTTONS):
      // this page was served by an earlier run of the server, so the bearer it
      // carries is no longer the live one and the open-mode gate refuses it.
      // The recovery itself is central (onShellTokenRejected, driven by the
      // fetch wrapper in init.js, which has already been handed this same 403);
      // all that is owed here is a status line that names the real cause. The
      // generic branch below would say "models unavailable (HTTP 403)", which
      // reads as a server fault rather than a page that needs reloading.
      // A 403 the auth layer did not mark as a shell-token refusal is a scope
      // refusal (a key without models:read); a shell refusal on a page that
      // was served no shell token at all is a server that needs a key.
      const shellRefused = !!(r.headers && r.headers.get && r.headers.get("X-Localm-Shell-Rejected"));
      setStatus("err", t(!shellRefused ? "sidebar.status.noModelsScope"
        : SHELL_TOKEN ? "sidebar.status.pageStale" : "sidebar.status.needsKey"));
      return;
    }
    if (!r.ok) {
      // A non-401 error (500, 503, ...) returns a body with no `models` array
      // (e.g. FastAPI's {"detail": ...}); the empty-list fallback below would
      // then silently show an empty dropdown + "ok / no model" status, masking
      // the server error. Surface it instead.
      setStatus("err", t("sidebar.status.modelsUnavailable", { status: r.status }));
      return;
    }
    const data = await r.json();
    const previousActive = modelCache.active;
    // Tolerate a malformed or empty payload (an old server or a proxy returning
    // {}). Without this, iterating an undefined model list throws and the model
    // dropdown silently breaks - fall back to an empty list instead.
    modelCache = (data && Array.isArray(data.models))
      ? data
      : { models: [], active: (data && data.active) || "" };
    if (_switchingTo && !modelCache.loading) modelCache.loading = _switchingTo;
    if (modelCache.active !== previousActive) _holdLoadFailed = false;
    // A model answered through an accepted peer route stays the selection
    // while the server still holds that route; a cleared route drops it.
    if (!_peerSelected && _peerRestore && data && data.peer_routes
        && data.peer_routes[_peerRestore]) _peerSelected = _peerRestore;
    _peerRestore = null;
    const peerRoute = _peerSelected && data && data.peer_routes
      ? data.peer_routes[_peerSelected] : null;
    if (!peerRoute) _peerSelected = null;
    else modelCache.active = _peerSelected;
    _storePeerSelected();
    // Don't rebuild the select while the user has it open
    if (document.activeElement !== modelSelect) {
      modelSelect.innerHTML = "";
      // NEW-MODEL-DROPDOWN-SHOWS-A-MODEL-THAT-IS-NOT-LOADED: without an
      // explicit placeholder, `opt.selected = true` below is set ONLY for the
      // active model - so when nothing is active, no <option> is ever marked
      // selected and the browser falls back to displaying OPTION INDEX 0 (an
      // arbitrary registered model) as if it were loaded. That also made the
      // most natural recovery a silent no-op: reopening the dropdown and
      // picking the model it already (wrongly) showed left `.value` unchanged,
      // so `change` never fired and nothing loaded.
      //
      // JUDGEMENT CALL: this option is DISABLED, i.e. purely a display state,
      // never an action - it never loads (there is nothing to load) and it
      // never unloads (that would make one option's meaning depend on
      // whether a model happens to be active, and an arrow-key slip while
      // browsing could then silently free VRAM). The dedicated Unload button
      // below is the actual affordance for that, and it is unambiguous in
      // both states: present only when there is something to unload.
      // `resumable` is the model an unnamed request would reload and be served
      // by (set by the server after an idle-unload, which keeps the Engine for
      // lazy reload). It is NOT resident, but it IS what the next message will
      // use, so it selects like the active model rather than falling through to
      // the placeholder - "No model loaded" would be the wrong answer to "what
      // serves my next message", and it is what made chat look broken whenever
      // idle_unload_seconds was enabled.
      //
      // This is the SAME shape the server is already in at startup, where the
      // configured model reports active with loaded false and the first message
      // loads it. Idle-unload returns the server to that state, so the sidebar
      // now renders it the same way instead of as a dead end.
      const willServe = modelCache.active || modelCache.resumable || "";
      const placeholder = document.createElement("option");
      placeholder.value = "";
      const notLoaded = !modelCache.active && !!modelCache.resumable && !peerRoute;
      placeholder.textContent = notLoaded
        ? t("sidebar.modelNotLoaded", { model: modelCache.resumable })
        : t("sidebar.noModelLoaded");
      placeholder.disabled = true;
      if (!willServe || notLoaded) placeholder.selected = true;
      modelSelect.appendChild(placeholder);
      for (const m of modelCache.models) {
        const opt = document.createElement("option");
        opt.value = m.name;
        const size = m.size_bytes ? ` (${(m.size_bytes / GIB).toFixed(1)} GB)` : "";
        opt.textContent = m.name + size;
        if (peerRoute ? m.name === _peerSelected
            : m.active) opt.selected = true;
        modelSelect.appendChild(opt);
      }
    }
    // ADR-0008 U4: do not clobber a deliberate busy state (e.g. a model load
    // still in flight) - this 30s poll has no idea one is running and used to
    // overwrite it unconditionally every time it landed mid-load.
    // "no model" only when nothing will serve the next message either - a
    // resumable model reloads on it, so reporting "no model" would contradict
    // both the dropdown above and what the server actually does.
    const polledLoading = !peerRoute && typeof data.loading === "string" ? data.loading : "";
    if (polledLoading && !_statusBusy) {
      setStatus("busy", t("sidebar.status.loading", { model: polledLoading }));
      _statusFromPoll = true;
    } else if (!polledLoading && _statusFromPoll) {
      _statusFromPoll = false;
      _statusBusy = false;
    }
    if (polledLoading && _loadingPollTimer === null) {
      _loadingPollTimer = setTimeout(() => {
        _loadingPollTimer = null;
        refreshModels();
      }, LOADING_POLL_MS);
    }
    if (!_statusBusy && !_holdLoadFailed) {
      if (peerRoute) {
        setStatus("note", t("sidebar.peerRoute.routedStatus", {
          model: _peerSelected, host: peerRoute.host, port: peerRoute.port,
        }));
      } else {
        setStatus("ok", data.active || data.resumable || t("sidebar.status.noModel"));
      }
    }
    // Present only when there is something to unload - not merely a courtesy,
    // the model this targets (modelCache.active) is the ONLY thing that makes
    // its click handler well-defined.
    if (sidebarUnloadBtn) sidebarUnloadBtn.hidden = !modelCache.active || !!peerRoute;
    renderModelSplitLine(data.active_gpu_split);
    refreshMediaRuntimes();
    // The active model can change from OUTSIDE this tab too - another tab,
    // another device, the CLI, an MCP client - while Settings is already open. A
    // same-tab switch refreshes the Live Tuning VRAM estimate itself (switchModel's
    // callers each call refreshPerfEstimate directly), but this 30s poll is the
    // only thing that ever learns about an external switch, so it has to carry the
    // estimate refresh too or the panel is left showing the previous model's
    // numbers indefinitely (it otherwise only touches the dropdown/status line).
    // Deliberately NOT `previousActive && modelCache.active && ...`: requiring
    // BOTH sides truthy looks like it merely skips the boot-time call, but "" is
    // also the legitimate value after an external UNLOAD, so a truthy-only guard
    // masks that transition AND poisons the next one - an unload then a reload
    // from elsewhere would both be silently dropped, since neither the outgoing
    // nor the incoming side of one of those two polls is truthy. _modelsEverRefreshed
    // is the only thing that actually means "not the first call"; every value
    // change after that (including through "") must refresh.
    if (_modelsEverRefreshed && modelCache.active !== previousActive) {
      refreshPerfEstimate();
      if (modelCache.active) {
        window.dispatchEvent(new CustomEvent("localm:model-switched",
          { detail: { model: modelCache.active } }));
      }
    }
    _modelsEverRefreshed = true;
    _publishActiveChange(previousActive);
    window.dispatchEvent(new CustomEvent("localm:models-refreshed"));
  } catch {
    setStatus("err", t("sidebar.status.serverUnreachable"));
    // A server that is really gone gets the reconnect overlay, which reloads
    // the page once the server answers again.
    if (window.serverReachable && window.onServerUnreachable && !(await window.serverReachable())) {
      window.onServerUnreachable({ sawDown: true });
    }
  }
}

/** Show the GPU split the ACTIVE model's load actually applied (auto
 *  free-VRAM-proportional, pinned ratios, or the equal fallback) under the
 *  model status - e.g. "Split: GPU 0 33% · GPU 1 67% (by free VRAM)". Hidden
 *  whenever /api/models carries no active_gpu_split (no split, no model, or
 *  a backend that does not record one), and re-hidden when a model switch
 *  drops it - a stale line would claim a distribution the current model does
 *  not have. */
export function renderModelSplitLine(split) {
  const line = document.getElementById("model-split-line");
  if (!line) return;
  const devices = split && Array.isArray(split.devices) ? split.devices : [];
  if (devices.length < 2) { line.hidden = true; line.textContent = ""; return; }
  const parts = devices.map(
    (d) => `GPU ${d.index} ${Math.round((d.share || 0) * 100)}%`).join(" · ");
  const key = split.source === "auto" ? "sidebar.split.lineAuto"
    : split.source === "pinned" ? "sidebar.split.linePinned"
    : split.source === "equal" ? "sidebar.split.lineEqual" : "sidebar.split.line";
  line.textContent = t(key, { parts });
  line.hidden = false;
}

// Set once a failed read of the media runtimes was logged.
let _mediaReadWarned = false;

/** Read GET /api/media/runtimes and show the native media runtimes this server
 *  holds loaded (see renderMediaRuntimes). A server without the route (404)
 *  shows none. Any other failed read keeps the list as it was and logs one
 *  console warning per page. Never throws. */
export async function refreshMediaRuntimes() {
  if (!document.getElementById("media-runtimes")) return;
  let rows;
  try {
    const r = await fetch("/api/media/runtimes", { headers: authHeaders() });
    if (r.status === 404) rows = [];
    else if (!r.ok) throw new Error("HTTP " + r.status);
    else {
      const d = await r.json();
      rows = d && Array.isArray(d.runtimes) ? d.runtimes : [];
    }
  } catch (e) {
    if (!_mediaReadWarned) console.warn("could not read the media runtimes:", e);
    _mediaReadWarned = true;
    return;
  }
  renderMediaRuntimes(rows);
}

/** List *rows* (the runtimes of GET /api/media/runtimes: name, state "busy",
 *  "idle" or "unknown", idle_unload_seconds) under the model status, one line
 *  each, plus one Unload button while any of them is idle. Hidden when *rows*
 *  is empty. */
export function renderMediaRuntimes(rows) {
  const box = document.getElementById("media-runtimes");
  if (!box) return;
  box.replaceChildren();
  for (const row of rows) {
    const name = String((row && row.name) || "");
    const secs = Number(row && row.idle_unload_seconds);
    const state = row && row.state;
    const key = state === "busy" ? "sidebar.media.busy"
      : state !== "idle" ? "sidebar.media.loaded"
      : secs > 0 ? "sidebar.media.idle" : "sidebar.media.idleNoTimer";
    box.appendChild(el("div", "media-runtime-row",
      t(key, { name, minutes: Math.max(1, Math.round(secs / 60)) })));
  }
  if (rows.some((row) => row && row.state === "idle")) {
    const btn = el("button", "btn-secondary", t("sidebar.media.unload"));
    btn.type = "button";
    btn.title = t("sidebar.media.unloadTitle");
    btn.onclick = () => unloadMediaRuntimes(btn);
    box.appendChild(btn);
  }
  box.hidden = rows.length === 0;
}

/** POST /api/media/runtimes/unload and say per runtime what happened (unloaded,
 *  kept running for a job, or failed), then read the list again. *btn* is
 *  disabled while the request runs. */
export async function unloadMediaRuntimes(btn) {
  if (btn) btn.disabled = true;
  try {
    const r = await fetch("/api/media/runtimes/unload", { method: "POST", headers: authHeaders() });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || r.statusText || "HTTP " + r.status);
    for (const o of Array.isArray(d.outcomes) ? d.outcomes : []) {
      if (o.outcome === "stopped") toast(t("sidebar.media.stopped", { name: o.name }));
      else if (o.outcome === "busy") toast(t("sidebar.media.stillBusy", { name: o.name }));
      else toast(t("sidebar.media.stopFailed", { name: o.name }), true);
    }
  } catch (e) {
    toast(t("sidebar.media.unloadFailed", { message: e.message }), true);
  } finally {
    if (btn) btn.disabled = false;
    await refreshMediaRuntimes();
  }
}

// Ask a live sibling instance to be routed to instead of loading a local
// copy: shows a modal naming the peer and resolves what the user chose:
// {route: key} for Route / Use it (key "" for a peer in open mode, which gets
// no key field), "local" for "Load here instead", or null for Cancel, Escape or
// the x. Route with an empty key field keeps the dialog open and says the key
// is needed. *error*, when given, is shown in the dialog (a failed earlier try).
function _offerPeerRoute(model, peer, error = "") {
  const needsKey = peer.requires_key !== false;
  return new Promise((resolve) => {
    let settled = false;
    let watch = null;
    let input;
    let errLine;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      if (watch) clearInterval(watch);
      $("modal").style.display = "none";
      resolve(value);
    };
    openModal(t("sidebar.peerRoute.title"), (body) => {
      body.appendChild(el("p", "", t(
        needsKey ? "sidebar.peerRoute.message" : "sidebar.peerRoute.messageOpen", {
          model, host: peer.host, port: peer.port,
        })));
      if (needsKey) {
        input = el("input");
        input.type = "password";
        input.placeholder = t("sidebar.peerRoute.keyPlaceholder");
        body.appendChild(input);
      }
      errLine = el("p", "key-gate-error peer-route-error", error);
      errLine.setAttribute("role", "alert");
      errLine.hidden = !error;
      body.appendChild(errLine);
      const accept = () => {
        if (!needsKey) { finish({ route: "" }); return; }
        const key = input.value.trim();
        if (key) { finish({ route: key }); return; }
        errLine.textContent = t("sidebar.peerRoute.keyRequired");
        errLine.hidden = false;
        input.focus();
      };
      const row = el("div", "actions");
      const cancel = el("button", "btn-secondary", t("common.modal.cancel"));
      cancel.onclick = () => finish(null);
      const local = el("button", "btn-secondary", t("sidebar.peerRoute.localButton"));
      local.onclick = () => finish("local");
      const ok = el("button", "btn-secondary",
        t(needsKey ? "sidebar.peerRoute.routeButton" : "sidebar.peerRoute.useButton"));
      ok.onclick = accept;
      row.appendChild(cancel);
      row.appendChild(local);
      row.appendChild(ok);
      body.appendChild(row);
      const onKey = (e) => {
        if (e.key === "Enter") { e.preventDefault(); accept(); }
        else if (e.key === "Escape") { e.preventDefault(); finish(null); }
      };
      if (input) input.onkeydown = onKey;
      else ok.onkeydown = onKey;
    });
    (input || document.querySelector("#modal .actions button:last-child"))?.focus();
    // Dismissing via the shared modal chrome (x / Escape) sets display:none;
    // poll for it and treat as cancel - same pattern promptText() uses.
    watch = setInterval(() => {
      if ($("modal").style.display === "none") finish(null);
    }, 200);
  });
}

// Check whether a live sibling instance already has *model* loaded
// (GET /v1/models/{model}/peer-offer) and, if so, offer the user routing
// there instead of a local load. Returns {routed: <the server's route
// result>} when the route is accepted, {cancelled: true} when the user
// cancelled the dialog, or {local: true} when there is no offer, the offer
// check itself fails, or the user chose to load it here. A refused route asks
// again with the server's reason shown in the dialog.
async function _maybeRoutePeer(model) {
  let offer;
  try {
    const r = await fetch(`/v1/models/${encodeURIComponent(model)}/peer-offer`,
                          { headers: authHeaders() });
    if (!r.ok) return { local: true };
    offer = await r.json();
  } catch {
    return { local: true };
  }
  if (!offer || !offer.available || !offer.peer) return { local: true };
  let error = "";
  let data;
  for (;;) {
    const choice = await _offerPeerRoute(model, offer.peer, error);
    if (choice === null) return { cancelled: true };
    if (choice === "local") return { local: true };
    let r2;
    try {
      r2 = await fetch(`/v1/models/${encodeURIComponent(model)}/peer-route`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ instance_id: offer.peer.instance_id, api_key: choice.route }),
      });
    } catch (e) {
      error = t("sidebar.peerRoute.routeFailed", { detail: e.message });
      continue;
    }
    if (r2.ok) {
      data = await r2.json();
      break;
    }
    const detail = (await r2.json().catch(() => ({}))).detail || r2.statusText;
    error = t("sidebar.peerRoute.routeFailed", { detail });
  }
  setStatus("note", t("sidebar.peerRoute.routedStatus", {
    model, host: offer.peer.host, port: offer.peer.port,
  }));
  _peerSelected = model;
  _storePeerSelected();
  const previousActive = modelCache.active;
  modelCache.active = model;
  _publishActiveChange(previousActive);
  _showServedModel();
  toast(t("sidebar.peerRoute.routedToast", { model }));
  window.dispatchEvent(new CustomEvent("localm:model-switched", { detail: { model } }));
  return { routed: data };
}

// The model the user chose to answer through a peer instance (its route was
// accepted in this tab), or null. /api/models reports the LOCAL active model,
// so refreshModels keeps this selected for as long as the server still holds
// its route (data.peer_routes).
let _peerSelected = null;

// sessionStorage key holding _peerSelected for this tab, so a reload keeps the
// selection while the server still holds the route. Not written while the server
// is in privacy mode or before its mode is known.
const PEER_SELECTED_KEY = "localm.peerSelected";

/** Write _peerSelected to sessionStorage, or remove the entry when nothing is
 *  selected or the server is in privacy mode. */
function _storePeerSelected() {
  try {
    if (!_peerSelected || chat.serverPrivacy) sessionStorage.removeItem(PEER_SELECTED_KEY);
    else if (chat.modeKnown) sessionStorage.setItem(PEER_SELECTED_KEY, _peerSelected);
  } catch { /* sessionStorage unavailable: the selection lasts until a reload */ }
}

// The selection a reload of this tab left in sessionStorage; refreshModels takes
// it over once when the server still holds its route.
let _peerRestore = (() => {
  try { return sessionStorage.getItem(PEER_SELECTED_KEY) || null; }
  catch { return null; }
})();

/** True while this tab's selection answers through an accepted peer route, so
 *  modelCache.active names a model that is not loaded on this instance. */
export function isPeerSelected() {
  return _peerSelected !== null;
}

// Point the model dropdown at the model modelCache says serves the next
// message (the active one, else the resumable one), or at the "No model
// loaded" placeholder when there is none, even while the dropdown has focus.
function _showServedModel() {
  const shown = modelCache.active || "";
  const placeholder = [...modelSelect.options].find((o) => o.value === "" && o.disabled);
  if (placeholder) {
    placeholder.textContent = !shown && modelCache.resumable
      ? t("sidebar.modelNotLoaded", { model: modelCache.resumable })
      : t("sidebar.noModelLoaded");
  }
  modelSelect.value = shown;
  if (modelSelect.value !== shown) modelSelect.value = "";
}

async function _postLoad(model, force, asChat = false, activate = true, keepOthers = false) {
  const r = await fetch("/api/models/load", {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({
      model, ...(force ? { force: true } : {}), ...(asChat ? { as_chat: true } : {}),
      ...(activate ? {} : { activate: false }), ...(keepOthers ? { keep_others: true } : {}),
    }),
  });
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  return await r.json().catch(() => ({ status: "loaded", model }));
}

// The names a confirm_required load result says would be unloaded: its
// unloaded models, the shared models (unloads_shared) and other instances'
// models (peers), quoted and in the GUI's language.
function _unloadNames(data) {
  const busy = Array.isArray(data.busy) ? data.busy : [];
  const names = (Array.isArray(data.unloaded) ? data.unloaded : []).map(
    (n) => (busy.includes(n) ? t("models.switch.busyModel", { model: n }) : `'${n}'`));
  if (data.unloads_shared) names.push(t("models.switch.sharedModels"));
  if (data.unloads_peers) names.push(t("models.switch.peerModels"));
  for (const p of Array.isArray(data.peers) ? data.peers : []) {
    names.push(t("models.switch.peerModel", { model: p }));
  }
  return names;
}

// The question a confirm_required load result asks, in the GUI's language:
// a "degraded_load" result is worded from its figures (plan_mb, full_mb,
// free_mb, moe, unloaded, unloads_shared, unloads_peers), an "unload_ask" or
// "unload_choice" result from what it would unload; any other result shows the
// server's detail.
export function confirmLoadText(data) {
  if (data && data.reason === "unload_ask") {
    const names = _unloadNames(data).join(", ");
    return data.freed_mb > 0
      ? t("models.switch.unloadAskFreed", { model: data.model, names, freed: data.freed_mb })
      : t("models.switch.unloadAsk", { model: data.model, names });
  }
  if (data && data.reason === "unload_choice") {
    return t(data.fits_with_experts_in_ram ? "models.switch.unloadChoiceMoe" : "models.switch.unloadChoice", {
      model: data.model, plan: data.plan_mb, full: data.full_mb, free: data.free_mb,
      names: _unloadNames(data).join(", ") });
  }
  if (!data || data.reason !== "degraded_load" || typeof data.plan_mb !== "number") {
    return (data && data.detail) || "";
  }
  const names = _unloadNames(data);
  const where = names.length
    ? t("models.switch.freeAfterUnload", { free: data.free_mb, names: names.join(", ") })
    : t("models.switch.freeNow", { free: data.free_mb });
  const worded = data.experts_on_disk ? "models.switch.degradedDisk"
    : (data.moe ? "models.switch.degradedMoe" : "models.switch.degradedDense");
  return t(worded, {
    model: data.model, plan: data.plan_mb, full: data.full_mb, where });
}

// Switch the active model. Returns the server status object
// ({status: "loaded" | "already_active" | "superseded" | "cancelled" |
// "peer_routed", model, ...}).
// "superseded" means another model was selected while this one was still loading;
// "cancelled" means the load was aborted for some other reason (including the
// user declining the confirm_required dialog below). Either way the server
// aborted this load, so we do NOT claim success or reset status here (that
// would flash a model that never actually loaded). Callers should skip their
// success toast for both. "peer_routed" means the user accepted an offer to
// route to a live sibling instance instead of loading a local copy - see
// _maybeRoutePeer, which always asks before ever touching /api/models/load.
//
// A "confirm_required" response (something else is in the way - a busy peer
// that will not clear, or a load the server's own estimate says will degrade
// to slow CPU offload) is never returned to the caller: it is resolved right
// here, so every caller gets the same final outcome without each needing its
// own confirm/force plumbing.
//
// On a failed load or peer route it sets the pill to "load failed", starts a
// refreshModels(), and rethrows; the caller only reports the error.
//
// *activate* false asks the server to load the model without making it the
// active model. *select* false leaves this tab's selection as it is: the peer
// route and modelCache.active are kept, no localm:model-switched is dispatched,
// and the status line returns to the selected model once modelCache is re-read.
/** End switchModel()'s own loading mark for *model*. */
function _endSwitch(model) {
  if (_switchingTo !== model) return;
  _switchingTo = "";
  if (modelCache.loading === model) modelCache.loading = "";
}

/** Undo switchModel()'s marks after the user declined a dialog of the switch,
 *  re-read the server's model state, and return {status: "cancelled"}. */
function _declinedSwitch(model) {
  _endSwitch(model);
  setStatus("ok", modelCache.active || t("sidebar.status.noModel"));
  _showServedModel();
  refreshModels().then(_showServedModel);
  return { status: "cancelled", reason: "declined by user", model };
}

export async function switchModel(model, { local = false, activate = true, select = true } = {}) {
  let data;
  try {
    const peer = local ? { local: true } : await _maybeRoutePeer(model);
    if (peer.routed) return peer.routed;
    if (peer.cancelled) return _declinedSwitch(model);
    if (select) {
      _peerSelected = null;
      _storePeerSelected();
    }
    setStatus("busy", t("sidebar.status.loading", { model }));
    _switchingTo = model;
    modelCache.loading = model;
    let force = false;
    let asChat = false;
    let keepOthers = false;
    data = await _postLoad(model, false, false, activate);
    while (data.status === "confirm_required") {
      const notChat = data.reason === "not_chat_model";
      const unloadAsk = data.reason === "unload_ask";
      // A confirm the server repeats after its flag was already sent is a failed load.
      if (notChat ? asChat : (force || keepOthers)) throw new Error(data.detail || t("sidebar.status.loadFailed"));
      if (data.reason === "unload_choice") {
        const names = _unloadNames(data).join(", ");
        const choice = await chooseAsync(t("models.switch.confirmTitle"), confirmLoadText(data), [
          { key: "beside", label: t("models.switch.loadBeside") },
          { key: "unload", label: t("models.switch.unloadForThis", { names }), danger: true },
        ]);
        if (!choice) return _declinedSwitch(model);
        if (choice === "beside") keepOthers = true;
        else force = true;
        setStatus("busy", t("sidebar.status.loading", { model }));
        data = await _postLoad(model, force, asChat, activate, keepOthers);
        continue;
      }
      const confirmed = await confirmDangerAsync(
        t(notChat ? "models.notChat.title"
          : unloadAsk ? "models.switch.unloadAskTitle" : "models.switch.confirmTitle"),
        confirmLoadText(data),
        t(notChat ? "models.notChat.confirmLabel"
          : unloadAsk ? "models.switch.unloadAskLabel" : "models.switch.confirmLabel"));
      if (!confirmed) return _declinedSwitch(model);
      if (notChat) asChat = true;
      else force = true;
      setStatus("busy", t("sidebar.status.loading", { model }));
      data = await _postLoad(model, force, asChat, activate, keepOthers);
    }
  } catch (e) {
    // A failed load ends the busy pill and re-reads the server's model state.
    _endSwitch(model);
    setStatus("err", t("sidebar.status.loadFailed"));
    _holdLoadFailed = true;
    _showServedModel();
    refreshModels().then(_showServedModel);
    throw e;
  }
  _endSwitch(model);
  if (data.status === "superseded" || data.status === "cancelled") return data;
  if (!select) {
    await refreshModels();
    setStatus("ok", modelCache.active || modelCache.resumable || t("sidebar.status.noModel"));
    _showServedModel();
    return data;
  }
  setStatus("ok", data.model || model);
  // Publish the newly active model NOW. refreshModels() is the only other writer
  // and it polls every 30s, so without this the client believes no model is
  // loaded until the next poll: sendChat's empty-model guard reads
  // modelCache.active and would falsely refuse "No model loaded" for up to ~30s
  // after a sidebar load actually succeeded (REG-471). Use the name the SERVER
  // reports (an alias may resolve to a different one); the next poll reconciles
  // the rest of the cache. Deliberately after the superseded/cancelled return:
  // that load was abandoned, so claiming it would publish a model that is not
  // loaded.
  const previousActive = modelCache.active;
  modelCache.active = data.model || model;
  _publishActiveChange(previousActive);
  _showServedModel();
  window.dispatchEvent(new CustomEvent("localm:model-switched", { detail: { model: data.model || model } }));
  return data;
}

// Unload *model* and load it again on this instance, so a reload picks up the
// saved load settings. Asks before unloading a model that is in use. Returns the
// switchModel() result, or {status: "cancelled"} when the user declined, or
// {status: "in_use"} when the model is busy and was not unloaded.
// Reloading a model that is not the active one leaves the active model, and
// this tab's selection, as they are, also when the reload fails or is cancelled.
// modelCache is re-read from the server first. Whether *model* is the server's
// active model is read from its modelCache entry; whether it is this tab's
// selection, from modelCache.active.
export async function reloadModel(model) {
  setStatus("busy", t("sidebar.status.unloading", { model }));
  await refreshModels();
  const wasActive = model === modelCache.active;
  const entry = (modelCache.models || []).find((m) => m.name === model);
  const serverActive = entry && typeof entry.active === "boolean" ? entry.active : wasActive;
  const idleStatus = () => (wasActive ? model
    : modelCache.active || modelCache.resumable || t("sidebar.status.noModel"));
  setStatus("busy", t("sidebar.status.unloading", { model }));
  let data;
  try {
    data = await _postUnload(model, false);
    if (data.status === "confirm_required") {
      const confirmed = await confirmDangerAsync(
        t("models.unload.confirmTitle"), data.detail, t("models.unload.confirmLabel"));
      if (!confirmed) {
        setStatus("ok", idleStatus());
        return { status: "cancelled", reason: "declined by user", model };
      }
      setStatus("busy", t("sidebar.status.unloading", { model }));
      data = await _postUnload(model, true);
    }
  } catch (e) {
    setStatus("err", t("sidebar.status.unloadFailed"));
    throw e;
  }
  if (data.status === "in_use") {
    setStatus("ok", idleStatus());
    return { status: "in_use", model };
  }
  let res;
  try {
    res = await switchModel(model, { local: true, activate: serverActive, select: wasActive });
  } catch (e) {
    if (wasActive) _markUnloaded(model);
    throw e;
  }
  if (res && res.status === "cancelled") {
    if (wasActive) _markUnloaded(model);
    setStatus("ok", idleStatus());
    refreshModels();
  }
  return res;
}

// Record that the active model *model* is no longer resident but still answers
// the next message (the server keeps it for lazy reload).
function _markUnloaded(model) {
  modelCache.active = "";
  modelCache.resumable = model;
  _publishActiveChange(model);
}

// Toast the outcome of a switchModel() call, headed by *head* ("Model switched
// to <model>" by default). A model too big to fully fit
// VRAM still loads with some layers on the CPU (gpu_layers_offloaded below
// gpu_layers_total): that toast is a warning. A Mixture-of-Experts load that
// keeps every layer on the GPU and only the routed experts of moe_cpu_layers
// layers in system RAM is its planned placement: that toast is information,
// naming the per-token RAM read (moe_ram_read_bytes). res.mmap_from_disk marks a
// model that may not fit in available RAM and runs from disk-backed memory,
// res.mmap_cpu_from_disk one whose CPU layers read it from disk on every token.
// res.mmap_ram_short marks a use_mmap off load larger than available RAM; its
// warning is res.mmap_note. A res.mmap_cpu_from_disk load with
// res.mmap_cpu_read_bytes and res.mmap_ram_available_bytes names both sizes.
export function toastLoadResult(res, model, head = t("models.switch.doneToast", { model })) {
  const fromDisk = !!(res && res.mmap_from_disk);
  const cpuFromDisk = !!(res && res.mmap_cpu_from_disk);
  const gib = (n) => (n / 1024 ** 3).toFixed(1);
  const sized = cpuFromDisk && res.mmap_cpu_read_bytes > 0 && res.mmap_ram_available_bytes >= 0;
  const diskNote = sized
    ? t("models.switch.mmapCpuFromDiskSized", {
      read: gib(res.mmap_cpu_read_bytes), avail: gib(res.mmap_ram_available_bytes) })
    : t(cpuFromDisk ? "models.switch.mmapCpuFromDisk" : "models.switch.mmapFromDisk");
  const moeText = res && res.moe_cpu_layers > 0
    ? (res.moe_ram_read_bytes > 0
      ? t("models.switch.moePlacement", {
        moe: res.moe_cpu_layers, total: res.gpu_layers_total,
        gb: (res.moe_ram_read_bytes / 1024 ** 3).toFixed(2) })
      : t("models.switch.moePlacementNoRead", {
        moe: res.moe_cpu_layers, total: res.gpu_layers_total }))
    : "";
  const extra = fromDisk ? `; ${diskNote}`
    : (res && res.mmap_ram_short && res.mmap_note ? `; ${res.mmap_note}` : "");
  if (res && res.degraded && res.gpu_layers_offloaded < res.gpu_layers_total) {
    const parts = [t("models.switch.layersOnGpu", { offloaded: res.gpu_layers_offloaded, total: res.gpu_layers_total })];
    if (moeText) parts.push(moeText);
    toast(`${head} (${t("models.switch.slower", { parts: parts.join("; ") })}${extra})`, true);
  } else if (res && res.degraded && moeText) {
    toast(`${head} (${moeText}${extra})`, !!extra);
  } else if (res && res.mmap_ram_short && res.mmap_note) {
    toast(`${head} (${res.mmap_note})`, true);
  } else if (fromDisk) {
    toast(`${head} (${diskNote})`, true);
  } else {
    toast(head);
  }
}

modelSelect.onchange = async () => {
  const model = modelSelect.value;
  // The placeholder option (value "") is disabled, so a real browser never
  // lets a user pick it - but this guard makes that inert semantics a
  // property of the CODE, not just of the DOM attribute, so it holds even if
  // something else ever forces the select's value.
  if (!model) return;
  try {
    const res = await switchModel(model);
    // Superseded: a newer selection is now loading - stay quiet and let its own
    // handler report when it lands, instead of toasting a model we abandoned.
    // Cancelled: the load was aborted for some other reason - also not a
    // success worth toasting.
    if (!res || (res.status !== "superseded" && res.status !== "cancelled")) {
      toastLoadResult(res, model);
      // The Settings "Live tuning" VRAM estimate defaults to the active model
      // server-side, but only re-fetches on its own slider input - refresh it
      // here too so a switch made from the model dropdown is reflected without
      // needing to nudge a slider or leave and re-enter Settings.
      refreshPerfEstimate();
    }
  } catch (e) {
    toast(t("models.switch.loadFailedToast", { message: e.message }), true);
  }
};

// Sidebar quick-unload (NEW-MODEL-DROPDOWN-SHOWS-A-MODEL-THAT-IS-NOT-LOADED,
// second ask): until now the only Unload control lived on the Models page,
// away from the CPU/RAM/VRAM line where a user actually notices a model is
// still resident. Targets modelCache.active specifically - the one model this
// sidebar shows - not "unload everything"; that broader action already
// exists as the Models page's own "Unload all" and would be a surprising
// thing for this control to do silently to OTHER loaded-but-inactive models
// the sidebar cannot even show.
async function _postUnload(model, force) {
  const r = await fetch("/api/models/unload", {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify(force ? { model, force: true } : { model }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || t("models.unload.failed"));
  return data;
}

if (sidebarUnloadBtn) {
  sidebarUnloadBtn.onclick = async () => {
    const model = modelCache.active;
    if (!model) return;   // hidden whenever there's nothing to unload; guard anyway
    sidebarUnloadBtn.disabled = true;
    setStatus("busy", t("sidebar.status.unloading", { model }));
    try {
      let data = await _postUnload(model, false);
      // unload_one_model() (http_server.py) answers HTTP 200 for a busy
      // engine too - it is a legitimate "not done yet, not an error" outcome
      // (a request is mid-generation against it right now, or a coder
      // session is using it), not the same thing as a real unload. Reporting
      // "Unloaded" here regardless of `status` would claim a VRAM release
      // that did not happen.
      if (data.status === "confirm_required") {
        const confirmed = await confirmDangerAsync(
          t("models.unload.confirmTitle"), data.detail, t("models.unload.confirmLabel"));
        if (!confirmed) {
          setStatus("ok", model);
          return;
        }
        setStatus("busy", t("sidebar.status.unloading", { model }));
        data = await _postUnload(model, true);
      }
      if (data.status === "in_use") {
        setStatus("ok", model);
        toast(t("models.unload.inUse", { name: model }), true);
        return;
      }
      toast(t("models.unload.toast", { name: model }));
      // Publish immediately, same reasoning as switchModel's own comment
      // above (REG-471): without this the dropdown/status would keep
      // showing `model` as active until the next 30s poll. setStatus also
      // resets _statusBusy, which refreshModels()'s own "ok" write is
      // gated on - skip it and the status line would stay stuck on
      // "unloading ...".
      // The server keeps the Engine for lazy reload, so the next chat request
      // still has a model to answer it.
      _markUnloaded(model);
      setStatus("ok", model);
      refreshModels();
      refreshPerfEstimate();
    } catch (e) {
      setStatus("err", t("sidebar.status.unloadFailed"));
      toast(t("models.unload.failedToast", { message: e.message }), true);
    } finally {
      sidebarUnloadBtn.disabled = false;
    }
  };
}

// The model dropdown's placeholder and the status pill are written by
// refreshModels, so they are redrawn when the interface language changes.
document.addEventListener("localm:language", () => {
  refreshModels();
  if (_installEnv) applyInstallUI(_installEnv);
});
