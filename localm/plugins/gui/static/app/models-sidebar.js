// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - models sidebar selector. */
"use strict";

// --- ES module imports ---
import { lsSetScoped } from "./chat.js";
import { $, GIB, authHeaders, el, fmtDuration, instanceCacheTrusted, openModal, streamJob, toast } from "./helpers.js";
import { t } from "./i18n.js";
import { refreshPerfEstimate } from "./settings-perf.js";

export const modelSelect = $("model-select");
export const sidebarUnloadBtn = $("sidebar-unload-btn");

// Whether the status pill is currently showing a caller's busy state (e.g.
// switchModel's "loading X…"). refreshModels()'s periodic "ok" write is gated on
// this so it does not clobber one.
let _statusBusy = false;

// State renders as a .job-state pill; busy maps to st-running.
const STATUS_STATE_CLASS = { ok: "st-ok", busy: "st-running", err: "st-error" };

export function setStatus(state, text) {
  _statusBusy = state === "busy";
  $("status-text").className = "job-state " + (STATUS_STATE_CLASS[state] || "");
  $("status-text").textContent = text;
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
        const label = d.index == null ? `GPU${i}` : `GPU${d.index}`;
        // Column 1 carries the card's IDENTITY, not a utilisation percent:
        // /api/stats reports gpu.percent as ONE system-wide figure (nvidia-smi,
        // NVIDIA only), which belongs to no single card.
        add("gpu", label);
        if (d.used != null) {
          add("vram", `${gib(d.used)}/${gib(d.total)} GB`, `vram-usage ${band(d.used, d.total)}`);
        } else {
          // The "total" word is required: the VRAM label means used/total
          // everywhere else, so a bare figure would read as a full card.
          add("vram", `${gib(d.total)} GB total`);
        }
      });
    } else if (v.used != null) {
      add("vram", `VRAM ${gib(v.used)}/${gib(v.total)} GB`, `vram-usage ${band(v.used, v.total)}`);
    } else {
      // The "total" word is required here for the same reason as the per-card
      // branch above.
      add("vram", `VRAM ${gib(v.total)} GB total`);
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
  try {
    const r = await fetch("/api/stats", { headers: authHeaders() });
    if (!r.ok) return;
    renderHwStats(await r.json());
  } catch (e) { /* transient - keep the last reading */ }
}

export let _hwStatsTimer = null;
export function startHwStats(intervalMs = 2500) {
  pollHwStats();
  pollActivity();   // ADR-0008 U4: folded into this existing poll, not a new timer
  if (_hwStatsTimer) clearInterval(_hwStatsTimer);
  _hwStatsTimer = setInterval(() => { pollHwStats(); pollActivity(); }, intervalMs);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) { pollHwStats(); pollActivity(); }   // refresh promptly on tab focus
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
const _ACTIVITY_MODAL_TITLE = "Activity";

export function isActivityBusy() {
  return _activityOps.some((op) => op.status === "running");
}

function activityLabel(op) {
  return (op && (op.label || op.kind)) || "operation";
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

export function renderActivityPill(ops) {
  const pill = $("activity-pill");
  if (!pill) return;
  const stale = _activityKnown && _activityStaleReads >= _ACTIVITY_STALE_AFTER_READS;
  const running = ops.filter((op) => op.status === "running");
  if (!running.length && !stale) {
    pill.style.display = "none";
    pill.classList.remove("st-unknown");
    pill.classList.add("st-running");
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
    pill.title = "Could not reach the server for an activity update - showing the last known state.";
    const label = running.length === 1 ? activityLabel(running[0])
      : running.length > 1 ? `${running.length} running` : null;
    pill.textContent = label ? `${label} - status unknown` : "Status unknown";
    return;
  }
  pill.classList.remove("st-unknown");
  pill.classList.add("st-running");
  pill.title = "Click for details";
  if (running.length === 1) {
    const op = running[0];
    const pct = typeof op.pct === "number" ? ` ${Math.round(op.pct)}%` : "";
    pill.textContent = activityLabel(op) + pct;
  } else {
    pill.textContent = `${running.length} running`;
  }
}

// Shared with the live-refresh path below so the modal never diverges from
// what openModal() rendered at open time.
function _buildActivityBody(body) {
  if (!_activityKnown) {
    // Never render "Nothing running." for a question that was never
    // successfully asked (R1) - this only shows before the very first
    // poll/reattach resolves, or if every attempt so far has failed.
    body.appendChild(el("p", "sub", "Checking…"));
    return;
  }
  if (_activityStaleReads >= _ACTIVITY_STALE_AFTER_READS) {
    body.appendChild(el("p", "sub",
      "Could not reach the server for an update - showing the last known state below."));
  }
  if (!_activityOps.length) {
    body.appendChild(el("p", "sub", "Nothing running."));
    return;
  }
  for (const op of _activityOps) {
    const row = el("div", "job-row");
    const head = el("div", "job-head");
    head.appendChild(el("span", "job-name", activityLabel(op)));
    head.appendChild(el("span", "job-state st-" + op.status, op.status));
    const age = activityAge(op);
    if (age) head.appendChild(el("span", "sub", age));
    row.appendChild(head);
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
}

export function showActivityDetails() {
  openModal(_ACTIVITY_MODAL_TITLE, _buildActivityBody);
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
  if (!title || title.textContent !== _ACTIVITY_MODAL_TITLE) return;
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
    const ops = Array.isArray(data.operations) ? data.operations : [];
    _activityOps = ops;
    _activityServerNow = typeof data.now === "number" ? data.now : null;
    _activityKnown = true;
    _activityStaleReads = 0;
    renderActivityPill(ops);
    _refreshActivityModalIfOpen();
    return ops;
  } catch (e) {
    return _activityReadFailed();
  }
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
        if (idx !== -1 && typeof ev.pct === "number") {
          _activityOps[idx] = { ..._activityOps[idx], pct: ev.pct };
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
      ? `Reattached to a running ${activityLabel(running[0])}`
      : `Reattached to ${running.length} running operations`);
  } catch (e) { /* server unreachable at boot - same as reattachSessions() */ }
}

// In-page API-key gate. Shown when an authed boot returns 401 and this browser
// has no working key - the network/phone case, where the loopback key is never
// auto-seeded. Replaces window.prompt() (suppressed by mobile/PWA browsers, the
// NET-1 white-page cause). Idempotent.
export function showKeyGate(message) {
  const gate = $("key-gate");
  if (!gate) return;
  if (message) { const m = $("key-gate-msg"); if (m) m.textContent = message; }
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
    input.focus();
  }
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
export async function loginWithKey(key) {
  try {
    const r = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    });
    if (r.ok) {
      try { window.__LOCALM_CSRF__ = (await r.json()).csrf || ""; } catch (e) { /* body optional */ }
    }
    return r.ok;
  } catch (e) { return false; }
}

// Submit the gate: log in with the entered key (trimmed) then reload. An empty
// entry just reloads (still unauthenticated -> the gate shows again).
export function submitKeyGate() {
  const input = $("key-gate-input");
  const key = (input ? input.value : "").trim();
  if (key) {
    loginWithKey(key).then((ok) => {
      // Mark a SUCCESSFUL login so a still-401 boot after the reload self-heals a
      // stale shell instead of looping the gate (AUTH-1b). A failed login (wrong
      // key / server down) sets nothing, so the gate just shows again.
      if (ok) { try { sessionStorage.setItem("localm.loginOk", "1"); } catch (e) { /* private mode */ } }
      location.reload();
    });
  } else {
    location.reload();
  }
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
  const btn = el("button", "reveal-btn", "show");
  btn.type = "button";          // never submit a surrounding form
  btn.setAttribute("aria-label", "Show or hide the key");
  btn.onclick = () => {
    const hidden = input.type === "password";
    input.type = hidden ? "text" : "password";
    btn.textContent = hidden ? "hide" : "show";
  };
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
  loginWithKey(key).then((ok) => {
    if (ok) { try { sessionStorage.setItem("localm.loginOk", "1"); } catch (e) { /* private mode */ } }
    location.reload();
  });
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
  if (status) status.textContent = "Starting camera…";

  // Prefer the native BarcodeDetector; otherwise load the bundled jsQR decoder.
  let detector = null;
  if ("BarcodeDetector" in window) {
    try { detector = new window.BarcodeDetector({ formats: ["qr_code"] }); }
    catch (e) { detector = null; }
  }
  let jsqr = null;
  if (!detector) {
    try { jsqr = await loadJsQR(); }
    catch (e) { if (status) status.textContent = "QR scanning is not available here."; return; }
  }

  try {
    _qrStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" }, audio: false });
  } catch (e) {
    if (status) status.textContent = "Could not open the camera (permission denied?).";
    return;
  }
  video.srcObject = _qrStream;
  try { await video.play(); } catch (e) { /* autoplay guard */ }
  if (status) status.textContent = "Point at the QR in the computer's Settings.";

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
    } catch (e) { /* transient detect error - keep scanning */ }
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
  } catch (e) { /* matchMedia absent (older/headless) */ }
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
export function applyInstallUI(env) {
  env = env || {};
  const btn = document.getElementById("install-app");
  const hint = document.getElementById("install-hint");
  const ios = document.getElementById("install-ios");
  if (btn) btn.style.display = "none";
  if (ios) ios.style.display = "none";
  if (!hint) return;
  hint.style.display = "none";
  if (env.standalone) {
    if (env.certUntrusted) {
      hint.textContent = "This device no longer trusts this server's certificate. "
        + "Open localm-ca.crt from the server (/localm-ca.crt) and install it, "
        + "then reload.";
      hint.style.display = "";
      return;
    }
    hint.textContent = "Running as an installed app.";
    hint.style.display = "";
    return;
  }
  if (env.canPrompt && btn) { btn.style.display = ""; return; }
  if (env.ios && ios) { ios.style.display = ""; return; }
  if (env.certNeeded) {
    // On HTTPS the service worker only registers behind a TRUSTED certificate;
    // without it the browser can only make a plain shortcut, not an installed
    // app. Point the user at the certificate step instead of the generic hint.
    hint.textContent = "To install localm as an app, trust this device's "
      + "certificate first (the “Install certificate” step above), then reload.";
    hint.style.display = "";
    return;
  }
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
  } catch (e) { /* storage blocked - treat as not onboarded */ }
  // Phones/tablets only: a touch device with a coarse pointer. A desktop
  // `localm gui` (fine pointer) opens straight into the app.
  return (navigator.maxTouchPoints || 0) > 0
    || !!(window.matchMedia && window.matchMedia("(pointer: coarse)").matches);
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

export async function refreshModels() {
  try {
    const r = await fetch("/api/models?type=llm", { headers: authHeaders() });
    if (r.status === 401) {
      // No working key (e.g. a network bind, where the loopback key is never
      // auto-seeded). Show the in-page key gate, not window.prompt() - mobile/PWA
      // browsers suppress prompt(), leaving a phone/LAN client on a blank page
      // (NET-1). The gate stores the key and reloads on submit.
      showKeyGate(t("models.keyRequired"));
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
      setStatus("err", t("sidebar.status.pageStale"));
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
      placeholder.textContent = t("sidebar.noModelLoaded");
      placeholder.disabled = true;
      if (!willServe) placeholder.selected = true;
      modelSelect.appendChild(placeholder);
      for (const m of modelCache.models) {
        const opt = document.createElement("option");
        opt.value = m.name;
        const size = m.size_bytes ? ` (${(m.size_bytes / GIB).toFixed(1)} GB)` : "";
        opt.textContent = m.name + size;
        if (m.active || (!modelCache.active && m.name === willServe)) opt.selected = true;
        modelSelect.appendChild(opt);
      }
    }
    // ADR-0008 U4: do not clobber a deliberate busy state (e.g. a model load
    // still in flight) - this 30s poll has no idea one is running and used to
    // overwrite it unconditionally every time it landed mid-load.
    // "no model" only when nothing will serve the next message either - a
    // resumable model reloads on it, so reporting "no model" would contradict
    // both the dropdown above and what the server actually does.
    if (!_statusBusy) setStatus("ok", data.active || data.resumable || t("sidebar.status.noModel"));
    // Present only when there is something to unload - not merely a courtesy,
    // the model this targets (modelCache.active) is the ONLY thing that makes
    // its click handler well-defined.
    if (sidebarUnloadBtn) sidebarUnloadBtn.hidden = !modelCache.active;
    renderModelSplitLine(data.active_gpu_split);
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
    }
    _modelsEverRefreshed = true;
  } catch (e) {
    setStatus("err", t("sidebar.status.serverUnreachable"));
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
    (d) => `GPU ${d.index} ${Math.round((d.share || 0) * 100)}%`);
  const why = split.source === "auto" ? " (by free VRAM)"
    : split.source === "pinned" ? " (pinned)"
    : split.source === "equal" ? " (equal)" : "";
  line.textContent = `Split: ${parts.join(" · ")}${why}`;
  line.hidden = false;
}

// Ask a live sibling instance to be routed to instead of loading a local
// copy: shows a password-input modal naming the peer and, on submit,
// resolves the entered API key. Resolves null on Cancel/Escape/backdrop
// dismiss. Modeled on promptText() in helpers.js, with a password-type
// input for the credential and its own message/title instead of a bare
// text prompt.
function _offerPeerRoute(model, peer) {
  return new Promise((resolve) => {
    let settled = false;
    let watch = null;
    let input;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      if (watch) clearInterval(watch);
      $("modal").style.display = "none";
      resolve(value);
    };
    openModal(t("sidebar.peerRoute.title"), (body) => {
      body.appendChild(el("p", "", t("sidebar.peerRoute.message", {
        model, host: peer.host, port: peer.port,
      })));
      input = el("input");
      input.type = "password";
      input.placeholder = t("sidebar.peerRoute.keyPlaceholder");
      body.appendChild(input);
      const row = el("div", "actions");
      const cancel = el("button", "btn-secondary", "Cancel");
      cancel.onclick = () => finish(null);
      const ok = el("button", "btn-secondary", t("sidebar.peerRoute.routeButton"));
      ok.onclick = () => finish(input.value.trim() || null);
      row.appendChild(cancel);
      row.appendChild(ok);
      body.appendChild(row);
      input.onkeydown = (e) => {
        if (e.key === "Enter") { e.preventDefault(); finish(input.value.trim() || null); }
        else if (e.key === "Escape") { e.preventDefault(); finish(null); }
      };
    });
    input.focus();
    // Dismissing via the shared modal chrome (x / backdrop) sets display:none;
    // poll for it and treat as cancel - same pattern promptText() uses.
    watch = setInterval(() => {
      if ($("modal").style.display === "none") finish(null);
    }, 200);
  });
}

// Check whether a live sibling instance already has *model* loaded
// (GET /v1/models/{model}/peer-offer) and, if so, offer the user routing
// there instead of a local load. Returns the server's route result
// ({status: "peer_routed", model, peer}) when the user accepts, or null
// when there is no offer, the offer check itself fails, or the user
// declines - the caller then proceeds with its own normal local load.
async function _maybeRoutePeer(model) {
  let offer;
  try {
    const r = await fetch(`/v1/models/${encodeURIComponent(model)}/peer-offer`,
                          { headers: authHeaders() });
    if (!r.ok) return null;
    offer = await r.json();
  } catch (e) {
    return null;
  }
  if (!offer || !offer.available || !offer.peer) return null;
  const apiKey = await _offerPeerRoute(model, offer.peer);
  if (!apiKey) return null;
  const r2 = await fetch(`/v1/models/${encodeURIComponent(model)}/peer-route`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ instance_id: offer.peer.instance_id, api_key: apiKey }),
  });
  if (!r2.ok) {
    const detail = (await r2.json().catch(() => ({}))).detail || r2.statusText;
    toast(t("sidebar.peerRoute.routeFailed", { detail }), true);
    return null;
  }
  const data = await r2.json();
  setStatus("ok", t("sidebar.peerRoute.routedStatus", {
    model, host: offer.peer.host, port: offer.peer.port,
  }));
  modelCache.active = model;
  toast(t("sidebar.peerRoute.routedToast", { model }));
  return data;
}

// Switch the active model. Returns the server status object
// ({status: "loaded" | "already_active" | "superseded" | "cancelled" |
// "peer_routed", model, ...}).
// "superseded" means another model was selected while this one was still loading;
// "cancelled" means the load was aborted for some other reason. Either way the
// server aborted this load, so we do NOT claim success or reset status here
// (that would flash a model that never actually loaded). Callers should skip
// their success toast for both. "peer_routed" means the user accepted an offer
// to route to a live sibling instance instead of loading a local copy - see
// _maybeRoutePeer, which always asks before ever touching /api/models/load.
export async function switchModel(model) {
  const routed = await _maybeRoutePeer(model);
  if (routed) return routed;
  setStatus("busy", t("sidebar.status.loading", { model }));
  const r = await fetch("/api/models/load", {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ model }),
  });
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  const data = await r.json().catch(() => ({ status: "loaded", model }));
  if (data.status === "superseded" || data.status === "cancelled") return data;
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
  modelCache.active = data.model || model;
  return data;
}

// Toast the outcome of a switchModel() call. A model too big to fully fit
// VRAM still loads (the backend deliberately defers to a partial/zero GPU
// offload rather than refusing), so a plain "switched" toast would read as
// success even when the load quietly fell back to (slow) CPU layers -
// res.degraded (from /api/models/load's gpu_layers_offloaded/gpu_layers_total)
// says so; warn instead of a bare success toast (AGENTS.md rule 5).
export function toastLoadResult(res, model) {
  if (res && res.degraded) {
    toast(`Model switched to ${model} (${res.gpu_layers_offloaded}/` +
          `${res.gpu_layers_total} layers on GPU, rest on CPU - slower)`, true);
  } else {
    toast("Model switched to " + model);
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
    setStatus("err", t("sidebar.status.loadFailed"));
    toast("Model load failed: " + e.message, true);
    refreshModels();
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
if (sidebarUnloadBtn) {
  sidebarUnloadBtn.onclick = async () => {
    const model = modelCache.active;
    if (!model) return;   // hidden whenever there's nothing to unload; guard anyway
    sidebarUnloadBtn.disabled = true;
    setStatus("busy", t("sidebar.status.unloading", { model }));
    try {
      const r = await fetch("/api/models/unload", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ model }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        setStatus("err", t("sidebar.status.unloadFailed"));
        toast(data.detail || "Unload failed", true);
        return;
      }
      // unload_one_model() (http_server.py) answers HTTP 200 for an in-use
      // engine too - it is a legitimate "not done yet, not an error" outcome
      // (a request is mid-generation against it right now), not the same
      // thing as a real unload. Reporting "Unloaded" here regardless of
      // `status` would claim a VRAM release that did not happen (AGENTS.md
      // rule 5). Reset the status line here too (still the same model,
      // still busy) - skip it and _statusBusy stays true forever, which
      // would leave refreshModels()'s own status write gated off below.
      if (data.status === "in_use") {
        setStatus("ok", model);
        toast(t("models.unload.inUse", { name: model }), true);
        refreshModels();
        return;
      }
      toast(`Unloaded '${model}'`);
      // Publish immediately, same reasoning as switchModel's own comment
      // above (REG-471): without this the dropdown/status would keep
      // showing `model` as active until the next 30s poll. setStatus also
      // resets _statusBusy, which refreshModels()'s own "ok" write is
      // gated on - skip it and the status line would stay stuck on
      // "unloading ...".
      modelCache.active = "";
      // The unload button's own tooltip promises "it reloads automatically on
      // the next chat request", and the server does exactly that (the Engine
      // stays in _engines for lazy reload). Publish that immediately alongside
      // clearing `active`, for the same reason the line above exists: until the
      // next poll lands, the chat gate would otherwise read no active and no
      // resumable model and refuse to send - turning the button's promise into
      // "load a model on the sidebar before chatting".
      modelCache.resumable = model;
      setStatus("ok", model);
      refreshModels();
      refreshPerfEstimate();
    } catch (e) {
      setStatus("err", t("sidebar.status.unloadFailed"));
      toast("Unload failed: " + e.message, true);
    } finally {
      sidebarUnloadBtn.disabled = false;
    }
  };
}

// The model dropdown's placeholder and the status pill are written by
// refreshModels, so they are redrawn when the interface language changes.
document.addEventListener("localm:language", () => { refreshModels(); });
