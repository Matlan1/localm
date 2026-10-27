// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - client error capture. */

"use strict";

// An in-memory ring of the last 50 JS errors, which the "Report a bug" control
// can attach. Loaded before the other app modules so early failures land here.

window.__localmClientLog = window.__localmClientLog || [];
export function __localmPushClientError(msg) {
  try {
    const line = String(msg).slice(0, 1000);
    const log = window.__localmClientLog;
    log.push(new Date().toISOString().slice(11, 19) + "  " + line);
    if (log.length > 50) log.splice(0, log.length - 50);
  } catch { /* ignored */ }
}
window.addEventListener("error", (e) => {
  __localmPushClientError(
    (e && e.message ? e.message : "error") +
    (e && e.filename ? "  (" + e.filename + ":" + (e.lineno || "?") + ")" : ""));
});
window.addEventListener("unhandledrejection", (e) => {
  const r = e && e.reason;
  __localmPushClientError("unhandledrejection: " + (r && r.message ? r.message : r));
});
(() => {
  const orig = console.error;
  console.error = function (...args) {
    __localmPushClientError(args.map((a) => {
      try { return typeof a === "string" ? a : (a && a.message) || JSON.stringify(a); }
      catch { return String(a); }
    }).join(" "));
    return orig.apply(console, args);
  };
})();


// --- GUI errors: the "Recent errors" list and the server error log ---------

export const RECENT_GUI_ERRORS_MAX = 5;
const GUI_ERROR_RESEND_MS = 5000;
const GUI_ERROR_SEND_PER_MIN = 30;
const GUI_ERROR_MAX_CHARS = 1000;

// Newest first: { text, time (ms), page, source, count }.
let _guiErrRecent = [];
const _guiErrListeners = new Set();
// { save(list) }, set once the server's privacy mode is known; null keeps the
// list in memory only.
let _guiErrStore = null;
let _guiErrAuth = () => ({});
const _guiErrLastSent = new Map();
let _guiErrSendTimes = [];

/** The recent GUI errors, newest first (a copy). */
export function recentGuiErrors() {
  return _guiErrRecent.map((e) => ({ ...e }));
}

/** Call *fn* after every change of the recent-errors list. */
export function onRecentGuiErrorsChange(fn) {
  _guiErrListeners.add(fn);
  return () => _guiErrListeners.delete(fn);
}

function _guiErrChanged() {
  if (_guiErrStore) {
    try { _guiErrStore.save(recentGuiErrors()); } catch { /* storage full/blocked */ }
  }
  for (const fn of _guiErrListeners) {
    try { fn(recentGuiErrors()); } catch { /* a broken listener must not stop the others */ }
  }
}

/** Empty the recent-errors list. */
export function clearRecentGuiErrors() {
  _guiErrRecent = [];
  _guiErrChanged();
}

/** Set where the list is kept: *store* ({ save(list) }) persists it, null keeps
 *  it in memory only. */
export function setRecentGuiErrorStore(store) {
  _guiErrStore = store || null;
}

/** Add stored entries (newest first) below the ones already in memory, keeping
 *  at most RECENT_GUI_ERRORS_MAX. Malformed entries are skipped. */
export function restoreRecentGuiErrors(list) {
  if (!Array.isArray(list)) return;
  const stored = list.filter((e) => e && typeof e.text === "string" && e.text)
    .map((e) => ({
      text: e.text.slice(0, GUI_ERROR_MAX_CHARS),
      time: Number(e.time) || 0,
      page: typeof e.page === "string" ? e.page.slice(0, 128) : "",
      source: typeof e.source === "string" ? e.source.slice(0, 64) : "",
      count: Math.max(1, parseInt(e.count, 10) || 1),
    }));
  _guiErrRecent = [..._guiErrRecent, ...stored].slice(0, RECENT_GUI_ERRORS_MAX);
  _guiErrChanged();
}

/** Set the function that returns the auth headers for the error report POST. */
export function setGuiErrorAuth(fn) {
  if (typeof fn === "function") _guiErrAuth = fn;
}

function _guiErrPage() {
  try { return location.hash || location.pathname || ""; } catch { return ""; }
}

// POST one report to /api/gui-error. Never throws, never logs, never retries:
// a failed report is dropped. The same message is sent at most once per
// GUI_ERROR_RESEND_MS, and at most GUI_ERROR_SEND_PER_MIN reports per minute.
function _guiErrSend(text, source, page) {
  try {
    const now = Date.now();
    const key = source + "\u0000" + text;
    const last = _guiErrLastSent.get(key);
    if (last !== undefined && now - last < GUI_ERROR_RESEND_MS) return;
    _guiErrSendTimes = _guiErrSendTimes.filter((t) => now - t < 60000);
    if (_guiErrSendTimes.length >= GUI_ERROR_SEND_PER_MIN) return;
    _guiErrSendTimes.push(now);
    _guiErrLastSent.set(key, now);
    if (_guiErrLastSent.size > 200) _guiErrLastSent.delete(_guiErrLastSent.keys().next().value);
    const send = window.__localmRawFetch || window.fetch;
    if (typeof send !== "function") return;
    let headers = {};
    try { headers = _guiErrAuth() || {}; } catch { headers = {}; }
    const p = send("/api/gui-error", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: JSON.stringify({ message: text, source, page, time: new Date(now).toISOString() }),
      keepalive: true,
    });
    if (p && typeof p.catch === "function") p.catch(() => {});
  } catch { /* dropped */ }
}

/** Record an error the GUI showed or threw: add it to the recent-errors list
 *  (an error equal to one already listed moves to the top and raises its
 *  count instead) and send it
 *  to the server log. *source* names where it came from ("toast", "status",
 *  "window", "promise"). Never throws. */
export function recordGuiError(message, source = "gui") {
  try {
    const text = String(message ?? "").trim().slice(0, GUI_ERROR_MAX_CHARS);
    if (!text) return;
    const page = _guiErrPage();
    const i = _guiErrRecent.findIndex((e) => e.text === text && e.source === source);
    const count = i >= 0 ? _guiErrRecent.splice(i, 1)[0].count + 1 : 1;
    _guiErrRecent.unshift({ text, time: Date.now(), page, source, count });
    if (_guiErrRecent.length > RECENT_GUI_ERRORS_MAX) _guiErrRecent.length = RECENT_GUI_ERRORS_MAX;
    _guiErrChanged();
    _guiErrSend(text, source, page);
  } catch { /* recording an error must never raise */ }
}

window.addEventListener("error", (e) => {
  if (e && e.target && e.target !== window && !e.message) return;
  recordGuiError((e && e.message ? e.message : "error") +
    (e && e.filename ? "  (" + e.filename + ":" + (e.lineno || "?") + ")" : ""), "window");
});
window.addEventListener("unhandledrejection", (e) => {
  const r = e && e.reason;
  recordGuiError("Unhandled rejection: " + (r && r.message ? r.message : String(r)), "promise");
});
