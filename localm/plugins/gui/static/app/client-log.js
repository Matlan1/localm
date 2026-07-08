// SPDX-License-Identifier: AGPL-3.0-or-later
/* localm GUI - vanilla JS, no build step.
   Talks to the localm FastAPI server: /v1 (OpenAI-compatible) + /api (GUI).
   All model/agent-originating strings go through textContent or DOMPurify -
   never raw innerHTML. pages.js builds on the helpers defined here. */

"use strict";

// Client error capture (for bug reports): an in-memory ring of recent JS errors
// the "Report a bug" control can attach. Client-side only; leaves the machine
// solely inside a report the user reviews and sends. Installed first to catch
// early failures.

window.__localmClientLog = window.__localmClientLog || [];
export function __localmPushClientError(msg) {
  try {
    const line = String(msg).slice(0, 1000);
    const log = window.__localmClientLog;
    log.push(new Date().toISOString().slice(11, 19) + "  " + line);
    if (log.length > 50) log.splice(0, log.length - 50);
  } catch (_) { /* never let logging break the app */ }
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
      catch (_) { return String(a); }
    }).join(" "));
    return orig.apply(console, args);
  };
})();

