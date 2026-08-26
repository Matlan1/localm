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
  } catch (_) { /* ignored */ }
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

