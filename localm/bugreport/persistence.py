# SPDX-License-Identifier: AGPL-3.0-or-later
"""Saving a report under the data dir, and building and saving a
user-initiated report.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from localm.bugreport.logs import _recent_hang_traces, _recent_log_tail_result
from localm.bugreport.assembly import build_report, report_title
import localm.bugreport as _br

def save_report(text: str, when: Optional[str] = None) -> Optional[Path]:
    """Write the report to the data dir and return its path (or None on failure).
    *when* is a caller-supplied timestamp string (kept injectable for tests)."""
    try:
        from localm.config import home_dir
        base = home_dir() / "bug-reports"
    except Exception:
        # Last-resort fallback (config import failed): stay CONTAINED inside the
        # install, never a shared ~/.localm outside it.
        base = Path(__file__).resolve().parents[2] / "home" / "bug-reports"
    try:
        base.mkdir(parents=True, exist_ok=True)
        if when is None:
            import time
            when = time.strftime("%Y%m%d-%H%M%S")
        path = base / f"bug-{when}.md"
        path.write_text(text, encoding="utf-8")
        # The report carries environment diagnostics and a traceback; on a
        # multi-user box it should not be world-readable. chmod is a no-op on
        # Windows, so this is safe cross-platform.
        try:
            path.chmod(0o600)
        except OSError:
            # chmod is a no-op on Windows and only fails on POSIX filesystems
            # that do not support per-file perms (e.g. a FAT or network mount),
            # where the data dir's own perms still apply. The report carries NO
            # secrets (no API key, env, config secrets, or chat content - see
            # the localm.bugreport package docstring).
            pass
        return path
    except Exception:
        return None


# The GUI route offloads save_user_report() to a real worker thread (it does
# blocking file I/O and must not stall the event loop), so two reports filed
# close together can run this function on two DIFFERENT threads at the same
# instant. Two hazards follow: _ring_activity() reads then deletes the shared
# pre_restart.log (a second reader can race the first's unlink), and
# save_report() can open the SAME same-second-timestamped filename for writing
# from two threads at once, interleaving their content into one corrupted file
# instead of cleanly losing one report. This lock serialises both. Scoped to
# this function, not a module-wide lock shared with report_failure's own
# build_report()/save_report() calls: report_failure is the CLI/crash path, a
# separate process per invocation with no in-process concurrency.
_SAVE_REPORT_LOCK = threading.Lock()


def save_user_report(description: str = "", *, summary: str = "",
                     what_i_expected: str = "", what_happened: str = "",
                     include_log: bool = False,
                     client: Optional[dict] = None,
                     extra_hang_trace: str = "") -> Optional[Path]:
    """Build and save a USER-initiated bug report and return its path.

    The shared backend for the GUI "Report a bug" control and the
    ``localm bug-report`` CLI: *description* fills the "What I was doing"
    section, *what_i_expected* and *what_happened* fill their own sections -
    three DISTINCT fields, not one string echoed three times. *summary*
    overrides the title; when omitted it is derived from *what_happened*
    (falling back to *description* when that too is empty), since "what
    happened" makes a more useful issue title than "what I was doing". The safe
    environment snapshot is collected as usual (loaded model, effective backend,
    session mode, a safe config subset, key dependency versions, and the
    in-memory recent-activity log), and with *include_log* the current run's
    on-disk debug log tail is attached (home-scrubbed at source, and with every
    chat-content record dropped by build_digest regardless of what it says -
    never the API key, config secrets, or chat content). *client* is an
    optional GUI-supplied browser context (user agent, page, viewport, recent
    JS console errors). *extra_hang_trace* lets the CLI supply a freeze trace
    it found in a DIFFERENT process (see live_server_hang_trace) - the self-pid
    check below only ever catches the caller's own freeze, which is never the
    CLI's own case. Returns None on a write failure (the caller surfaces that
    rather than reporting a false success). Serialized on
    ``_SAVE_REPORT_LOCK`` (see the comment above that lock) so two GUI-triggered
    calls running on different worker threads at once cannot race the shared
    pre_restart.log read+delete or collide on the same-second report filename."""
    with _SAVE_REPORT_LOCK:
        description = (description or "").strip()
        what_i_expected = (what_i_expected or "").strip()
        what_happened = (what_happened or "").strip()
        summary = report_title(summary, what_happened, description)
        context: dict = {"operation": "gui-bug-report"}
        if description:
            context["what_i_did"] = description
        if what_i_expected:
            context["what_i_expected"] = what_i_expected
        if what_happened:
            context["what_happened"] = what_happened
        if include_log:
            import os
            # The reason matters only when the user ASKED for the log: with
            # include_log false there is nothing to explain, and a "not
            # collected" line would misreport an opt-out as a failure.
            tail, log_unavailable = _recent_log_tail_result(pid=os.getpid())
            if tail:
                context["recent_log_tail"] = tail
            elif log_unavailable:
                context["log_unavailable"] = log_unavailable
        # Always attach a hang trace captured by THIS run (independent of
        # include_log): it only exists if the server actually froze. Scoped to
        # our own pid like the log tail above, so an old freeze from a previous
        # run is never presented as this report's diagnosis.
        import os as _os
        hang = _recent_hang_traces(pid=_os.getpid())
        # extra_hang_trace covers the ``localm bug-report`` CLI's own case: it is
        # a separate, short-lived process whose OWN pid never froze, while the
        # server that DID freeze is a different, still-running process - a pid-of-
        # self match finds nothing there, so the CLI looks the live server up itself
        # (live_server_hang_trace) and passes what it found in here instead.
        if extra_hang_trace:
            hang = f"{hang}\n\n{extra_hang_trace}" if hang else extra_hang_trace
        if hang:
            context["hang_traces"] = hang
        if isinstance(client, dict) and client:
            context["client"] = client
        text = build_report(summary, context=context)
        return _br.save_report(text)
