# SPDX-License-Identifier: AGPL-3.0-or-later
"""A run's own log digest, the reason it could not be collected, and the
event-loop hang traces matched to a run.
"""

from __future__ import annotations

import localm.bugreport as _br

# Outer safety cap on how much of the raw log FILE is even read before digesting -
# a per-run debug log is not expected to reach this, but a pathological long-lived
# run must not make digest-building read (and regex-scan) an unbounded file.
_LOG_TAIL_READ_BYTES = 2_000_000


def _find_run_log(home=None, pid=None):
    """The Path to a run's OWN log, matched by the pid embedded in the log
    filename (localm_<date>_<time>_<pid>.log), or None if none is found.
    Shared by _recent_log_tail (below) and _classify_prior_death's raw-tail
    truncation check, so the two never disagree about WHICH file a crash's
    evidence comes from. Never raises."""
    from pathlib import Path as _P
    if home is None:
        from localm.debuglog import logs_dir
        d = logs_dir()
    else:
        d = _P(home) / "logs"
    if not d.is_dir():
        return None
    logs = sorted(d.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pid is not None:
        for p in logs:
            if p.stem.endswith(f"_{pid}"):
                return p
    # No pid match (older marker): fall back to the most recent log that is
    # NOT this current run's, so we do not just echo the recovering start.
    import os as _os
    cur = f"_{_os.getpid()}.log"
    return next((p for p in logs if not p.name.endswith(cur)), None)


# Why the log digest came back empty. An empty digest has three unrelated
# causes - no log file matched this run's pid, the file was found but could not
# be read, or the run genuinely logged nothing notable. The first two are
# FAILURES to collect and the third is a clean result, so a caller attaches one
# of these reasons and the report SAYS which happened.
_LOG_UNAVAILABLE_NO_FILE = "no log file was found for that run"


def _log_failure_reason(exc: BaseException) -> str:
    """A one-line, PATH-FREE description of why reading the log failed, safe to
    put in a share-intended report.

    The obvious formattings leak the user's home directory, and therefore their
    username:

        str(exc)      "[Errno 13] Permission denied: " + the FULL absolute path
                      of the log file, which is under the user's home dir and so
                      contains their account name.                          LEAKS
        exc.filename  that same absolute path.                              LEAKS
        repr(exc)     "PermissionError(13, 'Permission denied')" - safe for
                      OSError ONLY, because its __repr__ drops the filename,
                      while a ValueError("boom " + path) reprs with the path
                      intact.

    So the reason is built from the exception CLASS NAME plus ``strerror`` - the
    OS's own errno text ("Permission denied"), which carries no path and is
    absent on non-OSError, where the class name alone is all we say. A detail
    containing a path separator is dropped outright rather than trusted, which
    makes a leak UNREPRESENTABLE. _scrub_secrets is a further backstop for the
    home-path form.
    """
    name = type(exc).__name__
    detail = getattr(exc, "strerror", None)
    if not isinstance(detail, str):
        detail = ""
    detail = detail.strip()
    # Structural guard, not a scrubber: anything path-shaped is discarded whole.
    if any(sep in detail for sep in ("/", "\\")):
        detail = ""
    reason = f"{name}: {detail}" if detail else name
    return _br._scrub_secrets(reason)[:200]


def _recent_log_tail_result(home=None, pid=None, max_chars: int = 6000) -> tuple:
    """``(digest, unavailable_reason)`` - the digest _recent_log_tail returns,
    plus WHY it is empty when it is empty.

    At most one of the two is ever non-empty. An empty reason alongside an empty
    digest is the honest third case: the log WAS collected and simply held
    nothing notable, which needs no line in the report. Never raises."""
    try:
        chosen = _find_run_log(home, pid)
        if chosen is None:
            return "", _LOG_UNAVAILABLE_NO_FILE
        raw = chosen.read_text(encoding="utf-8", errors="replace")
        truncated = len(raw) > _br._LOG_TAIL_READ_BYTES
        if truncated:
            raw = raw[-_br._LOG_TAIL_READ_BYTES:]
        from localm._log_digest import build_digest
        # start_tainted=truncated: a truncated tail can start mid-way through a
        # debug_content_enabled() write with no header of its own, so
        # build_digest must not trust whatever it finds first.
        return _br._scrub_secrets(
            build_digest(raw, max_chars=max_chars, start_tainted=truncated)), ""
    except Exception as e:
        # Swallowed: collecting a log must never break the report it is attached
        # to. The caller is TOLD instead, so an unreadable log stays
        # distinguishable from a clean one.
        return "", f"the log file could not be read ({_log_failure_reason(e)})"


def _recent_log_tail(home=None, pid=None, max_chars: int = 6000) -> str:
    """A digest of the crashed run's OWN log, matched by the pid embedded in the
    log filename (localm_<date>_<time>_<pid>.log): EVERY warning/error (with its
    full traceback) from the whole run, with runs of near-duplicate benign lines
    (e.g. routine ``GET /api/stats`` polling) collapsed to one line + a repeat
    count - see localm/_log_digest.py. The failure survives however much routine
    activity followed it before the report was filed. Chat content (a raw model
    reply, a memory-embed snippet, a web-tool query) is dropped by build_digest
    before this ever sees it, whatever the content itself says. Home paths are
    scrubbed. Never raises.

    Returns only the digest. A caller that renders a REPORT wants
    _recent_log_tail_result instead, so it can say why an empty digest is
    empty."""
    return _recent_log_tail_result(home, pid, max_chars)[0]


# A hang trace older than this cannot belong to the run being reported, whatever
# pid it carries (pids are reused). The pid match below is the precise filter;
# this bound only backstops a pid collision.
_HANG_TRACE_MAX_AGE_S = 24 * 3600


def _recent_hang_traces(home=None, max_chars: int = 8000, *, pid=None) -> str:
    """The event-loop hang trace captured by the run being reported
    (``<logs>/hang_<date>_<pid>.log``), if the always-on stall watchdog captured
    one - every thread's stack at the moment the server froze, which is exactly
    what diagnoses an intermittent "it hung" report. Kept HEAD-first (the first
    snapshot names where the loop first stuck). Home paths are scrubbed. Empty
    when that run captured no freeze. Never raises.

    Matched to *pid* - the run this report is about - the same way
    _recent_log_tail already matches its log, and additionally bounded by
    _HANG_TRACE_MAX_AGE_S. Both filters are needed and neither is enough alone:

      * The watchdog is ON by default, so any transient >10s stall (a big index
        build, VRAM pressure) writes a trace, and nothing ever prunes them. With
        no filter, an ordinary report about something else entirely ("the model
        gave a wrong answer") would attach whatever freeze happened to be
        newest, rendered under "## Server hang trace" asserting "the server
        froze".
      * A pid alone is not an identity (pids get reused), so an ancient trace
        carrying this run's pid would still slip through; the age bound stops it.
      * Age alone is not enough either: a recent freeze in a DIFFERENT localm
        instance is not this report's problem; the pid match stops that.

    A caller that cannot name the run (no pid) gets nothing rather than a
    guess."""
    try:
        import time as _time
        from pathlib import Path as _P
        if home is None:
            from localm.debuglog import logs_dir
            d = logs_dir()
        else:
            d = _P(home) / "logs"
        if not d.is_dir() or pid is None:
            return ""
        cutoff = _time.time() - _HANG_TRACE_MAX_AGE_S
        traces = sorted(
            (p for p in d.glob(f"hang_*_{pid}.log") if p.stat().st_mtime >= cutoff),
            key=lambda p: p.stat().st_mtime, reverse=True)
        if not traces:
            return ""
        text = traces[0].read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return ""
        scrubbed = _br._scrub_secrets(text)
        if len(scrubbed) > max_chars:
            scrubbed = scrubbed[:max_chars] + \
                "\n... (trace truncated - full stacks in the hang_*.log file)"
        return scrubbed
    except Exception:
        return ""


def live_server_hang_trace(home=None) -> str:
    """The event-loop hang trace captured by a LIVE server of this install, for a
    user-initiated ``localm bug-report`` filed from a SEPARATE process.

    The two in-process collectors each feed _recent_hang_traces a pid they already
    know: ``save_user_report`` (GUI) matches its own ``getpid()`` because the GUI IS
    the server, and crash recovery matches the crashed run's marker pid. The
    ``localm bug-report`` CLI has neither - it is its OWN short-lived process, whose
    pid never wrote a hang trace, while the frozen server is a DIFFERENT process
    still running, so a pid-of-self match finds nothing there.

    Instead we look the server up in the live instance registry
    (``<home>/run/*.json``), keep only entries whose pid is actually alive, and
    return that server's pid-matched trace - the SAME pid-scoped _recent_hang_traces
    collection the crash path uses, just sourced from the live registry rather than
    a crash marker. With more than one live server, the most recently started one
    wins: a single, stable choice (build_report renders one hang-trace section).
    Empty when no live server captured a freeze. Never raises - a registry hiccup
    must not break filing a bug report."""
    try:
        from localm import instances
        if home is None:
            from localm.config import HOME_DIR
            home = HOME_DIR
        entries = sorted(instances.list_entries(home),
                         key=lambda e: e.get("started") or "", reverse=True)
        for entry in entries:
            pid = entry.get("pid")
            if not isinstance(pid, int) or not instances.pid_alive(pid):
                continue
            trace = _recent_hang_traces(home, pid=pid)
            if trace:
                return trace
        return ""
    except Exception:
        return ""
