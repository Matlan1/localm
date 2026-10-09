# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reporting a prior server run that died hard: classifying the death from its
trace, log and exit code, and filing the report on the next start.
"""

from __future__ import annotations

import re
from typing import Optional

from localm.bugreport.logs import _find_run_log, _recent_hang_traces, _recent_log_tail_result
from localm.bugreport.assembly import trim_trace_for_report
from localm.bugreport.crash_guard import _all_crash_markers, _all_crash_stopping_records, _read_exit_record, _trace_path_for_marker, _trace_path_for_stopping_record
import localm.bugreport as _br


# A native/ggml status line printed via raw fprintf (no "TIMESTAMP LEVEL NAME:"
# prefix - see debuglog.py's dedup_native_stderr) - the vocabulary a hard native
# crash mid-load/mid-generate is seen stopping inside, e.g. a log stopping dead
# at "llama_co" during "llama_context: constructing llama_context".
_NATIVE_OP_LINE_RE = re.compile(
    r'^(llama_|ggml_|graph_reserve|sched_reserve|load_tensors|load_all_data|create_tensor)')


# How much of the raw log tail to read for the truncation check below, which
# needs only the last line rather than a digest.
_TRUNCATION_CHECK_READ_BYTES = 4000


# faulthandler's own fault headers: "Windows fatal exception: <description>"
# on Windows, "Fatal Python error: <signal>" everywhere else.
_WIN_FAULT_LINE_RE = re.compile(r"^Windows fatal exception:\s*(.+?)\s*$")


_POSIX_FAULT_LINE_RE = re.compile(r"^Fatal Python error:\s*(.+?)\s*$")


# A numeric Windows exception code, for the codes faulthandler has no name for.
_WIN_FAULT_CODE_RE = re.compile(r"^code 0x([0-9a-f]{8})$", re.I)


def _is_fatal_fault_description(desc: str) -> bool:
    """Whether a faulthandler fault description names a fault that actually
    kills the process.

    faulthandler's Windows handler LOGS an exception and then returns
    EXCEPTION_CONTINUE_SEARCH, so a trace is written for first-chance
    exceptions the process goes on to handle and survive. Measured
    2026-09-03: a healthy standalone app-window start writes
    "Windows fatal exception: code 0x8001010d"
    (RPC_E_CANTCALLOUT_ININPUTSYNCCALL, raised and handled inside
    WebView2/.NET while the window is created) and then runs normally to a
    clean exit. Treating any trace as proof of a crash therefore reports a
    crash that never happened.

    A description faulthandler NAMED (access violation, stack overflow, ...)
    is a genuine hardware/OS fault. A bare numeric code is fatal only in the
    NTSTATUS error range (0xC.......); the 0x8....... HRESULT range, the CLR's
    own exception code and the C++ throw code are all raised by software and
    routinely handled.
    """
    m = _WIN_FAULT_CODE_RE.match(desc.strip())
    if m is None:
        return True   # a named fault: access violation, stack overflow, ...
    return m.group(1).lower().startswith("c")


def _fatal_fault_line(native_trace: str) -> str:
    """The first line of *native_trace* naming a fault that actually killed the
    process, or "" when the trace holds none (see
    _is_fatal_fault_description). Never raises."""
    try:
        for line in (native_trace or "").splitlines():
            line = line.strip()
            m = _WIN_FAULT_LINE_RE.match(line)
            if m is not None:
                if _is_fatal_fault_description(m.group(1)):
                    return line
                continue
            if _POSIX_FAULT_LINE_RE.match(line):
                return line
    except Exception:
        pass
    return ""


def _raw_tail_truncation_signal(home=None, pid=None) -> "tuple[bool, str]":
    """(truncated, last_line): whether the crashed run's OWN raw log file (see
    _find_run_log - the SAME file _recent_log_tail digests, read raw here
    instead) ends WITHOUT a trailing newline, and what that final (possibly
    mid-word) line is. A clean shutdown's last logged line is always
    terminator-flushed with a trailing newline; a log that stops mid-line -
    literally mid-word, e.g. "...llama_co" - is the signature of the process
    dying between two writes, not of a clean exit. Never raises; returns
    (False, "") on any failure or when there is nothing to check."""
    try:
        chosen = _find_run_log(home, pid)
        if chosen is None:
            return False, ""
        with open(chosen, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - _TRUNCATION_CHECK_READ_BYTES))
            raw = fh.read().decode("utf-8", errors="replace")
        if not raw:
            return False, ""
        truncated = not raw.endswith(("\n", "\r"))
        last_line = raw.replace("\r\n", "\n").rstrip("\n").rsplit("\n", 1)[-1]
        return truncated, last_line
    except Exception:
        return False, ""


def _first_chance_codes(native_trace: str) -> "list[str]":
    """The distinct exception codes of the non-fatal fault headers in
    *native_trace*, in order of first appearance."""
    seen: "list[str]" = []
    for line in (native_trace or "").splitlines():
        m = _WIN_FAULT_LINE_RE.match(line.strip())
        if m is None:
            continue
        code = _WIN_FAULT_CODE_RE.match(m.group(1).strip())
        if code is not None and not _is_fatal_fault_description(m.group(1)):
            label = "0x" + code.group(1).lower()
            if label not in seen:
                seen.append(label)
    return seen


def _classify_prior_death(*, native_trace: str, hang_trace: str,
                          raw_tail_truncated: bool,
                          raw_tail_last_line: str,
                          exit_code: Optional[int] = None) -> "tuple[str, str]":
    """(summary, reason) for report_failure(), classified from the evidence
    actually collected. Pure function (no I/O) so the classification logic is
    directly unit-testable without real marker/log/trace files.

    Ordered by how direct the evidence is: a captured native trace beats an
    inferred mid-operation cutoff beats a captured hang, and only when NONE of
    those fired does it fall back to naming the remaining ambiguity (an OS kill
    / force-closed window legitimately leaves no trace at all).

    Only a trace naming a FATAL fault counts as the first kind of evidence: a
    trace holding nothing but survivable first-chance exceptions is not a
    crash report, it is noise the run wrote and lived through (see
    _fatal_fault_line). Such a trace is still attached to the report - it is
    context, just not the cause."""
    fatal_fault = _fatal_fault_line(native_trace)
    exit_text = ""
    exit_is_fault = False
    if exit_code is not None:
        from localm._mp_spawn import death_was_a_native_fault, describe_exit_code
        exit_text = describe_exit_code(exit_code, posix=False)
        exit_is_fault = death_was_a_native_fault(exit_code, posix=False)
    if fatal_fault:
        return (
            f"localm server crashed - native fault captured: {fatal_fault}",
            "a native crash was caught by the fault handler; see the captured "
            "trace below for the exact fault and thread/frame."
            + (f" The process exited with {exit_text}." if exit_text else "")
        )
    if exit_is_fault:
        survived = _first_chance_codes(native_trace)
        trace_note = (
            f"The fault handler's trace holds only first-chance exception(s) "
            f"{', '.join(survived)} that the process survived, and no fatal "
            f"fault line, so the fault that ended the process was not recorded "
            f"by the handler." if survived else
            "The fault handler recorded no fatal fault line.")
        return (
            f"localm server crashed - exited with {exit_text}",
            f"the previous server process ended with exit code {exit_text}, "
            f"a native fault. {trace_note}")
    if raw_tail_truncated and _NATIVE_OP_LINE_RE.match(raw_tail_last_line.strip()):
        return (
            "localm server crashed during model load/construction "
            "(native crash suspected, no trace captured)",
            f'the previous run\'s log stops abruptly mid-line inside '
            f'"{raw_tail_last_line.strip()}" - no faulthandler trace was '
            f'captured for this crash (a known open gap - see this report\'s '
            f'"Native fault trace" section, or its absence), but a log cut off '
            f'mid-operation during native model construction is the signature '
            f'of a hard native crash, not a clean stop.'
        )
    if hang_trace.strip():
        return (
            "localm server appeared frozen (unresponsive) before this run ended",
            "the always-on hang watchdog captured a stall before the process "
            "stopped - it was most likely force-closed after becoming "
            "unresponsive, not a native crash. See the captured stacks below."
        )
    return (
        "localm server crashed (recovered on the next start)",
        "the previous server run ended without a clean shutdown, and none of "
        "a native trace, a mid-operation log cutoff, or a captured hang were "
        "found in the evidence collected - most likely an OS kill or a "
        "force-closed window (both legitimately leave no trace)."
    )


def _report_one_crash_marker(d, marker, home, interactive: bool):
    """Report *marker* as a crash IF its recorded pid is no longer alive, then
    clear it (the marker and its trace file). The report is filed only when
    diagnostics are allowed now (``audit.diagnostics_allowed()``) and the
    marker does not record ``"diagnostics": false`` (a run armed in privacy
    mode); a marker without the key is reported like any other. A dead-pid
    marker that is not reported is still cleared, and logged at INFO. Returns
    the report path, or None if this marker was skipped (a live sibling
    instance, or not reported) or nothing could be filed. Never raises."""
    import json
    from localm.debuglog import logger
    from localm.instances import pid_alive
    try:
        if not marker.exists():
            return None
        info = {}
        try:
            info = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            # A corrupt/half-written marker (the process may have died mid-write)
            # still means a crash happened: fall back to empty context and report
            # it anyway rather than dropping the crash on the floor.
            pass
        pid = info.get("pid")
        try:
            if pid_alive(int(pid)):
                # The recorded pid is still running: this is a SIBLING instance
                # that is simply still up, not a crash. Its marker is left
                # alone - its own eventual disarm (clean exit) or a future check
                # here (real crash) handles it. PID reuse is an accepted
                # residual risk: the marker records no whoami-style identity
                # check to rule it out.
                return None
        except (TypeError, ValueError):
            # No/unparseable pid recorded: cannot confirm liveness, so treat it
            # like the corrupt-marker case above - report rather than drop it.
            pass
        report = _br._diagnostics_allowed() and info.get("diagnostics") is not False
        trace = ""
        tp = _trace_path_for_marker(d, marker)
        try:
            if report and tp.exists():
                trace = tp.read_text(encoding="utf-8").strip()
        except Exception:
            # Best-effort: the native traceback file is optional extra context.
            # If unreadable we report the crash without it rather than not at all.
            pass
        # Clear FIRST so a failure while reporting cannot loop the marker forever.
        try:
            marker.unlink(missing_ok=True)
        except Exception:
            # If the marker cannot be removed the next start may re-report this
            # same crash (a duplicate, not a lost one). Tolerable; do not abort the
            # report we are about to file over a failed unlink.
            pass
        # Delete the trace WITH its marker, now that its content (if any) has
        # been folded into ctx below. Best-effort, same as the marker unlink
        # above - a failure here must not block or duplicate the report.
        try:
            tp.unlink(missing_ok=True)
        except Exception:
            pass
        if not report:
            logger.info(
                "bugreport: prior hard crash detected (%s); not reported: "
                "privacy mode", marker.name)
            return None
        ctx = {"prior_run": info}
        exit_record = _read_exit_record(d, marker)
        if exit_record:
            ctx["prior_exit"] = exit_record
        if trace:
            ctx["native_trace"] = trim_trace_for_report(trace)
        # Attach the crashed run's own log tail (matched by pid) so the report is
        # actionable even with no native trace (window-close / OS-kill leave none).
        tail, log_unavailable = _recent_log_tail_result(home, pid=info.get("pid"))
        if tail:
            ctx["recent_log_tail"] = tail
        elif log_unavailable:
            # Say WHY the log is missing: a crash report with no native trace
            # and no log is otherwise contentless.
            ctx["log_unavailable"] = log_unavailable
        # If the watchdog captured a freeze before the run was force-killed, attach
        # its stacks too: a hang the user force-quit is exactly this recovery path.
        # The CRASHED run's freeze, matched by its pid like the log tail above -
        # not this recovering process's, and not some unrelated older one.
        hang = _recent_hang_traces(home, pid=info.get("pid"))
        if hang:
            ctx["hang_traces"] = hang
        # Classify the death from the evidence actually collected, rather than
        # a blind 3-way guess ("a native crash, an OS kill, or a force-closed
        # window") that names neither which one nor what was in flight.
        raw_truncated, raw_last_line = _raw_tail_truncation_signal(
            home, pid=info.get("pid"))
        summary, reason = _classify_prior_death(
            native_trace=trace, hang_trace=hang,
            raw_tail_truncated=raw_truncated, raw_tail_last_line=raw_last_line,
            exit_code=exit_record.get("exit_code"))
        return _br.report_failure(
            summary=summary, reason=reason,
            error=None, context=ctx, interactive=interactive)
    except Exception:
        return None


def _report_one_stopping_record(d, record, home, interactive: bool):
    """Handle a stopping record (see clear_crash_marker) whose run ended before
    release_crash_trace() ran: skipped while its recorded pid is still alive,
    otherwise deleted together with its trace file. When that trace names a
    fatal fault (see _fatal_fault_line), diagnostics are allowed now and the
    run did not arm in privacy mode (``"diagnostics": false``), it is reported
    as a crash during the stop. Returns the report path, or None. Never
    raises."""
    import json
    from localm.debuglog import logger
    from localm.instances import pid_alive
    try:
        info = {}
        try:
            info = json.loads(record.read_text(encoding="utf-8"))
        except Exception:
            # A corrupt record still pairs with its trace: handled like one
            # whose pid is gone.
            pass
        try:
            if pid_alive(int(info.get("pid"))):
                return None
        except (TypeError, ValueError):
            pass
        trace = _trace_path_for_stopping_record(d, record)
        text = ""
        try:
            if trace.exists():
                text = trace.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            pass
        try:
            record.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            trace.unlink(missing_ok=True)
        except Exception:
            pass
        fatal_fault = _fatal_fault_line(text)
        if not fatal_fault:
            return None
        if not (_br._diagnostics_allowed() and info.get("diagnostics") is not False):
            logger.info(
                "bugreport: native fault during a prior clean stop detected (%s); "
                "not reported: privacy mode", record.name)
            return None
        ctx = {"prior_run": dict(info, stage="stopping"),
               "native_trace": trim_trace_for_report(text)}
        tail, log_unavailable = _recent_log_tail_result(home, pid=info.get("pid"))
        if tail:
            ctx["recent_log_tail"] = tail
        elif log_unavailable:
            ctx["log_unavailable"] = log_unavailable
        return _br.report_failure(
            summary=("localm server crashed while stopping - native fault "
                     f"captured: {fatal_fault}"),
            reason=("a native crash was caught by the fault handler after the "
                    "server had begun a clean stop; see the captured trace below "
                    "for the exact fault and thread/frame."),
            error=None, context=ctx, interactive=interactive)
    except Exception:
        return None


def check_and_report_prior_crash(home=None, interactive: bool = False):
    """Scan every crash marker left under this LOCALM_HOME's run/ dir (one per
    instance that has ever armed here) and report+clear any whose recorded pid
    is no longer alive - a hard crash: a native fault, an OS kill, or a
    force-closed window. A marker whose pid IS still alive is a sibling
    instance simply still running (see the per-instance-scoping note in
    crash_guard.py) and is left untouched: not reported, not deleted. Then handle every
    stopping record left by a run that began a clean stop and ended before
    releasing its trace (see _report_one_stopping_record). Returns the last
    report path filed, or None if nothing was reported. Never raises into the
    caller."""
    try:
        d = _br._crash_dir(home)
        markers = _all_crash_markers(d)
    except Exception:
        return None
    filed = None
    for marker in markers:
        result = _report_one_crash_marker(d, marker, home, interactive)
        if result is not None:
            filed = result
    try:
        records = _all_crash_stopping_records(d)
    except Exception:
        records = []
    for record in records:
        result = _report_one_stopping_record(d, record, home, interactive)
        if result is not None:
            filed = result
    return filed
