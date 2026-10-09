# SPDX-License-Identifier: AGPL-3.0-or-later
"""The per-instance crash marker, stopping record and native-fault trace a
running server keeps under ``<home>/run``.
"""

from __future__ import annotations

from typing import Optional

import localm.bugreport as _br


# --------------------------------------------------------------------------- #
#  asyncio + native-crash net: "fire the bug reporter no matter what"          #
#                                                                              #
#  The excepthooks in handlers.py cover the main thread and background         #
#  threads. Two further failure modes: (1) an uncaught exception inside an     #
#  asyncio task (uvicorn's event loop) only logs "Task exception was never     #
#  retrieved";                                                                 #
#  (2) a NATIVE crash (a C-extension segfault, an OS kill, or a force-closed   #
#  console window) cannot be caught in-process at all. (1) is closed with an   #
#  asyncio exception handler and (2) with a crash marker: the server arms a    #
#  marker on start and disarms it on a clean shutdown, so a marker still       #
#  present on the NEXT start means the previous run died hard. The report is   #
#  filed then, with the native traceback faulthandler captured, when           #
#  audit.diagnostics_allowed() holds (log/full mode, or keep_diagnostics on);  #
#  in privacy mode the trace is never written and the crash is only logged.    #
# --------------------------------------------------------------------------- #


def _diagnostics_allowed() -> bool:
    """audit.diagnostics_allowed(), False if it cannot be resolved. Never raises."""
    try:
        from localm.audit import diagnostics_allowed
        return bool(diagnostics_allowed())
    except Exception:
        return False


def _crash_dir(home=None):
    from pathlib import Path
    if home is None:
        from localm.config import HOME_DIR
        home = HOME_DIR
    d = Path(home) / "run"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Running more than one localm server against the SAME LOCALM_HOME is a
# first-class, supported scenario (`localm ps` lists "running localm servers
# (per-directory instances)"; `serve --project/--new/--isolated`; the coder
# plugin self-starting its own backing server). The marker and its companion
# native-fault-trace file are therefore scoped per instance_id - the same
# per-process identity instances.py's registry uses (``run/<instance_id>.json``)
# - so each running instance only ever arms, reports, and disarms its OWN file.
# With one unscoped file per home, a second instance starting up would find the
# FIRST instance's still-armed marker, misread it as "the previous run died
# hard", and file a spurious crash report about a server that was never down;
# its own later clean-shutdown disarm would then delete whatever marker existed
# at that point, which could by then belong to a THIRD, still-live instance,
# silencing a real crash of that instance forever.


def _crash_marker_path(d, instance_id: Optional[str]):
    if instance_id:
        return d / f"server-crash.{instance_id}.marker"
    # No instance identity available (a bare create_app() test harness that
    # never went through instances.advertise()): fall back to the legacy
    # shared name rather than silently skipping the crash guard.
    return d / "server-crash.marker"


def _crash_trace_path(d, instance_id: Optional[str]):
    if instance_id:
        return d / f"server-crash-trace.{instance_id}.txt"
    return d / "server-crash-trace.txt"


def _exit_path_for_marker(d, marker):
    """The exit record the crash-recovery watchdog wrote for *marker*'s
    instance (``server-crash-exit.<instance_id>.json``: the dead process's pid
    and exit code), derived from the marker's own filename."""
    name = marker.name
    prefix, suffix = "server-crash.", ".marker"
    if name.startswith(prefix) and name.endswith(suffix) and name != "server-crash.marker":
        return d / f"server-crash-exit.{name[len(prefix):-len(suffix)]}.json"
    return d / "server-crash-exit.json"


def _read_exit_record(d, marker) -> dict:
    """The watchdog's exit record for *marker*'s instance, consumed (deleted)
    on read. ``{}`` when there is none or it is unreadable. Never raises."""
    import json
    path = _exit_path_for_marker(d, marker)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
    if not isinstance(data, dict) or not isinstance(data.get("exit_code"), int):
        return {}
    return data


def _trace_path_for_marker(d, marker):
    """The native-fault-trace file that belongs WITH *marker* (same instance),
    derived from the marker's own filename rather than a second parameter, so a
    caller iterating markers it did not itself name (check_and_report_prior_crash)
    never has to keep the two paths in sync by hand."""
    name = marker.name
    prefix, suffix = "server-crash.", ".marker"
    if name.startswith(prefix) and name.endswith(suffix) and name != "server-crash.marker":
        return d / f"server-crash-trace.{name[len(prefix):-len(suffix)]}.txt"
    return d / "server-crash-trace.txt"


def _all_crash_markers(d):
    """Every armed marker under *d*: one per instance that has ever armed
    against this LOCALM_HOME (plus the legacy unscoped name, if present)."""
    markers = list(d.glob("server-crash.*.marker"))
    legacy = d / "server-crash.marker"
    if legacy.exists():
        markers.append(legacy)
    return markers


def _crash_stopping_path(d, instance_id: Optional[str]):
    if instance_id:
        return d / f"server-crash.{instance_id}.stopping"
    return d / "server-crash.stopping"


def _trace_path_for_stopping_record(d, record):
    """The native-fault-trace file that belongs WITH stopping *record* (same
    instance), derived from the record's own filename."""
    name = record.name
    prefix, suffix = "server-crash.", ".stopping"
    if name.startswith(prefix) and name.endswith(suffix) and name != "server-crash.stopping":
        return d / f"server-crash-trace.{name[len(prefix):-len(suffix)]}.txt"
    return d / "server-crash-trace.txt"


def _all_crash_stopping_records(d):
    """Every stopping record under *d* (see clear_crash_marker), plus the legacy
    unscoped name if present."""
    records = list(d.glob("server-crash.*.stopping"))
    legacy = d / "server-crash.stopping"
    if legacy.exists():
        records.append(legacy)
    return records


def arm_crash_guard(context: Optional[dict] = None, home=None,
                    instance_id: Optional[str] = None) -> bool:
    """Mark that a server run is in progress, and when diagnostics are allowed
    (``audit.diagnostics_allowed()``: the log/full modes, or privacy mode with
    ``keep_diagnostics`` on) also open the native-fault trace file and enable
    faulthandler on it. The marker is written in EVERY mode: it is the
    liveness record the crash-recovery watchdog reads. If the process dies
    hard the marker survives; check_and_report_prior_crash() reports it on the
    next start. *instance_id* (``app.state.instance_id``, set by
    instances.advertise() before this is called) scopes the marker to THIS
    running instance so a sibling instance sharing the same LOCALM_HOME is
    never mistaken for a crash - see the module note above. The marker records
    ``"diagnostics"``, whether a trace was armed for this run. Returns True if
    armed. Fully guarded - never raises into the caller."""
    import faulthandler
    import json
    import os

    from localm.debuglog import logger
    try:
        d = _br._crash_dir(home)
        diagnostics = _br._diagnostics_allowed()
        if diagnostics:
            _br._crash_trace_fh = open(_crash_trace_path(d, instance_id), "w",
                                   encoding="utf-8")
            _br._crash_trace_instance_id = instance_id
            try:
                faulthandler.enable(file=_br._crash_trace_fh, all_threads=True)
                if not faulthandler.is_enabled():
                    # enable() can return without raising yet still not actually
                    # be armed on some platforms/file-object shapes - is_enabled()
                    # is the one call that tells the truth, not "no exception was
                    # raised", so an unarmed faulthandler is warned about rather
                    # than left silent.
                    logger.warning(
                        "bugreport: faulthandler.enable() returned without raising "
                        "but is_enabled() is False - a native crash will produce no "
                        "trace this run")
            except Exception as e:
                # Arming must not fail over this: the crash marker below is still
                # written, so a hard death is still reported next start, just
                # without the native traceback. The failure is logged rather than
                # swallowed, so a later empty trace file is diagnosable.
                logger.warning(
                    "bugreport: faulthandler could not attach (%s: %s) - a native "
                    "crash this run will produce no trace", type(e).__name__, e)
        _crash_marker_path(d, instance_id).write_text(
            json.dumps({"pid": os.getpid(), "context": context or {},
                        "diagnostics": diagnostics}),
            encoding="utf-8")
        _br._armed_instance_id = instance_id
        return True
    except Exception:
        return False


def armed_instance_id() -> Optional[str]:
    """The instance_id of the last arm_crash_guard() call in this process, or
    None if none has armed yet. For a caller with no other way to learn it -
    a console-close handler registered before app.state.instance_id exists,
    see gui/cli.py's _console_close_cleanup."""
    return _br._armed_instance_id


def clear_crash_marker(home=None, instance_id: Optional[str] = None) -> None:
    """Turn THIS instance's crash marker into its stopping record
    (``server-crash.<instance_id>.stopping``, the marker's own content),
    leaving its native-fault trace file and faulthandler attached (see
    release_crash_trace, which deletes both). Once the marker is gone the
    crash-recovery watchdog reads the run as a clean stop; a stopping record
    still present at the next start means the run ended before
    release_crash_trace() ran (see check_and_report_prior_crash). The marker is
    deleted instead when it cannot be renamed. *instance_id* must be the SAME
    id passed to the matching arm_crash_guard() call. armed_instance_id()
    returns None afterwards when *instance_id* is the one armed in this
    process. Never raises."""
    import os
    try:
        d = _br._crash_dir(home)
        marker = _crash_marker_path(d, instance_id)
        try:
            os.replace(marker, _crash_stopping_path(d, instance_id))
        except FileNotFoundError:
            pass
        except OSError:
            marker.unlink(missing_ok=True)
    except Exception:
        # Best-effort cleanup on a CLEAN shutdown. Worst case the marker survives
        # and the next start files one spurious "prior crash" report - annoying,
        # not unsafe. disarm must not raise during shutdown.
        pass
    if _br._armed_instance_id == instance_id:
        _br._armed_instance_id = None


def release_crash_trace(home=None, instance_id: Optional[str] = None) -> None:
    """Disable faulthandler and close this process's native-fault trace file
    when *instance_id* is the instance that armed it here, then delete that
    instance's trace file and then its stopping record (see
    clear_crash_marker). The handle is closed before the unlink (an open handle
    blocks the unlink on Windows). A trace owned by another instance id stays
    open and attached. Never raises."""
    import faulthandler
    try:
        if _br._crash_trace_fh is not None and _br._crash_trace_instance_id == instance_id:
            faulthandler.disable()
            _br._crash_trace_fh.close()
            _br._crash_trace_fh = None
            _br._crash_trace_instance_id = None
    except Exception:
        # Best-effort: releasing the faulthandler file on shutdown. A failure
        # leaks a file handle until process exit (imminent anyway); never raise.
        pass
    try:
        d = _br._crash_dir(home)
        _crash_trace_path(d, instance_id).unlink(missing_ok=True)
        _crash_stopping_path(d, instance_id).unlink(missing_ok=True)
    except Exception:
        # Best-effort: a trace file that cannot be removed is reported and
        # deleted by the next start's check_and_report_prior_crash().
        pass


def disarm_crash_guard(home=None, instance_id: Optional[str] = None) -> None:
    """Clean shutdown: drop THIS instance's own marker and its native-fault
    trace file (never a sibling's) so the next start does not report a crash
    and no trace is left behind: clear_crash_marker() then
    release_crash_trace(). *instance_id* must be the SAME id passed to the
    matching arm_crash_guard() call - see the module note above for why an
    unscoped delete is unsafe when more than one instance shares a LOCALM_HOME.
    The module-level faulthandler/trace handle is only released when
    *instance_id* matches the instance that armed it in THIS process; a
    sibling's own marker and trace file are still removed regardless, but this
    process's own live trace stays open and attached. Never raises."""
    _br.clear_crash_marker(home=home, instance_id=instance_id)
    release_crash_trace(home=home, instance_id=instance_id)
