# SPDX-License-Identifier: AGPL-3.0-or-later
"""Windows multiprocessing-spawn fix for the branded LocaLM.exe launcher (#617)."""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
from pathlib import Path
from typing import Optional

# Set once, inside a worker process, when the parent-death watchdog thread is
# running. Module-level so a second call in the same process is a cheap no-op;
# each spawned worker is a fresh process, so it always starts False there.
_parent_death_watchdog_installed = False

# Set once SetErrorMode has been applied in this process. Same per-process
# scoping rationale as the watchdog flag above.
_native_error_dialogs_suppressed = False


# Windows has no signals for this: a hard native fault surfaces as an NTSTATUS
# value in the process exit code. Only codes that are unambiguous and actually
# reachable from a native crash are listed - a wrong name here would be worse
# than a bare number, because a reader would act on it.
#
# 0xC0000409 is confirmed empirically, not just from documentation: os.abort()
# under this project's own CRT exits with 3221226505 (measured 2026-08-11 while
# building the worker crash-trace capture).
#
# 0xC0000139 earns its place twice over - it is the documented signature of the
# torch/HIP DLL-identity conflict this codebase already guards against (see
# discover.py's _torch_gpu_probe_known_doomed and _loader.native_lib_loaded), so
# decoding it makes a future report self-identifying instead of needing the same
# root-cause session again.
_NTSTATUS_CRASH_NAMES = {
    0xC0000005: "access violation",
    0xC000001D: "illegal instruction",
    0xC0000094: "integer divide by zero",
    0xC00000FD: "stack overflow",
    0xC0000135: "DLL not found",
    0xC0000139: "entry point not found (a native DLL version conflict)",
    0xC0000374: "heap corruption",
    0xC0000409: "stack buffer overrun (the usual shape of a native abort)",
}


# The crash-relevant POSIX signals, resolved WITHOUT the host's own signal enum.
#
# MEASURED 2026-08-11, and this is not a portability nicety - the host enum is
# actively WRONG for this job when the host is Windows:
#
#     Windows signal.Signals:  SIGILL 4, SIGFPE 8, SIGSEGV 11, SIGABRT 22
#     Linux   signal.Signals:  SIGILL 4, SIGFPE 8, SIGSEGV 11, SIGABRT  6,
#                              SIGKILL 9, SIGBUS 7
#
# So a POSIX -6 looked up in the Windows enum raises (6 is absent), and a POSIX
# -22 would come back "SIGABRT" when Linux 22 is really SIGTTOU. A code that
# reached us from a POSIX child must therefore be decoded with POSIX numbering,
# not with whatever enum this interpreter happens to ship.
#
# Only signals whose numbers are IDENTICAL across Linux and the BSD/macOS family
# are listed, so the table cannot itself become the wrong answer. SIGBUS is
# deliberately ABSENT: it is 7 on Linux but 10 on macOS, so it is left to the
# host enum below, which is authoritative when the host is the POSIX box in
# question (the production case - parent and child are always the same platform).
_POSIX_CRASH_SIGNALS = {
    4: "SIGILL",
    6: "SIGABRT",
    8: "SIGFPE",
    9: "SIGKILL",
    11: "SIGSEGV",
    13: "SIGPIPE",
    15: "SIGTERM",
}


def _posix_signal_name(number: int) -> str:
    """Name POSIX signal *number*, or return the bare number as a string."""
    name = _POSIX_CRASH_SIGNALS.get(number)
    if name:
        return name
    if os.name != "nt":
        try:
            import signal
            return signal.Signals(number).name
        except (ValueError, ImportError):
            pass
    return str(number)


def describe_exit_code(code, *, posix: Optional[bool] = None) -> str:
    """Render a dead child's exit *code* so a reader can act on it, e.g. ``'-4 (killed by signal SIGILL)'`` instead of ``'-4'``."""
    if code is None:
        return "unknown"
    if posix is None:
        posix = os.name != "nt"
    try:
        code = int(code)
    except (TypeError, ValueError):
        return str(code)

    if posix:
        # POSIX: multiprocessing reports -N when the child was killed by signal
        # N. A NON-negative code is an ordinary exit status and means nothing
        # about signals, so it is left alone.
        if code < 0:
            return f"{code} (killed by signal {_posix_signal_name(-code)})"
        return str(code)

    # Windows: a negative code is NOT a signal. Python's own
    # Process.terminate() calls TerminateProcess(handle, -1), so reading -1 as
    # "SIGHUP" would actively mislead - which is the whole reason the branches
    # are split rather than sharing the negative-means-signal rule.
    unsigned = code & 0xFFFFFFFF
    name = _NTSTATUS_CRASH_NAMES.get(unsigned)
    if name:
        return f"{code} (0x{unsigned:08X}, {name})"
    return str(code)


def death_was_a_native_fault(code, *, trace_captured: bool = False,
                             posix: Optional[bool] = None) -> bool:
    """Whether a dead child's *code* ESTABLISHES that it died from a native fault, as opposed to exiting with an ordinary status."""
    if trace_captured:
        return True
    if code is None:
        return False
    if posix is None:
        posix = os.name != "nt"
    try:
        code = int(code)
    except (TypeError, ValueError):
        return False
    if posix:
        return code < 0
    return (code & 0xFFFFFFFF) in _NTSTATUS_CRASH_NAMES


def real_base_python() -> Optional[Path]:
    """The real base interpreter directly under ``sys.base_prefix`` (``<base_prefix>/python.exe``) - a single hop, unaffected by CPython's ``sys._base_executable`` (which assumes the base install's binary keeps its original basename; wrong once the running exe has been renamed, e.g. localm's branded ``Loca..."""
    if sys.platform != "win32":
        return None
    base_python = Path(sys.base_prefix) / "python.exe"
    return base_python if base_python.is_file() else None


def ensure_spawn_uses_venv_python() -> None:
    """Make ``multiprocessing.get_context('spawn')`` children spawn via the base interpreter directly (never a venv trampoline, never a possibly-renamed ``sys.executable``) - see module docstring."""
    base_python = real_base_python()
    if base_python is not None:
        multiprocessing.set_executable(str(base_python))


def interpreter_for_localm_children() -> str:
    """Interpreter path for a PLAIN ``subprocess`` child that must import localm and its venv-installed packages (e.g. the VRAM-probe daemon, ``Popen([exe, '-m', 'localm....'])``)."""
    if sys.prefix != sys.base_prefix:
        return sys.executable
    for entry in sys.path:
        p = Path(entry)
        if p.name.lower() != "site-packages":
            continue
        # Windows: <venv>/Lib/site-packages (2 levels up); POSIX:
        # <venv>/lib/pythonX.Y/site-packages (3 levels up). pyvenv.cfg marks
        # the venv root either way.
        for root in list(p.parents)[:3]:
            if (root / "pyvenv.cfg").is_file():
                cand = (root / "Scripts" / "python.exe"
                        if sys.platform == "win32" else root / "bin" / "python")
                if cand.is_file():
                    return str(cand)
    return sys.executable


def install_parent_death_watchdog() -> bool:
    """Make THIS spawned worker process die when its parent dies - HOWEVER the parent died, including an uncatchable hard kill (Windows TerminateProcess / Task Manager 'End Task', POSIX SIGKILL) where NO parent-side code runs."""
    global _parent_death_watchdog_installed
    if _parent_death_watchdog_installed:
        return True
    try:
        parent = multiprocessing.parent_process()
    except Exception:
        return False
    if parent is None:
        return False   # the main process, not a spawned child - nothing to watch

    def _wait_and_die() -> None:
        try:
            parent.join()   # blocks on the kernel sentinel until the parent dies
        except Exception:
            # Could not wait on the parent sentinel. Do NOT assume the parent
            # died - leave the worker running (no watchdog) rather than kill a
            # worker whose parent is still alive. This keeps the "degrade safely,
            # never disrupt a normal worker" contract literally true.
            return
        # The parent is gone; exit now so this process (and its VRAM) does not
        # outlive it. os._exit, never a clean shutdown - see the docstring.
        os._exit(0)

    try:
        threading.Thread(target=_wait_and_die, daemon=True,
                         name="localm-parent-death-watch").start()
    except Exception:
        return False
    _parent_death_watchdog_installed = True
    return True


def suppress_native_error_dialogs() -> bool:
    """Stop Windows from popping a blocking modal dialog ('..."""
    global _native_error_dialogs_suppressed
    if _native_error_dialogs_suppressed:
        return True
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        SEM_FAILCRITICALERRORS = 0x0001
        SEM_NOGPFAULTERRORBOX = 0x0002
        ctypes.windll.kernel32.SetErrorMode(  # type: ignore[attr-defined]
            SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX)
    except Exception:
        return False
    _native_error_dialogs_suppressed = True
    return True
