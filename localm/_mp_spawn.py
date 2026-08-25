# SPDX-License-Identifier: AGPL-3.0-or-later
"""Windows multiprocessing-spawn fix for the branded LocaLM.exe launcher (#617).

``real_base_python()`` below is also reused by managed_comfy_fresh.py (#621): any
code that needs to hand a subprocess a plain, correctly-self-named interpreter -
not just multiprocessing's own internal spawn - hits the same class of bug for the
same reason (see its docstring). One resolver, reused, rather than a third
divergent attempt at this fix.

CPython's multiprocessing has a Windows-only optimization (bpo-35797):
whenever it detects the running interpreter differs from ``sys._base_executable``
(its own definition of "running inside a venv"), a spawned child is launched via
``sys._base_executable`` instead of ``sys.executable`` - see
``multiprocessing/popen_spawn_win32.py``'s ``WINENV`` check. CPython computes
``sys._base_executable`` as ``<base_prefix>/<basename of the running executable>``;
it does not look up what the base install's binary is actually named.

localm's branded launcher (``localm make-launcher`` -> ``<venv>/localm-app/
LocaLM.exe``, see applaunch.py) is a COPY of the base interpreter renamed to
LocaLM.exe. Running under that renamed copy, ``sys._base_executable`` becomes
``<base_prefix>/LocaLM.exe`` - a file that does not exist (the base install's
real file is named python.exe/python3) - so every
``multiprocessing.get_context("spawn")`` child (a GGUF model load, the voice/STT
worker) fails with ``FileNotFoundError: [WinError 2] The system cannot find the
file specified``. The GGUF loader (gguf.py) reports that as a misleading "Native
llama runtime failed to load" error that has nothing to do with the actual
llama.cpp runtime - confirmed live via GitHub issue #617.

FIRST FIX ATTEMPT (WRONG, do not repeat): redirect to the venv's own
``<prefix>/Scripts/python.exe``. That resolves the FileNotFoundError (spawn
succeeds), but breaks a SECOND, subtler thing: under a uv-managed Python,
``<prefix>/Scripts/python.exe`` is itself a TRAMPOLINE that re-spawns the real
base interpreter as ANOTHER, nested child process. Windows multiprocessing hands
a spawned child its Queue/Lock semaphore handles via a DIRECT
``DuplicateHandle`` call targeting that child's own process handle
(``Popen.duplicate_for_child``, in ``popen_spawn_win32.py`` - see
``synchronize.py``'s ``SemLock.__getstate__``). That handle is injected into the
TRAMPOLINE's process, not into the base interpreter it then spawns as its own
child - so the real worker process receives a handle that was never duplicated
into ITS OWN table, and its first ``Queue.get()``/``Lock`` use fails with
``OSError: [WinError 6] The handle is invalid``. Reproduced live: redirecting to
the trampoline fails this way; redirecting to the base interpreter directly does
not (confirmed with a real ``multiprocessing.Queue`` round trip, not just a bare
spawn-and-exit check - the earlier verification of the first fix only checked
that spawning succeeded, which is why this second bug was missed).

FIX: redirect straight to the base interpreter (``<sys.base_prefix>/python.exe``)
instead of the venv trampoline - a single hop, no nested re-spawn, so the
directly-duplicated handle lands in the same process that actually uses it.
``sys.base_prefix`` (a directory) is unaffected by the renamed-executable bug
above - only ``sys._base_executable`` (which also assumes a basename) is wrong -
so this sidesteps that bug too without needing to touch the broken attribute at
all. Calling this when NOT running under a renamed launcher is a harmless no-op
(it just repoints at the interpreter that is already running, one hop earlier).
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
from pathlib import Path
from typing import Optional

# Set inside a worker process once the parent-death watchdog thread is running,
# so a second call in the same process is a no-op.
_parent_death_watchdog_installed = False

# Set once SetErrorMode has been applied in this process. Same per-process
# scoping rationale as the watchdog flag above.
_native_error_dialogs_suppressed = False


# NTSTATUS exit codes a native crash produces on Windows, where there are no
# signals. Only unambiguous, reachable codes are listed.
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


# Crash-relevant POSIX signal numbers, resolved without the host's signal enum:
# a code from a POSIX child must be decoded with POSIX numbering, and the Windows
# enum numbers SIGABRT differently and lacks several entries. Only signals whose
# numbers are identical across Linux and the BSD/macOS family are listed. SIGBUS
# is absent (7 on Linux, 10 on macOS) and is left to the host enum.
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
    """Name POSIX signal *number*, or return the bare number as a string.

    Order matters. The universal table wins first so the answer is the same on
    every platform for the signals that actually kill a native worker; the host
    enum is consulted only on a POSIX host, where it is authoritative for
    everything else (SIGBUS, SIGUSR1, real-time signals) and cannot be wrong
    about its own box."""
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
    """Render a dead child's exit *code* so a reader can act on it, e.g.
    ``"-4 (killed by signal SIGILL)"`` instead of ``"-4"``.

    WHY THIS EXISTS: on the one crash that most needed diagnosing (issues
    1222/1223) the product reported ``worker exit -4`` and nothing else. That
    number is the single most discriminating fact available about a native death
    - it separates an illegal instruction from a segfault from an abort, which
    are different families of cause - and it had to be decoded by hand before the
    investigation could even choose a direction. Throwing it away on every native
    death is a diagnostic the product could give for free and did not.

    *posix* selects which OS convention the code follows, defaulting to this
    process's own. It is an explicit parameter rather than a bare ``os.name``
    read because the two conventions are mutually exclusive and each is
    unreachable from the other platform: without it, the POSIX branch (the one
    the field crash is on) could never be exercised by a test running on Windows,
    which is item-19 fixture blindness - the platform under test can never
    produce the value that matters.

    Never raises: this decorates an error message on a path that is already
    failing, so an unrecognised code degrades to the bare number.
    """
    if code is None:
        return "unknown"
    if posix is None:
        posix = os.name != "nt"
    try:
        code = int(code)
    except (TypeError, ValueError):
        return str(code)

    if posix:
        # POSIX: multiprocessing reports -N for death by signal N. A non-negative
        # code is an ordinary exit status and is left alone.
        if code < 0:
            return f"{code} (killed by signal {_posix_signal_name(-code)})"
        return str(code)

    # Windows: a negative code is not a signal. Process.terminate() calls
    # TerminateProcess(handle, -1), so -1 must not decode as a signal name.
    unsigned = code & 0xFFFFFFFF
    name = _NTSTATUS_CRASH_NAMES.get(unsigned)
    if name:
        return f"{code} (0x{unsigned:08X}, {name})"
    return str(code)


def death_was_a_native_fault(code, *, trace_captured: bool = False,
                             posix: Optional[bool] = None) -> bool:
    """Whether a dead child's *code* ESTABLISHES that it died from a native
    fault, as opposed to exiting with an ordinary status.

    WHY THIS EXISTS. Every runner reported ANY worker death detected via
    ``is_alive()`` as a "Native inference fault", including an ordinary uncaught
    Python exception - and this codebase already knew better in two places:

    * ``llamacpp/_runner.py``'s own docstring: "worker exit 1 ... is
      multiprocessing's own signature for exactly this case (an uncaught PYTHON
      exception), not a genuine native abort".
    * ``tests/test_image_decode_without_pillow.py``, whose whole subject is a
      missing Pillow surfacing as "Native inference fault (worker exit 1)" with
      a plain ``ModuleNotFoundError`` in the log: "Every clause of that was
      false. ... there was no native fault and no native stack trace, and the
      model was fine."

    That was fixed PER CAUSE (Pillow became a core dependency and the ImportError
    is guarded). The MISCLASSIFICATION survived for every other Python exception,
    because it lives one layer up - at the site that words the message - and a
    per-cause fix cannot reach it (diff-review-discipline item 23).

    EVIDENCE USED, strongest first:

    * *trace_captured* - a faulthandler trace exists. DEFINITIVE: faulthandler
      only fires on SIGSEGV/SIGFPE/SIGABRT/SIGBUS/SIGILL, so a trace means a
      native signal, on either platform.
    * a NEGATIVE code under the POSIX convention: death by signal.
    * a Windows NTSTATUS-shaped code we recognise as a crash status.

    Everything else answers FALSE - deliberately including codes we cannot
    classify. This is a "has it been ESTABLISHED" predicate, not a guess: exit 1
    is an uncaught Python exception, and a Windows abort under an ARMED
    faulthandler exits 3 (measured), which is indistinguishable from an ordinary
    exit 3 on the code alone. In that case the trace is what settles it, which is
    why *trace_captured* leads rather than being a tiebreak.

    Never raises - it decorates a message on an already-failing path."""
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
    """The real base interpreter directly under ``sys.base_prefix``
    (``<base_prefix>/python.exe``) - a single hop, unaffected by CPython's
    ``sys._base_executable`` (which assumes the base install's binary keeps
    its original basename; wrong once the running exe has been renamed, e.g.
    localm's branded ``LocaLM.exe`` copy - see module docstring and
    ``applaunch._base_interpreter``). Windows-only (the filename is
    hardcoded); returns None off Windows or if no such file exists."""
    if sys.platform != "win32":
        return None
    base_python = Path(sys.base_prefix) / "python.exe"
    return base_python if base_python.is_file() else None


def ensure_spawn_uses_venv_python() -> None:
    """Make ``multiprocessing.get_context("spawn")`` children spawn via the base
    interpreter directly (never a venv trampoline, never a possibly-renamed
    ``sys.executable``) - see module docstring. Windows-only; a no-op elsewhere.
    Best-effort: leaves multiprocessing's default untouched if the expected
    layout is not found - this must never block a normal launch, branded or
    not."""
    base_python = real_base_python()
    if base_python is not None:
        multiprocessing.set_executable(str(base_python))


def interpreter_for_localm_children() -> str:
    """Interpreter path for a PLAIN ``subprocess`` child that must import localm
    and its venv-installed packages (e.g. the VRAM-probe daemon,
    ``Popen([exe, "-m", "localm...."])``).

    ``sys.executable`` is correct for a process launched via the venv (or the
    branded launcher living inside it): the exe sits next to ``pyvenv.cfg``, so
    a child running the same exe re-discovers the venv on its own. It is WRONG
    inside a Windows multiprocessing-spawn worker: ``ensure_spawn_uses_venv_python``
    above deliberately spawns workers via the BASE interpreter (see the module
    docstring for why neither the renamed launcher nor the venv trampoline can
    be used THERE), and multiprocessing hands the worker the venv's ``sys.path``
    as spawn-prep data rather than via the exe - so inside the worker,
    ``sys.executable`` is a bare base python whose own children get no venv
    paths at all. Found live (2026-07-22): the GGUF worker's VRAM-probe daemon
    could not resolve the ``localm-llama-runtime`` wheel (nor ``localm`` itself
    on a non-dev install) and answered ERR on every query, silently costing the
    worker its only raw VRAM reading - which made every mid-generation
    context-grow KV-placement check unmeasurable.

    Resolution: a process already running inside the venv (``sys.prefix !=
    sys.base_prefix``) keeps ``sys.executable``, today's working behavior.
    Otherwise the venv is found via the site-packages entries multiprocessing
    injected into ``sys.path`` (the ancestor holding ``pyvenv.cfg``) and its
    ``Scripts/python.exe`` (``bin/python`` elsewhere) is returned. The venv
    trampoline is safe for THESE children: the WinError 6 failure that rules it
    out for multiprocessing workers is specific to mp's cross-process SEMAPHORE
    handle injection (``DuplicateHandle`` into the trampoline, not the real
    child - see module docstring); plain stdio pipes are standard handles,
    which the trampoline forwards to its child (verified live with a pipe
    round-trip). Falls back to ``sys.executable`` when no venv is found (a
    system-python setup with localm on ``PYTHONPATH`` keeps today's behavior)."""
    if sys.prefix != sys.base_prefix:
        return sys.executable
    for entry in sys.path:
        p = Path(entry)
        if p.name.lower() != "site-packages":
            continue
        # Windows: <venv>/Lib/site-packages (2 levels up); POSIX:
        # <venv>/lib/pythonX.Y/site-packages (3 levels up). pyvenv.cfg marks the root.
        for root in list(p.parents)[:3]:
            if (root / "pyvenv.cfg").is_file():
                cand = (root / "Scripts" / "python.exe"
                        if sys.platform == "win32" else root / "bin" / "python")
                if cand.is_file():
                    return str(cand)
    return sys.executable


def install_parent_death_watchdog() -> bool:
    """Make THIS spawned worker process die when its parent dies - HOWEVER the
    parent died, including an uncatchable hard kill (Windows TerminateProcess /
    Task Manager "End Task", POSIX SIGKILL) where NO parent-side code runs.

    Call at the very top of a worker's process-main. Returns True if the watchdog
    thread was installed, False if there is nothing to watch (the main process) or
    the mechanism is unavailable.

    WHY this is needed even though every worker is spawned ``daemon=True``:
    multiprocessing's daemon-child reclamation is an atexit hook
    (``multiprocessing.util._exit_function``), and atexit never runs under a hard
    kill. Every localm reclamation path (``ModelRunner.shutdown()`` on unload,
    ``embedder.release_for_exit()`` on stop/restart) is parent-side Python, which by
    definition does not run when the parent is force-killed. So without this, a
    force-closed / End-Task'd server leaves its model worker alive, holding its
    model resident in VRAM indefinitely, and the next start plans against a card
    that is mostly full (reproduced in the real product 2026-07-16).

    HOW: multiprocessing ALREADY hands a spawned child a waitable parent sentinel
    (Windows: an ``OpenProcess(SYNCHRONIZE)`` handle; POSIX: a dup'd pipe fd),
    exposed as ``multiprocessing.parent_process().join()`` - which blocks until the
    parent terminates, signalled by the kernel no matter how the parent died. A
    tiny daemon thread waits on it and, the instant it fires, ``os._exit(0)``s.

    ``os._exit`` (not ``worker.close()`` / ``sys.exit``) is deliberate: a model's
    ``close()`` takes the generation lock that is held during a native decode, so a
    polite close mid-generation would deadlock - and the parent is already gone, so
    an immediate process exit is both sufficient (it frees the VRAM) and the only
    thing that cannot hang. The native binding is a ctypes ``CDLL`` (verified),
    which releases the GIL during ``llama_decode``, so this thread still gets the
    GIL to run even while the worker is mid-token.

    Fully guarded and idempotent: a no-op in the main process (``parent_process()``
    is None there) and if anything is unavailable, so it can never block a normal
    worker start."""
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
            # Could not wait on the parent sentinel: leave the worker running
            # rather than kill one whose parent may still be alive.
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
    """Stop Windows from popping a blocking modal dialog ("... Entry Point Not
    Found", "... has stopped working") when a native DLL load fails in THIS
    process, so the failure surfaces as an ordinary catchable exception
    instead - matching what every ctypes.CDLL()/load_lib() caller in this
    codebase already assumes and handles (e.g.
    VramSizingMixin._free_total_vram_bytes).

    WHY this is needed even though the failure is already caught in Python:
    ctypes wraps a native DLL load in Windows SEH, so a bad DLL DOES raise a
    catchable exception rather than crashing the interpreter - but by default
    Windows' own critical-error handler shows its blocking dialog FIRST, before
    that exception ever reaches Python. Confirmed live: a worker still finished
    loading and replying after the box was manually dismissed - the code's
    "catch and log" handling was working exactly as designed, it just was not
    preventing the OS from ALSO blocking on a dialog nobody meant to show.
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX) makes the
    failing call return an error to its caller instead of ever presenting UI,
    which is what lets the existing try/except actually behave as intended.

    Call at the very top of a worker's process-main, alongside
    install_parent_death_watchdog() - a worker process's whole reason to exist
    is running native code, so suppressing the OS's error UI for its entire
    lifetime is always correct there, unlike the main process (which stays
    interactive and may want the standard OS UI for genuine hardware issues
    unrelated to this codebase's own native bindings).

    Windows-only; a no-op elsewhere. Idempotent (a second call in the same
    process is a cheap no-op, matching install_parent_death_watchdog's shape).
    Best-effort: never raises, so it can never block a normal worker start."""
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
