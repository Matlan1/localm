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

# Set once, inside a worker process, when the parent-death watchdog thread is
# running. Module-level so a second call in the same process is a cheap no-op;
# each spawned worker is a fresh process, so it always starts False there.
_parent_death_watchdog_installed = False


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
