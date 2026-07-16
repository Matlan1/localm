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
import sys
from pathlib import Path
from typing import Optional


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
