# SPDX-License-Identifier: AGPL-3.0-or-later
"""No test replaces the process that runs pytest with another program.

``os.execv`` and the rest of the ``os.exec*`` family end the test run in the middle (on
Windows they start a new process and end this one). :func:`arm` replaces each of them in
the process that runs pytest with a stand-in that fails the test at the call with
``pytest.fail``, naming the call site; a child forked from that process calls the real
one. A test that patches one (``monkeypatch.setattr(os,
"execv", ...)``) gets its own stand-in for that test, as before.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback

#: The functions of ``os`` that replace the process.
NAMES = ("execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe")
_ARMED = False


def _origin() -> str:
    """The innermost tests/ or localm/ frames that led here, newest first."""
    frames = []
    for frame in reversed(traceback.extract_stack()[:-2]):
        f = frame.filename.replace("\\", "/")
        if ("/tests/" in f or "/localm/" in f) and not f.endswith("/tests/_no_reexec.py"):
            frames.append(f"{os.path.basename(frame.filename)}:{frame.lineno}")
            if len(frames) >= 5:
                break
    origin = " <- ".join(frames) if frames else "unknown"
    thread = threading.current_thread()
    if thread is not threading.main_thread():
        origin += f" (thread {thread.name!r})"
    return origin


def _refusing(name: str):
    fail = sys.modules["pytest"].fail
    real, tests = getattr(os, name), os.getpid()

    def refuse(*args, **kwargs):
        if os.getpid() != tests:   # a child forked from the tests (pty.fork) runs its program
            return real(*args, **kwargs)
        fail(f"BLOCKED by tests/_no_reexec.py: os.{name}{args[:1]!r} would replace the "
             f"process that runs the tests\n    from {_origin()}\n    Patch os.{name} in "
             f"the test (monkeypatch.setattr(os, {name!r}, ...)).", pytrace=False)
    refuse.__name__ = refuse.__qualname__ = name
    refuse._localm_refuses_exec = True
    return refuse


def armed() -> bool:
    return _ARMED


def arm() -> bool:
    """Install the stand-ins once, in a process that has imported pytest. Returns True
    when it did."""
    global _ARMED
    if _ARMED or "pytest" not in sys.modules:
        return False
    for name in NAMES:
        if hasattr(os, name):
            setattr(os, name, _refusing(name))
    _ARMED = True
    return True
