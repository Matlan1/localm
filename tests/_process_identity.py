# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helpers for tests of decisions keyed on which process owns something: a
simulated system clock step, and a child interpreter running the tree under
test.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path


def step_the_clock(m, seconds: float) -> None:
    """Move every wall-clock reading by *seconds* on monkeypatch *m*, as a
    system clock step does on this platform.

    ``time.time`` and ``psutil.boot_time`` move everywhere. On Linux
    ``psutil.Process.create_time`` moves too: psutil adds the current boot time
    to the process's start ticks on every read, so a step changes it for a
    process that is already running. Elsewhere psutil returns the creation
    timestamp the OS stored when the process started, which a step leaves as
    it was.
    """
    import psutil
    real_time, boot = time.time, psutil.boot_time()
    m.setattr(time, "time", lambda: real_time() + seconds)
    m.setattr(psutil, "boot_time", lambda: boot + seconds)
    if sys.platform.startswith("linux"):
        real_create = psutil.Process.create_time
        m.setattr(psutil.Process, "create_time",
                  lambda self: real_create(self) + seconds)


def a_forward_step_past_boot() -> float:
    """Seconds for a forward step one hour larger than the machine's uptime,
    so a boot time read after it lies after every moment before it."""
    import psutil
    return time.time() - psutil.boot_time() + 3600.0


def start_identity_of(pid: int):
    """``_process_start_identity(pid)``, asserted readable on Linux and
    Windows; the calling test is skipped where the platform has none."""
    import pytest
    from localm.model_manager.pull import _process_start_identity
    ident = _process_start_identity(pid)
    if sys.platform.startswith("linux") or sys.platform == "win32":
        assert ident is not None, (
            f"no start identity for live pid {pid} on {sys.platform}")
    elif ident is None:
        pytest.skip(f"no process start identity on {sys.platform}")
    return ident


def started_an_hour_earlier(ident: dict) -> dict:
    """*ident* as it reads for a process started an hour earlier in the same
    boot."""
    if "ticks" in ident:
        return {**ident,
                "ticks": ident["ticks"] - 3600 * os.sysconf("SC_CLK_TCK")}
    return {**ident, "created": ident["created"] - 3600.0}


def tree_root() -> str:
    """The checkout this test process imported localm from."""
    import localm
    return str(Path(localm.__file__).resolve().parent.parent)


CHECK_TREE = textwrap.dedent('''
    import os
    import localm
    assert os.path.normcase(os.path.dirname(os.path.dirname(
        os.path.abspath(localm.__file__)))) == os.path.normcase(
        os.environ["EXPECT_ROOT"]), "child imported localm from " + localm.__file__
''')


def spawn_on_this_tree(script: str, home_dir, *args,
                       stdin=None) -> subprocess.Popen:
    """Run *script* in a child interpreter with ``LOCALM_HOME`` at *home_dir*.

    The child imports localm from :func:`tree_root` and first asserts that it
    did, so it never runs the venv's editable install of another checkout.
    """
    root = tree_root()
    env = dict(os.environ)
    env["LOCALM_HOME"] = str(home_dir)
    env["EXPECT_ROOT"] = root
    env["PYTHONPATH"] = root
    return subprocess.Popen(
        [sys.executable, "-c", CHECK_TREE + textwrap.dedent(script),
         *[str(a) for a in args]],
        cwd=root, env=env, stdin=stdin,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
