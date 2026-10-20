# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stub for the process launch behind ``tools.base.run_subprocess``.

``run_subprocess`` launches with ``subprocess.Popen`` and reads the pipes on
threads. :func:`patch_launch` replaces that launch with a fake process that
serves canned output, so a test can assert what was launched (argv, cwd, env)
and how the result is handled, without starting anything. It takes the same
``return_value`` / ``side_effect`` arguments as ``unittest.mock.patch``:

- ``return_value`` / a ``side_effect`` function returning an object with
  ``stdout``, ``stderr`` and ``returncode`` is a process that exits at once.
- A ``side_effect`` that is, or raises, ``subprocess.TimeoutExpired`` is a
  process that has printed ``exc.stdout`` / ``exc.stderr`` and never exits
  until killed; the clock the runner reads jumps forward instead of sleeping.
- Any other exception is raised by the launch itself (missing executable).
"""

import io
import subprocess
import time
from contextlib import contextmanager
from unittest.mock import patch


class _Clock:
    def __init__(self):
        self.offset = 0.0
        self._real = time.monotonic

    def __call__(self):
        return self._real() + self.offset


def _as_bytes(data):
    if data is None:
        return b""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    return str(data).encode("utf-8")


class FakeProcess:
    pid = 0

    def __init__(self, argv, clock, stdout=b"", stderr=b"", returncode=0, hangs=False):
        self.args = argv
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = None if hangs else returncode
        self._clock = clock
        self._hangs = hangs

    def wait(self, timeout=None):
        if self.returncode is None:
            self._clock.offset += timeout if timeout is not None else 1e6
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self.returncode

    def kill(self):
        self.returncode = -9

    def poll(self):
        return self.returncode


@contextmanager
def patch_launch(return_value=None, side_effect=None):
    clock = _Clock()

    def launch(argv, **kwargs):
        outcome = return_value
        if side_effect is not None:
            outcome = side_effect
            if isinstance(outcome, type) and issubclass(outcome, BaseException):
                outcome = outcome()
            elif callable(outcome):
                outcome = outcome(argv, **kwargs)
        if isinstance(outcome, subprocess.TimeoutExpired):
            return FakeProcess(argv, clock, _as_bytes(outcome.stdout),
                               _as_bytes(outcome.stderr), hangs=True)
        if isinstance(outcome, BaseException):
            raise outcome
        return FakeProcess(argv, clock, _as_bytes(outcome.stdout),
                           _as_bytes(outcome.stderr), outcome.returncode)

    with patch("subprocess.Popen", side_effect=launch), \
            patch("time.monotonic", clock), \
            patch("localm.plugins.coder.tools.base._kill_tree",
                  side_effect=lambda proc: proc.kill()):
        yield
