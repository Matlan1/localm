# SPDX-License-Identifier: AGPL-3.0-or-later
"""tests/_popen_scope.py: which threads a patched subprocess.Popen answers."""
import subprocess
import sys
import threading
from unittest.mock import MagicMock, patch

from tests._popen_scope import ignore_background_threads

_HARMLESS = [sys.executable, "-c", "pass"]


def _spawn_and_wait():
    proc = subprocess.Popen(_HARMLESS)
    proc.wait(timeout=60)
    return proc


def _run_in_thread(target, name=None):
    out = {}

    def body():
        out["proc"] = target()

    t = threading.Thread(target=body, name=name)
    t.start()
    t.join(timeout=60)
    assert not t.is_alive()
    return out["proc"]


def test_a_background_probe_never_reaches_the_fake():
    calls = []

    def fake(argv, cwd=None, **kw):
        calls.append((threading.current_thread().name, argv, cwd))
        return MagicMock()

    with patch("subprocess.Popen", ignore_background_threads(fake)):
        subprocess.Popen(["launcher"], cwd="the-workdir")
        probe = _run_in_thread(_spawn_and_wait, name="localm-gpu-probe")

    assert calls == [("MainThread", ["launcher"], "the-workdir")]
    assert isinstance(probe, subprocess.Popen)
    assert probe.returncode == 0


def test_a_thread_that_existed_before_the_fake_is_sent_to_the_real_popen():
    go, done = threading.Event(), {}

    def waiter():
        go.wait(timeout=60)
        done["proc"] = _spawn_and_wait()

    t = threading.Thread(target=waiter, name="earlier-test-thread")
    t.start()
    fake = MagicMock()
    with patch("subprocess.Popen", ignore_background_threads(fake)):
        go.set()
        t.join(timeout=60)
    fake.assert_not_called()
    assert isinstance(done["proc"], subprocess.Popen)


def test_a_thread_the_test_starts_reaches_the_fake():
    fake = MagicMock(return_value="faked")
    with patch("subprocess.Popen", ignore_background_threads(fake)):
        got = _run_in_thread(lambda: subprocess.Popen(["launcher"]))
    assert got == "faked"
    fake.assert_called_once_with(["launcher"])
