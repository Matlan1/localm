# SPDX-License-Identifier: AGPL-3.0-or-later
"""No test replaces the process that runs pytest (``tests/_no_reexec.py``): a real
``os.execv`` or ``os.execve`` fails the test at the call; a test that patches one gets
its own."""

from __future__ import annotations

import os
import sys

import pytest

from tests import _no_reexec


@pytest.mark.parametrize("name", ["execv", "execve"])
def test_a_real_reexec_fails_the_test_at_the_call(name):
    assert _no_reexec.armed()
    args = [sys.executable, [sys.executable, "-c", "pass"]]
    with pytest.raises(pytest.fail.Exception, match=r"BLOCKED by tests/_no_reexec\.py: "
                       rf"os\.{name}"):
        getattr(os, name)(*args, *([dict(os.environ)] if name == "execve" else []))


def test_the_server_restart_reaches_the_guard_when_a_test_does_not_patch_it():
    """``_execv_restart`` without its ``os.execve`` patched stops at the guard instead of
    replacing the test process."""
    from localm.inference import http_server
    with pytest.raises(pytest.fail.Exception, match=r"os\.execve"):
        http_server._execv_restart(None, None)


def test_a_test_that_patches_it_gets_its_own(monkeypatch):
    calls: list = []
    monkeypatch.setattr(os, "execv", lambda *a: calls.append(a))
    os.execv("x", ["x"])
    assert calls == [("x", ["x"])]
