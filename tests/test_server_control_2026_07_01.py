# SPDX-License-Identifier: AGPL-3.0-or-later
"""Server-control regression tests.

Covers: the restart re-exec marks fds non-inheritable, so no fd 3/4 leaks
across os.execv.
"""

import os
from types import SimpleNamespace

from localm.inference import http_server


# --------------------------------------------------------------------------- #
#  fds are marked non-inheritable before os.execv
# --------------------------------------------------------------------------- #

def test_do_restart_marks_fds_non_inheritable(monkeypatch):
    calls = []

    class _Stop(Exception):
        pass

    monkeypatch.setattr(os, "set_inheritable", lambda fd, inh: calls.append((fd, inh)))

    def _fake_execv(exe, argv):
        raise _Stop()

    monkeypatch.setattr(os, "execv", _fake_execv)
    monkeypatch.setattr(http_server, "_restart_argv",
                        lambda port=None: ["python", "-m", "localm"])
    monkeypatch.setattr(http_server, "_engine", None)

    try:
        http_server._do_restart()
    except _Stop:
        pass

    assert calls, "no fds were marked non-inheritable before execv"
    assert all(inh is False for _, inh in calls), "fds must be marked NON-inheritable"
    assert all(fd >= 3 for fd, _ in calls), "stdin/stdout/stderr (0-2) must be left alone"


def test_hang_restart_forced_fallback_marks_fds_non_inheritable(monkeypatch):
    """_hang_restart_action's FORCED fallback (the graceful _do_restart
    thread raised, or never finished) must mark fds non-inheritable before
    ITS OWN os.execv too - the same hygiene as the graceful path above,
    shared via http_server._mark_fds_noninheritable. This path is reached
    only when something is already known to be wedged, which is exactly
    when leaking the old listening socket into the re-exec'd image matters
    most (see PortInUseError / restart_grace_window in localm.config)."""
    calls = []

    class _Stop(Exception):
        pass

    def _graceful_fails(**kw):
        raise RuntimeError("graceful restart failed")

    monkeypatch.setattr(os, "set_inheritable", lambda fd, inh: calls.append((fd, inh)))
    monkeypatch.setattr(http_server, "_do_restart", _graceful_fails)
    monkeypatch.setattr(http_server, "_restart_argv",
                        lambda port=None: ["python", "-m", "localm"])

    def _fake_execv(exe, argv):
        raise _Stop()
    monkeypatch.setattr(os, "execv", _fake_execv)

    app = SimpleNamespace(state=SimpleNamespace(instance_port=None,
                                                 instance_id=None))
    try:
        http_server._hang_restart_action(app)
    except _Stop:
        pass

    assert calls, "no fds were marked non-inheritable before the forced os.execv"
    assert all(inh is False for _, inh in calls), "fds must be marked NON-inheritable"
    assert all(fd >= 3 for fd, _ in calls), "stdin/stdout/stderr (0-2) must be left alone"
