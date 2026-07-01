# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the 2026-07-01 server-control backlog cluster.

  NEW-G - restart re-exec marks fds non-inheritable (no fd 3/4 leak)

(SRV-CTRLC is deferred: it needs a portmux/serve refactor and a live Windows
Ctrl+C to verify. REC-OPEN-GET-GATE was reverted: a broad network-bind GET gate
broke legitimate discovery endpoints like /whoami, and a correct public-GET
allowlist needs the maintainer's route knowledge + live verification.)
"""

import os

from localm.inference import http_server


# --------------------------------------------------------------------------- #
#  NEW-G - fds are marked non-inheritable before os.execv
# --------------------------------------------------------------------------- #

def test_do_restart_marks_fds_non_inheritable(monkeypatch):
    calls = []

    class _Stop(Exception):
        pass

    monkeypatch.setattr(os, "set_inheritable", lambda fd, inh: calls.append((fd, inh)))

    def _fake_execv(exe, argv):
        raise _Stop()

    monkeypatch.setattr(os, "execv", _fake_execv)
    monkeypatch.setattr(http_server, "_restart_argv", lambda: ["python", "-m", "localm"])
    monkeypatch.setattr(http_server, "_engine", None)

    try:
        http_server._do_restart()
    except _Stop:
        pass

    assert calls, "no fds were marked non-inheritable before execv"
    assert all(inh is False for _, inh in calls), "fds must be marked NON-inheritable"
    assert all(fd >= 3 for fd, _ in calls), "stdin/stdout/stderr (0-2) must be left alone"
