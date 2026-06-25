# SPDX-License-Identifier: AGPL-3.0-or-later
"""R18: an in-app RESTART endpoint so the user can restart the server from Settings
(it comes back on the same port) instead of only being able to shut down. The
restart sequence unloads the model BEFORE relaunching, like the shutdown sequence."""

import os
import sys

from localm.inference import http_server


def test_restart_route_registered_and_gated():
    app = http_server.create_app(None)
    routes = {getattr(r, "path", None): r for r in app.routes}
    assert "/v1/server/restart" in routes
    route = routes["/v1/server/restart"]
    assert "POST" in route.methods
    # Server control must be auth-gated, not open to anyone on the network.
    assert route.dependencies, "restart endpoint must carry an auth dependency"


def test_restart_argv_is_canonical_python_m_localm():
    argv = http_server._restart_argv()
    assert argv[0] == sys.executable
    assert argv[1:3] == ["-m", "localm"]
    assert argv[3:] == sys.argv[1:]      # original subcommand + args preserved


def test_do_restart_unloads_before_relaunch(monkeypatch):
    order = []

    class _FakeEngine:
        def unload(self):
            order.append("unload")

    def _fake_relaunch(exe, argv):
        order.append(("relaunch", exe, tuple(argv)))
        raise SystemExit(0)   # stop _do_restart here instead of replacing pytest

    monkeypatch.setattr(http_server, "_engine", _FakeEngine())
    monkeypatch.setattr(os, "execv", _fake_relaunch)

    try:
        http_server._do_restart()
    except SystemExit:
        pass

    # Model unloaded BEFORE the relaunch (clean native teardown), then the canonical
    # re-launch command line.
    assert order and order[0] == "unload"
    assert order[-1][0] == "relaunch"
    assert order[-1][1] == sys.executable
    assert list(order[-1][2]) == http_server._restart_argv()


def test_do_restart_disarms_crash_guard_before_relaunch(monkeypatch):
    # An intentional restart must not be reported as a crash.
    disarmed = []

    def _boom(*_a):
        raise SystemExit(0)

    monkeypatch.setattr(http_server, "_engine", None)
    import localm.bugreport as bugreport
    monkeypatch.setattr(bugreport, "disarm_crash_guard",
                        lambda: disarmed.append(True))
    monkeypatch.setattr(os, "execv", _boom)
    try:
        http_server._do_restart()
    except SystemExit:
        pass
    assert disarmed == [True]
