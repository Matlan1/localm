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


def test_do_restart_releases_embedder(monkeypatch):
    """The shared embedder (localm.inference.embedder) is a separate lifecycle
    from _engines - it was previously never released before a restart's
    re-exec, leaking its native VRAM/RAM allocation across the restart."""
    from localm.inference import embedder as emb

    def _fake_relaunch(exe, argv):
        raise SystemExit(0)

    calls = []
    monkeypatch.setattr(emb, "reset_embedder", lambda: calls.append(1))
    monkeypatch.setattr(http_server, "_engine", None)
    monkeypatch.setattr(os, "execv", _fake_relaunch)

    try:
        http_server._do_restart()
    except SystemExit:
        pass

    assert calls == [1]


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


# ------------------------ LM-DA-011: post-update watchdog -------------------
#
# _do_restart's update_watchdog param is the ONLY thing that distinguishes the
# post-update auto-restart from a plain /v1/server/restart click; these tests
# prove the plain path is untouched and the update path wires through correctly.

def test_do_restart_spawns_watchdog_when_given(monkeypatch):
    from localm import updater
    calls = []
    monkeypatch.setattr(updater, "spawn_health_watchdog",
                        lambda **kw: calls.append(kw) or True)
    monkeypatch.setattr(http_server, "_engine", None)

    def _stop_here(*_a):
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", _stop_here)

    watchdog = {"host": "127.0.0.1", "port": 8642, "scheme": "http",
               "expect_version": "0.2.0"}
    try:
        http_server._do_restart(update_watchdog=watchdog)
    except SystemExit:
        pass
    assert calls == [{"host": "127.0.0.1", "port": 8642, "scheme": "http",
                      "expect_version": "0.2.0"}]


def test_do_restart_no_watchdog_by_default(monkeypatch):
    """The plain restart path (/v1/server/restart) must NEVER spawn a watchdog -
    it has no update to verify and no version to roll back to."""
    from localm import updater

    def _must_not_be_called(**_kw):
        raise AssertionError("spawn_health_watchdog must not fire for a plain restart")

    monkeypatch.setattr(updater, "spawn_health_watchdog", _must_not_be_called)
    monkeypatch.setattr(http_server, "_engine", None)

    def _stop_here(*_a):
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", _stop_here)
    try:
        http_server._do_restart()
    except SystemExit:
        pass   # no AssertionError means the watchdog was correctly never spawned


def test_do_restart_watchdog_spawn_exception_does_not_block_execv(monkeypatch):
    """A watchdog spawn failure must never prevent the restart itself - a broken
    watchdog must not make updates worse than having none at all."""
    from localm import updater

    def _boom(**_kw):
        raise RuntimeError("spawn blew up")

    monkeypatch.setattr(updater, "spawn_health_watchdog", _boom)
    monkeypatch.setattr(http_server, "_engine", None)
    reached = []

    def _fake_execv(exe, argv):
        reached.append(True)
        raise SystemExit(0)

    monkeypatch.setattr(os, "execv", _fake_execv)
    try:
        http_server._do_restart(update_watchdog={
            "host": "127.0.0.1", "port": 1, "scheme": "http", "expect_version": "x"})
    except SystemExit:
        pass
    assert reached == [True]   # execv still ran despite the watchdog spawn raising


def test_request_restart_threads_update_watchdog_through(monkeypatch):
    # Run the inner thread body synchronously so the test is not timing-dependent.
    class _SyncThread:
        def __init__(self, target, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr("threading.Thread", _SyncThread)
    captured = []
    monkeypatch.setattr(http_server, "_do_restart",
                        lambda **kw: captured.append(kw.get("update_watchdog")))

    watchdog = {"host": "127.0.0.1", "port": 9, "scheme": "https",
               "expect_version": "1.0.0"}
    http_server._request_restart(delay=0, update_watchdog=watchdog)
    assert captured == [watchdog]
