# SPDX-License-Identifier: AGPL-3.0-or-later
"""SRV-4: a direct shutdown endpoint so the user can stop the server cleanly
(rather than force-closing the window, which segfaults, or relying on a Ctrl+C
that sometimes does nothing). The stop sequence unloads the model BEFORE exiting
so the native context is freed cleanly."""

import os

from localm.inference import http_server


def test_shutdown_route_registered_and_gated():
    app = http_server.create_app(None)
    routes = {getattr(r, "path", None): r for r in app.routes}
    assert "/v1/server/shutdown" in routes
    route = routes["/v1/server/shutdown"]
    assert "POST" in route.methods
    # It must be auth-gated (a dependency), not open to anyone on the network.
    assert route.dependencies, "shutdown endpoint must carry an auth dependency"


def test_do_shutdown_unloads_before_exit(monkeypatch):
    order = []

    class _FakeEngine:
        def unload(self):
            order.append("unload")

    def _fake_exit(code):
        order.append(("exit", code))
        raise SystemExit(code)   # stop _do_shutdown here instead of killing pytest

    monkeypatch.setattr(http_server, "_engine", _FakeEngine())
    monkeypatch.setattr(os, "_exit", _fake_exit)

    try:
        http_server._do_shutdown()
    except SystemExit:
        pass

    # The model must be unloaded BEFORE the process exits (clean native teardown).
    assert order and order[0] == "unload"
    assert order[-1] == ("exit", 0)
