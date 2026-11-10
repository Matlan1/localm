# SPDX-License-Identifier: AGPL-3.0-or-later
"""portmux.stopping(): an endless response (the live stats stream) has to see
that the server was asked to stop, because uvicorn leaves an in-progress response
running for its whole graceful shutdown period and then logs an error."""

from __future__ import annotations

import ast
import inspect

from localm import portmux


class _Server:
    """A stand-in for uvicorn.Server (a SimpleNamespace cannot be weakly referenced)."""

    def __init__(self, should_exit):
        self.should_exit = should_exit


def test_stopping_follows_a_registered_servers_should_exit(monkeypatch):
    monkeypatch.setattr(portmux, "_servers", portmux.weakref.WeakSet())
    assert portmux.stopping() is False
    server = _Server(False)
    assert portmux._register_server(server) is server
    assert portmux.stopping() is False
    server.should_exit = True
    assert portmux.stopping() is True


def test_a_server_that_is_gone_no_longer_counts(monkeypatch):
    import gc
    monkeypatch.setattr(portmux, "_servers", portmux.weakref.WeakSet())
    server = _Server(True)
    portmux._register_server(server)
    assert portmux.stopping() is True
    del server
    gc.collect()
    assert portmux.stopping() is False


def test_every_uvicorn_server_portmux_builds_is_registered():
    tree = ast.parse(inspect.getsource(portmux))
    made, registered = 0, 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "uvicorn.Server":
            made += 1
        if (isinstance(node, ast.Call) and ast.unparse(node.func) == "_register_server"
                and node.args and isinstance(node.args[0], ast.Call)
                and ast.unparse(node.args[0].func) == "uvicorn.Server"):
            registered += 1
    assert made >= 4, "the sentinel found too few uvicorn.Server calls"
    assert registered == made, f"{made - registered} uvicorn.Server call(s) are not registered"


def test_a_server_that_finished_serving_stops_counting(monkeypatch):
    """A stale server of an earlier run in the same process must not end the
    streams of the one now serving."""
    monkeypatch.setattr(portmux, "_servers", portmux.weakref.WeakSet())
    monkeypatch.setattr(portmux, "_stop_hooks", [])
    monkeypatch.setattr(portmux, "_stop_requested", False)

    class Done:
        def __init__(self):
            self.callbacks = []

        def add_done_callback(self, cb):
            self.callbacks.append(cb)

    server = _Server(True)
    portmux._register_server(server)
    task = Done()
    portmux._track_server(server, task)
    assert portmux.stopping() is True
    for cb in task.callbacks:
        cb(task)
    assert portmux.stopping() is False
