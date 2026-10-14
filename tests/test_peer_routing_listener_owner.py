# SPDX-License-Identifier: AGPL-3.0-or-later
"""Peer routing sends a chat or a credential only to a port held by this OS user.
A process run by another account that answers ``/whoami`` as localm is never
offered, never sent the peer key, and a route to its port is refused."""

from __future__ import annotations

import socket
import sys

import pytest

from localm import listeners, peer_routing


@pytest.fixture(autouse=True)
def _no_cached_owner_answers():
    listeners._owned_cache.clear()
    yield
    listeners._owned_cache.clear()


def test_a_true_answer_is_reused_briefly_and_a_false_one_never(monkeypatch):
    calls = []
    answer = {"v": True}

    def counted(port, dial):
        calls.append(port)
        return answer["v"]
    monkeypatch.setattr(listeners, "_port_owned_by_current_user", counted)
    assert listeners.port_owned_by_current_user(9001) is True
    assert listeners.port_owned_by_current_user(9001) is True
    assert calls == [9001]
    answer["v"] = False
    assert listeners.port_owned_by_current_user(9002) is False
    assert listeners.port_owned_by_current_user(9002) is False
    assert calls == [9001, 9002, 9002]
    monkeypatch.setattr(listeners, "OWNED_CACHE_TTL_S", 0.0)
    listeners._owned_cache.clear()
    answer["v"] = True
    listeners.port_owned_by_current_user(9003)
    answer["v"] = False
    assert listeners.port_owned_by_current_user(9003) is False


@pytest.fixture
def own_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    try:
        yield s.getsockname()[1]
    finally:
        s.close()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_a_socket_this_process_listens_on_is_ours(own_port):
    assert listeners.port_owned_by_current_user(own_port) is True
    assert peer_routing.peer_listener_is_ours("127.0.0.1", own_port) is True


def test_a_port_nobody_listens_on_is_not_ours():
    assert listeners.port_owned_by_current_user(_free_port()) is False


@pytest.mark.parametrize("port", [None, True, "x", 0, 70000, -1, 3.5])
def test_a_malformed_port_is_not_ours(port):
    assert peer_routing.peer_listener_is_ours("127.0.0.1", port) is False


def test_an_unreadable_socket_table_is_not_ours(monkeypatch, own_port):
    for name in ("_windows_listener_owners", "_linux_listener_owners",
                 "_macos_listener_owners"):
        monkeypatch.setattr(listeners, name, lambda: None)
    assert listeners.port_owned_by_current_user(own_port) is False


def _foreign(monkeypatch, port):
    """The socket table reports *port* as held by an owner that is not this
    user: another uid on POSIX, a process of another account on Windows."""
    rows = [("127.0.0.1", port, 0 if sys.platform != "win32" else 4)]
    for name in ("_windows_listener_owners", "_linux_listener_owners",
                 "_macos_listener_owners"):
        monkeypatch.setattr(listeners, name, lambda rows=rows: rows)
    if sys.platform != "win32":
        monkeypatch.setattr(listeners.os, "getuid", lambda: 4242, raising=False)
    else:
        monkeypatch.setattr(listeners, "_windows_pid_is_current_user", lambda pid: False)


def test_a_wildcard_listener_of_another_user_on_the_port_is_not_ours(monkeypatch):
    rows = [("127.0.0.1", 9, 1), ("0.0.0.0", 9, 2)]
    for name in ("_windows_listener_owners", "_linux_listener_owners",
                 "_macos_listener_owners"):
        monkeypatch.setattr(listeners, name, lambda: rows)
    monkeypatch.setattr(listeners.os, "getuid", lambda: 1, raising=False)
    monkeypatch.setattr(listeners, "_windows_pid_is_current_user", lambda pid: pid == 1)
    assert listeners.port_owned_by_current_user(9) is False


def test_a_foreign_peer_is_never_offered(monkeypatch):
    from localm import gpu_registry
    _foreign(monkeypatch, 8642)
    monkeypatch.setattr(gpu_registry, "list_gpu_peers", lambda **kw: [{
        "instance_id": "impostor", "pid": 999999, "port": 8642, "host": "127.0.0.1",
        "scheme": "http", "model": "m", "models": [{"name": "m"}]}])
    assert peer_routing.find_offer("m", ()) is None


def test_a_foreign_peer_is_never_sent_the_key(monkeypatch):
    import requests
    _foreign(monkeypatch, 8642)
    sent = []
    monkeypatch.setattr(requests, "get", lambda *a, **kw: sent.append(a))
    with pytest.raises(peer_routing.PeerCredentialError, match="not run by"):
        peer_routing.verify_peer_credential(
            {"instance_id": "impostor", "host": "127.0.0.1", "port": 8642,
             "scheme": "http"}, "the-real-peers-key")
    assert sent == []


@pytest.mark.parametrize("mode", ["full", "log", "privacy"])
def test_a_route_to_a_foreign_port_is_refused_in_every_mode(monkeypatch, mode):
    import requests
    _foreign(monkeypatch, 8642)
    sent = []
    monkeypatch.setattr(requests, "get", lambda *a, **kw: sent.append(a))
    with pytest.raises(peer_routing.PeerUnreachable, match="not run by"):
        peer_routing.ensure_peer_keeps_mode(
            {"instance_id": "impostor", "host": "127.0.0.1", "port": 8642,
             "scheme": "http"}, "the-real-peers-key", mode)
    assert sent == []


def test_forward_to_a_foreign_port_clears_the_route_and_sends_nothing(monkeypatch):
    import asyncio

    import requests
    from fastapi import HTTPException
    _foreign(monkeypatch, 8642)
    sent = []
    monkeypatch.setattr(requests, "post", lambda *a, **kw: sent.append(a))
    monkeypatch.setattr(requests, "get", lambda *a, **kw: sent.append(a))
    route = peer_routing.PeerRoute(model="m", instance_id="impostor", host="127.0.0.1",
                                   port=8642, scheme="http", api_key="the-real-peers-key")
    peer_routing.set_route(route)

    class _Req:
        headers: dict = {}

        async def body(self):
            return b"{}"

    try:
        with pytest.raises(HTTPException) as ei:
            asyncio.run(peer_routing.forward(route, _Req(), "/v1/chat/completions"))
        assert ei.value.status_code == 502
        assert peer_routing.get_route("m") is None
        assert sent == []
    finally:
        peer_routing.clear_route("m")
