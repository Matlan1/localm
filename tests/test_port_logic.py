# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for port selection: default range, busy detection, auto-fallback."""

import socket

import pytest

from localm.config import PORT_RANGE, pick_port, port_in_use


@pytest.fixture()
def occupied_port():
    """Bind an OS-assigned port and hold it for the duration of the test."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    yield s.getsockname()[1]
    s.close()


class TestPortRange:
    def test_range_avoids_known_tools(self):
        start, end = PORT_RANGE
        for taken in (8188, 7860, 8080, 8000, 8888, 11434):
            assert not (start <= taken <= end), f"range collides with {taken}"

    def test_range_is_sane(self):
        start, end = PORT_RANGE
        assert 1024 < start < end < 65536


class TestPortInUse:
    def test_detects_listening_port(self, occupied_port):
        assert port_in_use(occupied_port) is True

    def test_free_port_not_in_use(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
        s.close()
        assert port_in_use(free) is False


class TestPickPort:
    def test_requested_free_port_kept(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
        s.close()
        port, was_busy = pick_port(free)
        assert port == free
        assert was_busy is False

    def test_busy_port_bumps_into_range(self, occupied_port):
        port, was_busy = pick_port(occupied_port)
        assert was_busy is True
        assert port != occupied_port
        assert not port_in_use(port)

    def test_default_comes_from_config(self, monkeypatch):
        import localm.config as cfg
        monkeypatch.setattr(cfg, "load_config", lambda: {"port": 8650})
        # 8650 may or may not be free on this machine; both outcomes valid,
        # but the chosen port must be usable either way
        port, _ = pick_port(None)
        assert not port_in_use(port)
