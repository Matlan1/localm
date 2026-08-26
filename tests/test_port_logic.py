# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for port selection: default range, busy detection, auto-fallback."""

import socket

import pytest

from localm.config import PORT_RANGE, PortInUseError, pick_port, port_in_use


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

    def test_busy_explicit_port_refuses(self, occupied_port):
        # An explicit port that is busy is refused, never silently relocated
        # onto a different (often the shared default) port.
        with pytest.raises(PortInUseError) as exc:
            pick_port(occupied_port)
        assert exc.value.port == occupied_port

    def test_busy_out_of_range_explicit_port_refuses(self):
        # An explicit port OUTSIDE localm's range, chosen to avoid a collision, is
        # not relocated back onto the default 8642; it refuses instead.
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy = s.getsockname()[1]
        try:
            assert not (PORT_RANGE[0] <= busy <= PORT_RANGE[1]), (
                "OS-assigned ephemeral port unexpectedly landed inside localm's "
                "range; test would not exercise the out-of-range path")
            with pytest.raises(PortInUseError) as exc:
                pick_port(busy)
            assert exc.value.port == busy
        finally:
            s.close()

    def test_default_port_auto_bumps_when_busy(self, monkeypatch):
        # No explicit port: the configured default auto-bumps to a free port
        # when busy.
        import localm.config as cfg
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy_default = s.getsockname()[1]
        try:
            monkeypatch.setattr(cfg, "load_config", lambda: {"port": busy_default})
            port, was_busy = pick_port(None)
            assert was_busy is True
            assert port != busy_default
            assert not port_in_use(port)
        finally:
            s.close()

    def test_default_comes_from_config(self, monkeypatch):
        import localm.config as cfg
        monkeypatch.setattr(cfg, "load_config", lambda: {"port": 8650})
        # 8650 may or may not be free on this machine; both outcomes valid,
        # but the chosen port must be usable either way
        port, _ = pick_port(None)
        assert not port_in_use(port)
