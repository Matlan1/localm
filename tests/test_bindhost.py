# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.bindhost holds the single loopback-host predicate. Every site that once
defined its own must re-export the SAME function object, not a copy that can
drift."""

import pytest

from localm.bindhost import is_loopback_host


@pytest.mark.parametrize("host,loop", [
    ("127.0.0.1", True), ("::1", True), ("localhost", True),
    ("127.0.0.5", True), ("10.0.0.7", False), ("192.168.1.4", False),
    ("0.0.0.0", False), ("testclient", False), ("", False),
])
def test_is_loopback_host(host, loop):
    assert is_loopback_host(host) is loop


def test_all_former_call_sites_reexport_the_same_function():
    import localm.inference.http_server as http_server
    import localm.inference.routes.keys as keys_routes
    import localm.plugins.deps_task as deps_task
    import localm.plugins.gui.web as gui_web

    assert http_server._is_loopback_host is is_loopback_host
    assert keys_routes._is_loopback is is_loopback_host
    assert deps_task.is_loopback_host is is_loopback_host
    assert gui_web._is_loopback_host is is_loopback_host


def test_system_route_uses_shared_predicate_not_a_drifted_inline_copy():
    # whoami() calls the shared helper for a loopback-shaped host that only the
    # ipaddress-based check recognizes.
    from localm.inference.routes import system as system_routes
    assert system_routes.is_loopback_host is is_loopback_host
    assert system_routes.is_loopback_host("127.0.0.5") is True


def test_interface_addresses_skip_an_address_that_does_not_parse(monkeypatch):
    import ipaddress
    import socket
    from types import SimpleNamespace

    import psutil

    from localm import bindhost
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: {"eth0": [
        SimpleNamespace(family=socket.AF_INET, address="192.0.2.7"),
        SimpleNamespace(family=socket.AF_INET6, address="not-an-address"),
        SimpleNamespace(family=socket.AF_INET6, address="fe80::1%12"),
    ]})
    assert bindhost._interface_addresses() == frozenset({
        ipaddress.ip_address("192.0.2.7"), ipaddress.ip_address("fe80::1")})


def test_an_interface_address_is_cut_at_its_first_zone_separator(monkeypatch):
    import socket
    from types import SimpleNamespace

    import psutil

    from localm import bindhost
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: {"eth0": [
        SimpleNamespace(family=socket.AF_INET6, address="fe80::2%a%b"),
    ]})
    assert bindhost.is_own_address("fe80::2") is True


@pytest.mark.parametrize("bind_host,expected", [
    (None, "127.0.0.1"), ("", "127.0.0.1"), ("   ", "127.0.0.1"),
    ("0.0.0.0", "127.0.0.1"), ("localhost", "127.0.0.1"), (" localhost ", "127.0.0.1"),
    ("::", "::1"), ("::1", "::1"), ("192.0.2.5", "192.0.2.5"), (" 192.0.2.5 ", "192.0.2.5"),
])
def test_self_connect_host_maps_a_wildcard_to_loopback_and_keeps_a_literal(bind_host, expected):
    from localm.bindhost import self_connect_host
    assert self_connect_host(bind_host) == expected


@pytest.mark.parametrize("host", [None, "", 12345, b"::1", "::1%1", "127.0.0.1%lo", "not-an-ip"])
def test_is_own_address_refuses_a_non_string_a_zone_id_and_a_non_literal(host, monkeypatch):
    import psutil

    from localm import bindhost
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: {})
    assert bindhost.is_own_address(host) is False


def test_is_own_address_accepts_loopback_and_only_listed_interface_addresses(monkeypatch):
    import socket
    from types import SimpleNamespace

    import psutil

    from localm import bindhost
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: {"eth0": [
        SimpleNamespace(family=socket.AF_INET, address="192.0.2.7")]})
    assert bindhost.is_own_address("127.0.0.1") is True
    assert bindhost.is_own_address("::1") is True
    assert bindhost.is_own_address("192.0.2.7") is True
    assert bindhost.is_own_address("192.0.2.8") is False
